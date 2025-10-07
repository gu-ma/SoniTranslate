from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger
import numpy as np


class Mixer:
    def __init__(self):
        self.parts = []

    def __len__(self):
        parts = self._sync()
        seg = parts[0][1]
        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        return int(1000.0 * frame_count / seg.frame_rate)

    def overlay(self, sound, position=0):
        self.parts.append((position, sound))
        return self

    def _sync(self):
        positions, segs = zip(*self.parts)

        frame_rate = segs[0].frame_rate
        array_type = segs[0].array_type # noqa

        offsets = [int(frame_rate * pos / 1000.0) for pos in positions]
        segs = AudioSegment.empty()._sync(*segs)
        return list(zip(offsets, segs))

    def append(self, sound):
        self.overlay(sound, position=len(self))

    def to_audio_segment(self):
        parts = self._sync()
        seg = parts[0][1]
        channels = seg.channels

        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        sample_count = int(frame_count * seg.channels)

        output = np.zeros(sample_count, dtype="int32")
        for offset, seg in parts:
            sample_offset = offset * channels

            # samples = np.frombuffer(seg.get_array_of_samples(), dtype="int32")
            # samples = np.int16(samples/np.max(np.abs(samples)) * 32767)

            sample_width = seg.sample_width
            if sample_width == 1:
                dtype = np.int8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            samples = np.frombuffer(seg.get_array_of_samples(), dtype=dtype)
            samples = samples.astype(np.int32)

            samples = np.int16(samples/np.max(np.abs(samples)) * 32767)

            start = sample_offset
            end = start + len(samples)
            output[start:end] += samples

        return seg._spawn(
            output, overrides={"sample_width": 4}).normalize(headroom=0.0)


def create_translated_audio(
    result_diarize, audio_files, final_file, concat=False, avoid_overlap=False,
):
    total_duration = result_diarize["segments"][-1]["end"]  # in seconds

    if concat:
        """
        file .\audio\1.ogg
        file .\audio\2.ogg
        file .\audio\3.ogg
        file .\audio\4.ogg
        ...
        """

        # Write the file paths to list.txt
        with open("list.txt", "w") as file:
            for i, audio_file in enumerate(audio_files):
                if i == len(audio_files) - 1:  # Check if it's the last item
                    file.write(f"file {audio_file}")
                else:
                    file.write(f"file {audio_file}\n")

        # command = f"ffmpeg -f concat -safe 0 -i list.txt {final_file}"
        command = (
            f"ffmpeg -f concat -safe 0 -i list.txt -c:a pcm_s16le {final_file}"
        )
        run_command(command)

    else:
        # silent audio with total_duration
        base_audio = AudioSegment.silent(
            duration=int(total_duration * 1000), frame_rate=41000
        )
        combined_audio = Mixer()
        combined_audio.overlay(base_audio)

        logger.debug(
            f"Audio duration: {total_duration // 60} "
            f"minutes and {int(total_duration % 60)} seconds"
        )

        last_end_time = 0
        previous_speaker = ""
        for line, audio_file in tqdm(
            zip(result_diarize["segments"], audio_files)
        ):
            start = float(line["start"])

            # Overlay each audio at the corresponding time
            try:
                audio = AudioSegment.from_file(audio_file)
                # audio_a = audio.speedup(playback_speed=1.5)

                if avoid_overlap:
                    speaker = line["speaker"]
                    if (last_end_time - 0.500) > start:
                        overlap_time = last_end_time - start
                        if previous_speaker and previous_speaker != speaker:
                            start = (last_end_time - 0.500)
                        else:
                            start = (last_end_time - 0.200)
                        if overlap_time > 2.5:
                            start = start - 0.3
                        logger.info(
                              f"Avoid overlap for {str(audio_file)} "
                              f"with {str(start)}"
                        )

                    previous_speaker = speaker

                    duration_tts_seconds = len(audio) / 1000.0  # to sec
                    last_end_time = (start + duration_tts_seconds)

                start_time = start * 1000  # to ms
                combined_audio = combined_audio.overlay(
                    audio, position=start_time
                )
            except Exception as error:
                logger.debug(str(error))
                logger.error(f"Error audio file {audio_file}")

        # combined audio as a file
        combined_audio_data = combined_audio.to_audio_segment()
        combined_audio_data.export(
            final_file, format="wav"
        )  # best than ogg, change if the audio is anomalous


# def generate_segment_based_volume_automation(
#     segments, duck_level=0.2, fade_duration=0.1
# ):
#     """
#     Generate FFmpeg volume filter string for segment-based ducking.

#     Args:
#         segments: List of segment dictionaries with 'start' and 'end' times
#         duck_level: Volume level during ducking (0.0 to 1.0)
#         fade_duration: Fade in/out duration in seconds

#     Returns:
#         String: FFmpeg volume filter chain
#     """
#     filters = []

#     for segment in segments:
#         start_time = max(0, segment["start"] - fade_duration)
#         end_time = segment["end"] + fade_duration

#         # Create volume filter for this segment
#         filter_str = (
#             f"volume=enable='between(t,{start_time},{end_time})':volume={duck_level}"
#         )
#         filters.append(filter_str)

#     return ",".join(filters) if filters else "anull"

def generate_segment_based_volume_automation(
    segments, duck_level=0.2, fade_duration=0.1
):
    """
    Build an FFmpeg volume expression that fades down to duck_level,
    holds at duck_level in [start, end], then fades back up.
    Returns: e.g. "volume=volume='...expr...':eval=frame" or "anull".
    """
    if not segments or duck_level >= 1.0:
        return "anull"

    def fnum(x):
        return f"{float(x):.6f}".rstrip("0").rstrip(".")

    duck = fnum(duck_level)
    one_minus_duck = fnum(1.0 - float(duck_level))
    f = max(0.0, float(fade_duration))

    envelopes = []
    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        if e <= s:
            continue

        if f > 0:
            s_f = max(0.0, s - f)
            e_f = e + f
            # Piecewise linear envelope:
            # < s_f -> 1
            # [s_f, s) -> ramp 1 -> duck
            # [s, e) -> duck
            # [e, e_f) -> ramp duck -> 1
            # >= e_f -> 1
            env = (
                f"if(lt(t,{fnum(s_f)}),1,"
                f" if(lt(t,{fnum(s)}), 1-({one_minus_duck})*((t-{fnum(s_f)})/{fnum(f)}),"
                f"  if(lt(t,{fnum(e)}), {duck},"
                f"   if(lt(t,{fnum(e_f)}), {duck}+({one_minus_duck})*((t-{fnum(e)})/{fnum(f)}),"
                f"    1))))"
            )
        else:
            # No fade: exactly duck_level in [s, e), 1 elsewhere
            env = f"1-({one_minus_duck})*between(t,{fnum(s)},{fnum(e)})"

        envelopes.append(env)

    if not envelopes:
        return "anull"

    # Use the minimum envelope so overlaps don’t multiply attenuation
    overall = envelopes[0]
    for env in envelopes[1:]:
        overall = f"min({overall},{env})"

    return f"volume=volume='{overall}':eval=frame"



def mix_audio_volume(
    base_audio_wav,
    dub_audio_file,
    mix_audio_file,
    volume_original_audio,
    volume_translated_audio,

):
    """Simple volume mixing without ducking."""

    filter_graph = (
        f"[0:a]volume={volume_original_audio}[a];"
        f"[1:a]volume={volume_translated_audio}[b];"
        " [a][b]amix=inputs=2:duration=longest"
    )

    command_volume_mix = [
        "ffmpeg", "-y",
        "-i", str(base_audio_wav),
        "-i", str(dub_audio_file),
        "-filter_complex", filter_graph,
        "-c:a", "libmp3lame",
        str(mix_audio_file),
    ]

    run_command(command_volume_mix)


def mix_audio_segment_ducking(
    base_audio_wav,
    dub_audio_file,
    mix_audio_file,
    volume_original_audio,
    volume_translated_audio,
    volume_automation
):
    """Segment-based ducking using precise timing from segments."""

    # Create filter graph with segment-based ducking
    filter_graph = (
        f"[0:a]{volume_automation}[bg_ducked];"
        f"[1:a]volume={volume_translated_audio}[dub_v];"
        "[bg_ducked][dub_v]amix=inputs=2:duration=longest,"
        "alimiter=limit=0.98,volume=2.0[final]"
    )

    command_segment_mix = [
        "ffmpeg", "-y",
        "-i", str(base_audio_wav),
        "-i", str(dub_audio_file),
        "-filter_complex", filter_graph,
        "-map", "[final]",
        "-ar", "48000", "-ac", "2",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(mix_audio_file),
    ]

    run_command(command_segment_mix)


def mix_audio_sidechain(
    base_audio_wav,
    dub_audio_file,
    mix_audio_file,
    volume_original_audio,
    volume_translated_audio,
):
    """Sidechain compression ducking with smooth fade transitions."""
    # targets
    target_bg_when_vo = volume_original_audio  # ~BG level during VO
    sc_mix = 1 - target_bg_when_vo  # 0.8 ≈ 20% BG during VO

    # compressor tuning (adjust to taste)
    threshold = 0.001  # ~ -26 dBFS; lower => ducks more readily
    ratio = 20  # heavy ducking
    attack_ms = 10
    release_ms = 300
    level_sc = 1.0  # raise if ducking feels too weak
    makeup = 2.0  # avoid auto gain-up of the BG

    filter_graph = (
        "[0:a]anull[bg_src];"
        f"[1:a]apad,volume={volume_translated_audio}[dub_v];"
        "[dub_v]asplit=2[sc][mix_src];"
        f"[bg_src][sc]sidechaincompress="
        f"threshold={threshold}:ratio={ratio}:attack={attack_ms}:release={release_ms}:"
        f"level_sc={level_sc}:mix={sc_mix}:makeup={makeup}[bg_duck];"
        "[bg_duck][mix_src]amix=inputs=2:duration=first:dropout_transition=0,"
        "alimiter=limit=0.98[final]"
    )

    command_background_mix = [
        "ffmpeg","-y",
        "-i", str(base_audio_wav),
        "-i", str(dub_audio_file),
        "-filter_complex", filter_graph,
        "-map", "[final]", "-shortest",
        "-ar","48000","-ac","2",
        "-c:a","libmp3lame","-b:a","192k",
        str(mix_audio_file),
    ]

    run_command(command_background_mix)
