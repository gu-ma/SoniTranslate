from scipy.ndimage import gaussian_filter1d
from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger
import numpy as np
import pyloudnorm as pyln
from pathlib import Path



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


            start = sample_offset
            end = start + len(samples)
            output[start:end] += samples

        return seg._spawn(
            output, overrides={"sample_width": 4}).normalize(headroom=1.0)


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
            duration=int(total_duration * 1000), frame_rate=48000
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

# Loudness normalization presets for different contexts
# 💡 Practical Notes
# * LUFS (Loudness Units relative to Full Scale) ≈ perceived volume.
# * TP ensures peaks never clip, even after encoding.
# * LRA keeps dynamics natural (lower → more compressed).

LOUDNORM_PRESETS = {
    # Broadcast (EBU R128 / ITU BS.1770-4)
    # Used by European TV, radio, and most traditional broadcast systems.
    # | Parameter | Typical Value | Meaning                                                 |
    # | --------- | ------------- | ------------------------------------------------------- |
    # | `I`       | **-23 LUFS**  | Integrated loudness target (overall perceived loudness) |
    # | `TP`      | **-1 dBTP**   | Maximum true peak (prevents clipping)                   |
    # | `LRA`     | **11 LU**     | Loudness range (dynamic range over time)                |
    # ✅ Produces consistent loudness across broadcast programs.
    # ⚠️ May sound quiet compared to streaming audio because of its lower LUFS target.
    "broadcast": {
        "I": -23.0,   # Integrated loudness target (LUFS)
        "TP": -1.0,   # True peak limit (dBTP)
        "LRA": 11.0,  # Loudness range (LU)
        "desc": "EBU R128 standard for broadcast TV and radio."
    },
    # Streaming / Music (YouTube, Spotify, Netflix, etc.)
    # Most modern platforms normalize to louder standards.
    # | Platform    | Integrated (I) | True Peak (TP) | Notes                              |
    # | ----------- | -------------- | -------------- | ---------------------------------- |
    # | YouTube     | -14 LUFS       | -1.0 dBTP      | Default playback normalization     |
    # | Spotify     | -14 LUFS       | -1.0 dBTP      | “Normal” playback mode             |
    # | Netflix     | -27 LUFS       | -2.0 dBTP      | Very dynamic, cinematic mixes      |
    # | Apple Music | -16 LUFS       | -1.0 dBTP      | Apple Sound Check                  |
    # | Podcasters  | -16 LUFS       | -1.5 dBTP      | Common standard for speech clarity |
    "streaming": {
        "I": -16.0,
        "TP": -1.5,
        "LRA": 11.0,
        "desc": "Balanced loudness for online content and podcasts."
    },
    "music": {
        "I": -14.0,
        "TP": -1.0,
        "LRA": 9.0,
        "desc": "Streaming platform standard for music (Spotify, YouTube)."
    },
    # Film / Cinema (Theatrical Mixes)
    # | Parameter | Typical Value     |
    # | --------- | ----------------- |
    # | `I`       | -24 to -27 LUFS   |
    # | `TP`      | -2.0 to -3.0 dBTP |
    # | `LRA`     | 15–20 LU          |
    # Uses a much wider dynamic range — louder peaks and softer dialogue.    
    "film": {
        "I": -24.0,
        "TP": -2.0,
        "LRA": 15.0,
        "desc": "Cinematic loudness range for theatrical or film mixes."
    },
    "audio_guide": {
        "I": -18.0,    # Same as podcast / "streaming speech" level
        "TP": -1.0,    # Safe headroom, avoids crunchy limiting
        "LRA": 7.0,    # Tighter dynamics so voice stays consistently intelligible
        "desc": "Voice-forward audio guides / VO with background music ducking."
    },
}


def generate_loudnorm_filter(preset_name="audio_guide"):
    """Return a formatted FFmpeg loudnorm filter string for a given preset."""
    preset = LOUDNORM_PRESETS.get(preset_name, LOUDNORM_PRESETS["streaming"])
    return f"loudnorm=I={preset['I']}:TP={preset['TP']}:LRA={preset['LRA']}"


def generate_segment_based_volume_automation_chunked(segments, chunk_size=50, **kwargs):
    chunks = [segments[i:i+chunk_size] for i in range(0, len(segments), chunk_size)]
    filters = []
    for chunk in chunks:
        filters.append(generate_segment_based_volume_automation(chunk, **kwargs))
    if len(filters) == 1:
        return filters[0]
    return ",".join(f for f in filters if f != "anull")


def generate_segment_based_volume_automation(
    segments, duck_level=0.2, fade_duration=0.1
):
    """
    Build an FFmpeg volume expression that fades down to duck_level,
    holds at duck_level in [start, end], then fades back up.
    Returns: e.g. "volume='...expr...':eval=frame" or "anull".
    """
    if not segments or duck_level >= 1.0:
        return "anull"

    def fnum(x):
        return f"{float(x):.6f}".rstrip("0").rstrip(".")

    duck = fnum(duck_level)
    one_minus_duck = fnum(1.0 - float(duck_level))
    f = max(0.0, float(fade_duration))
    EPS = 1e-6  # treat very small starts as 0

    envelopes = []
    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        if e <= s:
            continue

        if f > 0:
            s_f = max(0.0, s - f)
            e_f = e + f
            pre_dur = s - s_f  # in [0, f]

            if pre_dur > EPS:
                # Continuous ramp to duck exactly at t = s
                env = (
                    f"if(lt(t,{fnum(s_f)}),1,"
                    f" if(lt(t,{fnum(s)}), 1-({one_minus_duck})*((t-{fnum(s_f)})/{fnum(pre_dur)}),"
                    f"  if(lt(t,{fnum(e)}), {duck},"
                    f"   if(lt(t,{fnum(e_f)}), {duck}+({one_minus_duck})*((t-{fnum(e)})/{fnum(f)}),"
                    f"    1))))"
                )
            else:
                # No pre-fade possible (start ~0 or earlier than fade window)
                env = (
                    f"if(lt(t,{fnum(e)}), {duck},"
                    f" if(lt(t,{fnum(e_f)}), {duck}+({one_minus_duck})*((t-{fnum(e)})/{fnum(f)}),"
                    f"  1))"
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

    return f"volume='{overall}':eval=frame"


def mix_audio_volume(
    base_audio_wav,
    dub_audio_file,
    mix_audio_file,
    volume_original_audio,
    volume_translated_audio,
    loudnorm_preset="broadcast",
    enable_loudnorm=False,
    volume_boost_db=0.0,
    limiter=0.95,
):
    """
    Simple volume mixing without ducking.
    
    Args:
        loudnorm_preset: Loudness normalization preset
        enable_loudnorm: Enable/disable loudness normalization
        volume_boost_db: Additional volume boost in dB (e.g., 2.0 for +2dB)
        limiter: Loudness limiter    
    """

    # Optional loudnorm filter
    loudnorm_filter = generate_loudnorm_filter(loudnorm_preset) if enable_loudnorm else ""
    volume_boost_str = f"{volume_boost_db:+}dB"

    filter_parts = [
        f"[0:a]volume={volume_original_audio}[bg];"
        f"[1:a]volume={volume_translated_audio}[dub]",
        "[bg][dub]amix=inputs=2:normalize=0:duration=longest",
    ]

    if enable_loudnorm:
        filter_parts.append(loudnorm_filter)

    filter_parts.append(f"alimiter=limit={limiter}")
    filter_parts.append(f"volume={volume_boost_str}[final]")

    filter_graph = ",".join(filter_parts)

    # Choose codec based on output extension
    out_ext = Path(str(mix_audio_file)).suffix.lower()
    if out_ext in (".wav", ".wave"):
        codec_args = ["-c:a", "pcm_s16le"]
    else:
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]

    command_volume_mix = [
        "ffmpeg", "-y",
        "-i", str(base_audio_wav),
        "-i", str(dub_audio_file),
        "-filter_complex", filter_graph,
        "-map", "[final]",
        "-ar", "48000",
        "-ac", "2",
        *codec_args,
        str(mix_audio_file),
    ]

    run_command(command_volume_mix)


def mix_audio_segment_ducking(
    base_audio_wav,
    dub_audio_file,
    mix_audio_file,
    volume_translated_audio,
    volume_automation,
    loudnorm_preset="broadcast",
    enable_loudnorm=False,
    volume_boost_db=0.0,
    limiter=0.95,
):
    """
    Segment-based ducking using precise timing from segments with balanced output.
    
    Args:
        loudnorm_preset: Loudness normalization preset
        enable_loudnorm: Enable/disable loudness normalization
        volume_boost_db: Additional volume boost in dB (e.g., 2.0 for +2dB)
        limiter: Loudness limiter
    """

    # Optional loudnorm filter
    loudnorm_filter = generate_loudnorm_filter(loudnorm_preset) if enable_loudnorm else ""
    volume_boost_str = f"{volume_boost_db:+}dB"

    # Build the FFmpeg filter
    filter_parts = [
        f"[0:a]{volume_automation}[bg]",
        f"[1:a]volume={volume_translated_audio}[dub]",
        "[bg][dub]amix=inputs=2:normalize=0:duration=longest",
    ]

    if enable_loudnorm:
        filter_parts.append(loudnorm_filter)

    filter_parts.append(f"alimiter=limit={limiter}")
    filter_parts.append(f"volume={volume_boost_str}[final]")

    filter_graph = ",".join(filter_parts)
    logger.debug(filter_graph)

    # filter_graph = (
    #     f"[0:a]{volume_automation}[bg_ducked];"
    #     f"[1:a]volume={volume_translated_audio}[dub_v];"
    #     "[bg_ducked][dub_v]amix=inputs=2:duration=longest,"
    #     # f"{loudnorm_filter},"
    #     "alimiter=limit=0.95,"
    #     "volume=+2dB[final]"
    # )

    # Choose codec based on output extension
    out_ext = Path(str(mix_audio_file)).suffix.lower()
    if out_ext in (".wav", ".wave"):
        codec_args = ["-c:a", "pcm_s16le"]
    else:
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]

    command_segment_mix = [
        "ffmpeg", "-y",
        "-i", str(base_audio_wav),
        "-i", str(dub_audio_file),
        "-filter_complex", filter_graph,
        "-map", "[final]",
        "-ar", "48000", "-ac", "2",
        *codec_args,
        str(mix_audio_file),
    ]

    run_command(command_segment_mix)


def mix_audio_sidechain(
    base_audio_wav,
    dub_audio_file,
    mix_audio_file,
    volume_original_audio,
    volume_translated_audio,
    loudnorm_preset="broadcast",
    enable_loudnorm=False,
    volume_boost_db=0.0,
    limiter=0.95,
):
    """
    Sidechain compression ducking with smooth transitions, matching the structure of
    mix_audio_segment_ducking() with preset-based loudnorm, boost, limiter, and
    unified filter-building style.

    Args:
        loudnorm_preset: Loudness normalization preset
        enable_loudnorm: Enable/disable loudness normalization
        volume_boost_db: Additional volume boost in dB (e.g., 2.0 for +2dB)
        limiter: Loudness limiter
    """

    # Optional loudnorm filter
    loudnorm_filter = generate_loudnorm_filter(loudnorm_preset) if enable_loudnorm else ""
    volume_boost_str = f"{volume_boost_db:+}dB"

    # Sidechain parameters
    target_bg_when_vo = volume_original_audio  # ~BG level during VO
    sc_mix = 1 - target_bg_when_vo  # 0.8 ≈ 20% BG during VO

    # compressor tuning (adjust to taste)
    threshold = 0.001  # ~ -26 dBFS; lower => ducks more readily
    ratio = 20  # heavy ducking
    attack_ms = 10
    release_ms = 300
    level_sc = 1.0  # raise if ducking feels too weak
    makeup = 1.3  # balanced makeup gain

    # Build the FFmpeg filter
    filter_parts = [
        "[0:a]anull[bg_src]",  # Background source (anull for safety)
        f"[1:a]apad,volume={volume_translated_audio}[dub_v]",  # Dub/VO source with padding and volume
        "[dub_v]asplit=2[sc][mix_src]",  # Duplicate dub to feed sidechain + mix
        # Sidechain compressor
        (
            f"[bg_src][sc]sidechaincompress="
            f"threshold={threshold}:ratio={ratio}:attack={attack_ms}:release={release_ms}:"
            f"level_sc={level_sc}:mix={sc_mix}:makeup={makeup}[bg_duck]"
        ),
        
        "[bg_duck][mix_src]amix=inputs=2:normalize=0:duration=first:dropout_transition=0",  # Mix ducked BG with dub
    ]

    if enable_loudnorm:
        filter_parts.append(loudnorm_filter)

    filter_parts.append(f"alimiter=limit={limiter}")
    filter_parts.append(f"volume={volume_boost_str}[final]")

    filter_graph = ",".join(filter_parts)

    # Choose codec based on output extension
    out_ext = Path(str(mix_audio_file)).suffix.lower()
    if out_ext in (".wav", ".wave"):
        codec_args = ["-c:a", "pcm_s16le"]
    else:
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]

    command_sidechain_mix = [
        "ffmpeg", "-y",
        "-i", str(base_audio_wav),
        "-i", str(dub_audio_file),
        "-filter_complex", filter_graph,
        "-map", "[final]",
        "-ar", "48000", "-ac", "2",
        *codec_args,
        str(mix_audio_file),
    ]

    run_command(command_sidechain_mix)


# Convert AudioSegment → float32 numpy
def segment_to_float32(seg):
    samples = np.array(seg.get_array_of_samples())

    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels))

    # correct PCM scaling
    if seg.sample_width == 1:
        samples = (samples.astype(np.float32) - 128) / 128.0
    elif seg.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif seg.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0

    return samples.astype(np.float32)


# Convert numpy back → AudioSegment
def float32_to_segment(float_data, frame_rate, channels, sample_width):
    # Undo scaling (back to PCM)
    if sample_width == 2:
        pcm = np.int16(np.clip(float_data * 32767, -32768, 32767))
    elif sample_width == 4:
        pcm = np.int32(np.clip(float_data * 2147483647, -2147483648, 2147483647))
    elif sample_width == 1:
        pcm = np.uint8(np.clip((float_data * 128) + 128, 0, 255))

    if channels > 1:
        pcm = pcm.reshape(-1)

    return AudioSegment(
        pcm.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels
    )


def match_loudness_LUFS(ref_file, tgt_file, output_file, manual_offset_db=0.0):
    ref_seg = AudioSegment.from_file(ref_file)
    tgt_seg = AudioSegment.from_file(tgt_file)

    # Convert to float32
    ref_np = segment_to_float32(ref_seg)
    tgt_np = segment_to_float32(tgt_seg)

    meter = pyln.Meter(ref_seg.frame_rate)

    # Original loudness
    ref_lufs = meter.integrated_loudness(ref_np)
    tgt_lufs = meter.integrated_loudness(tgt_np)

    # Gain needed + manual correction
    gain_db = (ref_lufs - tgt_lufs) + manual_offset_db
    gain_linear = 10 ** (gain_db / 20)

    tgt_np_adjusted = tgt_np * gain_linear

    matched = float32_to_segment(
        tgt_np_adjusted,
        frame_rate=tgt_seg.frame_rate,
        channels=tgt_seg.channels,
        sample_width=tgt_seg.sample_width,
    )

    matched.export(output_file, format="mp3", bitrate="192k")
