import os
from datetime import datetime
from pydub import AudioSegment
from collections import Counter

from .logging_setup import logger
from .preprocessor import audio_video_preprocessor, audio_preprocessor
from .text_to_speech import (
    audio_segmentation_to_voice,
    accelerate_segments,
)
from .audio_segments import (
    mix_audio_volume,
    mix_audio_segment_ducking,
    mix_audio_sidechain,
    generate_segment_based_volume_automation_chunked,
    create_translated_audio,
    match_loudness_LUFS
)
from .utils import is_audio_file, load_state, save_state, remove_files, get_hash
from .llm_fit import select_segments_within_budget


def synthesize_and_mix(
    state_path: str,
    output_dir: str,
    output_file: str,
    config_kwargs: dict,
    config: dict,
    mapped_speakers_count: int,
):
    """
    Phase 3 — Synthesize & Mix

    Loads the Phase 2 JSON, performs TTS synthesis, applies optional
    acceleration and overlap handling, then mixes the translated track
    with the base audio if provided.

    Steps:
        1. Load Phase 2 state
        2. Preprocess audio
        3. Generate TTS voices
        4. Accelerate segments (optional)
        5. Create translated audio
        6. Mix with base audio
        7. Save Phase 3 JSON
    """

    # ------------------------------------------------------------------
    # Config extraction
    # ------------------------------------------------------------------

    # TTS config
    # Collect all tts_voiceXX keys
    tts_kwargs = {k: v for k, v in config_kwargs.items() if k.startswith("tts_voice")}
    # Ensure default empty strings for missing voices (up to tts_voice11)
    for i in range(12):
        tts_kwargs.setdefault(f"tts_voice{i:02}", "")
    max_accelerate_audio = float(config_kwargs.get("max_accelerate_audio", 1.2))
    min_accelerate_audio = float(config_kwargs.get("min_accelerate_audio", 1.0))
    acceleration_regulate = bool(
        config_kwargs.get("acceleration_rate_regulation", False)
    )
    avoid_overlap = bool(config_kwargs.get("avoid_overlap", False))

    # Mix config
    mix_method = config_kwargs.get(
        "mix_method_audio", "Segment-based ducking (precise timing)"
    )
    vol_original = float(config_kwargs.get("volume_original_audio", 0.15))
    vol_translated = float(config_kwargs.get("volume_translated_audio", 1.10))
    manual_offset_db = float(config_kwargs.get("manual_offset_db", 0.0))
    temp_dir = config_kwargs.get("temp_dir", "temp")


    # ------------------------------------------------------------------
    # Load previous phase
    # ------------------------------------------------------------------

    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Phase 2 state not found: {state_path}")

    state_phase2 = load_state(state_path)

    media_file = state_phase2.get("media_file", "")
    media_hash = state_phase2.get("media_hash", "")
    segments = state_phase2.get("segments", [])
    target_language = state_phase2.get("target_language", "en")

    logger.info(f"🏁 Starting Phase 3: Synthesis and Mix | {media_file} (hash {media_hash[:10]}...)")
    logger.info(f"Loaded {len(segments)} segments for synthesis.")


    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    # Filenames and paths (save permanent copies as well ??)
    base_audio_wav = os.path.join(temp_dir, "phase3_audio.wav")
    base_video_file = os.path.join(temp_dir, "phase3_video.mp4")
    dub_audio_file = os.path.join(temp_dir, "phase3_audio_dub.wav")
    mix_audio_file = os.path.join(temp_dir, "phase3_audio_mix.wav")
    # Final audio
    base_name = os.path.splitext(os.path.basename(media_file))[0]
    final_audio_filename = f"{base_name}_{target_language.upper()}.mp3"
    final_audio_file = os.path.join(output_dir, final_audio_filename)

    if not os.path.exists(media_file):
        raise FileNotFoundError(f"Input media not found: {media_file}")

    if is_audio_file(media_file):
        logger.info("[1] Extracting audio...")
        audio_preprocessor(False, media_file, base_audio_wav)
    else:
        logger.info("[1] Extracting audio and video streams...")
        audio_video_preprocessor(
            preview=False,
            video=media_file,
            OutputFile=base_video_file,
            audio_wav=base_audio_wav,
        )

    # ------------------------------------------------------------------
    # Generate TTS
    # ------------------------------------------------------------------

    logger.info("[2] Generating TTS for translated segments...")
    segments_speakers_count = len(Counter([s["speaker"] for s in segments]))
    logger.info(f"Found {segments_speakers_count} speakers in segments.")
    if mapped_speakers_count != segments_speakers_count:
        logger.warning(
            f"Number of speakers in segments ({segments_speakers_count}) does not match number of mapped speakers ({mapped_speakers_count})."
    )
    try:
        valid_speakers = audio_segmentation_to_voice(
            {"segments": segments},
            target_language,
            False,
            **tts_kwargs,
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise Exception(f"TTS generation process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Acceleration / overlap handling
    # ------------------------------------------------------------------

    logger.info("[3] Accelerating segments...")
    try:
        audio_files, _ = accelerate_segments(
            {"segments": segments},
            max_accelerate_audio,
            valid_speakers,
            min_accelerate_audio,
            acceleration_regulate,
        )
    except Exception as e:
        logger.warning(f"Acceleration failed, continuing without: {e}")
        audio_files = valid_speakers

    # ------------------------------------------------------------------
    # Create translated audio
    # ------------------------------------------------------------------

    logger.info("[4] Creating translated audio...")
    try:
        remove_files(dub_audio_file)
        create_translated_audio(
            {"segments": segments},
            audio_files,
            dub_audio_file,
            concat=False,
            avoid_overlap=avoid_overlap,
        )
    except Exception as e:
        logger.error(f"Failed to create translated audio: {e}")
        raise Exception(f"Translated audio creation process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Mix with base audio
    # ------------------------------------------------------------------

    logger.info("[5] Mixing audio...")
    try:
        remove_files(mix_audio_file)
        if base_audio_wav and os.path.exists(base_audio_wav):
            match mix_method:
                case "Adjusting volumes and mixing audio":
                    logger.info(f"Using mix method: {mix_method}")
                    mix_audio_volume(
                        base_audio_wav,
                        dub_audio_file,
                        mix_audio_file,
                        vol_original,
                        vol_translated,
                    )
                case "Segment-based ducking (precise timing)":
                    logger.info(f"Using mix method: {mix_method}")
                    volume_automation = (
                        generate_segment_based_volume_automation_chunked(
                            segments,
                            chunk_size=40,
                            duck_level=vol_original,
                            fade_duration=0.5,
                        )
                    )
                    mix_audio_segment_ducking(
                        base_audio_wav,
                        dub_audio_file,
                        mix_audio_file,
                        vol_original,
                        vol_translated,
                        volume_automation,
                        volume_boost_db=1.0,
                    )
                case "Mixing audio with sidechain compression":
                    logger.info(f"Using mix method: {mix_method}")
                    mix_audio_sidechain(
                        base_audio_wav,
                        dub_audio_file,
                        mix_audio_file,
                        vol_original,
                        vol_translated,
                    )
                case _:
                    logger.info(
                        f"Undefined mix method: {mix_method}, creating translated audio."
                    )
                    # Convert dub_audio_file (WAV) and save as mix_audio_file
                    audio = AudioSegment.from_wav(dub_audio_file)
                    audio.set_channels(2)
                    audio.export(
                        mix_audio_file,
                        format="wav",
                        parameters=["-ac", "2"]  # force 2 channels
                    )

        else:
            logger.info("No base audio provided — creating translated audio only.")
            # Convert dub_audio_file (WAV) and save as mix_audio_file
            audio = AudioSegment.from_wav(dub_audio_file)
            audio.set_channels(2)
            audio.export(
                mix_audio_file,
                format="wav",
                parameters=["-ac", "2"]  # force 2 channels
            )


    except Exception as e:
        logger.error(f"Audio mixing failed: {e}")
        raise Exception(f"Audio mixing process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Match amplitude
    # ------------------------------------------------------------------

    logger.info("[6] Matching loudness...")

    # def match_target_amplitude(sound, target_dBFS):
    #     change_in_dBFS = target_dBFS - sound.dBFS
    #     return sound.apply_gain(change_in_dBFS)

    # # Load MP3 files
    # ref = AudioSegment.from_file(base_audio_wav, format="wav")
    # tgt = AudioSegment.from_file(mix_audio_file, format="mp3")

    # # Match loudness
    # target_loudness = ref.dBFS
    # normalized_tgt = match_target_amplitude(tgt, target_loudness)
    # logger.info(f"Target loudness: {target_loudness} dBFS, Normalized loudness: {normalized_tgt.dBFS} dBFS")

    # # Export back to MP3
    # normalized_tgt.export(mix_audio_file, format="mp3", bitrate="192k")

    match_loudness_LUFS(
        ref_file=base_audio_wav,
        tgt_file=mix_audio_file,
        output_file=final_audio_file,
        manual_offset_db=manual_offset_db
    )


    # ------------------------------------------------------------------
    # 6️⃣ Save final state
    # ------------------------------------------------------------------

    logger.info("[7] Saving output state...")
    state_phase3 = {
        "created_at": datetime.now().isoformat(),
        "phase_version": "3.0",
        "media_file": media_file,
        "media_hash": media_hash,
        "target_language": target_language,
        "segments": segments,
        "tts_kwargs": tts_kwargs,
        "mix_method": mix_method,
        "output_audio": final_audio_file,
        "config": config,
    }

    output_path = os.path.join(output_dir, f"{output_file}.json")
    save_state(state_phase3, output_path)

    logger.info(f"✅ Phase 3 complete → {output_path} {final_audio_file}")
    return output_path
