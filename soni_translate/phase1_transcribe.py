import os
from datetime import datetime
from .logging_setup import logger
from .preprocessor import audio_video_preprocessor, audio_preprocessor
from .mdx_net import process_uvr_task
from .speech_segmentation import (
    transcribe_speech,
    align_speech,
    diarize_speech,
    diarization_models,
)
from .text_multiformat_processor import (
    linguistic_level_segments,
    break_aling_segments,
)
from .segments_utils import (
    merge_segments,
)
from .utils import (
    is_audio_file,
    remove_files,
    copy_files,
    get_hash,
    log_segments,
    save_state,
    get_audio_duration
)


def transcribe_and_analyze(
    media_file: str,
    output_dir: str,
    output_file: str,
    config_kwargs: dict,
    config: dict,
):
    """
    Phase 1 — Transcribe & Analyze

    Converts input media into structured speech segments with timestamps,
    speaker diarization and linguistic segmentation.

    Steps:
        1. Preprocess media (extract audio/video)
        2. Optional: vocal dereverb/refinement
        3. Transcription (Whisper / WhisperX)
        4. Alignment
        5. Segmentation
        6. Diarization
        7. Merge micro-segments
        8. Save output JSON state
    """

    # ------------------------------------------------------------------
    # Config extraction
    # ------------------------------------------------------------------

    # Transcribe config
    transcriber_model = config_kwargs.get("transcriber_model", "large-v3")
    compute_type = config_kwargs.get("compute_type", "float32")
    batch_size = config_kwargs.get("batch_size", 4)
    src_lang = config_kwargs.get("origin_language", None)
    literalize_numbers = config_kwargs.get("literalize_numbers", False)
    segment_duration_limit = config_kwargs.get("segment_duration_limit", 15)

    # Diarization config
    min_spk = config_kwargs.get("min_speakers", 1)
    max_spk = config_kwargs.get("max_speakers", 1)
    diar_model = config_kwargs.get("diarization_model", "pyannote_3.1")
    hf_token = config_kwargs.get("YOUR_HF_TOKEN", "")

    # Segmentation config
    seg_scale = config_kwargs.get("text_segmentation_scale", "sentence")
    divide_by = config_kwargs.get("divide_text_segments_by", "")

    logger.info(
        f"🏁 Starting Phase 1: Transcribe & Analyze | {media_file} (hash ...{output_file[-10:]})"
    )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    # Filenames and paths
    temp_dir = config_kwargs.get("temp_dir", "temp")
    base_audio_wav = os.path.join(temp_dir, "phase1_audio.wav")
    base_video_file = os.path.join(temp_dir, "phase1_video.mp4")

    # Create directories if not exists
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

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

    media_duration = get_audio_duration(base_audio_wav)
    logger.info(f"Media duration: {media_duration}")

    # ------------------------------------------------------------------
    # Vocal Refinement (optional)
    # ------------------------------------------------------------------

    vocals_audio_file = os.path.join(temp_dir, "phase1_audio_Vocals_DeReverb.wav")
    vocals = None
    if config_kwargs.get("vocal_refinement", False):
        try:
            logger.info("[2] Refining vocals (dereverb)...")
            _, _, _, _, file_vocals = process_uvr_task(
                orig_song_path=base_audio_wav,
                main_vocals=False,
                dereverb=True,
                remove_files_output_dir=True,
            )
            remove_files(vocals_audio_file)
            copy_files(file_vocals, temp_dir)
            vocals = vocals_audio_file
        except Exception as e:
            logger.warning(f"Vocal refinement failed: {e}")
            raise Exception(f"Vocal refinement process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    try:
        logger.info("[3] Transcribing speech...")
        audio_path = vocals or base_audio_wav
        audio, result = transcribe_speech(
            audio_path,
            transcriber_model,
            compute_type,
            batch_size,
            src_lang,
            literalize_numbers,
            segment_duration_limit,
        )
        log_segments(segments=result.get("segments"), title="Transcription results:")
        align_language = result.get("language")
        logger.info(f"Transcription done. Detected language: {align_language}")
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        raise Exception(f"Transcription process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Alignment (if supported)
    # ------------------------------------------------------------------

    logger.info("[4] Aligning speech...")
    if align_language not in ["vi"]:  # skip Vietnamese etc.
        try:
            result = align_speech(audio, result)
            log_segments(segments=result.get("segments"), title="Alignment results:")
        except Exception as e:
            logger.warning(f"Alignment skipped or failed: {e}")
            raise Exception(f"Alignment skipped process failed: {str(e)}") from e
    else:
        logger.warning(f"Skipping alignment for {align_language} (unsupported).")

    # ------------------------------------------------------------------
    # Text Segmentation
    # ------------------------------------------------------------------

    if align_language in ["ja", "zh", "zh-TW"]:
        divide_by += "|!|?|...|。"

    logger.info(f"[5] Segmenting text ({seg_scale})...")
    try:
        if seg_scale in ["word", "character"]:
            result = linguistic_level_segments(result, seg_scale)
        elif divide_by:
            result = break_aling_segments(result, break_characters=divide_by)
    except Exception as e:
        logger.warning(f"Segmentation failed: {e}")
        raise Exception(f"Segmentation process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Diarization
    # ------------------------------------------------------------------

    diarize_model_select = diarization_models[diar_model]
    logger.info("[6] Running speaker diarization...")
    try:
        result_diarize = diarize_speech(
            audio_path,
            result,
            min_spk,
            max_spk,
            hf_token,
            diarize_model_select,
        )
        log_segments(
            segments=result_diarize.get("segments"), title="Diarization results:"
        )
    except Exception as e:
        logger.warning(f"Diarization failed: {e}")
        raise Exception(f"Diarization process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Merge Segments
    # ------------------------------------------------------------------

    logger.info("[7] Merging micro-segments...")
    try:
        result_diarize = merge_segments(result_diarize, enabled=True)
        log_segments(segments=result_diarize.get("segments"), title="Merged segments:")
    except Exception as e:
        logger.warning(f"Segment merging failed: {e}")
        raise Exception(f"Segment merging process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Save Output State
    # ------------------------------------------------------------------

    logger.info("[8] Saving output state...")
    state = {
        "created_at": datetime.now().isoformat(),
        "phase_version": "1.0",
        "media_file": media_file,
        "media_hash": get_hash(media_file),
        "language": align_language,
        "media_duration": media_duration,
        "segments": result_diarize["segments"],
        "min_speakers": min_spk,
        "max_speakers": max_spk,
        "config": config,
    }

    output_path = os.path.join(output_dir, f"{output_file}.json")
    save_state(state, output_path)

    logger.info(f"✅ Phase 1 complete → {output_path}")
    return output_path
