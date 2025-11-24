import os
from datetime import datetime
from .logging_setup import logger
from .translate_segments import translate_text
from .llm_fit import per_segment_budgets, llm_fit, select_segments_within_budget
from .text_multiformat_processor import process_subtitles
from .utils import (
    log_segments,
    save_state,
    load_state,
    get_audio_duration,
)


def translate_and_fit(
    state_path: str,
    output_dir: str,
    output_file: str,
    config_kwargs: dict,
    config: dict,
):
    """
    Phase 2 — Translate & Fit

    Loads the Phase 1 JSON, performs text translation, applies optional
    LLM-based fitting (length and style adjustments), and optionally
    generates subtitles.

    Steps:
        1. Load Phase 1 state
        2. Translate segments
        3. Apply LLM fit (optional)
        4. Selecting segments within budget (optional)
        5. Save Phase 2 JSON
    """

    # ------------------------------------------------------------------
    # Config extraction
    # ------------------------------------------------------------------

    # Translation config
    translate_to = config_kwargs.get("target_language", "en")
    translate_process = config_kwargs.get("translate_process", "deep_translator")

    # LLM Fit config
    llm_fit_process = config_kwargs.get("llm_fit_process", "disable")
    llm_fit_model = config_kwargs.get("llm_fit_model", None)

    # ------------------------------------------------------------------
    # Load previous phase
    # ------------------------------------------------------------------

    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Phase 1 state not found: {state_path}")

    state_phase1 = load_state(state_path)

    media_file = state_phase1.get("media_file", "")
    media_hash = state_phase1.get("media_hash", "")
    source_lang = state_phase1.get("language", "auto")
    segments = state_phase1.get("segments", [])
    media_duration = state_phase1.get("media_duration", 0)
    if media_duration == 0 and os.path.exists(media_file):
        media_duration = get_audio_duration(media_file)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    logger.info(f"🏁 Starting Phase 2: Translate & Fit | {media_file} (hash {media_hash[:10]}...)")
    logger.info(f"Loaded {len(segments)} segments for translation.")
    logger.info(f"Source: {source_lang} → Target: {translate_to}")

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    logger.info("[1] Translating segments...")
    try:
        segments_translated = translate_text(
            segments,
            translate_to,
            translate_process,
            chunk_size=1800,
            source=source_lang,
        )
        log_segments(segments_translated, title="Translation results:")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise Exception(f"Translation process failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # LLM Fit (optional)
    # ------------------------------------------------------------------

    logger.info("[2] Applying LLM fit...")
    if llm_fit_process != "disable":
        try:
            budgets = per_segment_budgets(
                src_segments=segments,
                tgt_segments=segments_translated,
                cps_cap=19.0,
                safety=0.90,
                default_r=1.15,
                media_duration=media_duration,
            )
            llm_fit_result = llm_fit(
                src_segments=segments,
                tgt_segments=segments_translated,
                budgets=budgets,
                task_mode=llm_fit_process,
                target_lang=translate_to,
                source_lang=source_lang,
                model=llm_fit_model,
            )
            log_segments(
                segments=segments_translated,
                other_segments=llm_fit_result.segments,
                title="LLM fit results:",
            )
            segments_fitted = llm_fit_result.segments
        except Exception as e:
            logger.error(f"LLM fit failed: {e}")
            raise Exception(f"LLM fit process failed: {str(e)}") from e
    else:
        logger.info("LLM fit disabled — keeping raw translations.")
        segments_fitted = segments_translated
        budgets = None

    # ------------------------------------------------------------------
    # Force choosing segments within the source budget
    # ------------------------------------------------------------------

    sel_threshold = float(config_kwargs.get("fit_selection_threshold", 0.95))
    sel_budget_key = str(config_kwargs.get("fit_selection_budget_key", "char_src"))
    sel_buffer_threshold = float(config_kwargs.get("fit_selection_buffer_threshold", None))
    sel_buffer_override_max_ratio = float(config_kwargs.get("fit_selection_buffer_override_max_ratio", 1.3))

    logger.info(
        f"[3] Selecting segments within budget using {sel_budget_key} "
        + (f"with buffer>={sel_buffer_threshold}s override" if sel_buffer_threshold is not None else "(no buffer override)")
    )

    if segments_fitted and segments_translated and budgets:
        try:
            segments_selected, total_count, translated_used_count = select_segments_within_budget(
                budgets=budgets,
                segments_fitted=segments_fitted,
                segments_translated=segments_translated,
                threshold=sel_threshold,
                budget_key=sel_budget_key,
                buffer_threshold_s=sel_buffer_threshold,
                buffer_key="buffer",
                buffer_override_max_ratio=sel_buffer_override_max_ratio,
            )
            logger.debug(
                f"Segments chosen (translated/fitted) via selection — translated used: {translated_used_count} / {total_count}"
            )
        except Exception as e:
            logger.error(f"Segment selection failed: {e}")
            raise Exception(f"Segment selection process failed: {str(e)}") from e
    else:
        logger.info("Segment selection skipped — keeping all segments.")
        segments = segments_translated

    # ------------------------------------------------------------------
    # Save Phase 2 state
    # ------------------------------------------------------------------

    logger.info("[4] Saving output state...")
    state_phase2 = {
        "created_at": datetime.now().isoformat(),
        "phase_version": "2.0",
        "media_file": media_file,
        "media_hash": media_hash,
        "source_language": source_lang,
        "target_language": translate_to,
        "segments": segments_selected,
        "segments_translated": segments_translated,
        "segments_fitted": segments_fitted,
        "segments_original": segments,
        "budgets": budgets,
        "config": config,
    }

    output_path = os.path.join(output_dir, f"{output_file}.json")
    save_state(state_phase2, output_path)

    logger.info(f"✅ Phase 2 complete → {output_path}")
    return output_path
