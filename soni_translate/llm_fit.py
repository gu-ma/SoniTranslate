# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

# LangChain (>=0.2)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

# Pydantic for structured output
from pydantic import BaseModel, Field

# Local
from .llm_fit_configuration import (
    PUNCT_COST,
    SYSTEM_PROMPT_TMPL_TIGHTEN_AND_FIT,
    SYSTEM_PROMPT_TMPL_TRANSLATE_AND_FIT,
)
from .logging_setup import logger

LLM_FIT_OPTIONS = [
    "translate_and_fit",
    "tighten_and_fit",
    "disable",
]

LLM_FIT_MODELS = ["openai/gpt-5-mini", "openai/gpt-5", "google/gemini-2.5-pro"]


# -------------------------------
# Character cost / CPS utilities
# -------------------------------


def weighted_len(s: str) -> int:
    """
    Weighted character length to better correlate with TTS time:
      - normal char: 1
      - , ; : — - – : 2
      - . ? ! : 3
    """
    w = 0
    for ch in s:
        w += PUNCT_COST.get(ch, 1)
    return w


def cps(text: str, start: float, end: float, use_weighted: bool = False) -> float:
    dur = max(0.001, float(end) - float(start))
    L = weighted_len(text) if use_weighted else len(text.strip())
    return float(L) / dur


# -------------------------------
# Segment selection utilities
# -------------------------------


def select_segments_within_budget(
    budgets: Optional[List[Dict[str, Any]]],
    segments_fitted: Optional[List[Dict[str, Any]]],
    segments_translated: Optional[List[Dict[str, Any]]],
    threshold: float = 0.95,
    budget_key: str = "char_src",
    buffer_threshold_s: Optional[float] = None,
    buffer_key: str = "buffer",
    buffer_override_max_ratio: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Select between a fitted and a translated segment using a per-segment budget.

    Rules:
        1. Buffer override:
           If a time buffer exists (budget[buffer_key] >= buffer_threshold_s),
           prefer the translated segment as long as its weighted length does not
           exceed (budget * buffer_override_max_ratio). If max_ratio is None,
           the override is unconditional.

        2. Budget rule:
           Otherwise, pick the translated segment only if:
               weighted_len(translated) < budget * threshold
           Else use the fitted one.

        3. Fallback:
           If budgets or segments are missing, return the fitted list unchanged.

    Returns:
        (selected_segments, total_count, translated_used_count)
    """
    if not (budgets and segments_fitted and segments_translated):
        return segments_fitted or [], 0, 0

    selected_segments: List[Dict[str, Any]] = []
    total_count = 0
    translated_used_count = 0

    zip_object = zip(segments_fitted, segments_translated, budgets)
    for seg_fit, seg_trans, budget in zip_object:
        total_count += 1

        limit = float(budget.get(budget_key)) if isinstance(budget, dict) else None
        buffer_val = float(budget.get(buffer_key)) if isinstance(budget, dict) else None

        wl_trans = weighted_len(seg_trans.get("text", ""))
        wl_fit = weighted_len(seg_fit.get("text", ""))

        # ------------------------------------------------------------------
        # Buffer override: prefer translated if we have enough time buffer.
        # Safeguard: translated must be <= limit * buffer_override_max_ratio
        # (unless the ratio is None → unconditional override).
        # ------------------------------------------------------------------
        buffer_override_allowed = (
            buffer_threshold_s is not None
            and buffer_val is not None
            and buffer_val >= float(buffer_threshold_s)
        )

        if buffer_override_allowed and limit is not None:
            within_ratio = (
                buffer_override_max_ratio is None
                or wl_trans <= limit * float(buffer_override_max_ratio)
            )

            if within_ratio:
                logger.debug(
                    f"buffer override: buffer={buffer_val:.2f}s >= {float(buffer_threshold_s):.2f}s "
                    f"and wl_trans={wl_trans} <= {limit} * {buffer_override_max_ratio} -> using translated"
                )
                logger.debug(f"tra. text: {seg_trans.get('text', '')}")
                logger.debug(f"fit. text: {seg_fit.get('text', '')}")

                if isinstance(budget, dict):
                    logger.debug(
                        f"budget (src/tgt/budget/buffer): "
                        f"{budget.get('char_src')} / {budget.get('char_tgt')} / "
                        f"{budget.get('char_budget')} / {budget.get('buffer')}"
                    )

                selected_segments.append(seg_trans)
                translated_used_count += 1
                continue

            else:
                logger.debug(
                    f"buffer override skipped: buffer={buffer_val:.2f}s, wl_trans={wl_trans}, "
                    f"limit={limit}, max_ratio={buffer_override_max_ratio}"
                )

        # ------------------------------------------------------------------
        # Standard budget rule:
        # Choose translated only if it's strictly below (limit * threshold).
        # ------------------------------------------------------------------
        if limit is not None and wl_trans < limit * float(threshold):
            logger.debug(f"tra. text: {seg_trans.get('text', '')}")
            logger.debug(f"fit. text: {seg_fit.get('text', '')}")
            logger.debug(f"weighted_len (translated / fitted): {wl_trans} / {wl_fit}")

            if isinstance(budget, dict):
                logger.debug(
                    f"budget (src / tgt): {budget.get('char_src')} / {budget.get('char_tgt')}"
                )

            selected_segments.append(seg_trans)
            translated_used_count += 1
        else:
            selected_segments.append(seg_fit)

    return selected_segments, total_count, translated_used_count


# ---------------------------------------------
# Per-segment budgets from timing + source EN
# ---------------------------------------------


def per_segment_budgets(
    src_segments: List[Dict[str, Any]],  # [{'start','end','text'}] in SOURCE language
    tgt_segments: Optional[
        List[Dict[str, Any]]
    ] = None,  # optional target to estimate expansion ratio
    cps_cap: float = 19.0,  # upper cap for Latin scripts; use ~10–11 for CJK, ~16–18 AR/HE
    safety: float = 0.95,  # 5% margin
    default_r: float = 1.15,  # fallback expansion ratio (e.g., EN->IT ~1.1–1.2)
    media_duration: Optional[float] = None,  # optional media duration
) -> List[Dict[str, Any]]:
    """
    Returns a list with per-line cps_tgt and characters_budget (weighted chars).
    - If 'tgt_segments' provided, we compute a robust median expansion ratio r = wlen(IT)/wlen(EN)
    - Then per line: cps_tgt_i = min(cps_cap, cps_src_i * r * safety)
    - Budget_i = int(cps_tgt_i * duration_i)  (in WEIGHTED chars)
    """

    # 1) estimate expansion ratio r from the file (if we have target draft)
    r = default_r
    if tgt_segments:
        ratios = []
        for src, tgt in zip(src_segments, tgt_segments):
            src_len = max(1, weighted_len(src["text"]))
            tgt_len = max(1, weighted_len(tgt["text"]))
            # logger.info(f"EN: {src_len}, IT: {tgt_len}, ratio: {(tgt_len/src_len):.2f}")
            ratio = float(f"{tgt_len / src_len:.2f}")
            ratios.append(ratio)
        if ratios:
            r = float(f"{sum(ratios) / len(ratios):.2f}")

    # 2) compute per-line budgets
    budgets: List[Dict[str, Any]] = []
    for i, seg in enumerate(src_segments):

        # Start and end times
        start = float(seg["start"])
        end = float(seg["end"])

        # Determine the start time of the next segment
        if i < len(src_segments) - 1:
            next_start = float(src_segments[i + 1].get("start", end))
        else:
            # For the last segment, prefer media_duration (if provided) otherwise use end
            next_start = float(media_duration) if media_duration is not None else end
        # Guard against overlapping or misordered timestamps so buffer is never negative
        if next_start < end:
            next_start = end

        # Duration and buffer
        duration = max(0.001, end - start)
        buffer = max(0.001, next_start - end)

        # Ratio
        ratio = ratios[i] if ratios else 0

        # CPS
        cps_src = weighted_len(seg["text"]) / duration
        cps_tgt = cps_src * ratios[i] if ratios else 0
        cps_tgt_i = min(cps_cap, cps_src * r * safety)

        # Characters
        char_src = weighted_len(seg["text"])
        char_tgt = int(weighted_len(seg["text"]) * ratios[i]) if ratios else 0
        char_budget = int(cps_tgt_i * duration)
        duration_tgt = char_budget / cps_tgt_i

        budget = {
            "index": i,
            "start": start,
            "end": end,
            "duration": float(f"{duration:.2f}"),
            "buffer": float(f"{buffer:.2f}"),
            "ratio": ratio,
            "ratio_median": r,
            "cps_src": float(f"{cps_src:.2f}"),
            "cps_tgt": float(f"{cps_tgt:.2f}"),
            "cps_tgt_i": float(f"{cps_tgt_i:.2f}"),
            # "cps_src_raw": float(f"{cps(seg['text'], start, end):.2f}"),
            "char_src": char_src,
            "char_tgt": char_tgt,
            "char_budget": char_budget,
            "duration_tgt": float(f"{duration_tgt:.2f}"),
        }
        budgets.append(budget)

    return budgets


# --------------------------------
# LLM core (LangChain ChatOpenAI)
# --------------------------------


def _build_user_payload(
    segments_payload: List[Dict[str, Any]],
    task_mode: str,
    source_lang: Optional[str],
    target_lang: str,
    global_notes: Optional[List[str]],
) -> Dict[str, Any]:
    return {
        "task_mode": task_mode,  # "translate_and_fit" | "tighten_and_fit"
        "source_lang": source_lang,
        "target_lang": target_lang,
        "style_notes": global_notes or [],
        "segments": segments_payload,
    }


# --------------------------------
# Pydantic models for structured LLM output
# --------------------------------


class Candidate(BaseModel):
    """A candidate translation/tightening for a segment."""

    text: str = Field(..., description="Final text for the segment in target language.")
    notes: str = Field(..., description="Short note about editing decisions.")
    # ssml_suggestion: Optional[str] = Field(None, description="Optional SSML hint like <break time='...'/>.")


class SegmentOut(BaseModel):
    """Output structure for a single segment with candidates."""

    index: int
    start: float
    end: float
    candidates: List[Candidate] = Field(..., min_items=1, max_items=3)


class LLMResponse(BaseModel):
    """Complete LLM response structure."""

    segments: List[SegmentOut]
    global_notes: Optional[str] = None


@dataclass
class LLMFitResult:
    """Structured output for llm_fit()."""

    segments: List[Dict[str, Any]]  # [{'start','end','text'}] final target text
    diagnostics: List[Dict[str, Any]]  # cps, notes, ssml, speed suggestions, etc.
    model: str
    budgets: List[Dict[str, Any]]


def llm_fit(
    *,
    # INPUTS
    src_segments: List[Dict[str, Any]],  # [{'start','end','text'}]; source text
    tgt_segments: Optional[List[Dict[str, Any]]] = None,  # target text
    budgets: List[Dict[str, Any]],  # output from per_segment_budgets();
    task_mode: str,  # "translate_and_fit" or "tighten_and_fit"
    target_lang: str,
    source_lang: Optional[str] = None,
    # MODEL
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.25,
    max_retries: int = 2,
    # MISC
    style_notes: Optional[List[str]] = None,
    openai_api_key: Optional[str] = None,
) -> LLMFitResult:
    """
    Single call for the whole file (best for consistency).
    Returns chosen text per segment + diagnostics.
    """

    if task_mode not in LLM_FIT_OPTIONS:
        raise ValueError(f"task_mode must be one of {LLM_FIT_OPTIONS}.")

    # if task_mode is "disable" then return the source segments
    if task_mode == "disable":
        return LLMFitResult(
            segments=tgt_segments or src_segments,
            diagnostics=[],
            model=model,
            budgets=budgets,
        )

    # API key
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENROUTER_API_KEY")

    # Prepare custom headers for OpenRouter (optional)
    headers = {}
    if os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
    if os.getenv("OPENROUTER_SITE_NAME"):
        headers["X-Title"] = os.getenv("OPENROUTER_SITE_NAME")

    # Zip source & target segments
    zip_segments = (
        zip(src_segments, tgt_segments) if tgt_segments else zip(src_segments)
    )

    # Build per-segment payload combining text & budgets.
    segs_payload: List[Dict[str, Any]] = []
    for i, (seg, budget) in enumerate(zip(zip_segments, budgets)):
        # Sanity: indices must match
        assert budget["index"] == i, "Budget list must align with segments by index."
        # Extract src & tgt segments
        src_seg = seg[0]
        tgt_seg = seg[1] if len(seg) == 2 else None
        # Build payload
        item = {
            "index": i,
            "start": float(src_seg["start"]),
            "end": float(src_seg["end"]),
            "characters_budget": int(budget["char_budget"]),
            "characters_source": float(budget["char_src"]),
            "source_text": src_seg["text"],
            "original_target_text": tgt_seg["text"] if tgt_seg else None,
        }
        segs_payload.append(item)

    user_payload = _build_user_payload(
        segments_payload=segs_payload,
        task_mode=task_mode,
        source_lang=source_lang,
        target_lang=target_lang,
        global_notes=style_notes,
    )

    # Prepare JsonOutputParser with Pydantic schema
    parser = JsonOutputParser(pydantic_object=LLMResponse)
    format_instructions = parser.get_format_instructions()
    # logger.debug(format_instructions)

    reasoning = {
        "effort": "medium",  # 'low', 'medium', or 'high'
        "summary": "auto",  # 'detailed', 'auto', or None
    }

    extra_body = {"reasoning": {"effort": "medium"}}

    # Initialize OpenRouter client
    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=temperature,
        default_headers=headers,
        # use_responses_api=True,
        # seed=1,
        # reasoning=reasoning,
        # output_version="responses/v1",
        # extra_body=extra_body
        # model_kwargs={"temperature": temperature}
    )

    # Prepare messages with format instructions
    system_message = (
        SYSTEM_PROMPT_TMPL_TIGHTEN_AND_FIT
        if task_mode == "tighten_and_fit"
        else SYSTEM_PROMPT_TMPL_TRANSLATE_AND_FIT
    )
    messages = [
        SystemMessage(content=system_message + "\n\n" + format_instructions),
        HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
    ]
    # logger.debug(messages)

    # Parse with JsonOutputParser and retry if needed
    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        response = llm.invoke(messages)
        text = (response.content or "").strip()
        # logger.debug(f"LLM response: {text}")
        try:
            parsed = parser.parse(text)  # returns LLMResponse (Pydantic model)
            # parsed = json_repair.loads(parsed)
            data = parsed
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"LLM did not return valid JSON after retries: {last_err}")

    # Pick best candidate per segment w.r.t. WEIGHTED budget; compute diagnostics
    final_segments: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    segments_out = data["segments"]

    # Debug
    for segment_out, src_segment in zip(segments_out, src_segments):
        logger.debug("---")
        s = src_segment["text"].strip()
        for candidate in segment_out.get("candidates", []):
            t = candidate.get("text", "").strip()
            n = candidate.get("notes", "").strip()
            i = segment_out["index"]
            logger.debug(f"Segment {i}")
            logger.debug(f"source:\t\t{s}")
            logger.debug(f"candidate:\t{t}")
            logger.debug(f"notes:\t\t{n}")

    for i, (segment_out, budget, src_segment) in enumerate(
        zip(segments_out, budgets, src_segments)
    ):
        start, end = float(segment_out["start"]), float(segment_out["end"])
        speaker = src_segment.get("speaker", "")
        duration = max(0.001, end - start)
        budget_w = int(budget["char_budget"])

        # TODO: Choose the shortest candidate that is <= budget_w and >= src_len;
        # TODO: Choose the shortest candidate that is <= budget_w and >= src_len * .8;
        # Choose the longest candidate that is <= budget_w (by weighted_len);
        # else choose the candidate with minimal weighted_len.
        chosen_text: Optional[str] = None
        chosen_notes: str = ""
        best_len = 0
        best_over = 10**9

        for candidate in segment_out.get("candidates", []):
            t = candidate.get("text", "").strip()
            wl = weighted_len(t)
            if wl <= budget_w and wl > best_len:
                chosen_text = t
                chosen_notes = candidate.get("notes", "")
                best_len = wl
            else:
                over = wl - budget_w
                if over < best_over:
                    # fallback holder if none fit
                    fallback_text = t
                    fallback_notes = candidate.get("notes", "")
                    best_over = over
                    best_len_fallback = wl

        if chosen_text is None:
            # no candidate met budget; use fallback (shortest)
            chosen_text = fallback_text
            chosen_notes = fallback_notes
            best_len = best_len_fallback

        # Compute cps diagnostics (raw & weighted)
        cps_raw = len(chosen_text) / duration
        cps_w = best_len / duration

        # Suggest a small ElevenLabs speed tweak if still slightly over budget by weighted cps
        speed_suggestion = None
        # target cps from budget:
        cps_tgt_from_budget = budget_w / duration
        if cps_w > cps_tgt_from_budget:
            need = cps_w / cps_tgt_from_budget
            # Clamp modestly: 0.85–1.15
            speed_suggestion = round(max(0.85, min(1.15, 1.0 + (need - 1.0))), 2)

        final_segments.append(
            {
                "start": start,
                "end": end,
                "text": chosen_text,
                "speaker": speaker,
            }
        )

        diagnostics.append(
            {
                "index": i,
                "duration_s": round(duration, 3),
                "characters_budget_weighted": budget_w,
                "weighted_len": best_len,
                "cps_raw": round(cps_raw, 2),
                "cps_weighted": round(cps_w, 2),
                "notes": chosen_notes,
                "speed_suggestion_for_elevenlabs": speed_suggestion,
                "original_text": src_segment["text"],
            }
        )

    return LLMFitResult(
        segments=final_segments,
        diagnostics=diagnostics,
        model=model,
        budgets=budgets,
    )
