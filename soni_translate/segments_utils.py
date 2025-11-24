import re
from typing import List, Dict
from .logging_setup import logger

# -------------------------------------------------------------
# Pre-alignment merging to avoid ultra-short segments
# -------------------------------------------------------------

DEFAULT_MIN_WORDS = 12  # Previous 6
DEFAULT_MIN_DURATION = 6 # Previous 3
DEFAULT_MAX_BRIDGE_GAP = 1
DEFAULT_HARD_STOP_PATTERN = r"[.!?…]"


def _word_count(t: str) -> int:
    return len(re.findall(r"\w+", t or ""))


def merge_whisper_segments(
    segments: List[Dict],
    min_words: int = DEFAULT_MIN_WORDS,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_bridge_gap: float = DEFAULT_MAX_BRIDGE_GAP,
    hard_stop_pattern: str = DEFAULT_HARD_STOP_PATTERN,
    same_speaker_only: bool = True,
) -> List[Dict]:
    """
    Merge adjacent Whisper segments to reduce one-word/very short lines.

    Heuristics:
      - merge if the gap between segments <= max_bridge_gap
    #   - and if the buffered segment has fewer than min_words
    #   - or if the buffered segment duration < min_duration
      - but avoid merging across strong punctuation if the buffer is already
        "long enough" (meets min_words or min_duration)

    Returns a new list of dicts with updated start/end and concatenated text.
    Other keys are intentionally dropped to keep a clean pre-align structure.
    """
    if not segments:
        return []

    merged: List[Dict] = []
    buf: Dict | None = None

    def flush():
        nonlocal buf
        if buf is not None:
            buf["text"] = re.sub(r"\s+", " ", buf.get("text", "")).strip()
            merged.append(buf)
            buf = None

    for s in segments:
        seg = {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
            "speaker": s.get("speaker"),
        }

        if buf is None:
            buf = seg
            continue

        gap = seg["start"] - buf["end"]
        buf_wc = _word_count(buf["text"])
        buf_dur = max(0.0, buf["end"] - buf["start"])

        # Speaker-aware guard: do not merge across different speakers when requested.
        if same_speaker_only:
            buf_spk = buf.get("speaker")
            seg_spk = seg.get("speaker")
            # If either segment has a speaker label and they differ, do not merge.
            if not (buf_spk is None and seg_spk is None) and (buf_spk != seg_spk):
                flush()
                buf = seg
                continue

        # Merge if the gap between segments <= max_bridge_gap
        should_merge = gap <= max_bridge_gap

        # # Avoid merging across strong punctuation if buffer is already adequate
        # if should_merge and (
        #     re.search(hard_stop_pattern + r"\s*$", buf["text"])
        #     and (buf_wc >= min_words or buf_dur >= min_duration)
        # ):
        #     should_merge = False

        # Allow merging even across strong punctuation
        if should_merge and (buf_wc >= min_words or buf_dur >= min_duration):
            should_merge = False

        # logger.debug(
        #     f"\nbuf: [{buf['start']:0.2f} --> {buf['end']:0.2f}] {buf['text']}"
        #     f"\nSegment: [{seg['start']:0.2f} --> {seg['end']:0.2f}] {seg['text']}"
        #     f"\nConsidering merge: gap={gap:.2f}s, buf_wc={buf_wc}, buf_dur={buf_dur:.2f}s "
        #     f"=> should_merge={should_merge}"
        # )

        if should_merge:
            # Determine whether to insert a space when joining.
            # Default to a single space except for explicit connectors like hyphens/slashes.
            end_char = buf["text"][-1] if buf["text"] else ""
            start_char = seg["text"][0] if seg["text"] else ""
            if end_char in "-–—/":
                joiner = ""
            elif end_char.isalnum() or end_char in "%":
                joiner = " "
            elif start_char.isalnum():
                joiner = " "
            else:
                joiner = " "
            # Don't strip here — normalize whitespace on flush() to avoid removing
            # required inner spaces (e.g. around punctuation like "%").
            buf["text"] = f"{buf['text']}{joiner}{seg['text']}"
            buf["end"] = seg["end"]
        else:
            flush()
            buf = seg

    flush()
    return merged


def merge_segments(
    result: Dict,
    enabled: bool = True,
    min_words: int = DEFAULT_MIN_WORDS,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_bridge_gap: float = DEFAULT_MAX_BRIDGE_GAP,
    hard_stop_pattern: str = DEFAULT_HARD_STOP_PATTERN,
    same_speaker_only: bool = True,
) -> Dict:
    """Optionally apply merge_whisper_segments to result["segments"]."""
    if not enabled:
        return result
    if not result or "segments" not in result:
        return result
    merged_segments = merge_whisper_segments(
        result["segments"],
        min_words=min_words,
        min_duration=min_duration,
        max_bridge_gap=max_bridge_gap,
        hard_stop_pattern=hard_stop_pattern,
        same_speaker_only=same_speaker_only,
    )
    new_result = {**result}
    new_result["segments"] = merged_segments

    return new_result
