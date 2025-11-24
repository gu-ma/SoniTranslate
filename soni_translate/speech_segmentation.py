from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch
import gc
import os
import re
import soundfile as sf
from IPython.utils import capture  # noqa
from .language_configuration import EXTRA_ALIGN, INVERTED_LANGUAGES
from .logging_setup import logger
from .postprocessor import sanitize_file_name
from .utils import remove_directory_contents, run_command
from difflib import SequenceMatcher


ASR_MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "Systran/faster-distil-whisper-large-v3",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "distil-small.en",
    "distil-medium.en",
    "OpenAI_API_Whisper",
]

COMPUTE_TYPE_GPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "float16",
    "bfloat16",
    "float32",
]

COMPUTE_TYPE_CPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int16",
    "float32",
]

WHISPER_MODELS_PATH = "./WHISPER_MODELS"


def openai_api_whisper(input_audio_file, source_lang=None, chunk_duration=1800):

    info = sf.info(input_audio_file)
    duration = info.duration

    output_directory = "./whisper_api_audio_parts"
    os.makedirs(output_directory, exist_ok=True)
    remove_directory_contents(output_directory)

    if duration > chunk_duration:
        # Split the audio file into smaller chunks with 30-minute duration
        cm = f'ffmpeg -i "{input_audio_file}" -f segment -segment_time {chunk_duration} -c:a libvorbis "{output_directory}/output%03d.ogg"'
        run_command(cm)
        # Get list of generated chunk files
        chunk_files = sorted(
            [
                f"{output_directory}/{f}"
                for f in os.listdir(output_directory)
                if f.endswith(".ogg")
            ]
        )
    else:
        one_file = f"{output_directory}/output000.ogg"
        cm = f'ffmpeg -i "{input_audio_file}" -c:a libvorbis {one_file}'
        run_command(cm)
        chunk_files = [one_file]

    # Transcript
    segments = []
    language = source_lang if source_lang else None
    for i, chunk in enumerate(chunk_files):
        from openai import OpenAI

        client = OpenAI()

        audio_file = open(chunk, "rb")
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        try:
            transcript_dict = transcription.model_dump()
        except:  # noqa
            transcript_dict = transcription.to_dict()

        if language is None:
            logger.info(f'Language detected: {transcript_dict["language"]}')
            language = TO_LANGUAGE_CODE[transcript_dict["language"]]

        chunk_time = chunk_duration * (i)

        for seg in transcript_dict["segments"]:

            if "start" in seg.keys():
                segments.append(
                    {
                        "text": seg["text"],
                        "start": seg["start"] + chunk_time,
                        "end": seg["end"] + chunk_time,
                    }
                )

    audio = whisperx.load_audio(input_audio_file)
    result = {"segments": segments, "language": language}

    return audio, result


def find_whisper_models():
    path = WHISPER_MODELS_PATH
    folders = []

    if os.path.exists(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path) and "model.bin" in os.listdir(folder_path):
                folders.append(folder)
    return folders


def transcribe_speech(
    audio_wav,
    asr_model,
    compute_type,
    batch_size,
    SOURCE_LANGUAGE,
    literalize_numbers=True,
    segment_duration_limit=15,
):
    """
    Transcribe speech using a whisper model.

    Parameters:
    - audio_wav (str): Path to the audio file in WAV format.
    - asr_model (str): The whisper model to be loaded.
    - compute_type (str): Type of compute to be used (e.g., 'int8', 'float16').
    - batch_size (int): Batch size for transcription.
    - SOURCE_LANGUAGE (str): Source language for transcription.

    Returns:
    - Tuple containing:
        - audio: Loaded audio file.
        - result: Transcription result as a dictionary.
    """

    if asr_model == "OpenAI_API_Whisper":
        if literalize_numbers:
            logger.info(
                "OpenAI's API Whisper does not support "
                "the literalization of numbers."
            )
        return openai_api_whisper(audio_wav, SOURCE_LANGUAGE)

    # https://github.com/openai/whisper/discussions/277
    if SOURCE_LANGUAGE == "en":
        prompt = (
            "Transcribe the speech exactly as spoken, including filler words such as 'um', 'uh', and 'ah'. "
            "If the speaker mentions multiple-choice options like 'A', 'B', 'C', 'D', or 'E', "
            "write them as capital letters followed by periods (e.g., 'A.', 'B.', 'C.', 'D.', 'E.') "
            "and not as words."
        )
    elif SOURCE_LANGUAGE == "zh":
        prompt = (
            "以下是普通话的句子。"
            "逐字转录普通话语音，包括语气词（如“嗯”、“啊”、“呃”）。"
            "注意保留多项选择内容，例如“A”、“B”、“C”或“1”、“2”、“3”，完全按照听到的内容转录。"
        )
    else:
        prompt = None

    # prompt = "以下是普通话的句子。" if SOURCE_LANGUAGE == "zh" else None
    # prompt = "Transcribe the speech exactly as spoken, including filler words such as 'um', 'uh', 'ah', and hesitations." if SOURCE_LANGUAGE == "en" else None

    SOURCE_LANGUAGE = SOURCE_LANGUAGE if SOURCE_LANGUAGE != "zh-TW" else "zh"
    asr_options = {"initial_prompt": prompt, "suppress_numerals": literalize_numbers}

    if asr_model not in ASR_MODEL_OPTIONS:

        base_dir = WHISPER_MODELS_PATH
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        model_dir = os.path.join(base_dir, sanitize_file_name(asr_model))

        if not os.path.exists(model_dir):
            from ctranslate2.converters import TransformersConverter

            quantization = "float32"
            # Download new model
            try:
                converter = TransformersConverter(
                    asr_model,
                    low_cpu_mem_usage=True,
                    copy_files=["tokenizer_config.json", "preprocessor_config.json"],
                )
                converter.convert(model_dir, quantization=quantization, force=False)
            except Exception as error:
                if "File tokenizer_config.json does not exist" in str(error):
                    converter._copy_files = [
                        "tokenizer.json",
                        "preprocessor_config.json",
                    ]
                    converter.convert(model_dir, quantization=quantization, force=True)
                else:
                    raise error

        asr_model = model_dir
        logger.info(f"ASR Model: {str(model_dir)}")

    vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}

    model = whisperx.load_model(
        asr_model,
        os.environ.get("SONITR_DEVICE"),
        compute_type=compute_type,
        language=SOURCE_LANGUAGE,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=segment_duration_limit,
        print_progress=True,
    )

    if result["language"] == "zh" and not prompt:
        result["language"] = "zh-TW"
        logger.info("Chinese - Traditional (zh-TW)")

    del model
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return audio, result


def align_speech(audio, result):
    """
    Aligns speech segments based on the provided audio and result metadata.

    Parameters:
    - audio (array): The audio data in a suitable format for alignment.
    - result (dict): Metadata containing information about the segments
         and language.

    Returns:
    - result (dict): Updated metadata after aligning the segments with
        the audio. This includes character-level alignments if
        'return_char_alignments' is set to True.

    Notes:
    - This function uses language-specific models to align speech segments.
    - It performs language compatibility checks and selects the
        appropriate alignment model.
    - Cleans up memory by releasing resources after alignment.
    """
    DAMHF.update(DAMT)  # lang align

    if (
        not result["language"] in DAMHF.keys()
        and not result["language"] in EXTRA_ALIGN.keys()
    ):
        logger.warning("Automatic detection: Source language not compatible with align")
        raise ValueError(
            f"Detected language {result['language']}  incompatible, "
            "you can select the source language to avoid this error."
        )
    if (
        result["language"] in EXTRA_ALIGN.keys()
        and EXTRA_ALIGN[result["language"]] == ""
    ):
        lang_name = (
            INVERTED_LANGUAGES[result["language"]]
            if result["language"] in INVERTED_LANGUAGES.keys()
            else result["language"]
        )
        logger.warning(
            "No compatible wav2vec2 model found "
            f"for the language '{lang_name}', skipping alignment."
        )
        return result

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=os.environ.get("SONITR_DEVICE"),
        model_name=(
            None
            if result["language"] in DAMHF.keys()
            else EXTRA_ALIGN[result["language"]]
        ),
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        os.environ.get("SONITR_DEVICE"),
        return_char_alignments=True,
        print_progress=False,
    )
    del model_a
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return result


diarization_models = {
    "pyannote_3.1": "pyannote/speaker-diarization-3.1",
    "pyannote_2.1": "pyannote/speaker-diarization@2.1",
    "disable": "",
}


def reencode_speakers(result):

    if result["segments"][0]["speaker"] == "SPEAKER_00":
        return result

    speaker_mapping = {}
    counter = 0

    logger.debug("Reencode speakers")

    for segment in result["segments"]:
        old_speaker = segment["speaker"]
        if old_speaker not in speaker_mapping:
            speaker_mapping[old_speaker] = f"SPEAKER_{counter:02d}"
            counter += 1
        segment["speaker"] = speaker_mapping[old_speaker]

    return result


def diarize_speech(
    audio_wav,
    result,
    min_speakers,
    max_speakers,
    YOUR_HF_TOKEN,
    model_name="pyannote/speaker-diarization@2.1",
):
    """
    Performs speaker diarization on speech segments.

    Parameters:
    - audio_wav (array): Audio data in WAV format to perform speaker diarization.
    - result (dict): Metadata containing information about speech segments and alignments.
    - min_speakers (int): Minimum number of speakers expected in the audio.
    - max_speakers (int): Maximum number of speakers expected in the audio.
    - YOUR_HF_TOKEN (str): Your Hugging Face API token for model authentication.
    - model_name (str): Name of the speaker diarization model to be used
    (default: "pyannote/speaker-diarization@2.1").

    Returns:
    - result_diarize (dict): Updated metadata after assigning speaker labels to segments.

    Notes:
    - This function utilizes a speaker diarization model to label speaker segments in the audio.
    - It assigns speakers to word-level segments based on diarization results.
    - Cleans up memory by releasing resources after diarization.
    - If only one speaker is specified, each segment is automatically assigned
    as the first speaker, eliminating the need for diarization inference.
    """

    if max(min_speakers, max_speakers) > 1 and model_name:
        try:

            diarize_model = whisperx.DiarizationPipeline(
                model_name=model_name,
                use_auth_token=YOUR_HF_TOKEN,
                device=os.environ.get("SONITR_DEVICE"),
            )

        except Exception as error:
            error_str = str(error)
            gc.collect()
            torch.cuda.empty_cache()  # noqa
            if "'NoneType' object has no attribute 'to'" in error_str:
                if model_name == diarization_models["pyannote_2.1"]:
                    raise ValueError(
                        "Accept the license agreement for using Pyannote 2.1."
                        " You need to have an account on Hugging Face and "
                        "accept the license to use the models: "
                        "https://huggingface.co/pyannote/speaker-diarization "
                        "and https://huggingface.co/pyannote/segmentation "
                        "Get your KEY TOKEN here: "
                        "https://hf.co/settings/tokens "
                    )
                elif model_name == diarization_models["pyannote_3.1"]:
                    raise ValueError(
                        "New Licence Pyannote 3.1: You need to have an account"
                        " on Hugging Face and accept the license to use the "
                        "models: https://huggingface.co/pyannote/speaker-diarization-3.1 "  # noqa
                        "and https://huggingface.co/pyannote/segmentation-3.0 "
                    )
            else:
                raise error
        diarize_segments = diarize_model(
            audio_wav, min_speakers=min_speakers, max_speakers=max_speakers
        )

        result_diarize = whisperx.assign_word_speakers(diarize_segments, result)

        for segment in result_diarize["segments"]:
            if "speaker" not in segment:
                segment["speaker"] = "SPEAKER_00"
                logger.warning(
                    f"No speaker detected in {segment['start']}. First TTS "
                    f"will be used for the segment text: {segment['text']} "
                )

        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()  # noqa
    else:
        result_diarize = result
        result_diarize["segments"] = [
            {**item, "speaker": "SPEAKER_00"} for item in result_diarize["segments"]
        ]
    return reencode_speakers(result_diarize)

# --- Repetition detector ---

# def looks_repetitive(
#         text,
#         phrase_len=4,
#         repeat_threshold=2
#     ):
#     cleaned = re.sub(r"[^\w\s']", " ", text.lower())
#     for n in range(1, phrase_len + 1):
#         regex = (
#             r"(\b\w+(?:\s+\w+){0," + str(n - 1) + r"}\b)"
#             r"(?:\s+\1){" + str(repeat_threshold) + r",}"
#         )
#         if re.search(regex, cleaned):
#             return True
#     return False

def looks_repetitive(
        text,
        min_ngram=6,
        max_ngram=12,
        repeat_threshold=2
    ):
    """Detect repeated medium-length phrases (6-12 words)."""

    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    words = cleaned.split()
    n = len(words)

    if n < min_ngram * 2:
        return False  # too short to repeat anything meaningful

    for size in range(min_ngram, max_ngram + 1):
        seen = {}
        for i in range(n - size + 1):
            chunk = " ".join(words[i:i + size])
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= repeat_threshold:
                return True

    return False


def remove_hallucinated_segments(result, min_words_threshold=12, mode="remove"):
    """
    Detect or remove entire segments that look like hallucinations.

    Args:
        result: dict with a "segments" list.
        min_words_threshold: max word count for short hallucination phrases.
        mode: "remove" to delete segments, "detect" to mark them with a flag.

    A segment is flagged/removed if it:
      1. Contains known hallucination phrases (e.g. "Subscribe", "Thanks for watching")
         AND is shorter than `min_words_threshold` words.
      2. Contains strong internal repetition such as
         "I'm going to write it. I'm going to write it."

    Modifies and returns `result` in place.
    """

    # --- Known hallucination phrases ---
    hallucinations = [
        # YouTube / outros
        "Thanks for watching",
        "Thank you for watching",
        "Subscribe to my channel",
        "Don't forget to like and subscribe",
        "Leave a comment below",
        "Follow us on social media",
        "This video is sponsored by",
        "Support us on Patreon",
        "See you next time",
        "Stay tuned for more",
        "Video brought to you by",

        # Transcription / credits
        "Subtitles by",
        "Closed captions by",
        "Transcribed by",
        "Subtitles generated by",
        "Subtitles made by the community of Amara.org",
        "Auto-generated by YouTube",
        "Captions provided by",
        "transcript Emily Beynon",

        # Broadcast / intros
        "Welcome back to my channel",
        "Welcome to my channel",
        "You're watching",
        "Live from",
        "Previously on",
        "And now back to",

        # Self-referential / filler hallucinations
        "I'm going to",
        "I'll write it",
        "Let's get started",
        "I'll take this question",
    ]

    pattern = re.compile(r"(" + "|".join(re.escape(h) for h in hallucinations) + r")", re.IGNORECASE)

    segments = result.get("segments", [])
    filtered_segments = []
    removed_any = False
    detected_segments = []

    for seg in segments:
        text = seg.get("text", "") or ""
        match_known = pattern.search(text) and len(text.split()) < min_words_threshold

        match_repeat = looks_repetitive(text)

        if match_known or match_repeat:
            seg["hallucination_flag"] = True
            detected_segments.append(seg)
            logger.info(f"Potential hallucinated segment at {seg.get('start')}s: {text[:80]!r}")

            if mode == "remove":
                removed_any = True
                continue  # skip adding this segment to filtered list

        filtered_segments.append(seg)

    if removed_any:
        logger.debug(f"Removed {len(segments) - len(filtered_segments)} hallucinated segments.")

    # Update segments in-place
    result["segments"] = filtered_segments if mode == "remove" else segments

    # Return both the modified result and detected hallucinations
    return result, detected_segments
