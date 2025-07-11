from .logging_setup import logger

LANGUAGES_UNIDIRECTIONAL = {
    "Aymara (ay)": "ay",
    "Bambara (bm)": "bm",
    "Cebuano (ceb)": "ceb",
    "Chichewa (ny)": "ny",
    "Divehi (dv)": "dv",
    "Dogri (doi)": "doi",
    "Ewe (ee)": "ee",
    "Guarani (gn)": "gn",
    "Iloko (ilo)": "ilo",
    "Kinyarwanda (rw)": "rw",
    "Krio (kri)": "kri",
    "Kurdish (ku)": "ku",
    "Kirghiz (ky)": "ky",
    "Ganda (lg)": "lg",
    "Maithili (mai)": "mai",
    "Oriya (or)": "or",
    "Oromo (om)": "om",
    "Quechua (qu)": "qu",
    "Samoan (sm)": "sm",
    "Tigrinya (ti)": "ti",
    "Tsonga (ts)": "ts",
    "Akan (ak)": "ak",
    "Uighur (ug)": "ug",
}

UNIDIRECTIONAL_L_LIST = LANGUAGES_UNIDIRECTIONAL.keys()

LANGUAGES = {
    "Automatic detection": "Automatic detection",
    "Arabic (ar)": "ar",
    "Chinese - Simplified (zh-CN)": "zh",
    "Czech (cs)": "cs",
    "Danish (da)": "da",
    "Dutch (nl)": "nl",
    "English (en)": "en",
    "Finnish (fi)": "fi",
    "French (fr)": "fr",
    "German (de)": "de",
    "Greek (el)": "el",
    "Hebrew (he)": "he",
    "Hungarian (hu)": "hu",
    "Italian (it)": "it",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Persian (fa)": "fa",  # no aux gTTS
    "Polish (pl)": "pl",
    "Portuguese (pt)": "pt",
    "Russian (ru)": "ru",
    "Spanish (es)": "es",
    "Turkish (tr)": "tr",
    "Ukrainian (uk)": "uk",
    "Urdu (ur)": "ur",
    "Vietnamese (vi)": "vi",
    "Hindi (hi)": "hi",
    "Indonesian (id)": "id",
    "Bengali (bn)": "bn",
    "Telugu (te)": "te",
    "Marathi (mr)": "mr",
    "Tamil (ta)": "ta",
    "Javanese (jw|jv)": "jw",
    "Catalan (ca)": "ca",
    "Nepali (ne)": "ne",
    "Thai (th)": "th",
    "Swedish (sv)": "sv",
    "Amharic (am)": "am",
    "Welsh (cy)": "cy",  # no aux gTTS
    "Estonian (et)": "et",
    "Croatian (hr)": "hr",
    "Icelandic (is)": "is",
    "Georgian (ka)": "ka",  # no aux gTTS
    "Khmer (km)": "km",
    "Slovak (sk)": "sk",
    "Albanian (sq)": "sq",
    "Serbian (sr)": "sr",
    "Azerbaijani (az)": "az",  # no aux gTTS
    "Bulgarian (bg)": "bg",
    "Galician (gl)": "gl",  # no aux gTTS
    "Gujarati (gu)": "gu",
    "Kazakh (kk)": "kk",  # no aux gTTS
    "Kannada (kn)": "kn",
    "Lithuanian (lt)": "lt",  # no aux gTTS
    "Latvian (lv)": "lv",
    "Macedonian (mk)": "mk",  # no aux gTTS # error get align model
    "Malayalam (ml)": "ml",
    "Malay (ms)": "ms",  # error get align model
    "Romanian (ro)": "ro",
    "Sinhala (si)": "si",
    "Sundanese (su)": "su",
    "Swahili (sw)": "sw",  # error aling
    "Afrikaans (af)": "af",
    "Bosnian (bs)": "bs",
    "Latin (la)": "la",
    "Myanmar Burmese (my)": "my",
    "Norwegian (no|nb)": "no",
    "Chinese - Traditional (zh-TW)": "zh-TW",
    "Assamese (as)": "as",
    "Basque (eu)": "eu",
    "Hausa (ha)": "ha",
    "Haitian Creole (ht)": "ht",
    "Armenian (hy)": "hy",
    "Lao (lo)": "lo",
    "Malagasy (mg)": "mg",
    "Mongolian (mn)": "mn",
    "Maltese (mt)": "mt",
    "Punjabi (pa)": "pa",
    "Pashto (ps)": "ps",
    "Slovenian (sl)": "sl",
    "Shona (sn)": "sn",
    "Somali (so)": "so",
    "Tajik (tg)": "tg",
    "Turkmen (tk)": "tk",
    "Tatar (tt)": "tt",
    "Uzbek (uz)": "uz",
    "Yoruba (yo)": "yo",
    "Tagalog (tl)": "tl",
    **LANGUAGES_UNIDIRECTIONAL,
}

BASE_L_LIST = LANGUAGES.keys()
LANGUAGES_LIST = [list(BASE_L_LIST)[0]] + sorted(list(BASE_L_LIST)[1:])
INVERTED_LANGUAGES = {value: key for key, value in LANGUAGES.items()}

EXTRA_ALIGN = {
    "id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    "bn": "arijitx/wav2vec2-large-xlsr-bengali",
    "mr": "sumedh/wav2vec2-large-xlsr-marathi",
    "ta": "Amrrs/wav2vec2-large-xlsr-53-tamil",
    "jw": "cahya/wav2vec2-large-xlsr-javanese",
    "ne": "shniranjan/wav2vec2-large-xlsr-300m-nepali",
    "th": "sakares/wav2vec2-large-xlsr-thai-demo",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "am": "agkphysics/wav2vec2-large-xlsr-53-amharic",
    "cy": "Srulikbdd/Wav2Vec2-large-xlsr-welsh",
    "et": "anton-l/wav2vec2-large-xlsr-53-estonian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "is": "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-icelandic-ep10-1000h",
    "ka": "MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-Georgian",
    "km": "vitouphy/wav2vec2-xls-r-300m-khmer",
    "sk": "infinitejoy/wav2vec2-large-xls-r-300m-slovak",
    "sq": "Alimzhan/wav2vec2-large-xls-r-300m-albanian-colab",
    "sr": "dnikolic/wav2vec2-xlsr-530-serbian-colab",
    "az": "nijatzeynalov/wav2vec2-large-mms-1b-azerbaijani-common_voice15.0",
    "bg": "infinitejoy/wav2vec2-large-xls-r-300m-bulgarian",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "gu": "Harveenchadha/vakyansh-wav2vec2-gujarati-gnm-100",
    "kk": "aismlv/wav2vec2-large-xlsr-kazakh",
    "kn": "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560",
    "lt": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "lv": "anton-l/wav2vec2-large-xlsr-53-latvian",
    "mk": "",  # Konstantin-Bogdanoski/wav2vec2-macedonian-base
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "ms": "",  # Duy/wav2vec2_malay
    "ro": "anton-l/wav2vec2-large-xlsr-53-romanian",
    "si": "IAmNotAnanth/wav2vec2-large-xls-r-300m-sinhala",
    "su": "cahya/wav2vec2-large-xlsr-sundanese",
    "sw": "",  # Lians/fine-tune-wav2vec2-large-swahili
    "af": "",  # ylacombe/wav2vec2-common_voice-af-demo
    "bs": "",
    "la": "",
    "my": "",
    "no": "NbAiLab/wav2vec2-xlsr-300m-norwegian",
    "zh-TW": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "as": "",
    "eu": "",  # cahya/wav2vec2-large-xlsr-basque # verify
    "ha": "infinitejoy/wav2vec2-large-xls-r-300m-hausa",
    "ht": "",
    "hy": "infinitejoy/wav2vec2-large-xls-r-300m-armenian",  # no (.)
    "lo": "",
    "mg": "",
    "mn": "tugstugi/wav2vec2-large-xlsr-53-mongolian",
    "mt": "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h",
    "pa": "kingabzpro/wav2vec2-large-xlsr-53-punjabi",
    "ps": "aamirhs/wav2vec2-large-xls-r-300m-pashto-colab",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "sn": "",
    "so": "",
    "tg": "",
    "tk": "",  # Ragav/wav2vec2-tk
    "tt": "anton-l/wav2vec2-large-xlsr-53-tatar",
    "uz": "",  # Mekhriddin/wav2vec2-large-xls-r-300m-uzbek-colab
    "yo": "ogbi/wav2vec2-large-mms-1b-yoruba-test",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
}


def fix_code_language(translate_to, syntax="google"):
    if syntax == "google":
        # google-translator, gTTS
        replace_lang_code = {"zh": "zh-CN", "he": "iw", "zh-cn": "zh-CN"}
    elif syntax == "coqui":
        # coqui-xtts
        replace_lang_code = {"zh": "zh-cn", "zh-CN": "zh-cn", "zh-TW": "zh-cn"}

    new_code_lang = replace_lang_code.get(translate_to, translate_to)
    logger.debug(f"Fix code {translate_to} -> {new_code_lang}")
    return new_code_lang


BARK_VOICES_LIST = {
    "de_speaker_0-Male BARK": "v2/de_speaker_0",
    "de_speaker_1-Male BARK": "v2/de_speaker_1",
    "de_speaker_2-Male BARK": "v2/de_speaker_2",
    "de_speaker_3-Female BARK": "v2/de_speaker_3",
    "de_speaker_4-Male BARK": "v2/de_speaker_4",
    "de_speaker_5-Male BARK": "v2/de_speaker_5",
    "de_speaker_6-Male BARK": "v2/de_speaker_6",
    "de_speaker_7-Male BARK": "v2/de_speaker_7",
    "de_speaker_8-Female BARK": "v2/de_speaker_8",
    "de_speaker_9-Male BARK": "v2/de_speaker_9",
    "en_speaker_0-Male BARK": "v2/en_speaker_0",
    "en_speaker_1-Male BARK": "v2/en_speaker_1",
    "en_speaker_2-Male BARK": "v2/en_speaker_2",
    "en_speaker_3-Male BARK": "v2/en_speaker_3",
    "en_speaker_4-Male BARK": "v2/en_speaker_4",
    "en_speaker_5-Male BARK": "v2/en_speaker_5",
    "en_speaker_6-Male BARK": "v2/en_speaker_6",
    "en_speaker_7-Male BARK": "v2/en_speaker_7",
    "en_speaker_8-Male BARK": "v2/en_speaker_8",
    "en_speaker_9-Female BARK": "v2/en_speaker_9",
    "es_speaker_0-Male BARK": "v2/es_speaker_0",
    "es_speaker_1-Male BARK": "v2/es_speaker_1",
    "es_speaker_2-Male BARK": "v2/es_speaker_2",
    "es_speaker_3-Male BARK": "v2/es_speaker_3",
    "es_speaker_4-Male BARK": "v2/es_speaker_4",
    "es_speaker_5-Male BARK": "v2/es_speaker_5",
    "es_speaker_6-Male BARK": "v2/es_speaker_6",
    "es_speaker_7-Male BARK": "v2/es_speaker_7",
    "es_speaker_8-Female BARK": "v2/es_speaker_8",
    "es_speaker_9-Female BARK": "v2/es_speaker_9",
    "fr_speaker_0-Male BARK": "v2/fr_speaker_0",
    "fr_speaker_1-Female BARK": "v2/fr_speaker_1",
    "fr_speaker_2-Female BARK": "v2/fr_speaker_2",
    "fr_speaker_3-Male BARK": "v2/fr_speaker_3",
    "fr_speaker_4-Male BARK": "v2/fr_speaker_4",
    "fr_speaker_5-Female BARK": "v2/fr_speaker_5",
    "fr_speaker_6-Male BARK": "v2/fr_speaker_6",
    "fr_speaker_7-Male BARK": "v2/fr_speaker_7",
    "fr_speaker_8-Male BARK": "v2/fr_speaker_8",
    "fr_speaker_9-Male BARK": "v2/fr_speaker_9",
    "hi_speaker_0-Female BARK": "v2/hi_speaker_0",
    "hi_speaker_1-Female BARK": "v2/hi_speaker_1",
    "hi_speaker_2-Male BARK": "v2/hi_speaker_2",
    "hi_speaker_3-Female BARK": "v2/hi_speaker_3",
    "hi_speaker_4-Female BARK": "v2/hi_speaker_4",
    "hi_speaker_5-Male BARK": "v2/hi_speaker_5",
    "hi_speaker_6-Male BARK": "v2/hi_speaker_6",
    "hi_speaker_7-Male BARK": "v2/hi_speaker_7",
    "hi_speaker_8-Male BARK": "v2/hi_speaker_8",
    "hi_speaker_9-Female BARK": "v2/hi_speaker_9",
    "it_speaker_0-Male BARK": "v2/it_speaker_0",
    "it_speaker_1-Male BARK": "v2/it_speaker_1",
    "it_speaker_2-Female BARK": "v2/it_speaker_2",
    "it_speaker_3-Male BARK": "v2/it_speaker_3",
    "it_speaker_4-Male BARK": "v2/it_speaker_4",
    "it_speaker_5-Male BARK": "v2/it_speaker_5",
    "it_speaker_6-Male BARK": "v2/it_speaker_6",
    "it_speaker_7-Female BARK": "v2/it_speaker_7",
    "it_speaker_8-Male BARK": "v2/it_speaker_8",
    "it_speaker_9-Female BARK": "v2/it_speaker_9",
    "ja_speaker_0-Female BARK": "v2/ja_speaker_0",
    "ja_speaker_1-Female BARK": "v2/ja_speaker_1",
    "ja_speaker_2-Male BARK": "v2/ja_speaker_2",
    "ja_speaker_3-Female BARK": "v2/ja_speaker_3",
    "ja_speaker_4-Female BARK": "v2/ja_speaker_4",
    "ja_speaker_5-Female BARK": "v2/ja_speaker_5",
    "ja_speaker_6-Male BARK": "v2/ja_speaker_6",
    "ja_speaker_7-Female BARK": "v2/ja_speaker_7",
    "ja_speaker_8-Female BARK": "v2/ja_speaker_8",
    "ja_speaker_9-Female BARK": "v2/ja_speaker_9",
    "ko_speaker_0-Female BARK": "v2/ko_speaker_0",
    "ko_speaker_1-Male BARK": "v2/ko_speaker_1",
    "ko_speaker_2-Male BARK": "v2/ko_speaker_2",
    "ko_speaker_3-Male BARK": "v2/ko_speaker_3",
    "ko_speaker_4-Male BARK": "v2/ko_speaker_4",
    "ko_speaker_5-Male BARK": "v2/ko_speaker_5",
    "ko_speaker_6-Male BARK": "v2/ko_speaker_6",
    "ko_speaker_7-Male BARK": "v2/ko_speaker_7",
    "ko_speaker_8-Male BARK": "v2/ko_speaker_8",
    "ko_speaker_9-Male BARK": "v2/ko_speaker_9",
    "pl_speaker_0-Male BARK": "v2/pl_speaker_0",
    "pl_speaker_1-Male BARK": "v2/pl_speaker_1",
    "pl_speaker_2-Male BARK": "v2/pl_speaker_2",
    "pl_speaker_3-Male BARK": "v2/pl_speaker_3",
    "pl_speaker_4-Female BARK": "v2/pl_speaker_4",
    "pl_speaker_5-Male BARK": "v2/pl_speaker_5",
    "pl_speaker_6-Female BARK": "v2/pl_speaker_6",
    "pl_speaker_7-Male BARK": "v2/pl_speaker_7",
    "pl_speaker_8-Male BARK": "v2/pl_speaker_8",
    "pl_speaker_9-Female BARK": "v2/pl_speaker_9",
    "pt_speaker_0-Male BARK": "v2/pt_speaker_0",
    "pt_speaker_1-Male BARK": "v2/pt_speaker_1",
    "pt_speaker_2-Male BARK": "v2/pt_speaker_2",
    "pt_speaker_3-Male BARK": "v2/pt_speaker_3",
    "pt_speaker_4-Male BARK": "v2/pt_speaker_4",
    "pt_speaker_5-Male BARK": "v2/pt_speaker_5",
    "pt_speaker_6-Male BARK": "v2/pt_speaker_6",
    "pt_speaker_7-Male BARK": "v2/pt_speaker_7",
    "pt_speaker_8-Male BARK": "v2/pt_speaker_8",
    "pt_speaker_9-Male BARK": "v2/pt_speaker_9",
    "ru_speaker_0-Male BARK": "v2/ru_speaker_0",
    "ru_speaker_1-Male BARK": "v2/ru_speaker_1",
    "ru_speaker_2-Male BARK": "v2/ru_speaker_2",
    "ru_speaker_3-Male BARK": "v2/ru_speaker_3",
    "ru_speaker_4-Male BARK": "v2/ru_speaker_4",
    "ru_speaker_5-Female BARK": "v2/ru_speaker_5",
    "ru_speaker_6-Female BARK": "v2/ru_speaker_6",
    "ru_speaker_7-Male BARK": "v2/ru_speaker_7",
    "ru_speaker_8-Male BARK": "v2/ru_speaker_8",
    "ru_speaker_9-Female BARK": "v2/ru_speaker_9",
    "tr_speaker_0-Male BARK": "v2/tr_speaker_0",
    "tr_speaker_1-Male BARK": "v2/tr_speaker_1",
    "tr_speaker_2-Male BARK": "v2/tr_speaker_2",
    "tr_speaker_3-Male BARK": "v2/tr_speaker_3",
    "tr_speaker_4-Female BARK": "v2/tr_speaker_4",
    "tr_speaker_5-Female BARK": "v2/tr_speaker_5",
    "tr_speaker_6-Male BARK": "v2/tr_speaker_6",
    "tr_speaker_7-Male BARK": "v2/tr_speaker_7",
    "tr_speaker_8-Male BARK": "v2/tr_speaker_8",
    "tr_speaker_9-Male BARK": "v2/tr_speaker_9",
    "zh_speaker_0-Male BARK": "v2/zh_speaker_0",
    "zh_speaker_1-Male BARK": "v2/zh_speaker_1",
    "zh_speaker_2-Male BARK": "v2/zh_speaker_2",
    "zh_speaker_3-Male BARK": "v2/zh_speaker_3",
    "zh_speaker_4-Female BARK": "v2/zh_speaker_4",
    "zh_speaker_5-Male BARK": "v2/zh_speaker_5",
    "zh_speaker_6-Female BARK": "v2/zh_speaker_6",
    "zh_speaker_7-Female BARK": "v2/zh_speaker_7",
    "zh_speaker_8-Male BARK": "v2/zh_speaker_8",
    "zh_speaker_9-Female BARK": "v2/zh_speaker_9",
}

VITS_VOICES_LIST = {
    "ar-facebook-mms VITS": "facebook/mms-tts-ara",
    # 'zh-facebook-mms VITS': 'facebook/mms-tts-cmn',
    "zh_Hakka-facebook-mms VITS": "facebook/mms-tts-hak",
    "zh_MinNan-facebook-mms VITS": "facebook/mms-tts-nan",
    # 'cs-facebook-mms VITS': 'facebook/mms-tts-ces',
    # 'da-facebook-mms VITS': 'facebook/mms-tts-dan',
    "nl-facebook-mms VITS": "facebook/mms-tts-nld",
    "en-facebook-mms VITS": "facebook/mms-tts-eng",
    "fi-facebook-mms VITS": "facebook/mms-tts-fin",
    "fr-facebook-mms VITS": "facebook/mms-tts-fra",
    "de-facebook-mms VITS": "facebook/mms-tts-deu",
    "el-facebook-mms VITS": "facebook/mms-tts-ell",
    "el_Ancient-facebook-mms VITS": "facebook/mms-tts-grc",
    "he-facebook-mms VITS": "facebook/mms-tts-heb",
    "hu-facebook-mms VITS": "facebook/mms-tts-hun",
    # 'it-facebook-mms VITS': 'facebook/mms-tts-ita',
    # 'ja-facebook-mms VITS': 'facebook/mms-tts-jpn',
    "ko-facebook-mms VITS": "facebook/mms-tts-kor",
    "fa-facebook-mms VITS": "facebook/mms-tts-fas",
    "pl-facebook-mms VITS": "facebook/mms-tts-pol",
    "pt-facebook-mms VITS": "facebook/mms-tts-por",
    "ru-facebook-mms VITS": "facebook/mms-tts-rus",
    "es-facebook-mms VITS": "facebook/mms-tts-spa",
    "tr-facebook-mms VITS": "facebook/mms-tts-tur",
    "uk-facebook-mms VITS": "facebook/mms-tts-ukr",
    "ur_arabic-facebook-mms VITS": "facebook/mms-tts-urd-script_arabic",
    "ur_devanagari-facebook-mms VITS": "facebook/mms-tts-urd-script_devanagari",
    "ur_latin-facebook-mms VITS": "facebook/mms-tts-urd-script_latin",
    "vi-facebook-mms VITS": "facebook/mms-tts-vie",
    "hi-facebook-mms VITS": "facebook/mms-tts-hin",
    "hi_Fiji-facebook-mms VITS": "facebook/mms-tts-hif",
    "id-facebook-mms VITS": "facebook/mms-tts-ind",
    "bn-facebook-mms VITS": "facebook/mms-tts-ben",
    "te-facebook-mms VITS": "facebook/mms-tts-tel",
    "mr-facebook-mms VITS": "facebook/mms-tts-mar",
    "ta-facebook-mms VITS": "facebook/mms-tts-tam",
    "jw-facebook-mms VITS": "facebook/mms-tts-jav",
    "jw_Suriname-facebook-mms VITS": "facebook/mms-tts-jvn",
    "ca-facebook-mms VITS": "facebook/mms-tts-cat",
    "ne-facebook-mms VITS": "facebook/mms-tts-nep",
    "th-facebook-mms VITS": "facebook/mms-tts-tha",
    "th_Northern-facebook-mms VITS": "facebook/mms-tts-nod",
    "sv-facebook-mms VITS": "facebook/mms-tts-swe",
    "am-facebook-mms VITS": "facebook/mms-tts-amh",
    "cy-facebook-mms VITS": "facebook/mms-tts-cym",
    # "et-facebook-mms VITS": "facebook/mms-tts-est",
    # "ht-facebook-mms VITS": "facebook/mms-tts-hrv",
    "is-facebook-mms VITS": "facebook/mms-tts-isl",
    "km-facebook-mms VITS": "facebook/mms-tts-khm",
    "km_Northern-facebook-mms VITS": "facebook/mms-tts-kxm",
    # "sk-facebook-mms VITS": "facebook/mms-tts-slk",
    "sq_Northern-facebook-mms VITS": "facebook/mms-tts-sqi",
    "az_South-facebook-mms VITS": "facebook/mms-tts-azb",
    "az_North_script_cyrillic-facebook-mms VITS": "facebook/mms-tts-azj-script_cyrillic",
    "az_North_script_latin-facebook-mms VITS": "facebook/mms-tts-azj-script_latin",
    "bg-facebook-mms VITS": "facebook/mms-tts-bul",
    # "gl-facebook-mms VITS": "facebook/mms-tts-glg",
    "gu-facebook-mms VITS": "facebook/mms-tts-guj",
    "kk-facebook-mms VITS": "facebook/mms-tts-kaz",
    "kn-facebook-mms VITS": "facebook/mms-tts-kan",
    # "lt-facebook-mms VITS": "facebook/mms-tts-lit",
    "lv-facebook-mms VITS": "facebook/mms-tts-lav",
    # "mk-facebook-mms VITS": "facebook/mms-tts-mkd",
    "ml-facebook-mms VITS": "facebook/mms-tts-mal",
    "ms-facebook-mms VITS": "facebook/mms-tts-zlm",
    "ms_Central-facebook-mms VITS": "facebook/mms-tts-pse",
    "ms_Manado-facebook-mms VITS": "facebook/mms-tts-xmm",
    "ro-facebook-mms VITS": "facebook/mms-tts-ron",
    # "si-facebook-mms VITS": "facebook/mms-tts-sin",
    "sw-facebook-mms VITS": "facebook/mms-tts-swh",
    # "af-facebook-mms VITS": "facebook/mms-tts-afr",
    # "bs-facebook-mms VITS": "facebook/mms-tts-bos",
    "la-facebook-mms VITS": "facebook/mms-tts-lat",
    "my-facebook-mms VITS": "facebook/mms-tts-mya",
    # "no_Bokmål-facebook-mms VITS": "thomasht86/mms-tts-nob",  # verify
    "as-facebook-mms VITS": "facebook/mms-tts-asm",
    "as_Nagamese-facebook-mms VITS": "facebook/mms-tts-nag",
    "eu-facebook-mms VITS": "facebook/mms-tts-eus",
    "ha-facebook-mms VITS": "facebook/mms-tts-hau",
    "ht-facebook-mms VITS": "facebook/mms-tts-hat",
    "hy_Western-facebook-mms VITS": "facebook/mms-tts-hyw",
    "lo-facebook-mms VITS": "facebook/mms-tts-lao",
    "mg-facebook-mms VITS": "facebook/mms-tts-mlg",
    "mn-facebook-mms VITS": "facebook/mms-tts-mon",
    # "mt-facebook-mms VITS": "facebook/mms-tts-mlt",
    "pa_Eastern-facebook-mms VITS": "facebook/mms-tts-pan",
    # "pa_Western-facebook-mms VITS": "facebook/mms-tts-pnb",
    # "ps-facebook-mms VITS": "facebook/mms-tts-pus",
    # "sl-facebook-mms VITS": "facebook/mms-tts-slv",
    "sn-facebook-mms VITS": "facebook/mms-tts-sna",
    "so-facebook-mms VITS": "facebook/mms-tts-son",
    "tg-facebook-mms VITS": "facebook/mms-tts-tgk",
    "tk_script_arabic-facebook-mms VITS": "facebook/mms-tts-tuk-script_arabic",
    "tk_script_latin-facebook-mms VITS": "facebook/mms-tts-tuk-script_latin",
    "tt-facebook-mms VITS": "facebook/mms-tts-tat",
    "tt_Crimean-facebook-mms VITS": "facebook/mms-tts-crh",
    "uz_script_cyrillic-facebook-mms VITS": "facebook/mms-tts-uzb-script_cyrillic",
    "yo-facebook-mms VITS": "facebook/mms-tts-yor",
    "ay-facebook-mms VITS": "facebook/mms-tts-ayr",
    "bm-facebook-mms VITS": "facebook/mms-tts-bam",
    "ceb-facebook-mms VITS": "facebook/mms-tts-ceb",
    "ny-facebook-mms VITS": "facebook/mms-tts-nya",
    "dv-facebook-mms VITS": "facebook/mms-tts-div",
    "doi-facebook-mms VITS": "facebook/mms-tts-dgo",
    "ee-facebook-mms VITS": "facebook/mms-tts-ewe",
    "gn-facebook-mms VITS": "facebook/mms-tts-grn",
    "ilo-facebook-mms VITS": "facebook/mms-tts-ilo",
    "rw-facebook-mms VITS": "facebook/mms-tts-kin",
    "kri-facebook-mms VITS": "facebook/mms-tts-kri",
    "ku_script_arabic-facebook-mms VITS": "facebook/mms-tts-kmr-script_arabic",
    "ku_script_cyrillic-facebook-mms VITS": "facebook/mms-tts-kmr-script_cyrillic",
    "ku_script_latin-facebook-mms VITS": "facebook/mms-tts-kmr-script_latin",
    "ckb-facebook-mms VITS": "razhan/mms-tts-ckb",  # Verify w
    "ky-facebook-mms VITS": "facebook/mms-tts-kir",
    "lg-facebook-mms VITS": "facebook/mms-tts-lug",
    "mai-facebook-mms VITS": "facebook/mms-tts-mai",
    "or-facebook-mms VITS": "facebook/mms-tts-ory",
    "om-facebook-mms VITS": "facebook/mms-tts-orm",
    "qu_Huallaga-facebook-mms VITS": "facebook/mms-tts-qub",
    "qu_Lambayeque-facebook-mms VITS": "facebook/mms-tts-quf",
    "qu_South_Bolivian-facebook-mms VITS": "facebook/mms-tts-quh",
    "qu_North_Bolivian-facebook-mms VITS": "facebook/mms-tts-qul",
    "qu_Tena_Lowland-facebook-mms VITS": "facebook/mms-tts-quw",
    "qu_Ayacucho-facebook-mms VITS": "facebook/mms-tts-quy",
    "qu_Cusco-facebook-mms VITS": "facebook/mms-tts-quz",
    "qu_Cajamarca-facebook-mms VITS": "facebook/mms-tts-qvc",
    "qu_Eastern_Apurímac-facebook-mms VITS": "facebook/mms-tts-qve",
    "qu_Huamalíes_Dos_de_Mayo_Huánuco-facebook-mms VITS": "facebook/mms-tts-qvh",
    "qu_Margos_Yarowilca_Lauricocha-facebook-mms VITS": "facebook/mms-tts-qvm",
    "qu_North_Junín-facebook-mms VITS": "facebook/mms-tts-qvn",
    "qu_Napo-facebook-mms VITS": "facebook/mms-tts-qvo",
    "qu_San_Martín-facebook-mms VITS": "facebook/mms-tts-qvs",
    "qu_Huaylla_Wanca-facebook-mms VITS": "facebook/mms-tts-qvw",
    "qu_Northern_Pastaza-facebook-mms VITS": "facebook/mms-tts-qvz",
    "qu_Huaylas_Ancash-facebook-mms VITS": "facebook/mms-tts-qwh",
    "qu_Panao-facebook-mms VITS": "facebook/mms-tts-qxh",
    "qu_Salasaca_Highland-facebook-mms VITS": "facebook/mms-tts-qxl",
    "qu_Northern_Conchucos_Ancash-facebook-mms VITS": "facebook/mms-tts-qxn",
    "qu_Southern_Conchucos-facebook-mms VITS": "facebook/mms-tts-qxo",
    "qu_Cañar_Highland-facebook-mms VITS": "facebook/mms-tts-qxr",
    "sm-facebook-mms VITS": "facebook/mms-tts-smo",
    "ti-facebook-mms VITS": "facebook/mms-tts-tir",
    "ts-facebook-mms VITS": "facebook/mms-tts-tso",
    "ak-facebook-mms VITS": "facebook/mms-tts-aka",
    "ug_script_arabic-facebook-mms VITS": "facebook/mms-tts-uig-script_arabic",
    "ug_script_cyrillic-facebook-mms VITS": "facebook/mms-tts-uig-script_cyrillic",
    "tl-facebook-mms VITS": "facebook/mms-tts-tgl",
}

OPENAI_TTS_CODES = [
    "af",
    "ar",
    "hy",
    "az",
    "be",
    "bs",
    "bg",
    "ca",
    "zh",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "gl",
    "de",
    "el",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "mr",
    "mi",
    "ne",
    "no",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
    "zh-TW",
]

OPENAI_TTS_MODELS = [
    ">alloy OpenAI-TTS",
    ">echo OpenAI-TTS",
    ">fable OpenAI-TTS",
    ">onyx OpenAI-TTS",
    ">nova OpenAI-TTS",
    ">shimmer OpenAI-TTS",
    ">alloy HD OpenAI-TTS",
    ">echo HD OpenAI-TTS",
    ">fable HD OpenAI-TTS",
    ">onyx HD OpenAI-TTS",
    ">nova HD OpenAI-TTS",
    ">shimmer HD OpenAI-TTS",
]

ELEVENLABS_VOICES_LIST = {
    "MarcoTrox 11LABS": {
        "voice_id": "W71zT1VwIFFx3mMGH2uZ",
        "verified_languages": [
            "de",
            "pl",
            "hi",
            "pt",
            "it",
            "es",
            "ja",
            "fr",
            "ar",
            "da",
        ],
    },
    "Emilia 11LABS": {
        "voice_id": "Dt2jDzhoZC0pZw5bmy2S",
        "verified_languages": ["en", "pt", "it", "es", "cs", "pl", "hi", "fr"],
    },
    "Rachel 11LABS": {"voice_id": "21m00Tcm4TlvDq8ikWAM", "verified_languages": []},
    "Drew 11LABS": {"voice_id": "29vD33N1CtxCmqQRPOHJ", "verified_languages": []},
    "Clyde 11LABS": {"voice_id": "2EiwWnXFnvU5JabPnv8n", "verified_languages": []},
    "Paul 11LABS": {"voice_id": "5Q0t7uMcjvnagumLfvZi", "verified_languages": []},
    "Aria 11LABS": {
        "voice_id": "9BWtsMINqrJLrRacOk9x",
        "verified_languages": [
            "en",
            "en",
            "en",
            "en",
            "en",
            "en",
            "en",
            "fr",
            "zh",
            "tr",
        ],
    },
    "Domi 11LABS": {"voice_id": "AZnzlk1XvdvUeBnXmlld", "verified_languages": []},
    "Dave 11LABS": {"voice_id": "CYw3kZ02Hs0563khs1Fj", "verified_languages": []},
    "Roger 11LABS": {
        "voice_id": "CwhRBWXzGAHq8TQ4Fs17",
        "verified_languages": [
            "en",
            "en",
            "en",
            "en",
            "en",
            "en",
            "en",
            "fr",
            "de",
            "nl",
            "es",
        ],
    },
}

LANGUAGE_CODE_IN_THREE_LETTERS = {
    "Automatic detection": "aut",
    "ar": "ara",
    "zh": "chi",
    "cs": "cze",
    "da": "dan",
    "nl": "dut",
    "en": "eng",
    "fi": "fin",
    "fr": "fre",
    "de": "ger",
    "el": "gre",
    "he": "heb",
    "hu": "hun",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "fa": "per",
    "pl": "pol",
    "pt": "por",
    "ru": "rus",
    "es": "spa",
    "tr": "tur",
    "uk": "ukr",
    "ur": "urd",
    "vi": "vie",
    "hi": "hin",
    "id": "ind",
    "bn": "ben",
    "te": "tel",
    "mr": "mar",
    "ta": "tam",
    "jw": "jav",
    "ca": "cat",
    "ne": "nep",
    "th": "tha",
    "sv": "swe",
    "am": "amh",
    "cy": "cym",
    "et": "est",
    "hr": "hrv",
    "is": "isl",
    "km": "khm",
    "sk": "slk",
    "sq": "sqi",
    "sr": "srp",
}
