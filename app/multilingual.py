import re
import urllib.parse
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: Dict[str, Dict[str, Any]] = {
    "en": {
        "code": "en",
        "name": "English",
        "native": "English",
        "speech_locale": "en-IN",
        "default_voice": "en-IN-NeerjaNeural",
        "voices": [
            {"id": "en-IN-NeerjaNeural", "name": "Neerja (Indian English - Female)", "gender": "Female"},
            {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Indian English - Male)", "gender": "Male"},
            {"id": "en-US-JennyNeural", "name": "Jenny (US English - Female)", "gender": "Female"},
            {"id": "en-US-GuyNeural", "name": "Guy (US English - Male)", "gender": "Male"},
        ],
        "input_placeholder": "Ask a question about this college (e.g. admissions, courses, fees)...",
        "sample_questions": [
            "Tell me about this university",
            "What courses and degree programs are offered?",
            "What is the admission process and eligibility?",
            "How can I contact the administration?",
        ]
    },
    "ta": {
        "code": "ta",
        "name": "Tamil",
        "native": "தமிழ்",
        "speech_locale": "ta-IN",
        "default_voice": "ta-IN-PallaviNeural",
        "voices": [
            {"id": "ta-IN-PallaviNeural", "name": "பல்லவி - Pallavi (பெண் / Female)", "gender": "Female"},
            {"id": "ta-IN-ValluvarNeural", "name": "வள்ளுவர் - Valluvar (ஆண் / Male)", "gender": "Male"},
        ],
        "input_placeholder": "பல்கலைக்கழகம் பற்றி தமிழில் கேளுங்கள் (எ.கா: படிப்புகள், சேர்க்கை)...",
        "sample_questions": [
            "இந்த பல்கலைக்கழகத்தைப் பற்றி சொல்லுங்கள்",
            "இங்கு என்னென்ன படிப்புகள் மற்றும் பட்டங்கள் உள்ளன?",
            "சேர்க்கை செயல்முறை மற்றும் தகுதி என்ன?",
            "பல்கலைக்கழகத்தை எவ்வாறு தொடர்பு கொள்வது?",
        ]
    },
    "hi": {
        "code": "hi",
        "name": "Hindi",
        "native": "हिन्दी",
        "speech_locale": "hi-IN",
        "default_voice": "hi-IN-SwaraNeural",
        "voices": [
            {"id": "hi-IN-SwaraNeural", "name": "स्वरा - Swara (महिला / Female)", "gender": "Female"},
            {"id": "hi-IN-MadhurNeural", "name": "मधुर - Madhur (पुरुष / Male)", "gender": "Male"},
        ],
        "input_placeholder": "विश्वविद्यालय के बारे में हिन्दी में पूछें (उदा: पाठ्यक्रम, प्रवेश)...",
        "sample_questions": [
            "इस विश्वविद्यालय के बारे में बताइए",
            "यहाँ कौन-कौन से पाठ्यक्रम और डिग्रियां उपलब्ध हैं?",
            "प्रवेश प्रक्रिया और पात्रता क्या है?",
            "विश्वविद्यालय से संपर्क कैसे करें?",
        ]
    },
    "te": {
        "code": "te",
        "name": "Telugu",
        "native": "తెలుగు",
        "speech_locale": "te-IN",
        "default_voice": "te-IN-ShrutiNeural",
        "voices": [
            {"id": "te-IN-ShrutiNeural", "name": "శృతి - Shruti (మహిళ / Female)", "gender": "Female"},
            {"id": "te-IN-MohanNeural", "name": "మోహన్ - Mohan (పురుషుడు / Male)", "gender": "Male"},
        ],
        "input_placeholder": "విశ్వవిద్యాలయం గురించి తెలుగులో అడగండి...",
        "sample_questions": [
            "ఈ విశ్వవిద్యాలయం గురించి చెప్పండి",
            "ఏ కోర్సులు మరియు డిగ్రీలు అందుబాటులో ఉన్నాయి?",
            "ప్రవేశ ప్రక్రియ ఏమిటి?",
            "విశ్వవిద్యాలయాన్ని ఎలా సంప్రదించాలి?",
        ]
    },
    "kn": {
        "code": "kn",
        "name": "Kannada",
        "native": "ಕನ್ನಡ",
        "speech_locale": "kn-IN",
        "default_voice": "kn-IN-SapnaNeural",
        "voices": [
            {"id": "kn-IN-SapnaNeural", "name": "ಸಪ್ನಾ - Sapna (ಮಹಿಳೆ / Female)", "gender": "Female"},
            {"id": "kn-IN-GaganNeural", "name": "ಗಗನ್ - Gagan (ಪುರುಷ / Male)", "gender": "Male"},
        ],
        "input_placeholder": "ವಿಶ್ವವಿದ್ಯಾಲಯದ ಬಗ್ಗೆ ಕನ್ನಡದಲ್ಲಿ ಕೇಳಿ...",
        "sample_questions": [
            "ಈ ವಿಶ್ವವಿದ್ಯಾಲಯದ ಬಗ್ಗೆ ತಿಳಿಸಿ",
            "ಯಾವ ಕೋರ್ಸ್‌ಗಳು ಲಭ್ಯವಿವೆ?",
            "ಪ್ರವೇಶ ಪ್ರಕ್ರಿಯೆ ಏನು?",
            "ಸಂಪರ್ಕ ವಿವರಗಳು ಯಾವುವು?",
        ]
    },
    "ml": {
        "code": "ml",
        "name": "Malayalam",
        "native": "മലയാളം",
        "speech_locale": "ml-IN",
        "default_voice": "ml-IN-SobhanaNeural",
        "voices": [
            {"id": "ml-IN-SobhanaNeural", "name": "ശോഭന - Sobhana (സ്ത്രീ / Female)", "gender": "Female"},
            {"id": "ml-IN-MidhunNeural", "name": "മിഥുൻ - Midhun (പുരുഷൻ / Male)", "gender": "Male"},
        ],
        "input_placeholder": "സർവ്വകലാശാലയെക്കുറിച്ച് മലയാളത്തിൽ ചോദിക്കൂ...",
        "sample_questions": [
            "ഈ സർവ്വകലാശാലയെക്കുറിച്ച് പറയുക",
            "ഏതൊക്കെ കോഴ്സുകൾ ലഭ്യമാണ്?",
            "പ്രവേശന പ്രക്രിയ എങ്ങനെയാണ്?",
            "എങ്ങനെ ബന്ധപ്പെടാം?",
        ]
    },
    "bn": {
        "code": "bn",
        "name": "Bengali",
        "native": "বাংলা",
        "speech_locale": "bn-IN",
        "default_voice": "bn-IN-TanishaaNeural",
        "voices": [
            {"id": "bn-IN-TanishaaNeural", "name": "তানিশা - Tanishaa (মহিলা / Female)", "gender": "Female"},
            {"id": "bn-IN-BashkarNeural", "name": "ভাস্কর - Bashkar (পুরুষ / Male)", "gender": "Male"},
        ],
        "input_placeholder": "বিশ্ববিদ্যালয় সম্পর্কে বাংলায় জিজ্ঞাসা করুন...",
        "sample_questions": [
            "এই বিশ্ববিদ্যালয় সম্পর্কে বলুন",
            "কি কি কোর্স এবং ডিগ্রি দেওয়া হয়?",
            "ভর্তি প্রক্রিয়া কি?",
            "কীভাবে যোগাযোগ করবেন?",
        ]
    },
    "mr": {
        "code": "mr",
        "name": "Marathi",
        "native": "मराठी",
        "speech_locale": "mr-IN",
        "default_voice": "mr-IN-AarohiNeural",
        "voices": [
            {"id": "mr-IN-AarohiNeural", "name": "आरोही - Aarohi (महिला / Female)", "gender": "Female"},
            {"id": "mr-IN-ManoharNeural", "name": "मनोहर - Manohar (पुरुष / Male)", "gender": "Male"},
        ],
        "input_placeholder": "विद्यापीठाबद्दल मराठीत विचारा...",
        "sample_questions": [
            "या विद्यापीठाबद्दल माहिती सांगा",
            "येथे कोणते अभ्यासक्रम उपलब्ध आहेत?",
            "प्रवेश प्रक्रिया काय आहे?",
            "संपर्क कसा करावा?",
        ]
    },
    "gu": {
        "code": "gu",
        "name": "Gujarati",
        "native": "ગુજરાતી",
        "speech_locale": "gu-IN",
        "default_voice": "gu-IN-DhwaniNeural",
        "voices": [
            {"id": "gu-IN-DhwaniNeural", "name": "ધ્વનિ - Dhwani (મહિલા / Female)", "gender": "Female"},
            {"id": "gu-IN-NiranjanNeural", "name": "નિરંજન - Niranjan (પુરુષ / Male)", "gender": "Male"},
        ],
        "input_placeholder": "યુનિવર્સિટી વિશે ગુજરાતીમાં પૂછો...",
        "sample_questions": [
            "આ યુનિવર્સિટી વિશે જણાવો",
            "કયા અભ્યાસક્રમો ઉપલબ્ધ છે?",
            "પ્રવેશ પ્રક્રિયા શું છે?",
            "સંપર્ક કેવી રીતે કરવો?",
        ]
    }
}

# Offline keyword translation dictionary for domain-specific retrieval
REGIONAL_TOPIC_KEYWORDS = {
    # Tamil
    "ta": [
        (["சேர்க்கை", "விண்ணப்ப", "நுழைவு", "தகுதி", "விண்ணப்பிக்க"], "admission apply eligibility requirements deadline"),
        (["படிப்பு", "பாடப்பிரிவு", "துறை", "பட்டம்", "பாடத்திட்டம்", "பாடங்கள்", "பாடநெறி"], "courses programs degrees curriculum academic department syllabus"),
        (["கட்டணம்", "கட்டணங்கள்", "செலவு", "உதவித்தொகை", "கல்விக்கட்டணம்"], "tuition fees cost charges scholarship financial aid"),
        (["தொடர்பு", "முகவரி", "மின்னஞ்சல்", "தொலைபேசி", "எண்"], "contact address directory phone email office"),
        (["தேர்வு", "தேர்வுகள்", "மதிப்பெண்", "முடிவுகள்", "தேர்வுக்கட்டுப்பாட்டு"], "examinations exam results controller"),
        (["தொலைதூர", "ஆன்லைன்", "தூரக்கல்வி"], "distance education online cde"),
        (["விடுதி", "தங்குமிடம்", "வளாகம்", "இடம்", "எங்கு"], "hostel housing campus residence accommodation location"),
        (["வேலைவாய்ப்பு", "பிளேஸ்மென்ட்", "வேலை"], "placement careers recruitment training"),
        (["வரலாறு", "தொடக்கம்", "நிறுவப்பட்டது", "ஆரம்பிக்கப்பட்டது", "தொடங்கப்பட்டது", "பற்றி", "விவரம்", "விளக்கம்", "யார்", "வேந்தர்", "துணைவேந்தர்", "ஆண்டு"], "about history established founded overview founder leadership vice chancellor"),
    ],
    # Hindi
    "hi": [
        (["प्रवेश", "दाखिला", "आवेदन", "पात्रता"], "admission apply eligibility requirements deadline"),
        (["पाठ्यक्रम", "कोर्स", "डिग्री", "विभाग", "पढ़ाई"], "courses programs degrees curriculum academic department syllabus"),
        (["फीस", "शुल्क", "खर्च", "छात्रवृत्ति"], "tuition fees cost charges scholarship financial aid"),
        (["संपर्क", "पता", "ईमेल", "फोन", "नंबर"], "contact address directory phone email office"),
        (["परीक्षा", "परिणाम", "रिजल्ट"], "examinations exam results controller"),
        (["दूरस्थ", "ऑनलाइन"], "distance education online"),
        (["छात्रावास", "हॉस्टल", "परिसर", "कैंपस"], "hostel housing campus residence accommodation"),
        (["प्लेसमेंट", "रोजगार", "नौकरी"], "placement careers recruitment training"),
        (["इतिहास", "स्थापना", "के बारे में", "परिचय"], "about history established overview founder"),
    ],
    # Telugu
    "te": [
        (["ప్రవేశ", "దరఖాస్తు", "అర్హత"], "admission apply eligibility requirements deadline"),
        (["కోర్సులు", "డిగ్రీ", "విభాగం", "పాఠ్యాంశాలు"], "courses programs degrees curriculum academic department syllabus"),
        (["ఫీజు", "రుసుము", "ఖర్చు", "స్కాలర్‌షిప్"], "tuition fees cost charges scholarship financial aid"),
        (["సంప్రదించండి", "చిరునామా", "ఫోన్", "ఇమెయిల్"], "contact address directory phone email office"),
        (["పరీక్షలు", "ఫలితాలు"], "examinations exam results controller"),
        (["దూరవిద్య", "ఆన్‌లైన్"], "distance education online"),
        (["హాస్టల్", "వసతి", "క్యాంపస్"], "hostel housing campus residence accommodation"),
        (["ప్లేస్‌మెంట్", "ఉద్యోగం"], "placement careers recruitment training"),
        (["చరిత్ర", "గురించి"], "about history established overview founder"),
    ],
    # Kannada
    "kn": [
        (["ಪ್ರವೇಶ", "ಅರ್ಜಿ", "ಅರ್ಹತೆ"], "admission apply eligibility requirements deadline"),
        (["ಕೋರ್ಸ್", "ಪದವಿ", "ವಿಭಾಗ"], "courses programs degrees curriculum academic department syllabus"),
        (["ಶುಲ್ಕ", "ವೆಚ್ಚ", "ಸ್ಕಾಲರ್‌ಶಿಪ್"], "tuition fees cost charges scholarship financial aid"),
        (["ಸಂಪರ್ಕ", "ವಿಳಾಸ", "ಫೋನ್", "ಇಮೇಲ್"], "contact address directory phone email office"),
        (["ಪರೀಕ್ಷೆ", "ಫಲಿತಾಂಶ"], "examinations exam results controller"),
        (["ದೂರಶಿಕ್ಷಣ", "ಆನ್‌ಲೈನ್"], "distance education online"),
        (["ಹಾಸ್ಟೆಲ್", "ಕ್ಯಾಂಪಸ್"], "hostel housing campus residence accommodation"),
        (["ಬಗ್ಗೆ", "ಇತಿಹಾಸ"], "about history established overview founder"),
    ],
    # Malayalam
    "ml": [
        (["പ്രവേശനം", "അപേക്ഷ", "യോഗ്യത"], "admission apply eligibility requirements deadline"),
        (["കോഴ്സ്", "ബിരുദം", "വകുപ്പ്"], "courses programs degrees curriculum academic department syllabus"),
        (["ഫീസ്", "ചെലവ്", "സ്കോളർഷിപ്പ്"], "tuition fees cost charges scholarship financial aid"),
        (["ബന്ധപ്പെടുക", "വിലാസം", "ഫോൺ", "ഇമെയിൽ"], "contact address directory phone email office"),
        (["പരീക്ഷ", "ഫലം"], "examinations exam results controller"),
        (["വിദൂരവിദ്യാഭ്യാസം", "ഓൺലൈൻ"], "distance education online"),
        (["ഹോസ്റ്റൽ", "ക്യാമ്പസ്"], "hostel housing campus residence accommodation"),
        (["കുറിച്ച്", "ചരിത്രം"], "about history established overview founder"),
    ],
    # Bengali
    "bn": [
        (["ভর্তি", "আবেদন", "যোগ্যতা"], "admission apply eligibility requirements deadline"),
        (["কোর্স", "ডিগ্রি", "বিভাগ"], "courses programs degrees curriculum academic department syllabus"),
        (["ফি", "খরচ", "বৃত্তি"], "tuition fees cost charges scholarship financial aid"),
        (["যোগাযোগ", "ঠিকানা", "ফোন", "ইমেল"], "contact address directory phone email office"),
        (["পরীক্ষা", "ফলাফল"], "examinations exam results controller"),
        (["দূরশিক্ষা", "অনলাইন"], "distance education online"),
        (["হোস্টেল", "ক্যাম্পাস"], "hostel housing campus residence accommodation"),
        (["সম্পর্কে", "ইতিহাস"], "about history established overview founder"),
    ],
    # Marathi
    "mr": [
        (["प्रवेश", "अर्ज", "पात्रता"], "admission apply eligibility requirements deadline"),
        (["अभ्यासक्रम", "कोर्स", "पदवी", "विभाग"], "courses programs degrees curriculum academic department syllabus"),
        (["फी", "शुल्क", "खर्च", "शिष्यवृत्ती"], "tuition fees cost charges scholarship financial aid"),
        (["संपर्क", "पत्ता", "फोन", "ईमेल"], "contact address directory phone email office"),
        (["परीक्षा", "निकाल"], "examinations exam results controller"),
        (["दूरस्थ", "ऑनलाइन"], "distance education online"),
        (["वसतिगृह", "परिसर"], "hostel housing campus residence accommodation"),
        (["बद्दल", "इतिहास"], "about history established overview founder"),
    ],
    # Gujarati
    "gu": [
        (["પ્રવેશ", "અરજી", "લાયકાત"], "admission apply eligibility requirements deadline"),
        (["અભ્યાસક્રમ", "કોર્સ", "ડિગ્રી", "વિભાગ"], "courses programs degrees curriculum academic department syllabus"),
        (["ફી", "ખર્ચ", "શિષ્યવૃત્તિ"], "tuition fees cost charges scholarship financial aid"),
        (["સંપર્ક", "સરનામું", "ફોન", "ઇમેઇલ"], "contact address directory phone email office"),
        (["પરીક્ષા", "પરિણામ"], "examinations exam results controller"),
        (["વિશે", "ઇતિહાસ"], "about history established overview founder"),
    ]
}

def detect_language(text: str) -> str:
    """Detect language based on Unicode script block presence."""
    if not text:
        return "en"

    counts = {
        "ta": len(re.findall(r"[\u0B80-\u0BFF]", text)), # Tamil
        "hi": len(re.findall(r"[\u0900-\u097F]", text)), # Devanagari (Hindi / Marathi)
        "te": len(re.findall(r"[\u0C00-\u0C7F]", text)), # Telugu
        "kn": len(re.findall(r"[\u0C80-\u0CFF]", text)), # Kannada
        "ml": len(re.findall(r"[\u0D00-\u0D7F]", text)), # Malayalam
        "bn": len(re.findall(r"[\u0980-\u09FF]", text)), # Bengali
        "gu": len(re.findall(r"[\u0A80-\u0AFF]", text)), # Gujarati
    }

    max_lang = max(counts, key=counts.get)
    if counts[max_lang] > 0:
        # Disambiguate Devanagari between Marathi and Hindi
        if max_lang == "hi":
            marathi_indicators = ["आहे", "नाही", "सांगा", "बद्दल", "करावा", "विद्यापीठ", "अभ्यासक्रम"]
            if any(w in text for w in marathi_indicators):
                return "mr"
        return max_lang

    return "en"

def get_english_search_terms(query: str, lang: str) -> str:
    """Translate regional keywords into English query terms for database BM25/vector search."""
    if lang == "en":
        return query

    # 1. Try free Google Translate API if reachable
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={lang}&tl=en&dt=t&q={urllib.parse.quote(query)}"
        resp = requests.get(url, timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            translated = "".join(part[0] for part in data[0] if part[0])
            if translated and len(translated.strip()) > 3:
                return translated.strip()
    except Exception as e:
        logger.debug(f"Live translation of search query failed ({e}); using offline keyword mapper.")

    # 2. Offline keyword map fallback
    matched_terms = []
    patterns = REGIONAL_TOPIC_KEYWORDS.get(lang, [])
    for words, eng_equiv in patterns:
        if any(w in query for w in words):
            matched_terms.append(eng_equiv)

    # Extract any English alphanumeric words already present in the query (e.g. "Anna University", "MBA")
    latin_words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 1]

    combined = " ".join(latin_words + matched_terms)
    return combined if combined.strip() else query

_translation_cache: Dict[str, str] = {}

def translate_to_language(text: str, target_lang: str) -> str:
    """Translate text to the target language (supports free web API and offline fallback)."""
    if target_lang == "en" or not text.strip():
        return text

    cache_key = f"{target_lang}::{text}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    # Try live Google Translate endpoint with fast timeout
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={urllib.parse.quote(text)}"
        resp = requests.get(url, timeout=1.8)
        if resp.status_code == 200:
            data = resp.json()
            translated = "".join(part[0] for part in data[0] if part and part[0])
            if translated and len(translated.strip()) > 5:
                res = translated.strip()
                if len(_translation_cache) < 500:
                    _translation_cache[cache_key] = res
                return res
    except Exception as e:
        logger.debug(f"Live text translation to {target_lang} failed: {e}")

    return text

def get_all_voices() -> List[Dict[str, str]]:
    """Return flat list of all available voices across languages."""
    all_voices = []
    for lcode, ldata in SUPPORTED_LANGUAGES.items():
        all_voices.extend(ldata.get("voices", []))
    return all_voices
