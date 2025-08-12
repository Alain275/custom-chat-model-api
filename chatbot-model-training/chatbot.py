from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import logging
import traceback
import joblib
import json
import random
import numpy as np
import pandas as pd
import os
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from langdetect import detect, LangDetectException
from spellchecker import SpellChecker
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chatbot")

# Create output directory
os.makedirs("model_output", exist_ok=True)

class EnhancedLanguageDetection:
    """Comprehensive language detection with pattern matching for all languages"""
    
    def __init__(self):
        self.logger = logging.getLogger("language_detection")
        
        # Initialize language patterns for improved detection
        self.initialize_language_patterns()
        
    def initialize_language_patterns(self):
        """Initialize language patterns for improved detection"""
        # Dictionary of language patterns (common words and grammatical structures)
        self.language_patterns = {
            # AFRICAN LANGUAGES
            "rw": [  # Kinyarwanda
                r'\b(nkeneye|ndashaka|murakoze|amakuru|yego|oya|ndabona)\b',
                r'\b(urubuga|umurongo|interineti|telefoni|imodoka)\b',
                r'\b(umunsi|umugoroba|igitondo|ijoro|saa|isaha)\b',
                r'\b(kubonana|guhura|kuvugana|kwandika|gusoma)\b',
                r'\burwa\b', r'\bmwiza\b', r'\bmukuru\b', r'\bmuto\b',
                r'\b(kubaza|gusubiza|gutangira|kurangiza)\b'
            ],
            "sw": [  # Swahili
                r'\b(mimi|wewe|yeye|sisi|nyinyi|wao)\b',
                r'\b(habari|asante|tafadhali|samahani|karibu)\b',
                r'\b(ndiyo|hapana|labda|sawa|kwaheri)\b',
                r'\b(leo|jana|kesho|asubuhi|mchana|jioni)\b',
                r'\b(kazi|shule|nyumba|chakula|maji|simu)\b'
            ],
            "zu": [  # Zulu
                r'\b(mina|wena|yena|thina|nina|bona)\b',
                r'\b(sawubona|ngiyabonga|uxolo|kulungile|yebo|cha)\b',
                r'\b(namhlanje|izolo|kusasa|ekuseni|emini|kusihlwa)\b'
            ],
            "ln": [  # Lingala
                r'\b(ngai|yo|ye|biso|bino|bango)\b',
                r'\b(mbote|malamu|matondo|limbisa|boye|te)\b',
                r'\b(lelo|lobi|lobi|ntongo|midi|pokwa)\b'
            ],
            "lg": [  # Luganda
                r'\b(nze|ggwe|ye|ffe|mmwe|bo)\b',
                r'\b(ssebo|nnyabo|weebale|bambi|yee|nedda)\b',
                r'\bssaawa\b', r'\benkya\b', r'\bjjo\b', r'\bggulo\b'
            ],
            "am": [  # Amharic
                r'\b(እኔ|አንተ|እሱ|እኛ|እናንተ|እነሱ)\b',
                r'\b(ሰላም|እንደምን|አመሰግናለሁ|ይቅርታ|አዎ|አይ)\b',
                r'\b(ዛሬ|ትናንት|ነገ|ጠዋት|ከሰዓት|ማታ)\b'
            ],
            "ha": [  # Hausa
                r'\b(ni|kai|shi|mu|ku|su)\b',
                r'\b(sannu|nagode|don allah|barka|ee|a\'a)\b',
                r'\b(yau|jiya|gobe|safe|rana|dare)\b'
            ],
            
            # MIDDLE EASTERN LANGUAGES
            "ar": [  # Arabic
                r'\b(أنا|أنت|هو|نحن|أنتم|هم)\b',
                r'\b(مرحبا|شكرا|من فضلك|آسف|نعم|لا)\b',
                r'\b(اليوم|أمس|غدا|صباح|ظهر|مساء)\b',
                r'[\u0600-\u06FF]+'  # Arabic Unicode range
            ],
            "fa": [  # Persian
                r'\b(من|تو|او|ما|شما|آنها)\b',
                r'\b(سلام|متشکرم|لطفا|ببخشید|بله|نه)\b',
                r'\b(امروز|دیروز|فردا|صبح|ظهر|شب)\b',
                r'[\u0600-\u06FF]+'  # Persian uses Arabic script
            ],
            "he": [  # Hebrew
                r'\b(אני|אתה|הוא|אנחנו|אתם|הם)\b',
                r'\b(שלום|תודה|בבקשה|סליחה|כן|לא)\b',
                r'\b(היום|אתמול|מחר|בוקר|צהריים|ערב)\b',
                r'[\u0590-\u05FF]+'  # Hebrew Unicode range
            ],
            
            # SOUTH ASIAN LANGUAGES
            "hi": [  # Hindi
                r'\b(मैं|तुम|वह|हम|आप|वे)\b',
                r'\b(नमस्ते|धन्यवाद|कृपया|माफ़|हां|नहीं)\b',
                r'\b(आज|कल|परसों|सुबह|दोपहर|शाम)\b',
                r'[\u0900-\u097F]+'  # Devanagari Unicode range
            ],
            "bn": [  # Bengali
                r'\b(আমি|তুমি|সে|আমরা|তোমরা|তারা)\b',
                r'\b(নমস্কার|ধন্যবাদ|দয়া করে|দুঃখিত|হ্যাঁ|না)\b',
                r'\b(আজ|গতকাল|আগামীকাল|সকাল|দুপুর|সন্ধ্যা)\b',
                r'[\u0980-\u09FF]+'  # Bengali Unicode range
            ],
            "ta": [  # Tamil
                r'\b(நான்|நீ|அவன்|நாங்கள்|நீங்கள்|அவர்கள்)\b',
                r'\b(வணக்கம்|நன்றி|தயவுசெய்து|மன்னிக்கவும்|ஆம்|இல்லை)\b',
                r'\b(இன்று|நேற்று|நாளை|காலை|மதியம்|மாலை)\b',
                r'[\u0B80-\u0BFF]+'  # Tamil Unicode range
            ],
            "ur": [  # Urdu
                r'\b(میں|تم|وہ|ہم|آپ|وہ لوگ)\b',
                r'\b(سلام|شکریہ|براہ کرم|معاف کیجیے|ہاں|نہیں)\b',
                r'\b(آج|کل|پرسوں|صبح|دوپہر|شام)\b',
                r'[\u0600-\u06FF]+'  # Urdu uses Arabic script
            ],
            
            # EAST ASIAN LANGUAGES
            "zh-cn": [  # Chinese
                r'\b(我|你|他|我们|你们|他们)\b',
                r'\b(你好|谢谢|请|对不起|是|不)\b',
                r'\b(今天|昨天|明天|早上|中午|晚上)\b',
                r'[\u4E00-\u9FFF]+'  # Chinese Unicode range
            ],
            "ja": [  # Japanese
                r'\b(私|あなた|彼|私たち|あなたたち|彼ら)\b',
                r'\b(こんにちは|ありがとう|お願いします|すみません|はい|いいえ)\b',
                r'\b(今日|昨日|明日|朝|昼|夜)\b',
                r'[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]+'  # Japanese Unicode ranges
            ],
            "ko": [  # Korean
                r'\b(나|너|그|우리|너희|그들)\b',
                r'\b(안녕하세요|감사합니다|부탁합니다|죄송합니다|네|아니요)\b',
                r'\b(오늘|어제|내일|아침|점심|저녁)\b',
                r'[\uAC00-\uD7A3]+'  # Korean Unicode range
            ],
            
            # SOUTHEAST ASIAN LANGUAGES
            "th": [  # Thai
                r'\b(ผม|คุณ|เขา|เรา|พวกคุณ|พวกเขา)\b',
                r'\b(สวัสดี|ขอบคุณ|กรุณา|ขอโทษ|ใช่|ไม่)\b',
                r'\b(วันนี้|เมื่อวาน|พรุ่งนี้|เช้า|กลางวัน|เย็น)\b',
                r'[\u0E00-\u0E7F]+'  # Thai Unicode range
            ],
            "vi": [  # Vietnamese
                r'\b(tôi|bạn|anh ấy|chúng tôi|các bạn|họ)\b',
                r'\b(xin chào|cảm ơn|làm ơn|xin lỗi|có|không)\b',
                r'\b(hôm nay|hôm qua|ngày mai|buổi sáng|buổi trưa|buổi tối)\b'
            ],
            "id": [  # Indonesian
                r'\b(saya|kamu|dia|kami|kalian|mereka)\b',
                r'\b(halo|terima kasih|tolong|maaf|ya|tidak)\b',
                r'\b(hari ini|kemarin|besok|pagi|siang|malam)\b'
            ],
            
            # EUROPEAN LANGUAGES
            "fr": [  # French
                r'\b(je|tu|il|nous|vous|ils)\b',
                r'\b(bonjour|merci|s\'il vous plaît|pardon|oui|non)\b',
                r'\b(aujourd\'hui|hier|demain|matin|midi|soir)\b'
            ],
            "de": [  # German
                r'\b(ich|du|er|wir|ihr|sie)\b',
                r'\b(hallo|danke|bitte|entschuldigung|ja|nein)\b',
                r'\b(heute|gestern|morgen|vormittag|mittag|abend)\b'
            ],
            "es": [  # Spanish
                r'\b(yo|tú|él|nosotros|vosotros|ellos)\b',
                r'\b(hola|gracias|por favor|perdón|sí|no)\b',
                r'\b(hoy|ayer|mañana|mañana|mediodía|noche)\b'
            ],
            "it": [  # Italian
                r'\b(io|tu|lui|noi|voi|loro)\b',
                r'\b(ciao|grazie|per favore|scusa|sì|no)\b',
                r'\b(oggi|ieri|domani|mattina|mezzogiorno|sera)\b'
            ],
            "pt": [  # Portuguese
                r'\b(eu|tu|ele|nós|vós|eles)\b',
                r'\b(olá|obrigado|por favor|desculpe|sim|não)\b',
                r'\b(hoje|ontem|amanhã|manhã|meio-dia|noite)\b'
            ],
            "ru": [  # Russian
                r'\b(я|ты|он|мы|вы|они)\b',
                r'\b(привет|спасибо|пожалуйста|извините|да|нет)\b',
                r'\b(сегодня|вчера|завтра|утро|день|вечер)\b',
                r'[\u0400-\u04FF]+'  # Cyrillic Unicode range
            ],
            "pl": [  # Polish
                r'\b(ja|ty|on|my|wy|oni)\b',
                r'\b(cześć|dziękuję|proszę|przepraszam|tak|nie)\b',
                r'\b(dzisiaj|wczoraj|jutro|rano|południe|wieczór)\b'
            ],
            
            # INDIGENOUS LANGUAGES OF THE AMERICAS
            "qu": [  # Quechua
                r'\b(ñuqa|qam|pay|ñuqanchik|qamkuna|paykuna)\b',
                r'\b(napaykullayki|agradiseyki|ama hina kaspa|pampachaykuway|arí|mana)\b'
            ],
            "gn": [  # Guarani
                r'\b(che|nde|ha\'e|ñande|peẽ|ha\'ekuéra)\b',
                r'\b(mba\'éichapa|aguyje|ikatu|ñembyasy|heẽ|nahániri)\b'
            ]
        }
        
        # Define language detection overrides and corrections
        self.language_corrections = {
            # Languages that are commonly misidentified and their common false identifications
            "rw": ["id", "ms", "in", "jv", "su"],  # Kinyarwanda -> Indonesian family
            "sw": ["id", "ms", "in", "jv", "su"],  # Swahili -> Indonesian family
            "ln": ["id", "ms", "in", "jv", "su"],  # Lingala -> Indonesian family
            "lg": ["id", "ms", "in", "jv", "su"],  # Luganda -> Indonesian family
            "ny": ["id", "ms", "in", "jv", "su"],  # Chichewa -> Indonesian family
            "wo": ["id", "ms", "in", "jv", "su"],  # Wolof -> Indonesian family
            "ha": ["id", "ms", "in", "so"],  # Hausa -> Indonesian/Somali
            
            # Other common misidentifications
            "am": ["ar", "fa", "he"],  # Amharic -> Arabic/Persian/Hebrew
            "ti": ["ar", "fa", "he"],  # Tigrinya -> Arabic/Persian/Hebrew
            "ne": ["hi", "ur", "bn"],  # Nepali -> Hindi/Urdu/Bengali
            "hy": ["ru", "bg", "uk"],  # Armenian -> Russian/Bulgarian/Ukrainian
            "ka": ["ru", "bg", "uk"],  # Georgian -> Russian/Bulgarian/Ukrainian
            "my": ["th", "lo", "km"],  # Burmese -> Thai/Lao/Khmer
            "lo": ["th", "my", "km"],  # Lao -> Thai/Burmese/Khmer
            "km": ["th", "lo", "my"],  # Khmer -> Thai/Lao/Burmese
            
            # Indigenous languages are often misidentified as dominant regional languages
            "qu": ["es", "pt"],  # Quechua -> Spanish/Portuguese
            "gn": ["es", "pt"],  # Guarani -> Spanish/Portuguese
            "ay": ["es", "pt"],  # Aymara -> Spanish/Portuguese
            "nv": ["en", "es"],  # Navajo -> English/Spanish
            "kl": ["da", "no", "sv"],  # Greenlandic -> Danish/Norwegian/Swedish
        }
        
        # Script identification for languages with unique scripts
        self.script_identifiers = {
            # Define Unicode ranges for scripts that uniquely identify languages
            "ar": r'[\u0600-\u06FF]+',  # Arabic script
            "he": r'[\u0590-\u05FF]+',  # Hebrew script
            "hi": r'[\u0900-\u097F]+',  # Devanagari script
            "bn": r'[\u0980-\u09FF]+',  # Bengali script
            "ta": r'[\u0B80-\u0BFF]+',  # Tamil script
            "th": r'[\u0E00-\u0E7F]+',  # Thai script
            "zh-cn": r'[\u4E00-\u9FFF]+',  # Chinese characters
            "ja": r'[\u3040-\u30FF]+',  # Japanese Hiragana/Katakana
            "ko": r'[\uAC00-\uD7A3]+',  # Korean Hangul
            "my": r'[\u1000-\u109F]+',  # Myanmar script
            "ka": r'[\u10A0-\u10FF]+',  # Georgian script
            "hy": r'[\u0530-\u058F]+',  # Armenian script
            "el": r'[\u0370-\u03FF]+',  # Greek script
        }
    
    def get_script_match(self, text):
        """Identify language by script (for languages with unique scripts)"""
        for lang, pattern in self.script_identifiers.items():
            if re.search(pattern, text):
                return lang
        return None
    
    def count_pattern_matches(self, text, lang_code):
        """Count matches for a specific language's patterns"""
        if lang_code not in self.language_patterns:
            return 0
            
        matches = 0
        for pattern in self.language_patterns[lang_code]:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        return matches
    
    def detect_language(self, text):
        """Enhanced language detection with multiple fallback mechanisms"""
        # Skip detection for very short text
        if not text or len(text.strip()) < 3:
            return "en"
            
        # Step 1: Try to identify by script (for languages with unique scripts)
        script_lang = self.get_script_match(text)
        if script_lang:
            self.logger.info(f"Language detected by script: {script_lang}")
            return script_lang
            
        # Step 2: Use standard language detection
        try:
            detected_lang = detect(text)
            
            # Map Chinese variants
            if detected_lang in ["zh-cn", "zh-tw", "zh"]:
                detected_lang = "zh-cn"
                
            # Check if this is a commonly misidentified language
            potential_corrections = []
            for actual_lang, misidentified_as in self.language_corrections.items():
                if detected_lang in misidentified_as:
                    potential_corrections.append(actual_lang)
                    
            # Step 3: If potential correction exists, check patterns
            if potential_corrections:
                best_score = 0
                best_lang = None
                
                for lang in potential_corrections:
                    score = self.count_pattern_matches(text, lang)
                    if score > best_score:
                        best_score = score
                        best_lang = lang
                
                # If we have a good match with patterns, use it
                if best_score >= 2:
                    self.logger.info(f"Corrected language '{detected_lang}' to '{best_lang}' based on {best_score} pattern matches")
                    return best_lang
            
            # Step 4: Double-check detected language with its own patterns
            if detected_lang in self.language_patterns:
                score = self.count_pattern_matches(text, detected_lang)
                # If detected language doesn't match its own patterns well, try others
                if score < 1:
                    # Check other languages
                    best_score = 0
                    best_lang = detected_lang  # Default to keep original
                    
                    # Check all languages with patterns
                    for lang in self.language_patterns:
                        lang_score = self.count_pattern_matches(text, lang)
                        if lang_score > best_score:
                            best_score = lang_score
                            best_lang = lang
                    
                    if best_lang != detected_lang and best_score >= 2:
                        self.logger.info(f"Corrected language '{detected_lang}' to '{best_lang}' based on pattern analysis")
                        return best_lang
            
            return detected_lang
            
        except LangDetectException:
            self.logger.warning(f"Could not detect language for: {text}")
            # Fallback to pattern matching for all languages
            best_lang = "en"
            best_score = 0
            
            for lang, patterns in self.language_patterns.items():
                score = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
                if score > best_score:
                    best_score = score
                    best_lang = lang
            
            if best_score > 0:
                self.logger.info(f"Detected language '{best_lang}' using pattern matching after detection failure")
                return best_lang
            
            return "en"  # Default to English if all detection methods fail

class ComprehensiveMultilingualSupport:
    def __init__(self):
        """Initialize comprehensive multilingual support"""
        self.logger = logging.getLogger("multilingual")
        
        # Initialize enhanced language detection
        self.enhanced_detector = EnhancedLanguageDetection()
        
        # EXTENDED COMPREHENSIVE LANGUAGE MAP
        # This includes many more languages from all regions of the world
        self.language_map = {
            # AFRICA
            "am": {"name": "Amharic", "native": "አማርኛ", "region": "East Africa"},
            "ha": {"name": "Hausa", "native": "Hausa", "region": "West Africa"},
            "ig": {"name": "Igbo", "native": "Igbo", "region": "West Africa"},
            "yo": {"name": "Yoruba", "native": "Yorùbá", "region": "West Africa"},
            "sw": {"name": "Swahili", "native": "Kiswahili", "region": "East Africa"},
            "zu": {"name": "Zulu", "native": "isiZulu", "region": "Southern Africa"},
            "xh": {"name": "Xhosa", "native": "isiXhosa", "region": "Southern Africa"},
            "af": {"name": "Afrikaans", "native": "Afrikaans", "region": "Southern Africa"},
            "ln": {"name": "Lingala", "native": "Lingála", "region": "Central Africa"},
            "lg": {"name": "Luganda", "native": "Luganda", "region": "East Africa"},
            "rw": {"name": "Kinyarwanda", "native": "Kinyarwanda", "region": "Central Africa"},
            "sn": {"name": "Shona", "native": "chiShona", "region": "Southern Africa"},
            "so": {"name": "Somali", "native": "Soomaali", "region": "East Africa"},
            "ny": {"name": "Chichewa", "native": "Chichewa", "region": "Southeast Africa"},
            "mg": {"name": "Malagasy", "native": "Malagasy", "region": "East Africa"},
            "wo": {"name": "Wolof", "native": "Wolof", "region": "West Africa"},
            "ff": {"name": "Fulfulde", "native": "Fulfulde", "region": "West Africa"},
            "bm": {"name": "Bambara", "native": "Bamanankan", "region": "West Africa"},
            "ak": {"name": "Akan", "native": "Akan", "region": "West Africa"},
            "st": {"name": "Sesotho", "native": "Sesotho", "region": "Southern Africa"},
            "tn": {"name": "Setswana", "native": "Setswana", "region": "Southern Africa"},
            "ts": {"name": "Tsonga", "native": "Xitsonga", "region": "Southern Africa"},
            "om": {"name": "Oromo", "native": "Afaan Oromoo", "region": "East Africa"},
            "ti": {"name": "Tigrinya", "native": "ትግርኛ", "region": "East Africa"},
            
            # MIDDLE EAST & NORTH AFRICA
            "ar": {"name": "Arabic", "native": "العربية", "region": "Middle East/North Africa"},
            "fa": {"name": "Persian", "native": "فارسی", "region": "Middle East"},
            "he": {"name": "Hebrew", "native": "עברית", "region": "Middle East"},
            "mt": {"name": "Maltese", "native": "Malti", "region": "North Africa/Europe"},
            "dv": {"name": "Dhivehi", "native": "ދިވެހި", "region": "South Asia/Middle East"},
            "ku": {"name": "Kurdish", "native": "Kurdî", "region": "Middle East"},
            "ps": {"name": "Pashto", "native": "پښتو", "region": "Central/South Asia"},
            "sd": {"name": "Sindhi", "native": "سنڌي", "region": "South Asia"},
            "ug": {"name": "Uyghur", "native": "ئۇيغۇرچە", "region": "Central Asia"},
            "az": {"name": "Azerbaijani", "native": "Azərbaycan", "region": "Central Asia/Caucasus"},
            "hy": {"name": "Armenian", "native": "Հայերեն", "region": "Caucasus"},
            "ka": {"name": "Georgian", "native": "ქართული", "region": "Caucasus"},
            
            # SOUTH ASIA
            "bn": {"name": "Bengali", "native": "বাংলা", "region": "South Asia"},
            "hi": {"name": "Hindi", "native": "हिन्दी", "region": "South Asia"},
            "ur": {"name": "Urdu", "native": "اردو", "region": "South Asia"},
            "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "region": "South Asia"},
            "gu": {"name": "Gujarati", "native": "ગુજરાતી", "region": "South Asia"},
            "mr": {"name": "Marathi", "native": "मराठी", "region": "South Asia"},
            "ta": {"name": "Tamil", "native": "தமிழ்", "region": "South Asia"},
            "te": {"name": "Telugu", "native": "తెలుగు", "region": "South Asia"},
            "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "region": "South Asia"},
            "ml": {"name": "Malayalam", "native": "മലയാളം", "region": "South Asia"},
            "si": {"name": "Sinhala", "native": "සිංහල", "region": "South Asia"},
            "ne": {"name": "Nepali", "native": "नेपाली", "region": "South Asia"},
            "as": {"name": "Assamese", "native": "অসমীয়া", "region": "South Asia"},
            "or": {"name": "Odia", "native": "ଓଡ଼ିଆ", "region": "South Asia"},
            "sa": {"name": "Sanskrit", "native": "संस्कृतम्", "region": "South Asia"},
            "bho": {"name": "Bhojpuri", "native": "भोजपुरी", "region": "South Asia"},
            "mai": {"name": "Maithili", "native": "मैथिली", "region": "South Asia"},
            
            # EAST ASIA
            "zh-cn": {"name": "Chinese (Simplified)", "native": "简体中文", "region": "East Asia"},
            "zh-tw": {"name": "Chinese (Traditional)", "native": "繁體中文", "region": "East Asia"},
            "ja": {"name": "Japanese", "native": "日本語", "region": "East Asia"},
            "ko": {"name": "Korean", "native": "한국어", "region": "East Asia"},
            "mn": {"name": "Mongolian", "native": "Монгол", "region": "East Asia"},
            "bo": {"name": "Tibetan", "native": "བོད་སྐད་", "region": "East/Central Asia"},
            
            # SOUTHEAST ASIA
            "id": {"name": "Indonesian", "native": "Bahasa Indonesia", "region": "Southeast Asia"},
            "ms": {"name": "Malay", "native": "Bahasa Melayu", "region": "Southeast Asia"},
            "jv": {"name": "Javanese", "native": "Basa Jawa", "region": "Southeast Asia"},
            "su": {"name": "Sundanese", "native": "Basa Sunda", "region": "Southeast Asia"},
            "tl": {"name": "Tagalog", "native": "Tagalog", "region": "Southeast Asia"},
            "ceb": {"name": "Cebuano", "native": "Cebuano", "region": "Southeast Asia"},
            "ilo": {"name": "Ilocano", "native": "Ilokano", "region": "Southeast Asia"},
            "th": {"name": "Thai", "native": "ไทย", "region": "Southeast Asia"},
            "lo": {"name": "Lao", "native": "ລາວ", "region": "Southeast Asia"},
            "km": {"name": "Khmer", "native": "ខ្មែរ", "region": "Southeast Asia"},
            "my": {"name": "Burmese", "native": "မြန်မာဘာသာ", "region": "Southeast Asia"},
            "vi": {"name": "Vietnamese", "native": "Tiếng Việt", "region": "Southeast Asia"},
            
            # EUROPE
            "en": {"name": "English", "native": "English", "region": "Global/Europe"},
            "fr": {"name": "French", "native": "Français", "region": "Europe/Global"},
            "de": {"name": "German", "native": "Deutsch", "region": "Europe"},
            "es": {"name": "Spanish", "native": "Español", "region": "Europe/Latin America"},
            "it": {"name": "Italian", "native": "Italiano", "region": "Europe"},
            "pt": {"name": "Portuguese", "native": "Português", "region": "Europe/Latin America"},
            "ru": {"name": "Russian", "native": "русский", "region": "Europe/Asia"},
            "nl": {"name": "Dutch", "native": "Nederlands", "region": "Europe"},
            "sv": {"name": "Swedish", "native": "svenska", "region": "Europe"},
            "no": {"name": "Norwegian", "native": "norsk", "region": "Europe"},
            "da": {"name": "Danish", "native": "dansk", "region": "Europe"},
            "fi": {"name": "Finnish", "native": "suomi", "region": "Europe"},
            "pl": {"name": "Polish", "native": "polski", "region": "Europe"},
            "uk": {"name": "Ukrainian", "native": "українська", "region": "Europe"},
            "cs": {"name": "Czech", "native": "čeština", "region": "Europe"},
            "sk": {"name": "Slovak", "native": "slovenčina", "region": "Europe"},
            "hu": {"name": "Hungarian", "native": "magyar", "region": "Europe"},
            "ro": {"name": "Romanian", "native": "română", "region": "Europe"},
            "bg": {"name": "Bulgarian", "native": "български", "region": "Europe"},
            "el": {"name": "Greek", "native": "Ελληνικά", "region": "Europe"},
            "tr": {"name": "Turkish", "native": "Türkçe", "region": "Europe/Asia"},
            "sr": {"name": "Serbian", "native": "српски", "region": "Europe"},
            "hr": {"name": "Croatian", "native": "hrvatski", "region": "Europe"},
            "bs": {"name": "Bosnian", "native": "bosanski", "region": "Europe"},
            "sq": {"name": "Albanian", "native": "shqip", "region": "Europe"},
            "mk": {"name": "Macedonian", "native": "македонски", "region": "Europe"},
            "sl": {"name": "Slovenian", "native": "slovenščina", "region": "Europe"},
            "lv": {"name": "Latvian", "native": "latviešu", "region": "Europe"},
            "lt": {"name": "Lithuanian", "native": "lietuvių", "region": "Europe"},
            "et": {"name": "Estonian", "native": "eesti", "region": "Europe"},
            "is": {"name": "Icelandic", "native": "íslenska", "region": "Europe"},
            "ga": {"name": "Irish", "native": "Gaeilge", "region": "Europe"},
            "gd": {"name": "Scottish Gaelic", "native": "Gàidhlig", "region": "Europe"},
            "cy": {"name": "Welsh", "native": "Cymraeg", "region": "Europe"},
            "eu": {"name": "Basque", "native": "Euskara", "region": "Europe"},
            "ca": {"name": "Catalan", "native": "català", "region": "Europe"},
            "gl": {"name": "Galician", "native": "galego", "region": "Europe"},
            "oc": {"name": "Occitan", "native": "Occitan", "region": "Europe"},
            "br": {"name": "Breton", "native": "Brezhoneg", "region": "Europe"},
            "co": {"name": "Corsican", "native": "Corsu", "region": "Europe"},
            "fy": {"name": "Frisian", "native": "Frysk", "region": "Europe"},
            "lb": {"name": "Luxembourgish", "native": "Lëtzebuergesch", "region": "Europe"},
            
            # CENTRAL ASIA
            "kk": {"name": "Kazakh", "native": "қазақ тілі", "region": "Central Asia"},
            "ky": {"name": "Kyrgyz", "native": "Кыргызча", "region": "Central Asia"},
            "tg": {"name": "Tajik", "native": "тоҷикӣ", "region": "Central Asia"},
            "tk": {"name": "Turkmen", "native": "Türkmençe", "region": "Central Asia"},
            "uz": {"name": "Uzbek", "native": "O'zbek", "region": "Central Asia"},
            "tt": {"name": "Tatar", "native": "татарча", "region": "Central Asia/Europe"},
            
            # AMERICAS - INDIGENOUS
            "ay": {"name": "Aymara", "native": "Aymar aru", "region": "South America"},
            "gn": {"name": "Guarani", "native": "Avañe'ẽ", "region": "South America"},
            "qu": {"name": "Quechua", "native": "Runa Simi", "region": "South America"},
            "nv": {"name": "Navajo", "native": "Diné bizaad", "region": "North America"},
            "oj": {"name": "Ojibwe", "native": "ᐊᓂᔑᓈᐯᒧᐎᓐ", "region": "North America"},
            "cr": {"name": "Cree", "native": "ᓀᐦᐃᔭᐍᐏᐣ", "region": "North America"},
            "iu": {"name": "Inuktitut", "native": "ᐃᓄᒃᑎᑐᑦ", "region": "North America"},
            "kl": {"name": "Greenlandic", "native": "Kalaallisut", "region": "North America"},
            "ht": {"name": "Haitian Creole", "native": "Kreyòl ayisyen", "region": "Caribbean"},
            
            # PACIFIC
            "haw": {"name": "Hawaiian", "native": "ʻŌlelo Hawaiʻi", "region": "Pacific"},
            "mi": {"name": "Maori", "native": "Māori", "region": "Pacific"},
            "sm": {"name": "Samoan", "native": "Gagana Samoa", "region": "Pacific"},
            "to": {"name": "Tongan", "native": "Lea faka-Tonga", "region": "Pacific"},
            "ty": {"name": "Tahitian", "native": "Reo Tahiti", "region": "Pacific"},
            "fj": {"name": "Fijian", "native": "Na Vosa Vakaviti", "region": "Pacific"},
            "tet": {"name": "Tetum", "native": "Tetun", "region": "Pacific/Southeast Asia"},
            "gil": {"name": "Gilbertese", "native": "Taetae ni Kiribati", "region": "Pacific"},
            
            # ARTIFICIAL/CONSTRUCTED
            "eo": {"name": "Esperanto", "native": "Esperanto", "region": "Constructed"},
            "ia": {"name": "Interlingua", "native": "Interlingua", "region": "Constructed"},
            "io": {"name": "Ido", "native": "Ido", "region": "Constructed"},
            "vo": {"name": "Volapük", "native": "Volapük", "region": "Constructed"},
            
            # SIGN LANGUAGES (though text detection won't work for these)
            "ase": {"name": "American Sign Language", "native": "ASL", "region": "Sign Language"},
            "bfi": {"name": "British Sign Language", "native": "BSL", "region": "Sign Language"},
            "fsl": {"name": "French Sign Language", "native": "LSF", "region": "Sign Language"}
        }
        
    def detect_language(self, text):
        """Detect the language of input text with enhanced detection"""
        # Use the enhanced detector instead of the standard one
        return self.enhanced_detector.detect_language(text)
    
    def get_language_info(self, lang_code):
        """Get information about a language"""
        if lang_code in self.language_map:
            return self.language_map[lang_code]
        else:
            # For languages not in our map but detected by langdetect
            return {"name": f"Unknown ({lang_code})", "native": lang_code, "region": "Unknown"}
    
    def get_language_not_supported_response(self, lang_code):
        """Get response for languages that are not supported"""
        language_info = self.get_language_info(lang_code)
        
        # English response informing user that only English is supported
        response = f"I detected that you're writing in {language_info['name']} ({language_info['native']}). Currently, I only support English. Please write your message in English."
        
        return {
            "text": response,
            "intent": "language_not_supported",
            "detected_language": lang_code,
            "language_name": language_info["name"],
            "native_name": language_info["native"],
            "region": language_info["region"],
            "confidence": 0.99
        }

def train_model():
    """Train and save the model if it doesn't exist"""
    logger.info("Training new model...")
    
    # Load dataset
    df = pd.read_csv("data/chatbot_data.csv")
    logger.info(f"Loaded {len(df)} training examples across {df['intent'].nunique()} intents")
    
    # Prepare data
    X = df['query'].values
    y = df['intent'].values
    
    # Split data with stratification to ensure each intent is represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with TF-IDF and SVM
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LinearSVC(C=5))
    ])
    
    # Train the model
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    logger.info(f"Test set accuracy: {accuracy:.4f}")
    
    # Save the model
    joblib.dump(pipeline, "model_output/intent_classifier.pkl")
    logger.info("Model saved to model_output/intent_classifier.pkl")
    
    # Save model metadata
    intents = list(sorted(df['intent'].unique()))
    model_metadata = {
        "model_type": "sklearn_pipeline",
        "vectorizer": "tfidf",
        "classifier": "linear_svc",
        "num_intents": len(intents),
        "intents": intents
    }
    
    with open("model_output/model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info("Model metadata saved")
    
    return pipeline

# Simple chatbot model class
class SimpleChatbotModel:
    def __init__(self):
        # Initialize spell checker
        self.initialize_spellchecker()
        
        # Initialize multilingual support
        self.multilingual = ComprehensiveMultilingualSupport()
        
        # Check if model exists
        if not os.path.exists("model_output/intent_classifier.pkl"):
            logger.info("Model file not found, training new model...")
            self.model = train_model()
        else:
            logger.info("Loading existing model...")
            self.model = joblib.load("model_output/intent_classifier.pkl")
        
        # Load intent responses
        with open("data/intent_responses.json", "r") as f:
            intent_responses = json.load(f)
        
        # Create intent-response mapping
        self.responses = {}
        for intent_data in intent_responses:
            self.responses[intent_data["intent"]] = intent_data["responses"]
        
        logger.info(f"Loaded model with {len(self.responses)} intents")
    
    def initialize_spellchecker(self):
        """Initialize the spell checker with domain-specific words"""
        self.spell = SpellChecker()
        
        # Add domain-specific words to the dictionary
        domain_words = [
            'innovationhub', 'ecommerce', 'webapp', 'website', 'iOS', 
            'Android', 'React', 'Angular', 'Node.js', 'app', 'development',
            'consultation', 'meeting', 'pricing', 'contact', 'email',
            'mobile', 'application', 'membership', 'projects', 'demo',
            'homepage', 'schedule', 'hire', 'portfolio', 'responsive'
        ]
        
        # Use load_words instead of add_words
        for word in domain_words:
            self.spell.word_frequency.load_words([word])

    def preprocess_text(self, text):
        """Advanced spell checking and text normalization"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        # Common spelling corrections dictionary
        corrections = {
            'helo': 'hello',
            'helllo': 'hello',
            'hellow': 'hello',
            'thanx': 'thanks',
            'thx': 'thanks',
            'thankss': 'thanks',
            'websit': 'website',
            'webiste': 'website',
            'develoment': 'development',
            'developemnt': 'development',
            'aplications': 'applications',
            'aplication': 'application',
            'contct': 'contact',
            'scedule': 'schedule',
            'shcedule': 'schedule',
            'meetng': 'meeting',
            'whats': "what's",
            'ur': 'your',
            'plz': 'please',
            'pls': 'please',
            'u': 'you',
            'r': 'are',
            '2': 'to',
            '4': 'for'
        }
        
        # Apply direct corrections for known misspellings
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip very short words, URLs, or words with special characters
            if len(word) <= 2 or '/' in word or '@' in word:
                corrected_words.append(word)
                continue
                
            # Check if it's a known misspelling
            if word in corrections:
                corrected_words.append(corrections[word])
            # Apply spell checking
            elif word not in self.spell:
                correction = self.spell.correction(word)
                if correction:
                    corrected_words.append(correction)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        # Log if corrections were made
        if corrected_text != text:
            logger.debug(f"Corrected: '{text}' → '{corrected_text}'")
            
        return corrected_text
    
    def predict(self, text):
        """Safely predict intent with auto-spelling correction"""
        # Apply spelling correction for English text
        corrected_text = self.preprocess_text(text)
        
        try:
            # Use the pipeline's predict method which handles text inputs properly
            intent = self.model.predict([corrected_text])[0]
            
            # Use a fixed confidence value
            confidence = 0.8
            
            return intent, confidence
        except Exception as e:
            logger.error(f"Error predicting intent: {str(e)}")
            # Return a fallback intent if prediction fails
            return "greeting", 0.5
    
    def get_response(self, intent):
        """Get a response for the predicted intent"""
        if intent in self.responses:
            # Return a random response for the intent
            return random.choice(self.responses[intent])
        else:
            return "I'm not sure how to respond to that."
    
    def process_message(self, message):
        """Process a message and return a response - English only"""
        # Detect language first using enhanced detection
        lang_code = self.multilingual.detect_language(message)
        
        # If not English, return language not supported message
        if lang_code != "en":
            # Get language not supported response
            return self.multilingual.get_language_not_supported_response(lang_code)
        else:
            # For English, apply our intent recognition with spell correction
            intent, confidence = self.predict(message)
            response_text = self.get_response(intent)
            
            return {
                "text": response_text,
                "intent": intent,
                "confidence": confidence
            }

# Define the lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code - load model when the app starts
    global model
    try:
        logger.info("Loading chatbot model...")
        model = SimpleChatbotModel()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
    
    yield  # This is where FastAPI serves requests
    
    # Shutdown code (if needed)
    logger.info("Shutting down chatbot API")

# Initialize FastAPI with the lifespan
app = FastAPI(
    title="English-Only Chatbot API",
    description="API for the trained chatbot model - supports English only",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PredictRequest(BaseModel):
    message: str
    context: Optional[List[Dict[str, Any]]] = []
    options: Optional[Dict[str, Any]] = {}

# Response model
class PredictResponse(BaseModel):
    text: str
    confidence: float
    intent: str
    sources: List[str] = ["scheduling_system", "calendar_api"]
    processingTime: int
    detected_language: Optional[str] = None
    language_name: Optional[str] = None

# Global model instance
model = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Process the message
        result = model.process_message(request.message)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Create response
        response = PredictResponse(
            text=result["text"],
            confidence=result.get("confidence", 0.8),
            intent=result["intent"],
            sources=["scheduling_system", "calendar_api"],
            processingTime=processing_time,
            detected_language=result.get("detected_language", "en"),
            language_name=result.get("language_name", "English")
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Test function
def test_chatbot():
    chatbot = SimpleChatbotModel()
    test_queries = [
        "hello there",
        "can you build me a mobile app?",
        "how much do you charge for websites?",
        "I need to schedule a meeting",
        "what's your contact information",
        "thanks for your help",
        "goodbye",
        # Test spelling errors
        "helo frend",
        "websit develoment",
        "shedule meetng",
        # Test non-English
        "hola como estas",
        "bonjour comment ça va",
        "こんにちは",
        "مرحبا",
        "Habari yako?",  # Swahili - "How are you?"
        "Ninahitaji kupanga mkutano",  # Swahili - "I need to schedule a meeting"
        "Nkeneye urubuga rwa interineti",  # Kinyarwanda - "I need a website"
    ]
    
    print("\nTesting chatbot responses with English-only mode:")
    for query in test_queries:
        result = chatbot.process_message(query)
        print(f"\nQ: {query}")
        print(f"A: {result['text']}")
        print(f"Intent: {result['intent']} (Confidence: {result.get('confidence', 0.0):.2f})")
        if "detected_language" in result and result["detected_language"] != "en":
            print(f"Detected Language: {result.get('language_name', '')} ({result['detected_language']})")

# Run the server if this file is executed directly
if __name__ == "__main__":
    # First test the chatbot
    test_chatbot()
    
    # Then run the API server
    import uvicorn
    print("\nStarting API server...")
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=False)