# app.py
import os
import json
import logging
import time
import tempfile
import gc
import hashlib
import random
from datetime import datetime
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# -------------------------
# 基本設定 / ログ
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restee_safe")

# Config
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
REPO_ID = st.secrets.get("REPO_ID", os.environ.get("REPO_ID", ""))
DATA_DIR = "data_store"
MODEL_DIR = "models"
HISTORY_DIR = os.path.join(DATA_DIR, "history")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

DATA_PARQUET = os.path.join(DATA_DIR, "dataset.parquet")

FEATURE_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
PCA_FILE = os.path.join(MODEL_DIR, "pca.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

MEMORY_THRESHOLD_WARN = int(os.environ.get("MEM_WARN_MB", 2000))
MEMORY_THRESHOLD_CRITICAL = int(os.environ.get("MEM_CRIT_MB", 2400))

MODEL_CONFIG = {
    "body": {"model": os.path.join(MODEL_DIR, "body_model.txt"), "label": "label_body"},
    "brain": {"model": os.path.join(MODEL_DIR, "brain_model.txt"), "label": "label_brain"},
    "mental": {"model": os.path.join(MODEL_DIR, "mental_model.txt"), "label": "label_mental"},
}

OPENSMILE_VALID_RATIO = float(os.environ.get("OPENSMILE_VALID_RATIO", 0.6))
DEBUG = os.getenv("DEBUG", "0") == "1"

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)
sys_random = random.SystemRandom()

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

# -------------------------
# 品質統計
# -------------------------
QUALITY_STATS = {
    "total_attempts": 0,
    "quality_ok_count": 0,
    "quality_ng_count": 0,
    "smile_success_count": 0,
    "duplicate_blocked_count": 0,
}

def log_quality_stats():
    if DEBUG:
        logger.info(f"[QUALITY] total={QUALITY_STATS['total_attempts']}, ok={QUALITY_STATS['quality_ok_count']}, ng={QUALITY_STATS['quality_ng_count']}, smile_ok={QUALITY_STATS['smile_success_count']}, dup_blocked={QUALITY_STATS['duplicate_blocked_count']}")

# -------------------------
# FileLock
# -------------------------
try:
    from filelock import FileLock, Timeout
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False
    logger.warning("filelock not installed. Using fallback lock mechanism.")

def log_memory(label: str):
    try:
        gc.collect()
    except Exception:
        pass
    if HAS_PSUTIL:
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / (1024 ** 2)
        if mem_mb > MEMORY_THRESHOLD_CRITICAL:
            logger.critical(f"[MEM][CRIT] {label}: {mem_mb:.1f} MB")
        elif mem_mb > MEMORY_THRESHOLD_WARN:
            logger.warning(f"[MEM][WARN] {label}: {mem_mb:.1f} MB")
        else:
            logger.info(f"[MEM] {label}: {mem_mb:.1f} MB")
        return mem_mb
    return None

# -------------------------
# モデル / ライブラリ
# -------------------------
@st.cache_resource
def get_asr_model_safe():
    try:
        from transformers import MoonshineForConditionalGeneration, AutoProcessor
        import torch
        device = "cpu"
        torch_dtype = torch.float32
        model = MoonshineForConditionalGeneration.from_pretrained(
            "UsefulSensors/moonshine-tiny-ja",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny-ja")
        log_memory("moonshine_tiny_ja_loaded")
        return {"model": model, "processor": processor}
    except Exception as e:
        logger.warning(f"Moonshine Tiny JA load failed: {e}")
        return None

@st.cache_resource
def get_embedding_model_safe():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
        log_memory("embedding_loaded")
        return model
    except Exception as e:
        logger.warning(f"Embedding model load failed: {e}")
        return None

@st.cache_resource
def get_opensmile_cached():
    try:
        import opensmile
        m = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
        log_memory("opensmile_cached_loaded")
        return m
    except Exception as e:
        logger.warning(f"opensmile init failed: {e}")
        return None

def get_opensmile_model_safe(use_cache: bool = True):
    if use_cache:
        return get_opensmile_cached()
    else:
        try:
            import opensmile
            return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
        except Exception as e:
            logger.warning(f"opensmile init (nocache) failed: {e}")
            return None

@st.cache_resource
def load_models_safe():
    models = {}
    for cat, cfg in MODEL_CONFIG.items():
        if os.path.exists(cfg["model"]):
            try:
                models[cat] = lgb.Booster(model_file=cfg["model"])
            except Exception as e:
                logger.warning(f"Failed to load model {cat}: {e}")
                models[cat] = None
        else:
            models[cat] = None
    return models

@st.cache_resource
def load_preprocess_objects_safe():
    pca = joblib.load(PCA_FILE) if os.path.exists(PCA_FILE) else None
    scaler = joblib.load(SCALER_FILE) if os.path.exists(SCALER_FILE) else None
    feature_cols = None
    if os.path.exists(FEATURE_FILE):
        try:
            with open(FEATURE_FILE, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
        except Exception:
            feature_cols = None
    return pca, scaler, feature_cols

def load_metadata_safe():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# -------------------------
# Janome 形態素解析
# -------------------------
def get_word_count_janome(text: str) -> int:
    try:
        from janome.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        tokens = list(tokenizer.tokenize(text))
        content_words = [t for t in tokens if t.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
        return len(content_words)
    except Exception:
        return len(text.split())

# -------------------------
# 参考文書
# -------------------------
REFERENCE_DOCS = {
    "body": """passage: 身体的疲労とは、筋肉や全身のだるさ、重さ、倦怠感を指します。典型的な症状として、体が鉛のように重い、階段を上がるのがつらい、立っているだけで疲れる、肩や腰がこる、休んでも疲れが取れない、朝起きるのがつらい、手足に力が入りにくい、歩く速度が遅くなる、筋肉が張りつめる、動作がゆっくりになる、体がだるくて動きたくない、重いものを持つのがつらい、長時間立っていられない、階段を使うのを避けたい、関節が重く感じる、全身に倦怠感がある、体を動かすのがおっくう、疲れが蓄積している感じがする、などがあります。""",
    
    "brain": """passage: 脳疲労（認知疲労）とは、頭がぼーっとする、集中力が続かない、考えがまとまらない、ミスが増える、判断が遅くなる、記憶力が落ちた気がする、読んでも内容が入ってこない、単純な計算で間違える、注意力が散漫、アイデアが浮かびにくい、思考がクリアではない、物事を考えるのが面倒、頭を使う作業を避けたい、記憶を思い出すのが大変、複数の作業ができない、指示を覚えるのが難しい、判断を誤ることが多い、などがあります。""",
    
    "mental": """passage: 精神的疲労とは、やる気が出ない、気力がわかない、イライラする、不安を感じる、人と話したくない、心が疲れた、悲しい気持ちになる、未来が憂鬱、自分を責めてしまう、感情が不安定、何にも興味がわかない、楽しくない、孤独を感じる、プレッシャーを感じる、焦りを感じる、心がざわざわする、気分が落ち込む、自分に自信が持てない、休みたいと思うことが多い、横になりたい、何をするにも時間がかかる、朝から気分が重い、などがあります。""",
    
    "healthy": """passage: 健康で活力のある状態とは、体が軽く元気いっぱい、頭がすっきり集中できる、心が安定して前向き、よく眠れた、よく食べられた、朝すっきり起きられる、日中活動的、物事に興味がある、人との交流を楽しめる、未来に希望がある、などがあります。""",
}
REFERENCE_CATS = ["body", "brain", "mental", "healthy"]

@st.cache_resource
def get_reference_embeddings_safe():
    model = get_embedding_model_safe()
    if model is None:
        return None
    try:
        texts = [REFERENCE_DOCS[cat] for cat in REFERENCE_CATS]
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    except Exception:
        return None

# -------------------------
# 質問プール
# -------------------------
QUESTION_POOL = {
    "body": [
        "体が重いように感じることがあります",
        "いつもより動くことが負担に感じる瞬間があります",
        "体のどこかに張りや違和感を覚えることがあります",
        "だるさが気になることがあります",
        "普段より疲れやすいと感じることがあります",
        "長く活動すると疲れが残りやすいと感じます",
        "休んでもすっきりしないことがあります",
        "朝起きたときに軽さを感じにくいことがあります",
        "移動や階段がいつもより大変に感じることがあります",
        "体力が落ちたように思うことがあります",
        "筋肉のこわばりを感じることがあります",
        "体を休めたいと思うことがあります",
        "日常の動作が少し負担に感じることがあります",
        "身体的な疲れを意識する時間が増えています",
        "普段より姿勢を保つのが難しいと感じることがあります",
        "体を動かすとすぐに疲れを感じることがあります",
    ],
    "brain": [
        "集中が続きにくいと感じることがあります",
        "考えをまとめるのに時間がかかることがあります",
        "頭がぼんやりする瞬間があります",
        "注意がそれやすいと感じることがあります",
        "ミスが増えたように感じることがあります",
        "判断に時間がかかることがあります",
        "思い出す作業が難しく感じることがあります",
        "頭を使う作業で負担を感じることがあります",
        "読んだ内容が入りにくいと感じることがあります",
        "同時に複数のことを進めるのが大変に感じます",
        "アイデアが出にくいと感じることがあります",
        "思考が鈍くなったように感じることがあります",
        "勉強や仕事に集中しづらいと感じます",
        "頭の疲れを意識する時間が増えています",
        "理解に時間がかかることがあります",
        "情報を整理するのが難しく感じることがあります",
    ],
    "mental": [
        "気持ちが疲れていると感じることがあります",
        "ストレスを抱えやすいと感じることがあります",
        "気分が落ち込みやすい瞬間があります",
        "不安や心配が頭に残りやすいことがあります",
        "イライラしやすくなったと感じることがあります",
        "やる気が出にくいと感じることがあります",
        "気持ちの切り替えが難しいと感じることがあります",
        "人との関わりが負担に感じることがあります",
        "プレッシャーを感じることがあります",
        "心が休まらないと感じることがあります",
        "気分が安定しないと感じることがあります",
        "精神的な疲れを意識する時間が増えています",
        "何かに取り組む気力が出にくいことがあります",
        "リラックスする時間が不足していると感じます",
        "気分が揺れやすいと感じることがあります",
        "安心感を得にくいと感じることがあります",
    ],
}
FATIGUE_CATS = ["body", "brain", "mental"]

def select_random_questions(prev_questions: Optional[List[str]] = None) -> List[Dict[str, str]]:
    if prev_questions is None:
        prev_questions = []
    
    selected = []
    for cat in FATIGUE_CATS:
        pool = QUESTION_POOL[cat]
        available = [q for q in pool if q not in prev_questions]
        if len(available) < 2:
            available = pool
        
        questions = sys_random.sample(available, min(2, len(available)))
        for q in questions:
            selected.append({"category": cat, "question": q})
    
    sys_random.shuffle(selected)
    return selected

# -------------------------
# 音声特徴量抽出
# -------------------------
FATIGUE_KEYWORDS = [
    "疲れた", "だるい", "眠い", "重い", "しんどい", "やる気",
    "集中", "ぼーっと", "イライラ", "つらい",
    "疲れてる", "疲れてます", "疲れ", "疲労", "倦怠",
]

def extract_audio_features_safe(audio_path: str, opensmile_use_cache: bool = True) -> Dict:
    text = ""
    duration = 0.0

    try:
        asr = get_asr_model_safe()
        if asr is not None:
            import torch
            import librosa

            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = float(len(audio_array) / sr) if sr > 0 else 0.0

            processor = asr["processor"]
            model = asr["model"]

            inputs = processor(
                audio_array,
                return_tensors="pt",
                sampling_rate=processor.feature_extractor.sampling_rate,
            )
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            token_limit_factor = 13.0 / processor.feature_extractor.sampling_rate
            if "attention_mask" in inputs:
                seq_lens = inputs["attention_mask"].sum(dim=-1)
                max_length = max(10, int((seq_lens * token_limit_factor).max().item()))
            else:
                max_length = max(10, int(duration * 13) + 5)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_length=max_length)
            text = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

            del inputs, generated_ids, audio_array
    except Exception as e:
        logger.warning(f"Moonshine Tiny JA transcribe error: {e}")

    smile_features = {}
    smile_success = False
    smile = get_opensmile_model_safe(use_cache=opensmile_use_cache)

    if smile is not None:
        try:
            smile_df = smile.process_file(audio_path)
            if not smile_df.empty:
                valid_count = 0
                total_cols = len(smile_df.columns)
                for col in smile_df.columns:
                    col_key = "_".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
                    val = smile_df[col].iloc[0]
                    if np.isfinite(val):
                        valid_count += 1
                    smile_features[f"smile_{col_key}".replace(" ", "_")] = float(val) if np.isfinite(val) else 0.0
                
                valid_ratio = valid_count / total_cols if total_cols > 0 else 0.0
                smile_success = valid_ratio >= OPENSMILE_VALID_RATIO
            else:
                smile_success = False
        except Exception as e:
            logger.warning(f"OpenSMILE failed: {e}")
            smile_success = False
    else:
        smile_success = False

    text_len = len(text)
    unique_chars = len(set(text))
    speech_rate = text_len / max(duration, 1e-3) if duration > 0 else 0.0
    lexical_div = unique_chars / max(text_len, 1) if text_len > 0 else 0.0
    
    fatigue_word_count = sum(text.count(k) for k in FATIGUE_KEYWORDS)
    word_count = get_word_count_janome(text)

    duration_ok = duration >= 0.5
    text_len_ok = text_len >= 2
    
    quality_ok = duration_ok and text_len_ok
    
    quality_message = "OK"
    if not quality_ok:
        reasons = []
        if not duration_ok: reasons.append("録音時間が短すぎます")
        if not text_len_ok: reasons.append("声が認識できませんでした")
        quality_message = f"もう少しだけハッキリとお話しいただけますか？ ({' / '.join(reasons)})"

    return {
        "text": text, "duration": duration, "speech_rate": speech_rate,
        "fatigue_word_count": fatigue_word_count, "text_length": text_len,
        "lexical_diversity": lexical_div, "unique_chars": unique_chars,
        "word_count": word_count,
        "smile_features": smile_features,
        "quality_ok": quality_ok, "quality_message": quality_message,
        "smile_success": smile_success,
    }

def encode_text_safe(text: str):
    model = get_embedding_model_safe()
    if model is None: return np.array([], dtype=np.float32)
    try:
        return model.encode([f"query: {text}"], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    except Exception:
        return np.array([], dtype=np.float32)

def generate_sample_id(text: str, speech_rate: float, text_length: int, smile_features: Dict, choices: List[int], date_str: str) -> str:
    smile_sample = {k: round(v, 3) for k, v in list(smile_features.items())[:10]}
    content = f"{date_str}|{text}|{speech_rate:.3f}|{text_length}|{json.dumps(smile_sample, ensure_ascii=False, sort_keys=True)}|{choices}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# -------------------------
# 推論処理
# -------------------------
def get_confidence(scores: Dict[str, float], similarities: Dict, audio_feat: Dict) -> str:
    if not scores: return "低"
    mx = max(scores.values())
    sorted_scores = sorted(scores.values(), reverse=True)
    diff = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 1.0
    
    max_sim = max(similarities.get(f"sim_{cat}", 0.0) for cat in ["body", "brain", "mental"])
    healthy_sim = similarities.get("sim_healthy", 0.0)
    smile_ok = audio_feat.get("smile_success", False)
    text_len = audio_feat.get("text_length", 0)
    
    if mx >= 4.0 and max_sim > healthy_sim and diff > 0.3:
        return "高" if (smile_ok or text_len >= 15) else "中"
    elif mx >= 2.5:
        return "中"
    return "低"

def get_confidence_percent(scores: Dict[str, float], similarities: Dict, audio_feat: Dict) -> float:
    score = 0.0
    duration = audio_feat.get("duration", 0)
    score += min(duration / 10.0, 1.0) * 30
    text_length = audio_feat.get("text_length", 0)
    score += min(text_length / 30.0, 1.0) * 30
    if audio_feat.get("smile_success", False):
        score += 20
    fatigue_sim = max(
        similarities.get("sim_body",0),
        similarities.get("sim_brain",0),
        similarities.get("sim_mental",0)
    )
    score += max(0, min(fatigue_sim,1.0)) * 20
    return round(score,1)

def get_fatigue_type(scores: Dict[str, float]) -> str:
    if not scores: return "不明"
    max_score = max(scores.values())
    max_cats = [cat for cat, score in scores.items() if score == max_score]
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and (sorted_scores[0] - sorted_scores[1]) <= 0.4:
        if len([cat for cat, score in scores.items() if score >= 3.5]) >= 2:
            return "複合的な疲れ傾向"
    
    type_map = {"body": "身体の疲れ傾向", "brain": "頭の疲れ傾向", "mental": "心の疲れ傾向"}
    return type_map.get(max_cats[0], "不明")

def generate_all_comments(scores: Dict[str, float]) -> Dict[str, str]:
    if not scores: return {"body": "", "brain": "", "mental": ""}
    comments = {
        "body": {
            "high": [
                "体がかなりお疲れのようです。休息や軽いストレッチを意識してみてください。",
                "身体に重さが残っている感じがします。今日は少し早めに休むのもおすすめです。",
                "筋肉や関節に疲れが溜まっているかも。ゆっくり動く時間を大切に。",
            ],
            "mid": [
                "身体に少し疲れが溜まっているかも。軽い運動や休憩がおすすめです。",
                "体が「もう少し休みたい」と言っているかもしれません。無理は禁物です。",
                "日常の動作に少し負担を感じやすい状態のようです。",
            ],
            "low": [
                "体の調子は良さそうです。",
                "身体は比較的軽やかな状態みたいです。この調子をキープを。",
                "体のエネルギーはまずまず安定しています。",
            ],
        },
        "brain": {
            "high": [
                "頭をたくさん使ったあとみたいです。短い休憩を挟むとスッキリしやすいかも。",
                "思考が少し重めのようです。5分でも目を閉じる時間を作ってみてください。",
                "集中力が消耗している傾向があります。情報を減らす時間も大事です。",
            ],
            "mid": [
                "頭が少しお疲れ気味のようです。リフレッシュの時間を取ってみてください。",
                "考えをまとめるのに少し時間がかかる状態かも。焦らずいきましょう。",
                "脳が「ちょっと休憩したい」サインを出しているかもしれません。",
            ],
            "low": [
                "頭はすっきりしているようです。",
                "思考は比較的クリアな状態みたいです。良い調子です。",
                "頭の疲れは少なめのようです。",
            ],
        },
        "mental": {
            "high": [
                "心に負担がかかっている傾向が出ています。無理せずリラックスする時間を大切に。",
                "気持ちが少し重めかもしれません。好きな音楽や深呼吸を試してみてください。",
                "心が「休みたい」と感じているようです。自分を責めないでくださいね。",
            ],
            "mid": [
                "心が少しお疲れのようです。深呼吸や好きなことをする時間を取ってみてください。",
                "気持ちの切り替えがやや難しい状態かも。小さな休息を重ねましょう。",
                "心のエネルギーが少し低下気味のようです。",
            ],
            "low": [
                "心は落ち着いていて安定しているようです。",
                "気持ちは比較的穏やかな状態みたいです。このまま大切に。",
                "心の疲れは少なめのようです。",
            ],
        },
    }
    result = {}
    for cat in ["body", "brain", "mental"]:
        score = scores.get(cat, 0.0)
        level = "high" if score >= 4.0 else "mid" if score >= 2.5 else "low"
        options = comments.get(cat, {}).get(level, [""])
        result[cat] = sys_random.choice(options) if options else ""
    return result

def generate_summary_comment(scores: Dict[str, float]) -> str:
    if not scores: return ""
    high_cats = [cat for cat, score in scores.items() if score >= 4.0]
    mid_cats = [cat for cat, score in scores.items() if 2.5 <= score < 4.0]

    templates = []
    if len(high_cats) >= 2:
        templates = [
            "複数の疲れ傾向が同時に出ています。全体を休める時間を意識してみてください。",
            "身体・頭・心のうち複数がお疲れ気味です。今日は無理をしない選択を。",
            "複合的な疲れが見られます。小さな休息を積み重ねるのがおすすめです。",
        ]
    elif len(high_cats) == 1:
        if high_cats[0] == "body":
            templates = [
                "身体の疲れ傾向が特に強く出ています。体を優先して休めてあげてください。",
                "体が一番お疲れのようです。動かなくてもいい時間を作ってみましょう。",
                "身体面の負担が目立ちます。軽いストレッチや早めの就寝が助けになるかも。",
            ]
        elif high_cats[0] == "brain":
            templates = [
                "頭の疲れ傾向が特に出ています。情報を減らす時間を意識してみてください。",
                "思考の消耗が目立ちます。短い休憩を挟むと回復しやすいです。",
                "脳がお疲れ気味です。ぼーっとする時間も立派な休息ですよ。",
            ]
        elif high_cats[0] == "mental":
            templates = [
                "心の疲れ傾向が特に出ています。気持ちを大切にする時間を作ってみてください。",
                "心の負担が大きめです。好きなことに触れる時間を意識してみましょう。",
                "心がお疲れのようです。自分を労わる選択を優先してみてください。",
            ]
    elif len(mid_cats) >= 2:
        templates = [
            "全体的に疲れが少しずつ蓄積している可能性があります。",
            "複数の領域で中程度の疲れが見られます。早めのケアがおすすめです。",
            "バランスよくお疲れ気味です。今日は少しペースを落としてみましょう。",
        ]
    else:
        templates = [
            "比較的良好な状態のようです。この調子を大切に。",
            "全体的に安定した状態です。無理のない範囲で活動を続けてみてください。",
            "疲れは少なめのようです。今日の自分を褒めてあげましょう。",
        ]
    return sys_random.choice(templates)

def generate_rest_suggestions(scores: Dict[str, float], fatigue_type: str) -> List[Dict[str, str]]:
    """疲れタイプに応じた休息提案を返す（タイトル + 説明）"""
    suggestions = {
        "body": [
            {"title": "軽いストレッチやヨガを5分だけ", "desc": "筋肉をゆるめて、体の重さをほぐしましょう。"},
            {"title": "温かいお風呂や足湯で体をほぐす", "desc": "副交感神経を優位にして、深いリラックスを得られます。"},
            {"title": "早めに横になって体を休める", "desc": "動かなくてもいい時間を作ることが回復につながります。"},
            {"title": "ゆっくり散歩して筋肉をゆるめる", "desc": "軽い運動で血行を促し、体のこわばりを和らげます。"},
            {"title": "姿勢を意識して深呼吸を数回", "desc": "呼吸を整えるだけで体の緊張が少しずつ解けていきます。"},
        ],
        "brain": [
            {"title": "スマホやPCから5〜10分離れる", "desc": "情報入力を止めて、脳に休む時間を与えましょう。"},
            {"title": "目を閉じてぼーっとする時間を作る", "desc": "何も考えない時間が、思考の疲労回復に効きます。"},
            {"title": "好きな音楽を聴きながら何もしない", "desc": "受動的に音楽を楽しむだけでも脳がリセットされます。"},
            {"title": "簡単なメモや日記で頭の中を整理する", "desc": "書き出すことで思考の渋滞を解消しやすくなります。"},
            {"title": "自然の音や映像を眺めて脳をリセット", "desc": "自然の刺激は脳のリラックスに効果的です。"},
        ],
        "mental": [
            {"title": "深呼吸をゆっくり10回繰り返す", "desc": "呼吸を整えることで、心のざわつきを落ち着かせます。"},
            {"title": "好きな飲み物を丁寧に味わう時間を作る", "desc": "小さな「丁寧な時間」が心の余裕を生みます。"},
            {"title": "信頼できる人に短いメッセージを送る", "desc": "つながりを感じるだけで、孤独感が和らぎます。"},
            {"title": "好きな写真や動画を眺めて気分転換", "desc": "ポジティブな刺激で気持ちを切り替えやすくなります。"},
            {"title": "「今日はこれで十分」と自分に声をかける", "desc": "自分を認める言葉が、心の負担を軽くしてくれます。"},
        ],
        "common": [
            {"title": "水分をしっかり摂る", "desc": "脱水は疲労感を増幅させます。こまめな水分補給を。"},
            {"title": "睡眠時間を少しでも確保する", "desc": "質の良い睡眠が、体・頭・心すべての回復に効きます。"},
            {"title": "今日の自分を「よく頑張った」と認める", "desc": "自己肯定感を高めることが、回復の第一歩です。"},
            {"title": "無理に予定を入れず余白を残す", "desc": "空いた時間があるだけで、心に余裕が生まれます。"},
        ],
    }

    selected = []
    top_cat = None
    if scores:
        top_cat = max(scores, key=scores.get)
        pool = suggestions.get(top_cat, [])
        if pool:
            selected.extend(sys_random.sample(pool, k=min(2, len(pool))))
    selected.append(sys_random.choice(suggestions["common"]))
    other_cats = [c for c in ["body", "brain", "mental"] if c != top_cat]
    if other_cats:
        other = sys_random.choice(other_cats)
        selected.append(sys_random.choice(suggestions[other]))
    return selected[:4]

def predict_fatigue_safe(audio_path: Optional[str]) -> Dict:
    log_memory("predict_before")
    
    if not audio_path:
        return {"success": False, "message": "音声ファイルが指定されていません", "scores": {}, "confidence": "低", "confidence_percent": 20.0}

    mem_mb = log_memory("predict_memcheck")
    if mem_mb is not None and mem_mb > MEMORY_THRESHOLD_CRITICAL:
        return {"success": False, "message": "システム負荷が高いため処理を中断しました。少し待って再試行してください。", "scores": {}, "confidence": "低", "confidence_percent": 20.0}

    audio_feat = extract_audio_features_safe(audio_path)
    if not audio_feat["quality_ok"]:
        return {"success": False, "message": audio_feat["quality_message"], "scores": {}, "confidence": "低", "confidence_percent": 20.0, "audio_feat": audio_feat}

    query_emb = encode_text_safe(audio_feat["text"])
    ref_emb = get_reference_embeddings_safe()

    similarities = {}
    if query_emb.size > 0 and ref_emb is not None:
        for i, cat in enumerate(REFERENCE_CATS):
            try:
                similarities[f"sim_{cat}"] = float(cosine_similarity(query_emb.reshape(1, -1), ref_emb[i].reshape(1, -1))[0][0])
            except Exception:
                similarities[f"sim_{cat}"] = 0.0
    else:
        similarities = {f"sim_{cat}": 0.0 for cat in REFERENCE_CATS}

    features = {
        "speech_rate": audio_feat["speech_rate"],
        "fatigue_word_count": audio_feat["fatigue_word_count"],
        "text_length": audio_feat["text_length"],
        "lexical_diversity": audio_feat["lexical_diversity"],
        **audio_feat["smile_features"], **similarities,
    }

    models = load_models_safe()
    pca, scaler, feature_cols = load_preprocess_objects_safe()
    metadata = load_metadata_safe()

    models_ready = all(models.get(k) is not None for k in ["body", "brain", "mental"]) and pca is not None and scaler is not None and feature_cols is not None

    if not models_ready:
        fallback_scores = {}
        for cat in ["body", "brain", "mental"]:
            raw = max(similarities.get(f"sim_{cat}", 0.0) - similarities.get("sim_healthy", 0.0) * 0.5, 0.0)
            fallback_scores[cat] = round(float(np.clip(raw * 5.0 + 1.0, 1.0, 5.0)), 1)
        
        try:
            del ref_emb
        except:
            pass
        gc.collect()
        
        return {
            "success": True, "scores": fallback_scores, "confidence": get_confidence(fallback_scores, similarities, audio_feat),
            "confidence_percent": get_confidence_percent(fallback_scores, similarities, audio_feat),
            "message": "簡易推定", "features": features, "audio_quality": f"{audio_feat['duration']:.1f}秒",
            "audio_feat": audio_feat, "similarities": similarities, "query_embedding": query_emb
        }

    X_base = np.array([[features.get(col, 0.0) for col in feature_cols]], dtype=np.float32)
    try:
        X_base_scaled = scaler.transform(X_base)
    except Exception:
        return {"success": False, "message": "前処理に失敗しました", "scores": {}, "confidence": "低", "confidence_percent": 20.0}

    pca_n_comp = getattr(pca, "n_components_", getattr(pca, "n_components", 0))
    if query_emb.size == 0:
        emb_pca = np.zeros((1, int(pca_n_comp) if pca_n_comp is not None else 0), dtype=np.float32)
    else:
        try:
            emb_pca = pca.transform(query_emb.reshape(1, -1))
        except Exception as e:
            logger.warning(f"PCA transform failed: {e}")
            return {"success": False, "message": "PCA 前処理に失敗しました", "scores": {}, "confidence": "低", "confidence_percent": 20.0}

    X = np.hstack([X_base_scaled, emb_pca]) if emb_pca.size > 0 else X_base_scaled

    scores = {}
    for cat in ["body", "brain", "mental"]:
        model = models.get(cat)
        if model:
            try:
                pred_raw = float(model.predict(X)[0])
                scores[cat] = round(float(np.clip(pred_raw * 5.0 / 9.0, 1.0, 5.0)), 1)
            except Exception as e:
                scores[cat] = 1.0
        else:
            scores[cat] = 1.0
            
    try:
        del ref_emb, X_base, X_base_scaled, emb_pca, X
    except:
        pass
    gc.collect()
    log_memory("predict_after")
    
    return {
        "success": True, "scores": scores, "confidence": get_confidence(scores, similarities, audio_feat),
        "confidence_percent": get_confidence_percent(scores, similarities, audio_feat), "message": "OK",
        "features": features, "audio_quality": f"{audio_feat['duration']:.1f}秒",
        "audio_feat": audio_feat, "similarities": similarities, "query_embedding": query_emb
    }

# -------------------------
# データ保存
# -------------------------
def save_to_hf_dataset_with_retry_safe(file_path: str, repo_path: str, max_retries: int = 3):
    if not HF_TOKEN or not REPO_ID:
        return False
    for attempt in range(1, max_retries + 1):
        try:
            from huggingface_hub import HfApi
            HfApi(token=HF_TOKEN).upload_file(
                path_or_fileobj=file_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            return True
        except Exception as e:
            logger.warning(f"HF upload failed (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    return False

def save_data_with_result_safe(audio_feat, query_emb, pred_scores: Dict[str, float], similarities: Dict, 
                                questions: List[Dict[str, str]], choices: List[int]) -> str:
    if not audio_feat or not audio_feat.get("quality_ok", False):
        return "音声が短すぎるため保存をスキップしました"
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    sample_id = generate_sample_id(
        audio_feat.get("text", ""), audio_feat.get("speech_rate", 0.0),
        audio_feat.get("text_length", 0), audio_feat.get("smile_features", {}), choices, date_str
    )
    
    lock_path = DATA_PARQUET + ".lock"

    def _do_save() -> str:
        existing = pd.DataFrame()
        if os.path.exists(DATA_PARQUET) and os.path.getsize(DATA_PARQUET) > 0:
            try:
                existing = pd.read_parquet(DATA_PARQUET)
                if len(existing) > 10000:
                    logger.warning(f"Dataset is large ({len(existing)} rows). Consider migrating to a database.")
            except Exception as e:
                logger.warning(f"Failed to read existing parquet: {e}")
                existing = pd.DataFrame()
        
        if "sample_id" in existing.columns and sample_id in existing["sample_id"].values:
            QUALITY_STATS["duplicate_blocked_count"] += 1
            return "同じ内容のデータはすでに送信されています"
        
        category_choices = {"body": [], "brain": [], "mental": []}
        category_questions = {"body": [], "brain": [], "mental": []}
        for i, q in enumerate(questions):
            cat = q["category"]
            if i < len(choices):
                category_choices[cat].append(choices[i])
                category_questions[cat].append(q["question"])
        
        avg_body = np.mean(category_choices["body"]) if category_choices["body"] else 0.0
        avg_brain = np.mean(category_choices["brain"]) if category_choices["brain"] else 0.0
        avg_mental = np.mean(category_choices["mental"]) if category_choices["mental"] else 0.0
        
        df_row = {
            "text": audio_feat.get("text", ""),
            "speech_rate": float(audio_feat.get("speech_rate", 0.0)),
            "fatigue_word_count": float(audio_feat.get("fatigue_word_count", 0.0)),
            "text_length": int(audio_feat.get("text_length", 0)),
            "lexical_diversity": float(audio_feat.get("lexical_diversity", 0.0)),
            "unique_chars": int(audio_feat.get("unique_chars", 0)),
            "word_count": int(audio_feat.get("word_count", 0)),
            **{k: float(v) for k, v in audio_feat.get("smile_features", {}).items()},
            **{k: float(v) for k, v in similarities.items()},
            "label_body_avg": float(avg_body), "label_brain_avg": float(avg_brain), "label_mental_avg": float(avg_mental),
            "label_body_q1": float(category_choices["body"][0]) if len(category_choices["body"]) > 0 else 0.0,
            "label_body_q2": float(category_choices["body"][1]) if len(category_choices["body"]) > 1 else 0.0,
            "label_brain_q1": float(category_choices["brain"][0]) if len(category_choices["brain"]) > 0 else 0.0,
            "label_brain_q2": float(category_choices["brain"][1]) if len(category_choices["brain"]) > 1 else 0.0,
            "label_mental_q1": float(category_choices["mental"][0]) if len(category_choices["mental"]) > 0 else 0.0,
            "label_mental_q2": float(category_choices["mental"][1]) if len(category_choices["mental"]) > 1 else 0.0,
            "label_body_q1_text": category_questions["body"][0] if len(category_questions["body"]) > 0 else "",
            "label_body_q2_text": category_questions["body"][1] if len(category_questions["body"]) > 1 else "",
            "label_brain_q1_text": category_questions["brain"][0] if len(category_questions["brain"]) > 0 else "",
            "label_brain_q2_text": category_questions["brain"][1] if len(category_questions["brain"]) > 1 else "",
            "label_mental_q1_text": category_questions["mental"][0] if len(category_questions["mental"]) > 0 else "",
            "label_mental_q2_text": category_questions["mental"][1] if len(category_questions["mental"]) > 1 else "",
            "pred_body": float(pred_scores.get("body", 0.0)),
            "pred_brain": float(pred_scores.get("brain", 0.0)),
            "pred_mental": float(pred_scores.get("mental", 0.0)),
            "timestamp": datetime.now().isoformat(),
            "smile_success": bool(audio_feat.get("smile_success", False)),
            "sample_id": sample_id,
        }
        
        combined = pd.concat([existing, pd.DataFrame([df_row])], ignore_index=True, sort=False).fillna(0)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(DATA_PARQUET))
        os.close(tmp_fd)
        try:
            combined.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, DATA_PARQUET)
            history_file = os.path.join(HISTORY_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M')}.parquet")
            combined.to_parquet(history_file, index=False)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        
        return f"無事に保存されました（累計データ数：{len(combined)}件）"

    if HAS_FILELOCK:
        lock = FileLock(lock_path, timeout=10)
        try:
            with lock:
                result_msg = _do_save()
        except Timeout:
            return "保存ロックの取得に失敗しました"
    else:
        start = time.time()
        acquired = False
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                acquired = True
                break
            except FileExistsError:
                if time.time() - start > 10:
                    return "保存ロックの取得に失敗しました"
                time.sleep(0.1)
            except OSError:
                return "保存ロックの取得に失敗しました"
        
        try:
            result_msg = _do_save()
        finally:
            try:
                os.remove(lock_path)
            except Exception:
                pass

    save_to_hf_dataset_with_retry_safe(DATA_PARQUET, "dataset.parquet")
    return result_msg

@st.cache_data(ttl=60)
def get_current_data_count() -> int:
    """
    HuggingFace Datasetを正として件数取得
    """

    # HF優先
    if HF_TOKEN and REPO_ID:
        try:
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq

            path = hf_hub_download(
                repo_id=REPO_ID,
                filename="dataset.parquet",
                repo_type="dataset",
                token=HF_TOKEN,
                force_download=True
            )

            pf = pq.ParquetFile(path)
            return int(pf.metadata.num_rows)

        except Exception as e:
            logger.warning(f"HF count fetch failed: {e}")


    # HF失敗時だけlocal fallback
    try:
        if os.path.exists(DATA_PARQUET):
            df = pd.read_parquet(DATA_PARQUET)
            return int(len(df))
    except Exception:
        pass

    return 0

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(
    page_title="Restee - 休息のデザイン",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- ダークテーマ + レポート用スタイル ----
st.markdown("""
<style>
    /* 全体の背景と文字色 */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    
    /* メトリックカード */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 18px 16px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #f59e0b !important;
        font-size: 0.9rem !important;
    }
    
    /* 見出し */
    h1, h2, h3 {
        color: #fafafa !important;
    }
    
    /* アラート / 情報ボックス */
    .stAlert {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
    }
    
    /* 区切り線 */
    hr {
        border-color: #374151;
    }
    
    /* キャプション */
    .stCaption {
        color: #9ca3af !important;
    }
    
    /* 進捗バー */
    .stProgress > div > div {
        background-color: #60a5fa;
    }
    
    /* ボタン */
    .stButton > button {
        border-radius: 8px;
    }
    
    /* レポートカード風 */
    .report-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        height: 100%;
    }
    .report-card h4 {
        margin: 0 0 8px 0;
        color: #fafafa;
        font-size: 1.05rem;
    }
    .report-card p {
        margin: 0;
        color: #d1d5db;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    /* 休息提案カード */
    .tip-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .tip-card strong {
        color: #fafafa;
        font-size: 0.98rem;
    }
    .tip-card span {
        color: #9ca3af;
        font-size: 0.88rem;
        display: block;
        margin-top: 4px;
    }
    
    /* 余白調整 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌱 Restee - AI モデル開発用")
st.markdown("音声での対話から疲労の種類を分析して休み方を提案するアプリを作りたいです。ご協力お願いします！！")

# 件数表示
count = get_current_data_count()
if count > 0:
    st.info(f"📊 現在 **{count} 件** のデータが集まっています！ご協力ありがとうございます🌱")
else:
    st.info("📊 現在 0 件 のデータが集まっています。最初のデータ提供者になりませんか？")

# セッション初期化
if "questions" not in st.session_state: 
    st.session_state["questions"] = select_random_questions()
    st.session_state["prev_questions"] = [q["question"] for q in st.session_state["questions"]]
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "data_saved" not in st.session_state:
    st.session_state["data_saved"] = False
if "save_msg" not in st.session_state:
    st.session_state["save_msg"] = ""
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = None
if "audio_hash" not in st.session_state:
    st.session_state["audio_hash"] = None

# アンケートカード
with st.container():
    st.subheader("📝 1. 今の感覚に近いものを教えてください")
    st.caption("1: 全く当てはまらない 〜 5: 非常に当てはまる")
    
    for i, q in enumerate(st.session_state["questions"]):
        st.write(f"**{q['question']}**")
        st.session_state[f"choice_{i}"] = st.slider(
            label="度合い", min_value=1, max_value=5, value=st.session_state.get(f"choice_{i}", 3),
            key=f"slider_{i}", label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
    if st.button("🔄 質問を変える", help="別の質問にシャッフルします"):
        prev_qs = st.session_state.get("prev_questions", [])
        st.session_state["questions"] = select_random_questions(prev_qs)
        st.session_state["prev_questions"] = [q["question"] for q in st.session_state["questions"]]
        for i in range(len(st.session_state["questions"])): 
            st.session_state.pop(f"choice_{i}", None)
        st.rerun()

st.divider()

# 音声録音
st.subheader("🎤 2. 今の気持ちを声で残してみる")
st.markdown("「今日はちょっと疲れたな」「なんだか体が重い」など、今の状態を数秒つぶやいてみてください。")
st.caption("🔒 音声データそのものは保存されません。匿名化された特徴量のみが研究目的で利用されます。個人を特定できる情報は収集しません。")

audio_value = st.audio_input("音声を録音してください")

if audio_value is not None:
    audio_value.seek(0)
    audio_bytes = audio_value.read()
    audio_value.seek(0)
    
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    
    if st.session_state["audio_hash"] != audio_hash:
        st.session_state["audio_hash"] = audio_hash
        st.session_state["audio_data"] = audio_value
        st.session_state["analyzed"] = False
        st.session_state["data_saved"] = False
        st.session_state["save_msg"] = ""
    
    st.audio(st.session_state["audio_data"])
    
    analyzed = st.session_state.get("analyzed", False)
    
    if st.button("✨ 解析する！", type="primary", use_container_width=True, disabled=analyzed):
        with st.spinner("声のトーンや言葉を紐解いています..."):
            tmp_path = None
            try:
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, f"restee_rec_{int(time.time()*1000)}.wav")
                st.session_state["audio_data"].seek(0)
                with open(tmp_path, "wb") as f:
                    f.write(st.session_state["audio_data"].read())
                
                QUALITY_STATS["total_attempts"] += 1
                
                st.session_state["last_result"] = predict_fatigue_safe(tmp_path)
                st.session_state["analyzed"] = True
                st.session_state["data_saved"] = False
            except Exception as e:
                logger.exception("解析エラー")
                st.error(f"ごめんなさい、解析中にエラーが起きました：{e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

# ============================================================
# 結果表示（レポート風に整理）
# ============================================================
if st.session_state["analyzed"] and st.session_state["last_result"]:
    res = st.session_state["last_result"]
    st.divider()
    
    # ---- ヘッダー ----
    st.markdown("# 🍀 あなたの疲労レポート")
    st.caption("※本レポートは研究用のものであり、医学的な診断・助言・判断の代わりにはなりません。あくまで参考情報としてご利用ください。")
    
    if not res.get("success", False):
        st.warning(f"{res.get('message', '')}")
    else:
        scores = res.get("scores", {})
        similarities = res.get("similarities", {})
        audio_feat = res.get("audio_feat", {})
        
        fatigue_type = get_fatigue_type(scores)
        summary = generate_summary_comment(scores)
        
        # ---- 疲れタイプ ----
        st.markdown(f"### あなたの疲れタイプ：**{fatigue_type}**")
        if summary:
            st.info(summary)
        
        # ---- メトリクス（3列） ----
        col1, col2, col3 = st.columns(3)
        metrics = [
            ("💪 身体", "body"),
            ("🧠 頭", "brain"),
            ("💙 心", "mental"),
        ]
        max_score = max(scores.values()) if scores else 1.0
        avg_score = sum(scores.values()) / 3 if scores else 1.0

        for col, (label, cat) in zip([col1, col2, col3], metrics):
            val = scores.get(cat, 1.0)
            fatigue_pct = float(np.clip((val - 1.0) / 4.0 * 100.0, 0.0, 100.0))
            delta_txt = f"疲労度 {fatigue_pct:.0f}%"
            if val == max_score and max_score - avg_score > 0.3:
                delta_txt = f"▲ 最も高い {fatigue_pct:.0f}%"
            with col:
                st.metric(label, f"{val:.1f} / 5", delta=delta_txt)
                st.progress(float(np.clip((val - 1.0) / 4.0, 0.0, 1.0)))

        st.markdown("")  # 余白

        # ---- レーダーチャート / 棒グラフ ----
        st.markdown("### 疲労度グラフ")
        try:
            import matplotlib.pyplot as plt

            cats_jp = ["身体", "頭", "心"]
            vals = [scores.get("body", 1.0), scores.get("brain", 1.0), scores.get("mental", 1.0)]
            vals += vals[:1]
            angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
            ax.plot(angles, vals, "o-", linewidth=2.5, color="#60a5fa", markersize=8)
            ax.fill(angles, vals, alpha=0.3, color="#60a5fa")
            ax.set_thetagrids(np.degrees(angles[:-1]), cats_jp, fontsize=12, color="#e5e7eb")
            ax.set_ylim(1, 5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#9ca3af", fontsize=9)
            ax.spines["polar"].set_color("#4b5563")
            ax.grid(color="#4b5563", linestyle="--", alpha=0.6)
            ax.set_facecolor("#1f2937")
            fig.patch.set_facecolor("#0e1117")
            ax.set_title("疲れ度レーダー（1〜5）", va="bottom", fontsize=13, color="#fafafa", pad=12)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        except Exception:
            chart_df = pd.DataFrame({
                "カテゴリ": ["身体", "頭", "心"],
                "疲れ度": [scores.get("body", 1.0), scores.get("brain", 1.0), scores.get("mental", 1.0)]
            }).set_index("カテゴリ")
            st.bar_chart(chart_df, height=240)

        st.markdown("")  # 余白

        # ---- カテゴリー別コメント（3列カード） ----
        st.markdown("### カテゴリー別のコメント")
        all_comments = generate_all_comments(scores)
        
        c1, c2, c3 = st.columns(3)
        comment_data = [
            (c1, "💪 身体", "body", scores.get("body", 1.0)),
            (c2, "🧠 頭", "brain", scores.get("brain", 1.0)),
            (c3, "💙 心", "mental", scores.get("mental", 1.0)),
        ]
        for col, title, cat, val in comment_data:
            with col:
                st.markdown(f"""
                <div class="report-card">
                    <h4>{title}</h4>
                    <p style="color:#60a5fa; font-weight:600; margin-bottom:8px;">{val:.1f} / 5</p>
                    <p>{all_comments.get(cat, "")}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")  # 余白

        # ---- おすすめ休息提案（2×2カード） ----
        st.markdown("### 🌿 今日のおすすめ休息")
        tips = generate_rest_suggestions(scores, fatigue_type)
        
        # 2列レイアウトでカード表示
        for i in range(0, len(tips), 2):
            cols = st.columns(2)
            for j, tip in enumerate(tips[i:i+2]):
                with cols[j]:
                    st.markdown(f"""
                    <div class="tip-card">
                        <strong>{tip["title"]}</strong>
                        <span>{tip["desc"]}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("")  # 余白

        # ---- 解析品質 ----
        confidence_percent = res.get("confidence_percent", 20.0)
        st.markdown("---")
        st.caption(f"解析に利用できた情報量: **{confidence_percent:.0f}%**")
        st.caption("音声の長さ・文字数・特徴量の充実度などから算出した参考指標です。モデルの自信度ではありません。")
        
        if DEBUG:
            with st.expander("🛠️ 解析の詳細データを見る"):
                st.write("各スコア (1-5):", {k: f"{v:.1f}" for k, v in scores.items()})
                st.write("オーディオ品質:", res.get("audio_quality", ""))
                st.json(res.get("features", {}))
            
        # ---- データ送信 ----
        st.divider()
        st.subheader("🤝 開発へのご協力のお願い")
        st.markdown("""
        差し支えなければ、今回の結果を匿名データとして送信してください。
        
        - 🔒 音声データそのものは保存されません
        - 📊 匿名化された特徴量のみがモデル開発目的で利用されます
        - 👤 個人を特定できる情報は収集しません
        
        安心してご協力ください！
        """)
        
        if st.session_state["data_saved"]:
            st.success(f"送信済みです🌱\n\n({st.session_state.get('save_msg', '')})")
        else:
            if st.button("💾 この結果を匿名で送信する", type="secondary"):
                with st.spinner("データを送っています..."):
                    choices = []
                    for i in range(len(st.session_state["questions"])):
                        choices.append(st.session_state.get(f"choice_{i}", 3))
                    
                    audio_feat = res.get("audio_feat")
                    
                    save_msg = save_data_with_result_safe(
                        audio_feat=audio_feat,
                        query_emb=res.get("query_embedding"),
                        pred_scores=scores,
                        similarities=res.get("similarities", {}),
                        questions=st.session_state["questions"],
                        choices=choices
                    )
                    st.session_state["save_msg"] = save_msg
                    st.session_state["data_saved"] = True
                    st.rerun()

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("Restee - Designing rest through dialogue.")
