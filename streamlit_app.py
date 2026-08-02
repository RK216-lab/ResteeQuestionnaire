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
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = os.environ.get("REPO_ID", "")
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

# ☁️ Streamlit Community Cloud に合わせたメモリ閾値に変更
MEMORY_THRESHOLD_WARN = int(os.environ.get("MEM_WARN_MB", 2000))
MEMORY_THRESHOLD_CRITICAL = int(os.environ.get("MEM_CRIT_MB", 2400))

MODEL_CONFIG = {
    "body": {"model": os.path.join(MODEL_DIR, "body_model.txt"), "label": "label_body"},
    "brain": {"model": os.path.join(MODEL_DIR, "brain_model.txt"), "label": "label_brain"},
    "mental": {"model": os.path.join(MODEL_DIR, "mental_model.txt"), "label": "label_mental"},
}

OPENSMILE_VALID_RATIO = float(os.environ.get("OPENSMILE_VALID_RATIO", 0.6))
DEBUG = os.getenv("DEBUG", "0") == "1"

# 機械学習用のシードは固定しつつ、質問選定などは SystemRandom で動かす
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)
sys_random = random.SystemRandom() # アクセスごとのランダム担保用

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
# ロック管理
# -------------------------
def _acquire_lock(lock_path: str, timeout: int = 10) -> bool:
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        except OSError:
            return False

def _release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except Exception:
        pass

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
def get_whisper_model_safe():
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
        log_memory("whisper_loaded")
        return model
    except Exception as e:
        logger.warning(f"Whisper load failed: {e}")
        return None

@st.cache_resource
def get_embedding_model_safe():
    try:
        from sentence_transformers import SentenceTransformer
        # ☁️ メモリ大幅削減 (約470MB -> 約50MB) かつ精度を維持する日本語特化の軽量モデルに変更
        model = SentenceTransformer("oshizo/sbert-jsnli-luke-japanese-base-lite", device="cpu")
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

# -------------------------
# 質問選択（アクセスごとに必ずシャッフル）
# -------------------------
def select_random_questions(prev_questions: Optional[List[str]] = None) -> List[Dict[str, str]]:
    if prev_questions is None:
        prev_questions = []
    
    selected = []
    for cat in FATIGUE_CATS:
        pool = QUESTION_POOL[cat]
        available = [q for q in pool if q not in prev_questions]
        if len(available) < 2:
            available = pool
        
        # 🚀 random.SystemRandom() を使うことで、キャッシュやシード固定を無視してシャッフル
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
    whisper = None
    text = ""
    duration = 0.0

    try:
        whisper = get_whisper_model_safe()
        if whisper is not None:
            segments, info = whisper.transcribe(audio_path, language="ja", beam_size=1, vad_filter=True, condition_on_previous_text=False)
            text = "".join(s.text for s in segments).strip()
            duration = float(info.duration) if (info and hasattr(info, "duration")) else 0.0
    except Exception as e:
        logger.warning(f"Whisper transcribe error: {e}")

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

    # 🚀 品質チェックを大幅に緩和し、「品質が低くて送れません」を防ぐ
    duration_ok = duration >= 0.5  # 0.5秒以上あればOK
    text_len_ok = text_len >= 2    # 2文字以上あればOK
    
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
    
    if mx >= 7 and max_sim > healthy_sim and diff > 0.3:
        return "高" if (smile_ok or text_len >= 15) else "中"
    elif mx >= 4:
        return "中"
    return "低"

def get_confidence_percent(scores: Dict[str, float], similarities: Dict, audio_feat: Dict) -> float:
    confidence_str = get_confidence(scores, similarities, audio_feat)
    if confidence_str == "高": return 84.0
    elif confidence_str == "中": return 50.0
    else: return 20.0

def get_fatigue_type(scores: Dict[str, float]) -> str:
    if not scores: return "不明"
    max_score = max(scores.values())
    max_cats = [cat for cat, score in scores.items() if score == max_score]
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and (sorted_scores[0] - sorted_scores[1]) <= 0.5:
        if len([cat for cat, score in scores.items() if score >= 6.0]) >= 2:
            return "複合疲労"
    
    type_map = {"body": "身体疲れ", "brain": "脳疲れ", "mental": "心疲れ"}
    return type_map.get(max_cats[0], "不明")

def generate_all_comments(scores: Dict[str, float]) -> Dict[str, str]:
    if not scores: return {"body": "", "brain": "", "mental": ""}
    comments = {
        "body": {"high": "体がかなりお疲れのようです🛌 筋肉の疲労が蓄積している可能性があります。","mid": "身体に少し疲れが溜まっているかも。軽いストレッチがおすすめです。","low": "体の調子は良さそうです！✨"},
        "brain": {"high": "頭をたくさん使いましたね🧠 認知機能が疲労しているサインです。","mid": "脳が少しお疲れ気味です。短い休憩を挟むとスッキリしますよ🌱","low": "頭は冴えているようです！💡"},
        "mental": {"high": "心に負担がかかっているサインです☁️ 情緒的な疲労が見られます。","mid": "心が少しお疲れのようです。深呼吸してリラックスしましょう🍀","low": "心は落ち着いていて安定しています🌸"},
    }
    result = {}
    for cat in ["body", "brain", "mental"]:
        score = scores.get(cat, 0.0)
        level = "high" if score >= 7 else "mid" if score >= 4 else "low"
        result[cat] = comments.get(cat, {}).get(level, "")
    return result

def generate_summary_comment(scores: Dict[str, float]) -> str:
    if not scores: return ""
    high_cats = [cat for cat, score in scores.items() if score >= 7.0]
    mid_cats = [cat for cat, score in scores.items() if 4.0 <= score < 7.0]
    
    parts = []
    if len(high_cats) >= 2: parts.append("複数の疲労が見られます。")
    elif len(high_cats) == 1:
        if high_cats[0] == "body": parts.append("身体の疲れが特に目立ちます。")
        elif high_cats[0] == "brain": parts.append("脳の疲れが特に目立ちます。")
        elif high_cats[0] == "mental": parts.append("心の疲れが特に目立ちます。")
    if len(mid_cats) >= 2: parts.append("全体的に疲労が蓄積しています。")
    
    if not parts: parts.append("良好な状態です。")
    return " ".join(parts)

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
            fallback_scores[cat] = round(float(np.clip(raw * 9.0, 0.0, 9.0)), 1)
        
        try: del ref_emb
        except: pass
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
            
    try: del ref_emb, X_base, X_base_scaled, emb_pca, X
    except: pass
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
    if not HF_TOKEN or not REPO_ID: return False
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
    
    lock = DATA_PARQUET + ".lock"
    if not _acquire_lock(lock, timeout=10):
        return "保存ロックの取得に失敗しました"
    
    try:
        existing = pd.read_parquet(DATA_PARQUET) if os.path.exists(DATA_PARQUET) and os.path.getsize(DATA_PARQUET) > 0 else pd.DataFrame()
        
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
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
    finally:
        _release_lock(lock)

    save_to_hf_dataset_with_retry_safe(DATA_PARQUET, "dataset.parquet")
    return f"無事に保存されました🌱（累計データ数：{len(combined)}件）"

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Restee - 休息のデザイン", page_icon="🌱", layout="centered")

st.title("🌱 Restee - AI モデル開発用")
st.markdown("音声での対話から疲労の種類を分析して休み方を提案するアプリを作りたいです。ご協力お願いします！！")

try:
    if os.path.exists(DATA_PARQUET) and os.path.getsize(DATA_PARQUET) > 0:
        df_tmp = pd.read_parquet(DATA_PARQUET)
        st.info(f"📊 現在 **{len(df_tmp)} 件** のデータが集まっています！ご協力ありがとうございます🌱")
    else:
        st.info("📊 現在 0 件 のデータが集まっています。最初のデータ提供者になりませんか？")
except Exception:
    pass

# セッション初期化（アクセスごとに新しい質問を生成）
if "questions" not in st.session_state: 
    st.session_state["questions"] = select_random_questions()
    st.session_state["prev_questions"] = [q["question"] for q in st.session_state["questions"]]
if "analyzed" not in st.session_state: st.session_state["analyzed"] = False
if "last_result" not in st.session_state: st.session_state["last_result"] = None
if "data_saved" not in st.session_state: st.session_state["data_saved"] = False
if "save_msg" not in st.session_state: st.session_state["save_msg"] = ""
if "audio_data" not in st.session_state: st.session_state["audio_data"] = None
if "audio_hash" not in st.session_state: st.session_state["audio_hash"] = None

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
    audio_hash = hashlib.sha256(audio_value.read()).hexdigest()
    audio_value.seek(0)
    
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
                    try: os.remove(tmp_path)
                    except: pass

# 結果表示
if st.session_state["analyzed"] and st.session_state["last_result"]:
    res = st.session_state["last_result"]
    st.divider()
    st.subheader("🍀 あなたの今の状態")
    
    if not res.get("success", False):
        st.warning(f"{res.get('message', '')}")
    else:
        scores = res.get("scores", {})
        similarities = res.get("similarities", {})
        audio_feat = res.get("audio_feat", {})
        
        fatigue_type = get_fatigue_type(scores)
        st.success(f"📈 あなたの疲れタイプ：**{fatigue_type}**")
        
        summary = generate_summary_comment(scores)
        if summary:
            st.info(f"📊 {summary}")
        
        all_comments = generate_all_comments(scores)
        with st.expander("💬 各カテゴリのコメントを見る"):
            st.write(f"💪 身体: {all_comments.get('body', '')}")
            st.write(f"🧠 脳: {all_comments.get('brain', '')}")
            st.write(f"💙 心: {all_comments.get('mental', '')}")
        
        col1, col2, col3 = st.columns(3)
        metrics = [("身体の疲れ", "body", "💪"), ("頭の疲れ", "brain", "🧠"), ("心の疲れ", "mental", "💙")]
        
        for col, (label, cat, icon) in zip([col1, col2, col3], metrics):
            val = scores.get(cat, 0.0)
            display_val = val * 9.0 / 5.0 if val > 0 else 0.0
            with col:
                st.metric(f"{icon} {label}", f"{display_val:.1f} / 9")
                st.progress(float(np.clip(display_val / 9.0, 0.0, 1.0)))
        
        confidence = res.get("confidence", "低")
        confidence_percent = res.get("confidence_percent", 20.0)
        st.write(f"**解析への自信度**: {confidence} ({confidence_percent:.0f}%)")
        
        if DEBUG:
            with st.expander("🛠️ 解析の詳細データを見る"):
                st.write("各スコア (0-9):", {k: f"{v:.1f}" for k, v in scores.items()})
                st.write("オーディオ品質:", res.get("audio_quality", ""))
                st.json(res.get("features", {}))
            
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
