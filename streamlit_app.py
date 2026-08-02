# app.py
import os
import json
import logging
import time
import tempfile
import gc
import hashlib
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
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Parquet のみで運用（軽量・高速）
DATA_PARQUET = os.path.join(DATA_DIR, "dataset.parquet")

FEATURE_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
PCA_FILE = os.path.join(MODEL_DIR, "pca.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

# Memory thresholds (MB)
MEMORY_THRESHOLD_WARN = int(os.environ.get("MEM_WARN_MB", 2200))
MEMORY_THRESHOLD_CRITICAL = int(os.environ.get("MEM_CRIT_MB", 2500))

MODEL_CONFIG = {
    "body": {"model": os.path.join(MODEL_DIR, "body_model.txt"), "label": "label_body"},
    "brain": {"model": os.path.join(MODEL_DIR, "brain_model.txt"), "label": "label_brain"},
    "mental": {"model": os.path.join(MODEL_DIR, "mental_model.txt"), "label": "label_mental"},
}

OPENSMILE_VALID_RATIO = float(os.environ.get("OPENSMILE_VALID_RATIO", 0.6))

# DEBUG モードフラグ
DEBUG = os.getenv("DEBUG", "0") == "1"

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

# -------------------------
# 品質統計カウンター（デバッグ用）
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

def embed_to_json(arr: Any) -> str:
    if arr is None: return "[]"
    if isinstance(arr, np.ndarray):
        return json.dumps(arr.astype(float).tolist(), ensure_ascii=False)
    return "[]"

# -------------------------
# モデル / ライブラリの遅延ロード
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
# 参考文書 embedding
# -------------------------
REFERENCE_DOCS = {
    "body": "passage: 身体的疲労。体がだるい、重い、筋肉に力が入らない、へとへと、動きたくない、休みたい。",
    "brain": "passage: 認知の疲労（脳疲労）。頭がぼーっとする、集中できない、考えがまとまらない、ミスが増える。",
    "mental": "passage: 精神的疲労。やる気が出ない、気力がない、イライラする、不安、人と話したくない、心が疲れた。",
    "healthy": "passage: 健康で活力のある状態。体が軽い、元気いっぱい、すっきり、よく眠れた、集中できる、前向き。",
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
# 質問プール（各カテゴリ 2 問）
# -------------------------
QUESTION_POOL = {
    "body": [
        "今日は体がだるくて重いです", "筋肉に力が入らない感じがあります",
        "動きたくないくらい疲れています", "階段を上がるのがつらいです",
        "立っているだけで疲れます", "体が重いと感じます",
        "肩や腰がこっています", "全身が疲れ切っています"
    ],
    "brain": [
        "頭がぼーっとして集中できません", "考えがまとまらない感じです",
        "ミスが増えていると感じます", "記憶力が落ちた気がします",
        "判断が遅くなったと感じます", "頭が回らない感じです",
        "集中力が続かないです", "思考がクリアではありません"
    ],
    "mental": [
        "やる気が出ない感じです", "イライラしやすいです",
        "人と話したくない気分です", "不安を感じることが多いです",
        "気力がわいてきません", "心が疲れた感じがします",
        "何もしたくない気分です", "心が休まらない感じです"
    ],
}
FATIGUE_CATS = ["body", "brain", "mental"]

import random

def select_random_questions() -> List[Dict[str, str]]:
    selected = []
    for cat in FATIGUE_CATS:
        questions = random.sample(QUESTION_POOL[cat], min(2, len(QUESTION_POOL[cat])))
        for q in questions:
            selected.append({"category": cat, "question": q})
    random.shuffle(selected)
    return selected

# -------------------------
# 音声特徴量抽出 + 品質チェック強化
# -------------------------
def extract_audio_features_safe(audio_path: str, min_duration: float = 3.0, opensmile_use_cache: bool = True) -> Dict:
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

    # 品質チェック強化
    text_len = len(text)
    unique_chars = len(set(text))
    speech_rate = text_len / max(duration, 1e-3) if duration > 0 else 0.0
    lexical_div = unique_chars / max(text_len, 1) if text_len > 0 else 0.0
    
    keywords = ["疲れた", "しんどい", "だるい", "眠い", "つらい", "無理"]
    fatigue_word_count = sum(text.count(k) for k in keywords)

    # 品質条件：音声長 >= 3 秒, 文字数 >= 10, ユニーク文字数 >= 5, (OpenSMILE 成功 or 十分な文字起こし)
    duration_ok = duration >= min_duration
    text_len_ok = text_len >= 10
    unique_ok = unique_chars >= 5
    smile_or_text = smile_success or (text_len >= 20 and unique_chars >= 10)
    
    quality_ok = duration_ok and text_len_ok and unique_ok and smile_or_text
    
    quality_message = "OK"
    if not quality_ok:
        reasons = []
        if not duration_ok: reasons.append("音声短")
        if not text_len_ok: reasons.append("文字少")
        if not unique_ok: reasons.append("単調")
        if not smile_or_text: reasons.append("品質不足")
        quality_message = f"品質 NG: {', '.join(reasons)}"

    return {
        "text": text, "duration": duration, "speech_rate": speech_rate,
        "fatigue_word_count": fatigue_word_count, "text_length": text_len,
        "lexical_diversity": lexical_div, "unique_chars": unique_chars,
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

# -------------------------
# サンプル ID 生成（重複対策）
# -------------------------
def generate_sample_id(text: str, speech_rate: float, text_length: int, smile_features: Dict) -> str:
    """テキスト + 特徴量から SHA256 サンプル ID を生成"""
    # smile_features の一部をサンプリング（軽量に）
    smile_sample = {k: round(v, 3) for k, v in list(smile_features.items())[:10]}
    content = f"{text}|{speech_rate:.3f}|{text_length}|{json.dumps(smile_sample, ensure_ascii=False, sort_keys=True)}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# -------------------------
# 推論処理
# -------------------------
def get_confidence(scores: Dict[str, float]) -> str:
    if not scores:
        return "低"
    mx = max(scores.values())
    if mx >= 7:
        return "高"
    elif mx >= 4:
        return "中"
    return "低"

def predict_fatigue_safe(audio_path: Optional[str]) -> Dict:
    log_memory("predict_before")
    
    if not audio_path:
        return {"success": False, "message": "音声ファイルが指定されていません", "scores": {}, "confidence": "低"}

    mem_mb = log_memory("predict_memcheck")
    if mem_mb is not None and mem_mb > MEMORY_THRESHOLD_CRITICAL:
        return {"success": False, "message": "メモリ不足のため処理を中断しました", "scores": {}, "confidence": "低"}

    audio_feat = extract_audio_features_safe(audio_path)
    if not audio_feat["quality_ok"]:
        return {"success": False, "message": audio_feat["quality_message"], "scores": {}, "confidence": "低", "audio_feat": audio_feat}

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
            "success": True, "scores": fallback_scores, "confidence": f"{get_confidence(fallback_scores)}（推定）",
            "message": "簡易推定", "features": features, "audio_quality": f"{audio_feat['duration']:.1f}秒",
            "audio_feat": audio_feat, "similarities": similarities, "query_embedding": query_emb
        }

    X_base = np.array([[features.get(col, 0.0) for col in feature_cols]], dtype=np.float32)
    try:
        X_base_scaled = scaler.transform(X_base)
    except Exception:
        return {"success": False, "message": "前処理に失敗しました", "scores": {}, "confidence": "低"}

    pca_n_comp = getattr(pca, "n_components_", getattr(pca, "n_components", 0))
    if query_emb.size == 0:
        emb_pca = np.zeros((1, int(pca_n_comp) if pca_n_comp is not None else 0), dtype=np.float32)
    else:
        try:
            emb_pca = pca.transform(query_emb.reshape(1, -1))
        except Exception as e:
            logger.warning(f"PCA transform failed: {e}")
            emb_pca = np.zeros((1, int(pca_n_comp) if pca_n_comp is not None else 0), dtype=np.float32)

    X = np.hstack([X_base_scaled, emb_pca]) if emb_pca.size > 0 else X_base_scaled

    if metadata:
        expected_features = int(metadata.get("model_feature_count", X.shape[1]))
        if X.shape[1] != expected_features:
            return {"success": False, "message": f"特徴量数不一致：{X.shape[1]} != {expected_features}", "scores": {}, "confidence": "低"}

    scores = {}
    for cat in ["body", "brain", "mental"]:
        model = models.get(cat)
        if model:
            try:
                scores[cat] = round(float(np.clip(float(model.predict(X)[0]), 0.0, 9.0)), 1)
            except Exception as e:
                logger.warning(f"Predict error for {cat}: {e}")
                scores[cat] = 0.0
        else:
            scores[cat] = 0.0
            
    confidence = get_confidence(scores)
    
    try: del ref_emb, X_base, X_base_scaled, emb_pca, X
    except: pass
    gc.collect()
    log_memory("predict_after")
    
    return {
        "success": True, "scores": scores, "confidence": confidence, "message": "OK",
        "features": features, "audio_quality": f"{audio_feat['duration']:.1f}秒",
        "audio_feat": audio_feat, "similarities": similarities, "query_embedding": query_emb
    }

# -------------------------
# データ保存（重複チェック + embedding 削除 + ラベル平均化）
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

def save_data_with_result_safe(audio_feat, query_emb, user_labels: Dict[str, float], pred_scores: Dict[str, float], similarities: Dict, questions: List[Dict[str, str]], choices: List[int]) -> str:
    """
    user_labels: 未使用（新方式では choices から計算）
    questions: [{"category": "body", "question": "..."}, ...]
    choices: [4, 2, 5, 4, 3, 2] などのスライダー値
    """
    if not audio_feat or not audio_feat.get("quality_ok", False):
        return "音声品質不足のため保存をスキップしました"
    
    # ① ラベル集約：各カテゴリの平均値を計算
    category_scores = {"body": [], "brain": [], "mental": []}
    for i, q in enumerate(questions):
        cat = q["category"]
        if i < len(choices):
            raw_val = choices[i]
            # 1-5 スケールを 0-9 スケールに変換
            scaled = (raw_val - 1) * 2.25
            category_scores[cat].append(scaled)
    
    # 平均値を計算（0-9 スケール）
    label_body = np.mean(category_scores["body"]) if category_scores["body"] else 0.0
    label_brain = np.mean(category_scores["brain"]) if category_scores["brain"] else 0.0
    label_mental = np.mean(category_scores["mental"]) if category_scores["mental"] else 0.0
    
    # ③ 重複チェック：サンプル ID 生成
    sample_id = generate_sample_id(
        audio_feat.get("text", ""),
        audio_feat.get("speech_rate", 0.0),
        audio_feat.get("text_length", 0),
        audio_feat.get("smile_features", {})
    )
    
    # 既存データに同じ sample_id がないか確認
    lock = DATA_PARQUET + ".lock"
    if not _acquire_lock(lock, timeout=10):
        return "保存ロックの取得に失敗しました"
    
    try:
        existing = pd.read_parquet(DATA_PARQUET) if os.path.exists(DATA_PARQUET) and os.path.getsize(DATA_PARQUET) > 0 else pd.DataFrame()
        
        # 重複チェック
        if "sample_id" in existing.columns and sample_id in existing["sample_id"].values:
            QUALITY_STATS["duplicate_blocked_count"] += 1
            log_quality_stats()
            return "同じ内容のデータはすでに送信されています"
        
        # ④ text_embedding を保存しない
        df_row = {
            "text": audio_feat.get("text", ""),
            # "text_embedding": embed_to_json(query_emb),  # 削除
            "speech_rate": float(audio_feat.get("speech_rate", 0.0)),
            "fatigue_word_count": float(audio_feat.get("fatigue_word_count", 0.0)),
            "text_length": int(audio_feat.get("text_length", 0)),
            "lexical_diversity": float(audio_feat.get("lexical_diversity", 0.0)),
            "unique_chars": int(audio_feat.get("unique_chars", 0)),
            **{k: float(v) for k, v in audio_feat.get("smile_features", {}).items()},
            **{k: float(v) for k, v in similarities.items()},
            "label_body": float(label_body),
            "label_brain": float(label_brain),
            "label_mental": float(label_mental),
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
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        
        # 品質統計更新
        QUALITY_STATS["quality_ok_count"] += 1
        if audio_feat.get("smile_success", False):
            QUALITY_STATS["smile_success_count"] += 1
        log_quality_stats()
        
    finally:
        _release_lock(lock)

    save_to_hf_dataset_with_retry_safe(DATA_PARQUET, "dataset.parquet")
    
    return f"無事に保存されました🌱（累計データ数：{len(combined)}件）"

# -------------------------
# UI コンポーネント
# -------------------------
def generate_all_comments(scores: Dict[str, float]) -> Dict[str, str]:
    if not scores:
        return {"body": "", "brain": "", "mental": ""}
    
    comments = {
        "body": {
            "high": "体がかなりお疲れのようです🛌 今日は早めにお布団に入って、ゆっくり休んでくださいね。",
            "mid": "身体に少し疲れが溜まっているかも。温かい飲み物でもいかがですか？☕",
            "low": "体の調子は良さそうです！✨"
        },
        "brain": {
            "high": "頭をたくさん使いましたね🧠 画面から少し離れて、目を閉じる時間を作ってみてください。",
            "mid": "脳が少しお疲れ気味です。短い休憩を挟むとスッキリしますよ🌱",
            "low": "頭は冴えているようです！💡"
        },
        "mental": {
            "high": "心に負担がかかっているサインです☁️ 好きな音楽を聴いたり、まずは自分を甘やかしてあげてくださいね。",
            "mid": "心が少しお疲れのようです。深呼吸して、リラックスする時間をとりましょう🍀",
            "low": "心は落ち着いていて安定しています🌸"
        },
    }
    
    result = {}
    for cat in ["body", "brain", "mental"]:
        score = scores.get(cat, 0.0)
        level = "high" if score >= 7 else "mid" if score >= 4 else "low"
        result[cat] = comments.get(cat, {}).get(level, "")
    return result

def get_fatigue_type(scores: Dict[str, float]) -> str:
    if not scores:
        return "不明"
    max_cat = max(scores, key=scores.get)
    type_map = {"body": "身体疲れ", "brain": "脳疲れ", "mental": "心疲れ"}
    return type_map.get(max_cat, "不明")

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Restee - 休息のデザイン", page_icon="🌱", layout="centered")

st.title("🌱 Restee - AI モデル開発用")
st.markdown("音声での対話から疲労の種類を分析して休み方を提案するアプリを作りたいです。ご協力お願いします！！")

# 冒頭にデータ数表示
try:
    if os.path.exists(DATA_PARQUET) and os.path.getsize(DATA_PARQUET) > 0:
        df_tmp = pd.read_parquet(DATA_PARQUET)
        st.info(f"📊 現在 **{len(df_tmp)} 件** のデータが集まっています！ご協力ありがとうございます🌱")
    else:
        st.info("📊 現在 0 件 のデータが集まっています。最初のデータ提供者になりませんか？")
except Exception:
    pass

# セッション初期化
if "questions" not in st.session_state: st.session_state["questions"] = select_random_questions()
if "analyzed" not in st.session_state: st.session_state["analyzed"] = False
if "last_result" not in st.session_state: st.session_state["last_result"] = None
if "data_saved" not in st.session_state: st.session_state["data_saved"] = False
if "save_msg" not in st.session_state: st.session_state["save_msg"] = ""
if "uploaded_file_hash" not in st.session_state: st.session_state["uploaded_file_hash"] = None

# アンケートカード（各カテゴリ 2 問、計 6 問）
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
        st.session_state["questions"] = select_random_questions()
        for i in range(len(st.session_state["questions"])): st.session_state.pop(f"choice_{i}", None)
        st.rerun()

st.divider()

# 音声アップロードと解析
st.subheader("🎤 2. 今の気持ちを声で残してみる")
st.markdown("「今日はちょっと疲れたな」「なんだか体が重い」など、今の状態を数秒つぶやいてみてください。")
st.caption("🔒 音声データそのものは保存されません。匿名化された特徴量のみが研究目的で利用されます。個人を特定できる情報は収集しません。")

uploaded_file = st.file_uploader("音声ファイルを選択 (WAV / MP3 等)", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    
    if st.session_state["uploaded_file_hash"] != file_hash:
        st.session_state["uploaded_file_hash"] = file_hash
        st.session_state["analyzed"] = False
        st.session_state["data_saved"] = False
        st.session_state["save_msg"] = ""
        
    st.audio(uploaded_file)
    
    analyzed = st.session_state.get("analyzed", False)
    data_saved = st.session_state.get("data_saved", False)
    
    if st.button("✨ 解析する！", type="primary", use_container_width=True, disabled=analyzed):
        with st.spinner("声のトーンや言葉を紐解いています..."):
            tmp_path = None
            try:
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, f"restee_upload_{int(time.time()*1000)}_{uploaded_file.name}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 品質統計：total_attempts カウント
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

# 結果表示エリア
if st.session_state["analyzed"] and st.session_state["last_result"]:
    res = st.session_state["last_result"]
    st.divider()
    st.subheader("🍀 あなたの今の状態")
    
    if not res.get("success", False):
        st.warning(f"うまく読み取れませんでした：{res.get('message', '')}")
        # 品質 NG もカウント
        if "quality_ok" in res.get("audio_feat", {}):
            QUALITY_STATS["quality_ng_count"] += 1
            log_quality_stats()
    else:
        scores = res.get("scores", {})
        
        fatigue_type = get_fatigue_type(scores)
        st.success(f"📈 あなたの疲れタイプ：**{fatigue_type}**")
        
        all_comments = generate_all_comments(scores)
        with st.expander("💬 各カテゴリのコメントを見る"):
            st.write(f"**💪 身体**: {all_comments.get('body', '')}")
            st.write(f"**🧠 脳**: {all_comments.get('brain', '')}")
            st.write(f"**💙 心**: {all_comments.get('mental', '')}")
        
        col1, col2, col3 = st.columns(3)
        metrics = [("身体の疲れ", "body", "💪"), ("頭の疲れ", "brain", "🧠"), ("心の疲れ", "mental", "💙")]
        
        for col, (label, cat, icon) in zip([col1, col2, col3], metrics):
            val = scores.get(cat, 0.0)
            with col:
                st.metric(f"{icon} {label}", f"{val:.1f} / 9")
                st.progress(float(np.clip(val / 9.0, 0.0, 1.0)))
        
        st.write(f"**信頼度**: {res.get('confidence', '低')}")

        with st.expander("🛠️ 解析の詳細データを見る"):
            st.write("各スコア (0-9):", {k: f"{v:.1f}" for k, v in scores.items()})
            st.write("オーディオ品質:", res.get("audio_quality", ""))
            if DEBUG:
                st.json(res.get("features", {}))
            
        st.divider()
        st.subheader("🤝 開発へのご協力のお願い")
        st.markdown("""
        差し支えなければ、今回の結果を匿名データとして送信してください。
        
        - 🔒 **音声データそのものは保存されません**
        - 📊 **匿名化された特徴量のみが研究目的で利用されます**
        - 👤 **個人を特定できる情報は収集しません**
        
        安心してご協力ください！
        """)
        
        if st.session_state["data_saved"]:
            st.success(f"送信済みです🌱\n\n({st.session_state.get('save_msg', '')})")
        else:
            if st.button("💾 この結果を匿名で送信する", type="secondary"):
                with st.spinner("データを送っています..."):
                    # ① 回答をリストで取得
                    choices = []
                    for i in range(len(st.session_state["questions"])):
                        choices.append(st.session_state.get(f"choice_{i}", 3))
                    
                    audio_feat = res.get("audio_feat")
                    
                    save_msg = save_data_with_result_safe(
                        audio_feat=audio_feat,
                        query_emb=res.get("query_embedding"),
                        user_labels={},  # 未使用（新方式では choices から計算）
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
