"""
Restee - Streamlit Cloud 安定稼働版（研究データ収集・完全修正版）
目的:
- Streamlit Cloud（RAM 制限環境）で安定稼働
- 研究用データセットとして品質の高いデータのみ蓄積
- 音声特徴量の品質判定を厳格化（80% ルール）
- Whisper 失敗時も OpenSMILE 成功なら保存（声質重視）

主な変更点
- バグ修正: del segments, info の UnboundLocalError 対策
- OpenSMILE 成功判定: 80% 以上の特徴量が有効な場合のみ成功
- quality_ok 条件: Whisper 失敗でも OpenSMILE 成功なら保存
- 研究データ品質: 壊れた音声・無音・短すぎる録音を弾く
- 既存機能（fallback 推定、HF 保存、cache 構造等）は維持
"""

import os
import json
import logging
import time
import tempfile
import gc
from datetime import datetime
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# Optional imports guarded
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restee_safe")

# Config
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = os.environ.get("REPO_ID", "")
DATA_DIR = "data_store"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PARQUET = os.path.join(DATA_DIR, "dataset.parquet")
DATA_CSV = os.path.join(DATA_DIR, "dataset.csv")
FEATURE_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
PCA_FILE = os.path.join(MODEL_DIR, "pca.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

# Memory thresholds (MB) tuned per request
MEMORY_THRESHOLD_WARN = int(os.environ.get("MEM_WARN_MB", 2200))
MEMORY_THRESHOLD_CRITICAL = int(os.environ.get("MEM_CRIT_MB", 2500))

MODEL_CONFIG = {
    "body": {"model": os.path.join(MODEL_DIR, "body_model.txt"), "label": "label_body"},
    "brain": {"model": os.path.join(MODEL_DIR, "brain_model.txt"), "label": "label_brain"},
    "mental": {"model": os.path.join(MODEL_DIR, "mental_model.txt"), "label": "label_mental"},
}

# HF login guard
HF_READY = False

# Simple file lock to avoid concurrent writes (not robust across NFS)
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
    else:
        logger.debug(f"[MEM] psutil not available for {label}")
        return None

# Embedding helpers (保存は JSON 文字列)
def embed_to_json(arr: np.ndarray) -> str:
    return json.dumps(arr.astype(float).tolist(), ensure_ascii=False)

def json_to_embed(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    if isinstance(obj, list):
        return np.array(obj, dtype=np.float32)
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return np.array(parsed, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)
    return np.array([], dtype=np.float32)

# -------------------------
# モデル / ライブラリの遅延ロード（例外処理付き）
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
        m = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        log_memory("opensmile_cached_loaded")
        return m
    except Exception as e:
        logger.warning(f"opensmile init failed: {e}")
        return None

def get_opensmile_model_safe(use_cache: bool = True):
    try:
        if use_cache:
            return get_opensmile_cached()
        else:
            try:
                import opensmile
                m = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals
                )
                log_memory("opensmile_loaded_nocache")
                return m
            except Exception as e:
                logger.warning(f"opensmile init (nocache) failed: {e}")
                return None
    except Exception as e:
        logger.warning(f"get_opensmile_model_safe error: {e}")
        return None

@st.cache_resource
def load_models_safe():
    models = {}
    for cat, cfg in MODEL_CONFIG.items():
        path = cfg["model"]
        if os.path.exists(path):
            try:
                models[cat] = lgb.Booster(model_file=path)
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
        except Exception as e:
            logger.warning(f"Failed to load feature file: {e}")
            feature_cols = None
    return pca, scaler, feature_cols

def load_metadata_safe():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None
    return None

# -------------------------
# 参考文書 embedding（キャッシュ）
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
        logger.warning("Embedding model unavailable for reference embeddings")
        return None
    try:
        texts = [REFERENCE_DOCS[cat] for cat in REFERENCE_CATS]
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        return emb
    except Exception as e:
        logger.warning(f"Failed to create reference embeddings: {e}")
        return None

# -------------------------
# 音声特徴量抽出（堅牢化・研究データ品質版）
# -------------------------
QUESTION_ORDER = ["body", "brain", "mental"]

def extract_audio_features_safe(
    audio_path: str,
    min_duration: float = 3.0,
    opensmile_use_cache: bool = True
) -> Dict:
    """
    - Whisper で文字起こし（失敗時は空文字）
    - OpenSMILE で音響特徴量（失敗時は空辞書）
    - 音声品質チェック（duration >= min_duration）
    - smile_success を返す（OpenSMILE 80% 以上の特徴量が有効な場合のみ True）
    - 研究データ品質: 壊れた音声・無音・短すぎる録音を弾く
    """
    log_memory("extract_before")

    # 初期化: UnboundLocalError 対策
    whisper = None
    segments = None
    info = None

    text = ""
    duration = 0.0

    try:
        whisper = get_whisper_model_safe()
        if whisper is not None:
            segments, info = whisper.transcribe(
                audio_path,
                language="ja",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )
            text = "".join(s.text for s in segments).strip()
            duration = float(info.duration) if (info and hasattr(info, "duration")) else 0.0
        else:
            logger.info("Whisper model not available; skipping transcription")
    except Exception as e:
        logger.warning(f"Whisper transcribe error: {e}")
        text = ""
        duration = 0.0

    # Whisper 後の不要変数を安全に解放
    try:
        del whisper
    except Exception:
        pass
    try:
        del segments, info
    except Exception:
        pass

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
                    # MultiIndex 対策: タプルなら結合
                    if isinstance(col, tuple):
                        col_key = "_".join(str(c) for c in col)
                    else:
                        col_key = str(col)

                    val = smile_df[col].iloc[0]
                    if np.isfinite(val):
                        valid_count += 1
                    key = f"smile_{col_key}".replace(" ", "_")
                    smile_features[key] = float(val) if np.isfinite(val) else 0.0

                # 80% ルール: 88 特徴量中 80% 以上が有効な場合のみ成功
                valid_ratio = valid_count / total_cols if total_cols > 0 else 0.0
                if valid_ratio >= 0.8:
                    smile_success = True
                else:
                    smile_success = False
                    smile_features = {}
                    logger.info(f"OpenSMILE valid_ratio={valid_ratio:.2f} < 0.8, marking as failed")
            else:
                smile_success = False
        except Exception as e:
            logger.warning(f"OpenSMILE processing failed: {e}")
            smile_success = False
    else:
        logger.info("OpenSMILE not available; skipping acoustic features")
        smile_success = False

    keywords = ["疲れた", "しんどい", "だるい", "眠い", "つらい", "無理"]
    text_len = len(text)
    speech_rate = len(text) / max(duration, 1e-3) if duration > 0 else 0.0
    lexical_div = len(set(text)) / max(text_len, 1) if text_len > 0 else 0.0
    fatigue_word_count = sum(text.count(k) for k in keywords)

    # quality_ok 条件: Whisper 失敗でも OpenSMILE 成功なら保存（声質重視）
    quality_ok = (
        duration >= min_duration
        and (
            text_len >= 3
            or smile_success
        )
    )
    quality_message = "OK" if quality_ok else "音声が短すぎるか文字起こし失敗"

    log_memory("extract_after")

    return {
        "text": text,
        "duration": duration,
        "speech_rate": speech_rate,
        "fatigue_word_count": fatigue_word_count,
        "text_length": text_len,
        "lexical_diversity": lexical_div,
        "smile_features": smile_features,
        "quality_ok": quality_ok,
        "quality_message": quality_message,
        "smile_success": smile_success,
    }

def encode_text_safe(text: str):
    model = get_embedding_model_safe()
    if model is None:
        return np.array([], dtype=np.float32)
    try:
        emb = model.encode(
            [f"query: {text}"],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)[0]
        return emb
    except Exception as e:
        logger.warning(f"Text encoding failed: {e}")
        return np.array([], dtype=np.float32)

# -------------------------
# 推論処理（安全化）
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
        return {
            "success": False,
            "message": "音声ファイルが指定されていません",
            "scores": {},
            "confidence": "低"
        }

    mem_mb = log_memory("predict_memcheck")
    if mem_mb is not None and mem_mb > MEMORY_THRESHOLD_CRITICAL:
        return {
            "success": False,
            "message": "メモリ不足のため処理を中断しました",
            "scores": {},
            "confidence": "低"
        }

    audio_feat = extract_audio_features_safe(audio_path)
    if not audio_feat["quality_ok"]:
        return {
            "success": False,
            "message": audio_feat["quality_message"],
            "scores": {},
            "confidence": "低",
            "audio_feat": audio_feat
        }

    query_emb = encode_text_safe(audio_feat["text"])
    ref_emb = get_reference_embeddings_safe()

    similarities = {}
    if query_emb.size > 0 and ref_emb is not None:
        for i, cat in enumerate(REFERENCE_CATS):
            try:
                sim = cosine_similarity(
                    query_emb.reshape(1, -1),
                    ref_emb[i].reshape(1, -1)
                )[0][0]
            except Exception:
                sim = 0.0
            similarities[f"sim_{cat}"] = float(sim)
    else:
        for cat in REFERENCE_CATS:
            similarities[f"sim_{cat}"] = 0.0

    features = {
        "speech_rate": audio_feat["speech_rate"],
        "fatigue_word_count": audio_feat["fatigue_word_count"],
        "text_length": audio_feat["text_length"],
        "lexical_diversity": audio_feat["lexical_diversity"],
        **audio_feat["smile_features"],
        **similarities,
    }

    models = load_models_safe()
    pca, scaler, feature_cols = load_preprocess_objects_safe()
    metadata = load_metadata_safe()

    models_ready = (
        all(models.get(k) is not None for k in ["body", "brain", "mental"])
        and pca is not None
        and scaler is not None
        and feature_cols is not None
    )

    if not models_ready:
        fallback_scores = {}
        for cat in ["body", "brain", "mental"]:
            sim_fatigue = similarities.get(f"sim_{cat}", 0.0)
            sim_healthy = similarities.get("sim_healthy", 0.0)
            raw = (sim_fatigue - sim_healthy + 1.0) / 2.0
            fallback_scores[cat] = round(float(np.clip(raw * 9.0, 0.0, 9.0)), 1)
        return {
            "success": True,
            "scores": fallback_scores,
            "confidence": f"{get_confidence(fallback_scores)}（モデル未ロード・類似度ベース）",
            "message": "簡易推定（類似度ベース）",
            "features": features,
            "audio_quality": f"{audio_feat['duration']:.1f}秒",
            "audio_feat": audio_feat,
            "similarities": similarities,
        }

    # 特徴量ベクトル作成（float32 徹底）
    X_base = np.array(
        [[features.get(col, 0.0) for col in feature_cols]],
        dtype=np.float32
    )
    try:
        X_base_scaled = scaler.transform(X_base)
    except Exception as e:
        logger.warning(f"Scaler transform failed: {e}")
        return {
            "success": False,
            "message": "前処理に失敗しました",
            "scores": {},
            "confidence": "低"
        }

    # PCA の n_components 安全参照
    pca_n_comp = getattr(pca, "n_components_", None)
    if pca_n_comp is None:
        pca_n_comp = getattr(pca, "n_components", 0)

    if query_emb.size == 0:
        emb_pca = np.zeros(
            (1, int(pca_n_comp) if pca_n_comp is not None else 0),
            dtype=np.float32
        )
    else:
        try:
            emb_pca = pca.transform(query_emb.reshape(1, -1))
        except Exception as e:
            logger.warning(f"PCA transform failed: {e}")
            emb_pca = np.zeros(
                (1, int(pca_n_comp) if pca_n_comp is not None else 0),
                dtype=np.float32
            )

    X = np.hstack([X_base_scaled, emb_pca]) if emb_pca.size > 0 else X_base_scaled

    if metadata is not None and "model_feature_count" in metadata:
        expected = int(metadata["model_feature_count"])
        if X.shape[1] != expected:
            return {
                "success": False,
                "message": f"特徴量数不一致：{X.shape[1]} != {expected}",
                "scores": {},
                "confidence": "低"
            }

    scores = {}
    for cat in ["body", "brain", "mental"]:
        model = models.get(cat)
        if model is None:
            scores[cat] = 0.0
            continue
        try:
            raw = float(model.predict(X)[0])
            scores[cat] = round(float(np.clip(raw, 0.0, 9.0)), 1)
        except Exception as e:
            logger.warning(f"Model predict failed for {cat}: {e}")
            scores[cat] = 0.0

    # 解放
    del X_base, X_base_scaled, emb_pca, X
    gc.collect()
    log_memory("predict_after")

    return {
        "success": True,
        "scores": scores,
        "confidence": get_confidence(scores),
        "message": "OK",
        "features": features,
        "audio_quality": f"{audio_feat['duration']:.1f}秒",
        "audio_feat": audio_feat,
        "similarities": similarities,
    }

# -------------------------
# データ保存（原子操作・リトライ） JSON 埋め込みで Parquet 安定化
# -------------------------
def _atomic_write_csv(df: pd.DataFrame, target_path: str):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(target_path))
    os.close(tmp_fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def save_to_hf_dataset_with_retry_safe(df_row: Dict, max_retries: int = 3):
    global HF_READY
    if not HF_TOKEN or not REPO_ID:
        logger.warning("HF_TOKEN or REPO_ID not set; skipping HF upload")
        return False

    lock = DATA_CSV + ".lock"
    if not _acquire_lock(lock, timeout=10):
        logger.warning("Could not acquire lock for CSV write")
        return False
    try:
        if os.path.exists(DATA_CSV) and os.path.getsize(DATA_CSV) > 0:
            try:
                existing = pd.read_csv(DATA_CSV)
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        new_df = pd.DataFrame([df_row])
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        # 数値列のみ fillna(0)、text_embedding はそのまま
        if "text_embedding" in combined.columns:
            cols_to_fill = combined.columns.difference(["text_embedding"])
            combined[cols_to_fill] = combined[cols_to_fill].fillna(0)
        else:
            combined = combined.fillna(0)
        _atomic_write_csv(combined, DATA_CSV)
    finally:
        _release_lock(lock)

    for attempt in range(1, max_retries + 1):
        try:
            from huggingface_hub import HfApi, login
            if not HF_READY:
                login(token=HF_TOKEN)
                HF_READY = True
            api = HfApi()
            api.upload_file(
                path_or_fileobj=DATA_CSV,
                path_in_repo="dataset.csv",
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            logger.info("Uploaded CSV to HF")
            return True
        except Exception as e:
            logger.warning(f"HF upload attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)
    logger.error("HF upload failed after retries")
    return False

def save_data_with_result_safe(
    audio_feat,
    query_emb,
    body_score: float,
    brain_score: float,
    mental_score: float,
    similarities: Dict
):
    """
    保存前に品質チェックを行い、品質不足の場合は保存しない。
    保存時は text_embedding を JSON 文字列で保存し、Parquet の text_embedding 列には fillna を適用しない。
    """
    # 保存禁止条件チェック
    if audio_feat is None:
        return "解析品質不足のため保存しません（audio_feat が None）"
    if not audio_feat.get("quality_ok", False):
        return "解析品質不足のため保存しません（音声品質不足）"
    if not audio_feat.get("smile_success", False):
        return "解析品質不足のため保存しません（OpenSMILE による音響特徴抽出失敗）"
    if query_emb is None or (isinstance(query_emb, np.ndarray) and query_emb.size == 0):
        return "解析品質不足のため保存しません（埋め込み取得失敗）"

    df_row = {
        "text": audio_feat.get("text", ""),
        "text_embedding": (
            embed_to_json(query_emb)
            if query_emb is not None and getattr(query_emb, "size", 0) > 0
            else embed_to_json(np.array([], dtype=np.float32))
        ),
        "speech_rate": float(audio_feat.get("speech_rate", 0.0)),
        "fatigue_word_count": float(audio_feat.get("fatigue_word_count", 0.0)),
        "text_length": int(audio_feat.get("text_length", 0)),
        "lexical_diversity": float(audio_feat.get("lexical_diversity", 0.0)),
        **{k: float(v) for k, v in audio_feat.get("smile_features", {}).items()},
        **{k: float(v) for k, v in similarities.items()},
        "label_body": float(body_score),
        "label_brain": float(brain_score),
        "label_mental": float(mental_score),
        "timestamp": datetime.now().isoformat(),
        "smile_success": bool(audio_feat.get("smile_success", False)),
        "audio_quality": audio_feat.get("quality_message", ""),
    }

    lock = DATA_PARQUET + ".lock"
    if not _acquire_lock(lock, timeout=10):
        logger.warning("Could not acquire lock for parquet write")
        return "ロック取得失敗"

    try:
        if os.path.exists(DATA_PARQUET) and os.path.getsize(DATA_PARQUET) > 0:
            try:
                existing = pd.read_parquet(DATA_PARQUET)
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        new_df = pd.DataFrame([df_row])
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        # text_embedding 列は fillna を適用しない
        if "text_embedding" in combined.columns:
            cols_to_fill = combined.columns.difference(["text_embedding"])
            combined[cols_to_fill] = combined[cols_to_fill].fillna(0)
        else:
            combined = combined.fillna(0)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(DATA_PARQUET))
        os.close(tmp_fd)
        try:
            combined.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, DATA_PARQUET)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    finally:
        _release_lock(lock)

    # HF へ CSV 形式でアップロード（text_embedding は JSON 文字列のまま）
    save_to_hf_dataset_with_retry_safe({
        k: (v if k != "text_embedding" else df_row["text_embedding"])
        for k, v in df_row.items()
    })
    return f"データ保存完了（計 {len(combined)} 件）"

# -------------------------
# 学習処理はローカル推奨。必要なら安全ラッパーを別途実装
# -------------------------
def train_model_safe(df: pd.DataFrame):
    raise NotImplementedError("学習処理はローカルでの実行を推奨します。必要なら安全ラッパーを追加します。")

# -------------------------
# ユーティリティ（UI 向け）
# -------------------------
def score_from_choice(choice: str) -> float:
    return {"弱い": 1.0, "ふつう": 5.0, "強い": 9.0}.get(choice, 5.0)

def generate_comment(scores: Dict[str, float]) -> str:
    if not scores:
        return ""
    max_cat = max(scores, key=scores.get)
    max_score = scores[max_cat]
    if max_score >= 7:
        level = "high"
    elif max_score >= 4:
        level = "mid"
    else:
        level = "low"
    comments = {
        "body": {
            "high": "身体疲労が高めです。睡眠や軽いストレッチを取り入れてみましょう。",
            "mid": "身体疲労が見られます。適度な休息を。",
            "low": "身体的には良好です。"
        },
        "brain": {
            "high": "脳疲労が高めです。少し画面から離れて休憩すると効果的です。",
            "mid": "脳疲労が見られます。短い休憩を挟みましょう。",
            "low": "認知機能は良好です。"
        },
        "mental": {
            "high": "精神疲労が高めです。無理をせずリラックスする時間を作りましょう。",
            "mid": "精神疲労が見られます。好きなことをして気分転換を。",
            "low": "精神的には安定しています。"
        },
    }
    return comments.get(max_cat, {}).get(level, "")
