"""
Restee - 音声・テキスト特徴量を用いた疲労状態推定システム
Streamlit Community Cloud 対応 + HF Dataset(CSV) 蓄積
3 分類対応（身体・脳・精神）
"""
import random
import os
import json
import logging
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

from huggingface_hub import HfApi, login, hf_hub_download

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = os.environ.get("REPO_ID", "your-username/fatigue-dataset")
USE_GPU = os.environ.get("USE_GPU", "false").lower() in ("1", "true", "yes")

DATA_DIR = "data_store"
MODEL_DIR = "models"
DATA_PARQUET = os.path.join(DATA_DIR, "dataset.parquet")
DATA_CSV = os.path.join(DATA_DIR, "dataset.csv")

MODEL_CONFIG = {
    "body": {"model": os.path.join(MODEL_DIR, "body_model.txt"), "label": "label_body"},
    "brain": {"model": os.path.join(MODEL_DIR, "brain_model.txt"), "label": "label_brain"},
    "mental": {"model": os.path.join(MODEL_DIR, "mental_model.txt"), "label": "label_mental"},
}
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_FILE = os.path.join(MODEL_DIR, "pca.pkl")
FEATURE_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info(f"Logged in to Hugging Face as {REPO_ID}")
        try:
            downloaded_csv = hf_hub_download(
                repo_id=REPO_ID,
                filename="dataset.csv",
                repo_type="dataset",
            )
            shutil.copy(downloaded_csv, DATA_CSV)
            df_init = pd.read_csv(DATA_CSV)
            if "text_embedding" in df_init.columns:
                df_init["text_embedding"] = df_init["text_embedding"].apply(
                    lambda x: np.array(json.loads(x), dtype=np.float32)
                    if isinstance(x, str) and x
                    else x
                )
            df_init.to_parquet(DATA_PARQUET, index=False)
            logger.info(f"Successfully loaded existing dataset from HF ({len(df_init)} rows).")
        except Exception as e:
            logger.warning(f"No existing dataset found or download failed: {e}")
    except Exception as e:
        logger.warning(f"HF login failed: {e}")

REFERENCE_DOCS = {
    "body": "passage: 身体的疲労。体がだるい、重い、筋肉に力が入らない、へとへと、動きたくない、休みたい。",
    "brain": "passage: 認知の疲労（脳疲労）。頭がぼーっとする、集中できない、考えがまとまらない、ミスが増える。",
    "mental": "passage: 精神的疲労。やる気が出ない、気力がない、イライラする、不安、人と話したくない、心が疲れた。",
    "healthy": "passage: 健康で活力のある状態。体が軽い、元気いっぱい、すっきり、よく眠れた、集中できる、前向き。",
}

QUESTION_BANK = {
    "body": [
        "体を動かした後のように、体が重く感じることがある。",
        "朝起きたときに、体が十分に休めた感じがしない。",
        "少し動いただけでも、体力を使ったと感じやすい。",
        "同じ姿勢を続けると、体のだるさを感じやすい。",
        "階段を上るなどの軽い動作で、いつもより疲れやすい。",
        "最近、体の回復に時間がかかると感じる。",
        "運動していなくても、体に疲れを感じることがある。",
        "長時間座ったあとに、体のだるさが強くなる。",
        "首や肩、脚などに、全身的な疲れを感じやすい。",
        "体を休めても、疲れが抜けにくいと感じる。",
        "日中に、体の重さで動きたくないと感じる。",
        "立ち上がるときに、体がすぐに反応しにくい。",
        "いつもより体の持久力が落ちたと感じる。",
        "細かな動作でも、体に負担を感じやすい。",
        "全身のエネルギーが不足しているように感じる。",
    ],
    "brain": [
        "集中しようとしても、考えがまとまりにくい。",
        "読んでいる内容が頭に入りにくいと感じる。",
        "普段より、作業の手順を思い出しにくい。",
        "一度に複数のことを考えるのが難しい。",
        "作業中に、何をしていたか忘れやすい。",
        "短い説明でも、理解に時間がかかることがある。",
        "考えごとを続けると、頭がすぐにいっぱいになる。",
        "新しい情報を覚えるのに、いつもより時間がかかる。",
        "注意を向け続けるのが難しいと感じる。",
        "作業の途中で、思考が止まりやすい。",
        "計算や整理をすると、頭が疲れやすい。",
        "考える作業のあと、頭がぼんやりしやすい。",
        "人の話を聞きながら内容を追うのが難しい。",
        "判断に普段より時間がかかると感じる。",
        "頭を使ったあとの回復に時間がかかる。",
    ],
    "mental": [
        "気分が落ち着かず、そわそわしやすい。",
        "小さなことが気になってしまう。",
        "やる気が出にくいと感じることがある。",
        "気持ちの切り替えに時間がかかる。",
        "普段より、気分が重く感じる。",
        "集中したいのに、気持ちがそれやすい。",
        "作業に対して、前向きな気持ちを保ちにくい。",
        "ストレスを感じやすい状態だと感じる。",
        "気分の上下で、行動が左右されやすい。",
        "何かを始める気持ちを作りにくい。",
        "心が休まらないと感じることがある。",
        "些細なことで気持ちが疲れやすい。",
        "作業後に、気分的な消耗を強く感じる。",
        "落ち着いて考えるより先に焦りが出やすい。",
        "精神的に回復していないと感じることがある。",
    ],
}

QUESTION_ORDER = ["body", "brain", "mental"]


class AudioProcessor:
    def __init__(self):
        self._whisper = None
        self._smile = None
        self._device = "cuda" if USE_GPU else "cpu"

    def _load_whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel

            compute_type = "float16" if USE_GPU else "int8"
            self._whisper = WhisperModel("base", device=self._device, compute_type=compute_type)
        return self._whisper

    def _load_smile(self):
        if self._smile is None:
            import opensmile

            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        return self._smile

    @staticmethod
    def _sanitize(col: str) -> str:
        return f"smile_{re.sub(r'[^\\w]', '_', col)}"

    def extract(self, audio_path: str) -> Dict:
        whisper = self._load_whisper()
        smile = self._load_smile()

        smile_df = smile.process_file(audio_path)
        smile_features = {}
        if not smile_df.empty:
            for col in smile_df.columns:
                val = smile_df[col].iloc[0]
                smile_features[self._sanitize(col)] = float(val) if np.isfinite(val) else 0.0

        try:
            segments, info = whisper.transcribe(audio_path, language="ja")
            text = "".join([s.text for s in segments]).strip()
            duration = float(info.duration) if info else 0.0
        except Exception as e:
            logger.warning(f"Whisper error: {e}")
            text = ""
            duration = 0.0

        keywords = ["疲れた", "しんどい", "だるい", "眠い", "つらい", "無理"]
        return {
            "text": text,
            "duration": duration,
            "speech_rate": len(text) / max(duration, 1e-3),
            "fatigue_word_count": sum(text.count(k) for k in keywords),
            "text_length": len(text),
            "lexical_diversity": len(set(text)) / max(len(text), 1),
            "smile_features": smile_features,
            "quality_ok": len(text) > 0 and duration >= 3.0,
            "quality_message": "OK" if len(text) > 0 and duration >= 3.0 else "音声が短すぎるか文字起こし失敗",
        }


class EmbeddingProcessor:
    def __init__(self):
        self._model = None
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("intfloat/multilingual-e5-small", device=self._device)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def get_confidence(scores: Dict[str, float]) -> str:
    if not scores:
        return "低"
    mx = max(scores.values())
    if mx >= 7:
        return "高"
    elif mx >= 4:
        return "中"
    return "低"


def build_question_items():
    question_items = []
    for cat in QUESTION_ORDER:
        items = QUESTION_BANK[cat].copy()
        random.shuffle(items)
        for i, q in enumerate(items[:3], 1):
            question_items.append((f"{cat}_q{i}", cat, q))
    return question_items


def train_model(data: pd.DataFrame) -> Dict:
    if len(data) < 30:
        raise ValueError("30 件以上必要です")

    feature_cols = sorted(
        [
            c
            for c in data.columns
            if c.startswith("smile_")
            or c.startswith("sim_")
            or c in {"speech_rate", "fatigue_word_count", "text_length", "lexical_diversity"}
        ]
    )

    X_base = data[feature_cols].fillna(0).astype(np.float32).values
    emb = np.vstack([np.array(e, dtype=np.float32) for e in data["text_embedding"]])

    n_comp = min(32, len(data) - 1, emb.shape[1])
    if n_comp < 2:
        raise ValueError("PCA に必要なデータが不足しています")

    pca = PCA(n_components=n_comp, random_state=42)
    emb_pca = pca.fit_transform(emb)

    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)
    X = np.hstack([X_base_scaled, emb_pca])

    y_data = {}
    for category, config in MODEL_CONFIG.items():
        label_col = config["label"]
        if label_col in data.columns:
            y_data[category] = data[label_col].astype(np.float32).values

    for col in ["label_body", "label_brain", "label_mental"]:
        if col in data.columns and data[col].nunique() < 2:
            raise ValueError(f"{col} のラベル種類が不足しています")

    if len(data) < 50:
        n_splits = 3
    elif len(data) < 200:
        n_splits = 5
    else:
        n_splits = 10

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 10,
        "max_depth": 4,
        "min_child_samples": 3,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.6,
        "bagging_freq": 2,
        "lambda_l1": 0.2,
        "lambda_l2": 0.2,
        "verbose": -1,
        "seed": 42,
    }

    for category, y in y_data.items():
        val_scores = []
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_set_cat = lgb.Dataset(X_train, label=y_train)
            val_set_cat = lgb.Dataset(X_val, label=y_val, reference=train_set_cat)

            model = lgb.train(
                params,
                train_set_cat,
                num_boost_round=100,
                valid_sets=[val_set_cat],
                callbacks=[lgb.early_stopping(15), lgb.log_evaluation(period=0)],
            )
            val_pred = model.predict(X_val)
            val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            val_scores.append(val_rmse)

        avg_rmse = float(np.mean(val_scores))
        results[category] = {"val_rmse": avg_rmse}

        final_train = lgb.Dataset(X, label=y)
        final_model = lgb.train(
            params,
            final_train,
            num_boost_round=100,
        )
        final_model.save_model(MODEL_CONFIG[category]["model"])

    joblib.dump(pca, PCA_FILE)
    joblib.dump(scaler, SCALER_FILE)

    with open(FEATURE_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_timestamp": timestamp,
                "embedding_model": "intfloat/multilingual-e5-small",
                "pca_components": n_comp,
                "feature_count": len(feature_cols),
                "feature_columns": feature_cols,
                "model_feature_count": X.shape[1],
                "data_count": len(data),
                "categories": list(MODEL_CONFIG.keys()),
                "body_rmse": results.get("body", {}).get("val_rmse", 0.0),
                "brain_rmse": results.get("brain", {}).get("val_rmse", 0.0),
                "mental_rmse": results.get("mental", {}).get("val_rmse", 0.0),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "results": results,
        "data_count": len(data),
        "feature_count": len(feature_cols),
    }


class InferenceEngine:
    def __init__(self):
        self.models = {"body": None, "brain": None, "mental": None}
        self.pca = None
        self.scaler = None
        self.feature_cols = None
        self.audio_processor = AudioProcessor()
        self.embedding_processor = EmbeddingProcessor()
        self.reference_embeddings = None
        self._load_models()
        self._load_reference_embeddings()

    def _load_reference_embeddings(self):
        ref_texts = list(REFERENCE_DOCS.values())
        self.reference_embeddings = self.embedding_processor.encode(ref_texts)

    def _load_models(self):
        self.models = {"body": None, "brain": None, "mental": None}
        for category, config in MODEL_CONFIG.items():
            model_path = config["model"]
            if os.path.exists(model_path):
                try:
                    self.models[category] = lgb.Booster(model_file=model_path)
                except Exception as e:
                    logger.warning(f"Failed to load {category} model: {e}")

        if os.path.exists(PCA_FILE):
            self.pca = joblib.load(PCA_FILE)
        if os.path.exists(SCALER_FILE):
            self.scaler = joblib.load(SCALER_FILE)
        if os.path.exists(FEATURE_FILE):
            with open(FEATURE_FILE, "r", encoding="utf-8") as f:
                self.feature_cols = json.load(f)

    def _models_ready(self) -> bool:
        return all(m is not None for m in self.models.values())

    def predict(self, audio_path: Optional[str] = None) -> Dict:
        if not audio_path:
            return {"success": False, "scores": {}, "confidence": "低", "message": "音声なし"}

        audio_feat = self.audio_processor.extract(audio_path)
        if not audio_feat["quality_ok"]:
            return {
                "success": False,
                "scores": {},
                "confidence": "低",
                "message": audio_feat["quality_message"],
            }

        query_emb = self.embedding_processor.encode([f"query: {audio_feat['text']}"])[0]
        similarities = {}
        for i, cat in enumerate(REFERENCE_DOCS.keys()):
            sim = cosine_similarity(
                query_emb.reshape(1, -1),
                self.reference_embeddings[i].reshape(1, -1),
            )[0][0]
            similarities[f"sim_{cat}"] = float(sim)

        features = {
            "speech_rate": audio_feat["speech_rate"],
            "fatigue_word_count": audio_feat["fatigue_word_count"],
            "text_length": audio_feat["text_length"],
            "lexical_diversity": audio_feat["lexical_diversity"],
            **audio_feat["smile_features"],
            **similarities,
        }

        if not self._models_ready() or self.pca is None or self.scaler is None or self.feature_cols is None:
            fallback_scores = {}
            for cat in ["body", "brain", "mental"]:
                sim_fatigue = similarities.get(f"sim_{cat}", 0.0)
                sim_healthy = similarities.get("sim_healthy", 0.0)
                raw = (sim_fatigue - sim_healthy + 1.0) / 2.0
                fallback_scores[cat] = round(float(np.clip(raw * 9.0, 0.0, 9.0)), 1)

            return {
                "success": True,
                "scores": fallback_scores,
                "confidence": f"{get_confidence(fallback_scores)}（モデル未学習・類似度ベース）",
                "message": "モデル未学習。類似度ベースの簡易推定。",
                "features": features,
                "audio_quality": f"{audio_feat['duration']:.1f}秒",
                "audio_feat": audio_feat,
                "query_emb": query_emb,
                "similarities": similarities,
            }

        X_base = np.array([[features.get(col, 0.0) for col in self.feature_cols]], dtype=np.float32)
        if len(self.feature_cols) != X_base.shape[1]:
            raise ValueError("特徴量数が一致しません")

        X_base_scaled = self.scaler.transform(X_base)
        emb_pca = self.pca.transform(query_emb.reshape(1, -1))
        X = np.hstack([X_base_scaled, emb_pca])

        scores = {}
        for category in ["body", "brain", "mental"]:
            if self.models[category] is not None:
                raw_score = float(self.models[category].predict(X)[0])
                scores[category] = round(float(np.clip(raw_score, 0.0, 9.0)), 1)
            else:
                scores[category] = 0.0

        return {
            "success": True,
            "scores": scores,
            "confidence": get_confidence(scores),
            "message": "OK",
            "features": features,
            "audio_quality": f"{audio_feat['duration']:.1f}秒",
            "audio_feat": audio_feat,
            "query_emb": query_emb,
            "similarities": similarities,
        }


def save_to_hf_dataset(df_row: Dict):
    if not HF_TOKEN or not REPO_ID:
        logger.warning("HF_TOKEN または REPO_ID が設定されていません")
        return

    try:
        api = HfApi()
        df = pd.DataFrame([df_row])
        if os.path.exists(DATA_CSV):
            existing = pd.read_csv(DATA_CSV)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(DATA_CSV, index=False)

        api.upload_file(
            path_or_fileobj=DATA_CSV,
            path_in_repo="dataset.csv",
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        logger.info(f"Uploaded to HF Dataset: {REPO_ID}")
    except Exception as e:
        logger.error(f"Failed to upload to HF Dataset: {e}")


engine = InferenceEngine()


def save_data_with_result(audio_feat, query_emb, body_score, brain_score, mental_score, similarities):
    if audio_feat is None:
        return "音声ファイルが必要です"

    sim_dict = similarities

    df_row = {
        "text": audio_feat["text"],
        "text_embedding": json.dumps(query_emb.tolist()),
        "speech_rate": audio_feat["speech_rate"],
        "fatigue_word_count": audio_feat["fatigue_word_count"],
        "text_length": audio_feat["text_length"],
        "lexical_diversity": audio_feat["lexical_diversity"],
        **audio_feat["smile_features"],
        **sim_dict,
        "label_body": float(body_score),
        "label_brain": float(brainscore),
        "label_mental": float(mental_score),
        "timestamp": datetime.now().isoformat(),
    }

    df = pd.DataFrame([df_row])
    df["text_embedding"] = df["text_embedding"].apply(
        lambda x: np.array(json.loads(x), dtype=np.float32)
    )
    if os.path.exists(DATA_PARQUET):
        existing = pd.read_parquet(DATA_PARQUET)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(DATA_PARQUET, index=False)

    df_row["text_embedding"] = json.dumps(query_emb.tolist())
    save_to_hf_dataset(df_row)

    return f"データ保存完了（計 {len(df)} 件）"


def run_training():
    if not os.path.exists(DATA_PARQUET):
        return "学習データがありません"

    df = pd.read_parquet(DATA_PARQUET)
    df["text_embedding"] = df["text_embedding"].apply(
        lambda x: np.array(x, dtype=np.float32) if isinstance(x, (list, np.ndarray)) else np.array(json.loads(x), dtype=np.float32)
    )

    if len(df) < 30:
        return f"データ不足：{len(df)} 件（学習はスキップしました）"

    try:
        metrics = train_model(df)
        engine._load_models()

        result_lines = [
            f"学習完了\nデータ数：{metrics['data_count']}\n特徴量数：{metrics['feature_count']}"
        ]
        for cat, res in metrics["results"].items():
            result_lines.append(f"{cat}: 検証 RMSE={res['val_rmse']:.2f}")
        return "\n".join(result_lines)
    except Exception as e:
        return f"学習はスキップ/失敗しました：{e}"


def score_from_choice(choice: str) -> float:
    return {"弱い": 1.0, "ふつう": 5.0, "強い": 9.0}.get(choice, 5.0)


def generate_comment(scores: Dict[str, float]) -> str:
    if not scores:
        return ""
    max_cat = max(scores, key=scores.get)
    max_score = scores[max_cat]
    cat_names = {"body": "身体", "brain": "脳", "mental": "精神"}
    comments = {
        "body": {
            "high": "身体疲労が高めです。睡眠や軽いストレッチを取り入れてみましょう。",
            "mid": "身体疲労が見られます。適度な休息を。",
            "low": "身体的には良好です。",
        },
        "brain": {
            "high": "脳疲労が高めです。少し画面から離れて休憩すると効果的です。",
            "mid": "脳疲労が見られます。短い休憩を挟みましょう。",
            "low": "認知機能は良好です。",
        },
        "mental": {
            "high": "精神疲労が高めです。無理をせずリラックスする時間を作りましょう。",
            "mid": "精神疲労が見られます。好きなことをして気分転換を。",
            "low": "精神的には安定しています。",
        },
    }
    if max_score >= 7:
        return comments.get(max_cat, {}).get("high", "")
    elif max_score >= 4:
        return comments.get(max_cat, {}).get("mid", "")
    else:
        return comments.get(max_cat, {}).get("low", "")


def full_flow(audio_path, answers):
    result = engine.predict(audio_path)
    if not result["success"]:
        return f"推定失敗：{result['message']}", ""

    scores = result["scores"]
    similarities = result.get("similarities", {})

    score_text = (
        f"身体疲労：{scores.get('body', 0) * 100 / 9:.0f}%\n"
        f"脳疲労：{scores.get('brain', 0) * 100 / 9:.0f}%\n"
        f"精神疲労：{scores.get('mental', 0) * 100 / 9:.0f}%\n"
        f"信頼度：{result['confidence']}\n"
        f"{result['message']}"
    )

    comment = generate_comment(scores)

    audio_feat = result["audio_feat"]
    query_emb = result["query_emb"]

    body_score = float(np.mean([score_from_choice(x) for x in answers[0:3]]))
    brain_score = float(np.mean([score_from_choice(x) for x in answers[3:6]]))
    mental_score = float(np.mean([score_from_choice(x) for x in answers[6:9]]))

    save_msg = save_data_with_result(audio_feat, query_emb, body_score, brain_score, mental_score, similarities)

    return f"{score_text}\n\n{comment}\n\n{save_msg}", ""


def main():
    st.set_page_config(page_title="Restee", layout="wide")
    st.title("Restee - 疲労状態推定システム")

    st.markdown("### 音声ファイル")
    audio_file = st.file_uploader("音声をアップロードしてください", type=["wav", "mp3", "m4a"])

    st.markdown("### アンケート（各 3 問）")
    question_inputs = []
    for cat in QUESTION_ORDER:
        items = QUESTION_BANK[cat].copy()
        random.shuffle(items)
        for i, q in enumerate(items[:3], 1):
            q_key = f"{cat}_q{i}"
            val = st.radio(q, ["弱い", "ふつう", "強い"], key=q_key, horizontal=True)
            question_inputs.append(val)

    if st.button("解析して保存"):
        if audio_file is None:
            st.error("音声ファイルがアップロードされていません")
        else:
            temp_path = os.path.join(DATA_DIR, "temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())

            output, status = full_flow(temp_path, question_inputs)
            st.text_area("結果", value=output, height=200)
            if status:
                st.text_area("ステータス", value=status, height=100)

    if st.button("学習（管理者用）"):
        result = run_training()
        st.text_area("学習結果", value=result, height=200)


if __name__ == "__main__":
    main()
