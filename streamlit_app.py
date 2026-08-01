import streamlit as st
import requests
import random
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Restee - 疲労度分析", layout="centered")

# --- 自覚症状質問プール（カテゴリをユーザーに伏せてランダム出題） ---
# カテゴリは「脳疲労」「精神疲労」「身体疲労」の3つ
# 各カテゴリから1問ずつランダムに抽出し、ユーザーに順番を悟られないようシャッフルして出題する
# 質問は0~3のスライダーで回答する（0:なし、1:やや感じる、2:感じる、3:強く感じる）
# 問題数を増やす場合は、QUESTION_POOLに追加するだけでOK
QUESTION_POOL = [
    # 脳疲労項目
    {"id": "b1", "cat": "brain", "text": "最近、考えがまとまりにくく感じることはありますか？"},
    {"id": "b2", "cat": "brain", "text": "文字や文章を読むのが億劫に感じますか？"},
    {"id": "b3", "cat": "brain", "text": "ケアレスミスが増えたように感じますか？"},
    # 精神疲労項目
    {"id": "m1", "cat": "mental", "text": "ふとした瞬間にため息が出ることが増えましたか？"},
    {"id": "m2", "cat": "mental", "text": "少しのことで気持ちがモヤモヤしたり焦ったりしますか？"},
    {"id": "m3", "cat": "mental", "text": "趣味や好きなことに対するワクワク感が薄れていますか？"},
    # 身体疲労項目
    {"id": "p1", "cat": "body", "text": "朝起きた時に、体に重だるさを感じますか？"},
    {"id": "p2", "cat": "body", "text": "首や肩、目の奥などにコリや違和感はありますか？"},
    {"id": "p3", "cat": "body", "text": "階段の上り下りなどで足腰の疲労を感じやすいですか？"}
]

st.title("🌿 Restee 疲労分析チャット")

# Colabで発行されたlocaltunnel URLの入力欄
api_url = st.sidebar.text_input("Colab API URL", placeholder="https://ten-planes-double.loca.lt/")

# 質問のランダム抽出し保持（セッション状態）
if "selected_questions" not in st.session_state:
    # 各カテゴリから1問ずつランダム抽出し、さらに全体の順番をシャッフル
    brain_q = random.choice([q for q in QUESTION_POOL if q["cat"] == "brain"])
    mental_q = random.choice([q for q in QUESTION_POOL if q["cat"] == "mental"])
    body_q = random.choice([q for q in QUESTION_POOL if q["cat"] == "body"])
    
    questions = [brain_q, mental_q, body_q]
    random.shuffle(questions) # ユーザーにカテゴリ順を悟らせない
    st.session_state.selected_questions = questions

st.subheader("1. 以下の質問に直感でお答えください (0:なし ~ 3:強く感じる)")

answers = {}
for i, q in enumerate(st.session_state.selected_questions):
    answers[q["cat"]] = st.slider(f"Q{i+1}. {q['text']}", 0, 3, 1, key=q["id"])

st.subheader("2. 今の気分や出来事を声で教えてください")
audio_bytes = audio_recorder(text="タップして録音開始 / 停止", icon_name="microphone")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    if st.button("疲労度を解析する"):
        if not api_url:
            st.error("サイドバーに Colab の API URL を入力してください！")
        else:
            with st.spinner("AIが音声と感情・疲労度を解析中..."):
                try:
                    files = {"file": ("voice.wav", audio_bytes, "audio/wav")}
                    data = {
                        "ans_brain": answers["brain"],
                        "ans_mental": answers["mental"],
                        "ans_body": answers["body"]
                    }
                    
                    # localtunnel対策ヘッダーを追加
                    headers = {"Bypass-Tunnel-Reminder": "true"}
                    
                    response = requests.post(f"{api_url.rstrip('/')}/predict", files=files, data=data, headers=headers)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.success("解析が完了しました！")
                        
                        st.markdown(f"> **文字起こし結果**: 「{res['transcription']}」")
                        
                        st.markdown("---")
                        st.header("📊 疲労度アナリティクス")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("総合疲労度", f"{res['scores']['total']}%")
                        col2.metric("脳疲労", f"{res['scores']['brain']}%")
                        col3.metric("精神疲労", f"{res['scores']['mental']}%")
                        col4.metric("身体疲労", f"{res['scores']['body']}%")
                        
                    else:
                        st.error(f"APIエラーが発生しました: {response.status_code}")
                except Exception as e:
                    st.error(f"通信エラー: {e}")

