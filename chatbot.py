import streamlit as st
import os
import joblib
from datetime import datetime
from collections import Counter
import random
import csv
import pandas as pd

# ---------- BASE DIRECTORY ----------
base_dir = os.path.dirname(os.path.abspath(__file__))

# ---------- LOAD MODELS ----------
emotion_model = joblib.load(os.path.join(base_dir, 'emotion_model.pkl'))
emotion_vec = joblib.load(os.path.join(base_dir, 'emotion_vectorizer.pkl'))

sentiment_model = joblib.load(os.path.join(base_dir, 'sentiment_model.pkl'))
sentiment_vec = joblib.load(os.path.join(base_dir, 'sentiment_vectorizer.pkl'))

# ---------- STORAGE ----------
if 'daily_logs' not in st.session_state:
    st.session_state.daily_logs = []

# ---------- SIDEBAR ----------
st.sidebar.title("Mental Health Companion")
st.sidebar.info(
    "⚠️ **Disclaimer**\n\n"
    "This chatbot is **not a medical professional**.\n"
    "It provides emotional support only.\n\n"
    "If you feel unsafe or overwhelmed, please reach out to a "
    "mental health professional or a trusted person."
)

# ---------- RESPONSE MAP ----------
response_map = {
    'sadness': [
        "I'm sorry you're feeling this way.",
        "That sounds really difficult.",
        "It's okay to feel sad sometimes.",
        "You're not alone in feeling this.",
        "Thanks for sharing how you feel."
    ],
    'anger': [
        "It sounds like something upset you.",
        "That seems frustrating.",
        "Strong feelings can be hard to handle.",
        "It's okay to pause and breathe.",
        "Thanks for expressing this."
    ],
    'fear': [
        "That sounds worrying.",
        "Feeling anxious can be overwhelming.",
        "It's okay to feel unsure sometimes.",
        "You're not alone in this feeling.",
        "Thank you for sharing this."
    ],
    'boredom': [
        "Feeling empty can be hard.",
        "That sounds dull or draining.",
        "It's okay to feel disengaged sometimes.",
        "Thanks for being honest about this.",
        "You're not alone in feeling this way."
    ],
    'positive': [
        "It's good to hear that.",
        "That sounds positive.",
        "I'm glad you're feeling this way.",
        "Thanks for sharing something good.",
        "It's nice to hear this today."
    ],
    'neutral': [
        "Thank you for sharing.",
        "I'm here to listen.",
        "Thanks for telling me how you feel.",
        "I appreciate you sharing this.",
        "Feel free to continue."
    ]
}

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Daily Summary", "🗂 History"])

# ================= CHAT TAB =================
with tab1:
    st.header("Talk to the chatbot")

    user_input = st.text_area("Type your thoughts here:")

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Vectorize input
            emo_vec = emotion_vec.transform([user_input])
            sent_vec = sentiment_vec.transform([user_input])

            # Predict
            emotion = emotion_model.predict(emo_vec)[0]
            sentiment = sentiment_model.predict(sent_vec)[0]

            # Generate empathetic response
            response = random.choice(
                response_map.get(emotion, ["Thank you for sharing."])
            )

            # Display output
            st.success(response)
            st.write(f"**Emotion detected:** {emotion}")
            st.write(f"**Sentiment tone:** {sentiment}")

            # Save to session (today's logs)
            st.session_state.daily_logs.append({
                'time': datetime.now(),
                'text': user_input,
                'emotion': emotion,
                'sentiment': sentiment
            })

# ================= SUMMARY TAB =================
with tab2:
    st.header("Today's Emotional Summary")

    summary_path = os.path.join(base_dir, "daily_summary.csv")

    if not st.session_state.daily_logs:
        st.info("No messages yet today.")
    else:
        emotions = [x['emotion'] for x in st.session_state.daily_logs]
        sentiments = [x['sentiment'] for x in st.session_state.daily_logs]

        dominant_emotion = Counter(emotions).most_common(1)[0][0]
        avg_sentiment = sum(sentiments) / len(sentiments)

        st.write(f"**Dominant Emotion:** {dominant_emotion}")
        st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
        st.write(f"**Messages today:** {len(st.session_state.daily_logs)}")

        if st.button("Save Today's Summary"):
            file_exists = os.path.exists(summary_path)

            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "Date_Time",
                        "Dominant_Emotion",
                        "Avg_Sentiment",
                        "Messages_Today"
                    ])

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    dominant_emotion,
                    round(avg_sentiment, 2),
                    len(st.session_state.daily_logs)
                ])

            st.success("Summary saved successfully!")

# ================= HISTORY TAB =================
with tab3:
    st.header("Previous Daily Summaries")

    summary_path = os.path.join(base_dir, "daily_summary.csv")

    if os.path.exists(summary_path):
        history = pd.read_csv(summary_path)
        st.dataframe(history)

        st.download_button(
            label="Download Summary Report",
            data=history.to_csv(index=False),
            file_name="mental_health_summary.csv",
            mime="text/csv"
        )
    else:
        st.info("No past summaries available yet.")