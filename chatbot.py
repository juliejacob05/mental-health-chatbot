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

# ---------- FILE PATH ----------
summary_path = os.path.join(base_dir, "daily_summary.csv")

# ---------- SESSION STORAGE ----------
if 'daily_logs' not in st.session_state:
    st.session_state.daily_logs = []

# ---------- CRISIS KEYWORDS ----------
CRISIS_KEYWORDS = [
    "suicide", "self harm", "end my life", 
    "kill myself", "want to die", "no reason to live"
]

# ---------- ACTIVITY SUGGESTIONS ----------
activity_suggestions = [
    "Try 5 slow deep breaths.",
    "Take a 10-minute walk.",
    "Write down what you're feeling.",
    "Drink a glass of water.",
    "Listen to calming music."
]

# ---------- EMERGENCY MESSAGES ----------
emergency_messages = [
    "If you're in danger, please contact a trusted person immediately.",
    "You can reach a local mental health helpline in your country.",
    "Consider speaking to a counselor or therapist."
]

# ---------- RESPONSE MAP ----------
response_map = {
    'sadness': ["I'm sorry you're feeling this way.","You're not alone.","Thanks for sharing."],
    'anger': ["That seems frustrating.","It's okay to pause and breathe.","Thanks for expressing this."],
    'fear': ["Feeling anxious can be overwhelming.","You're not alone.","Thank you for sharing."],
    'boredom': ["Feeling empty can be hard.","Thanks for being honest.","You're not alone."],
    'positive': ["It's good to hear that.","Thanks for sharing something good."],
    'neutral': ["Thank you for sharing.","I'm here to listen."]
}

# ---------- SIDEBAR ----------
st.sidebar.title("Mental Health Companion")
st.sidebar.info(
    "⚠️ **Disclaimer**\n"
    "This chatbot is **not a medical professional**.\n"
    "It provides emotional support only.\n"
    "If you feel unsafe, reach out to a mental health professional."
)

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
            # ---- CRISIS CHECK ----
            crisis_detected = any(word in user_input.lower() for word in CRISIS_KEYWORDS)
            if crisis_detected:
                st.error("I’m really sorry you’re feeling this way.")
                for msg in emergency_messages:
                    st.write("⚠️ " + msg)
                # Override emotion & sentiment for crisis
                emotion = 'fear'
                sentiment = -1.0
            else:
                # ---- PREDICTION ----
                emo_vec_input = emotion_vec.transform([user_input])
                sent_vec_input = sentiment_vec.transform([user_input])
                emotion = emotion_model.predict(emo_vec_input)[0]
                sentiment = sentiment_model.predict(sent_vec_input)[0]

            # ---- RESPONSE ----
            response = random.choice(response_map.get(emotion, ["Thank you for sharing."]))
            st.success(response)
            st.write(f"**Emotion detected:** {emotion}")
            st.write(f"**Sentiment tone:** {sentiment}")

            # ---- ACTIVITY SUGGESTION IF NEGATIVE ----
            if sentiment < 0:
                st.info("Here are some things that might help:")
                for act in random.sample(activity_suggestions, 2):
                    st.write("• " + act)

            # ---- SAVE TO SESSION LOGS ----
            st.session_state.daily_logs.append({
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': user_input,
                'emotion': emotion,
                'sentiment': sentiment,
                'crisis': crisis_detected
            })

# ================= SUMMARY TAB =================
with tab2:
    st.header("Today's Emotional Summary")
    if not st.session_state.daily_logs:
        st.info("No messages yet today.")
    else:
        emotions = [x['emotion'] for x in st.session_state.daily_logs]
        sentiments = [x['sentiment'] for x in st.session_state.daily_logs]
        crisis_flags = [x['crisis'] for x in st.session_state.daily_logs]

        dominant_emotion = Counter(emotions).most_common(1)[0][0]
        avg_sentiment = sum(sentiments) / len(sentiments)

        st.write(f"**Dominant Emotion:** {dominant_emotion}")
        st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
        st.write(f"**Messages today:** {len(st.session_state.daily_logs)}")

        # Highlight if crisis detected today
        if any(crisis_flags) or dominant_emotion == 'fear':
            st.warning("⚠️ Crisis detected today. Please consider seeking help immediately.")

        if st.button("Save Today's Summary"):
            file_exists = os.path.exists(summary_path)
            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date_Time", "Dominant_Emotion", "Avg_Sentiment", "Messages_Today"])
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
    if os.path.exists(summary_path):
        history = pd.read_csv(summary_path)
        
        # Ensure 'Crisis' column exists for highlighting
        if 'Crisis' not in history.columns:
            history['Crisis'] = 0  # default no crisis
        
        # Highlight crisis rows
        def highlight_crisis(row):
            if row['Crisis'] == 1 or row['Dominant_Emotion'] == 'fear' or row['Avg_Sentiment'] < 0:
                return ['color: red']*len(row)
            else:
                return ['']*len(row)

        st.dataframe(history.style.apply(highlight_crisis, axis=1))

        st.download_button(
            label="Download Summary Report",
            data=history.to_csv(index=False),
            file_name="mental_health_summary.csv",
            mime="text/csv"
        )
    else:
        st.info("No past summaries available yet.")