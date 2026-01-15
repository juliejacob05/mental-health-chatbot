import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

#------------------
#emotion datasets
#-----------------

e_data = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\processed_emotion_data.csv')
# Prepare emotion data
e_x = e_data['final_text']
e_y = e_data['mapped_emotion']

e_x_tr, e_x_te, e_y_tr, e_y_te = train_test_split(e_x, e_y, test_size=0.2, random_state=42)

e_vec = TfidfVectorizer(max_features=5000)
e_x_tr_vec = e_vec.fit_transform(e_x_tr)
e_x_te_vec = e_vec.transform(e_x_te)

emotion_model = LogisticRegression(max_iter=1000)
emotion_model.fit(e_x_tr_vec, e_y_tr)

print(accuracy_score(e_y_te, emotion_model.predict(e_x_te_vec)))

# Save emotion model and vectorizer
joblib.dump(emotion_model, 'emotion_model.pkl')
joblib.dump(e_vec, 'emotion_vectorizer.pkl')




#-------------------
#sentiment datasets
#-------------------

s_data = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\processed_sentiment_data.csv')
# Prepare sentiment data
s_x = s_data['final_text']
s_y = s_data['mapped_sentiment']

s_x_tr, s_x_te, s_y_tr, s_y_te = train_test_split(s_x, s_y, test_size=0.2, random_state=42)

s_vec = TfidfVectorizer(max_features=3000)
X_tr_s_vec = s_vec.fit_transform(s_x_tr)
X_te_s_vec = s_vec.transform(s_x_te)
sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_tr_s_vec, s_y_tr)

print(accuracy_score(s_y_te, sentiment_model.predict(X_te_s_vec)))

# Save sentiment model and vectorizer
joblib.dump(sentiment_model, 'sentiment_model.pkl')
joblib.dump(s_vec, 'sentiment_vectorizer.pkl')