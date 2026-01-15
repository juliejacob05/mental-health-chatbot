import pandas as pd

#------------------
#emotion datasets
#-----------------

e1 = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\emotion1.csv')
e2 = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\emotion2.csv')
e3 = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\emotion3.csv')
e4 = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\emotion4.csv')

e2 = e2.drop(columns='tweet_id')
e3 = e3[1:]

emotion_mapping = {
    # Core negative emotions
    'sadness': 'sadness',
    'sad': 'sadness',
    'empty': 'sadness',
    'boredom': 'boredom',
    'anger': 'anger',
    'hate': 'anger',

    # Anxiety-related
    'worry': 'fear',
    'fear': 'fear',
    'stress': 'fear',

    # Positive emotions
    'enthusiasm': 'positive',
    'happiness': 'positive',
    'joy': 'positive',
    'love': 'positive',
    'fun': 'positive',
    'relief': 'positive',

    # Neutral / ambiguous
    'neutral': 'neutral',
    'surprise': 'neutral'
}
e_data = pd.concat([e1, e2, e3, e4], ignore_index=True)
e_data['final_text'] = (
    e_data['text']
    .combine_first(e_data['content'])
    .combine_first(e_data['sentence'])
)
e_data['raw_emotion'] = (
    e_data['emotion']
    .combine_first(e_data['Emotion'])
    .combine_first(e_data['sentiment'])
)
e_data['mapped_emotion'] = e_data['raw_emotion'].map(emotion_mapping)

e_data = e_data[['final_text', 'mapped_emotion']]

# Remove rows with missing text or labels
e_data = e_data.dropna(subset=['final_text', 'mapped_emotion'])

# Remove empty or whitespace-only text
e_data = e_data[e_data['final_text'].str.strip() != ""]

# ----- STUDENT_STRESS DATASETS ----
stud_data = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\student_stress.csv')
stud_data['mapped_emotion'] = stud_data['emotion'].map(emotion_mapping)
stud_data = stud_data[['text', 'mapped_emotion']]
stud_data = stud_data.rename(columns={'text': 'final_text'})

# Combine with main emotion data
e_data = pd.concat([e_data, stud_data], ignore_index=True)

# Remove any possible empty or null rows
e_data = e_data.dropna(subset=['final_text', 'mapped_emotion'])
e_data = e_data[e_data['final_text'].str.strip() != ""]

#training emotion datasets
e_data.to_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\processed_emotion_data.csv', index=False)

print("Emotion distribution:")
print(e_data['mapped_emotion'].value_counts())




#------------------
#sentiment datasets
#-----------------

s1 = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\senti1.csv')
s2 = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\senti2.csv')
s3 = pd.read_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\senti3.csv')

s1 = s1.iloc[1:]
s1 = s1.drop(columns=['Unnamed: 0', 'textID', 'selected_text', 'Time of Tweet', 'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'])
s2 = s2.iloc[1:]
s2 = s2.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Timestamp', 'User', 'Platform', 'Hashtags', 'Retweets', 'Likes', 'Country', 'Year', 'Month', 'Day', 'Hour'])

sentiment_mapping = {
    'positive': 1,
    'negative': -1,
    'neutral': 0
}

s_data = pd.concat([s1, s2, s3], ignore_index=True)
s_data['final_text'] = (
    s_data['text']
    .combine_first(s_data['Text'])
    .combine_first(s_data['clean_comment'])
)
s_data['raw_sentiment'] = (
    s_data['sentiment']
    .combine_first(s_data['Sentiment'])
    .combine_first(s_data['category'])
)
s_data['mapped_sentiment'] = s_data['raw_sentiment'].map(sentiment_mapping)

s_data = s_data[['final_text', 'mapped_sentiment']]

# Remove rows with missing text or labels
s_data = s_data.dropna(subset=['final_text', 'mapped_sentiment'])

# Remove empty or whitespace-only text
s_data = s_data[s_data['final_text'].str.strip() != ""]

print("\nSentiment distribution:")
print(s_data['mapped_sentiment'].value_counts())

#training sentiment datasets
s_data.to_csv(r'D:\Providence\extras\internship\25.12 asap edunet\programs\datasets\processed_sentiment_data.csv', index=False)