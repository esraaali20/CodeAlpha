from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
import re

# Initialize the stemmer
stemmer = PorterStemmer()

# Load dataset
df = pd.read_csv('spotify_millsongdata.csv')

# Sample and preprocess the dataset
df = df.sample(3000).drop('link', axis=1).reset_index(drop=True)

print(df.head())


# Normalize the text
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex=True)

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df['text'] = df['text'].apply(lambda x: tokenization(x))

# Normalize the song titles
df['normalized_song'] = df['song'].str.lower().str.strip()

# Vectorize the text
tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)

def recommendation(song_title):
    # Normalize the input song title
    normalized_title = song_title.lower().strip()
    
    # Check if the song title exists in the dataset
    if normalized_title not in df['normalized_song'].values:
        return f"Error: The song title '{song_title}' does not exist in the dataset."
    
    idx = df[df['normalized_song'] == normalized_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].song)
        
    return songs

# Example usage
print(recommendation('Come Back To Me'))

# Save the similarity matrix and DataFrame
import pickle
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(df, open('df.pkl', 'wb'))
