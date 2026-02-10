#Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

#DATA COLLECTION
#Reading the csv file
df=pd.read_csv("spotify_tracks.csv")
print(df.shape)
print(df.columns)

#Retrieve or filter the Tamil music song
final_df = (df[df['language'] == 'Tamil'].copy())
final_df.reset_index(drop=True,inplace=True)
print(final_df.shape)
print(final_df.head(5))
print(final_df['language'].value_counts())

#PREPROCESSING
#Checking Duplicates and null values
print(final_df.isnull().sum())
print("Duplicated values : " ,final_df.duplicated().sum())
final_df.drop_duplicates(inplace=True)
final_df.reset_index(drop=True, inplace=True)
final_df = final_df.drop_duplicates(subset='track_name').reset_index(drop=True)
print("After cleaning Duplicated values :",final_df.duplicated().sum())
final_df.drop_duplicates(subset='track_name', inplace=True)
final_df.reset_index(drop=True, inplace=True)

#EDA-VISUALISATION
#Display Correlation matrix
plt.figure(figsize=(14, 10))
corr_matrix = final_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

#Display Top 10 Most Prolific Artists
#print(final_df['artist_name'].value_counts().head(10))
top_artists = final_df['artist_name'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_artists.values, y=top_artists.index,hue=top_artists.index, palette='viridis',legend=False)
plt.title('Top 10 Most Prolific Artists')
plt.xlabel('Number of Tracks')
plt.ylabel('Artist')
plt.show()

#Display Number of auditions by year
#print(final_df.groupby('year')['popularity'].sum())
yearly_popularity =final_df.groupby('year')['popularity'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(yearly_popularity['year'], yearly_popularity['popularity'], color='skyblue')
plt.title('Number of auditions by year')
plt.xlabel('Year')
plt.ylabel('Number of auditions')
plt.xticks(yearly_popularity['year'], rotation = 90)
plt.grid(axis='y')
plt.show()

# Plot the distribution of track popularity
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True, color='purple')
plt.title('Distribution of Track Popularity')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

#Cleaning the track name
final_df.loc[:,'track_name'] = final_df['track_name'].str.lower().str.strip()
print(final_df['track_name'])

#Select Features
features = ['danceability', 'energy', 'loudness','speechiness', 'acousticness','valence', 'tempo']
X = final_df[features]

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Cosine Similarity
similarity = cosine_similarity(X_scaled)

#Recommendation Function
def recommend_songs(song_name, n=5):
    song_name = song_name.lower().strip()
    if song_name not in final_df['track_name'].values:
        print("Song not found")
        return []

    idx = final_df[final_df['track_name'] == song_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(f"\nSongs similar to: {song_name.upper()}\n")
    songs = [final_df.iloc[i[0]]['track_name'] for i in scores[1:n+1]]
    return songs

#calling or testing
result = recommend_songs('life of bachelor',5)

for i, song in enumerate(result, start=1):
    print(f"{i}. {song}")

