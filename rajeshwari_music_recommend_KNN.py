import pandas as pd

df = pd.read_csv("rajeshwari_music_recommendation.csv")

df = df[df['language'] == 'Tamil'].reset_index(drop=True)

#PREPROCESSING
#Checking Duplicates and null values
print(df.isnull().sum())
print("Duplicated values : " ,df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.drop_duplicates(subset='track_name').reset_index(drop=True)
print("After cleaning Duplicated values :",df.duplicated().sum())
df.drop_duplicates(subset='track_name', inplace=True)
df.reset_index(drop=True, inplace=True)


df['track_name'] = df['track_name'].str.lower().str.strip()
print(df['track_name'].head(5))

features = ['danceability', 'energy', 'tempo', 'loudness', 'valence']
X = df[features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(
    n_neighbors=6,
    metric='cosine'
)

knn.fit(X_scaled)

def recommend_songs(song_name):
    song_name = song_name.lower().strip()

    if song_name not in df['track_name'].str.lower().values:
        print("Song not found")
        return

    idx = df[df['track_name'].str.lower() == song_name].index[0]

    distances, indices = knn.kneighbors([X_scaled[idx]])

    print("Recommended Songs:")
    for i in indices[0][1:]:
        print(df.loc[i, 'track_name'])

#K-Means
# from sklearn.cluster import KMeans
#
# kmeans = KMeans(n_clusters=10, random_state=42)
# df['cluster'] = kmeans.fit_predict(X_scaled)
#
#
# def recommend_songs(song_name, df):
#     song_name = song_name.lower().strip()
#
#     if song_name not in df['track_name'].values:
#         print(f"❌ Song '{song_name}' dataset-la illa")
#         return
#
#     cluster = df[df['track_name'] == song_name]['cluster'].values[0]
#     print("Song Cluster:", cluster)
#
# from sklearn.metrics import silhouette_score
# score = silhouette_score(X_scaled, df['cluster'])
# print(score)
#
recommend_songs("life of bachelor")

