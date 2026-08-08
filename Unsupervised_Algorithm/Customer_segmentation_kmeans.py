#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

#Load Dataset
cust_segment = pd.read_csv("Mall_Customers.csv")
print(cust_segment.head())

#Preprocessing
print(cust_segment.describe())
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(cust_segment.info())

#checking null and duplicated values
print(cust_segment.isnull().sum())
print(cust_segment.shape)
print(cust_segment.duplicated().sum())

#Encoding
le = LabelEncoder()
cust_segment["Gender"]=le.fit_transform(cust_segment["Gender"])
print(cust_segment)

#Visualisation
sns.pairplot(cust_segment)
plt.show()

plt.scatter(x="Annual Income (k$)",y="Spending Score (1-100)",data=cust_segment)
plt.show()

#Select Features
X = cust_segment[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss,marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

cust_segment['Cluster'] = clusters
print(cust_segment['Cluster'].unique())

score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

cust_segment['TSNE1'] = X_tsne[:,0]
cust_segment['TSNE2'] = X_tsne[:,1]
# print(cust_segment['TSNE1'])
# print(cust_segment['TSNE2'])

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='TSNE1',
    y='TSNE2',
    hue='Cluster',
    palette='Set1',
    data=cust_segment
)
plt.title("Customer Segmentation using KMeans + t-SNE")
plt.show()