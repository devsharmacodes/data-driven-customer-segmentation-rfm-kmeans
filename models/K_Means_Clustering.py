from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# 1. Load Data

df = pd.read_csv('rfm_dataset.csv')

print("Dataset Shape:", df.shape)
print(df.head())


# 2. Select Features

X = df[['Recency', 'Frequency', 'Monetary']]


# 3. Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. Choose Optimal K (I used the elbow method for selecting K)

k = 4  


# 5. Apply KMeans

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())


# 6. PCA for Visualization

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Transform centroids also
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Plot clusters
plt.figure(figsize=(8,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'])
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, marker='X')

plt.title('Customer Segments (PCA Visualization)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# 7. Cluster Interpretation

cluster_summary = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\nCluster Summary:")
print(cluster_summary)


# 8. Label Clusters (Optional but powerful)

def label_customer(row):
    if row['Cluster'] == 0:
        return 'High Value'
    elif row['Cluster'] == 1:
        return 'Regular'
    elif row['Cluster'] == 2:
        return 'At Risk'
    else:
        return 'Low Value'

df['Customer_Segment'] = df.apply(label_customer, axis=1)

print("\nFinal Data:")
print(df.head())


# 9. Save Output

df.to_csv('rfm_with_clusters.csv', index=False)

# 10. Turn this file into .pkl
with open("kmeans_model.pkl", "wb") as file:
    pickle.dump(kmeans, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Model and scaler saved successfully")