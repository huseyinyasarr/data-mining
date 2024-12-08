import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('datasets/Mall_Customers.csv')


# Checking for missing values
print("Missing Values:\n", df.isnull().sum())

# Encoding categorical data
df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})

# Selecting and scaling only numerical columns
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Determining the Optimal K Value using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=61)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Choosing K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Choosing K with the Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=61)
    kmeans.fit(scaled_features)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(scaled_features, labels))

# Plot the Silhouette Score Graph
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Optimal K Value Based on Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Scoore')
plt.show()

# Select the optimal k value
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=61)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Extracting cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
print("\nCluster Centers:\n", cluster_centers_df)

# Visualization of Results
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    palette='viridis', 
    s=100
)
plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# Additionally, a scatter plot based on age
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='Age', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    palette='viridis', 
    s=100
)
plt.title('Customer Clusters by Age')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
