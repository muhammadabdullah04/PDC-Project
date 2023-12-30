import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import pymp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

# Load the dataset
# Replace 'your_dataset.csv' with the actual file path or data loading method
df = pd.read_csv('ds_salaries.csv')

# Select relevant features for clustering
features = df[['work_year', 'experience_level', 'salary_in_usd', 'remote_ratio']]

# Assuming 'remote_ratio' is already converted to a numeric type
# If it's not, you can still use the code to convert it (as you did in your original code)
# features['remote_ratio'] = features['remote_ratio'].str.rstrip('%').astype('float') / 100.0

# Encode categorical variables if needed (e.g., 'experience_level' and 'work_year')
label_encoder = LabelEncoder()
features.loc[:, 'experience_level'] = label_encoder.fit_transform(features['experience_level'].values)
features.loc[:, 'work_year'] = label_encoder.fit_transform(features['work_year'].values)


# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters (K) using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (K)
optimal_k = 3  # You can adjust this based on the visual inspection of the Elbow Method graph

# Apply K-Means clustering with the optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clusters
sns.pairplot(df, hue='cluster', palette='Dark2')
plt.show()

# Print the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:")
print(pd.DataFrame(cluster_centers, columns=features.columns))

# Add the 'cluster' column to the original dataset
df.to_csv('clustered_dataset.csv', index=False)

# Load the dataset
# Replace 'your_dataset.csv' with the actual file path or data loading method
df = pd.read_csv('ds_salaries.csv')

# Select relevant features for clustering
features = df[['work_year', 'experience_level', 'salary_in_usd', 'remote_ratio']]

# Encode categorical variables if needed (e.g., 'experience_level' and 'work_year')
label_encoder = LabelEncoder()
features['experience_level'] = label_encoder.fit_transform(features['experience_level'])
features['work_year'] = label_encoder.fit_transform(features['work_year'])

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters (K) using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares

def fit_kmeans(num_clusters, data):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    return kmeans.inertia_

# Use joblib for parallelization
num_threads = 4  # Adjust based on your system capabilities
wcss = Parallel(n_jobs=num_threads)(delayed(fit_kmeans)(i, scaled_features) for i in range(1, 11))

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (K)
optimal_k = 3  # You can adjust this based on the visual inspection of the Elbow Method graph

# Apply K-Means clustering with the optimal K using joblib
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clusters
sns.pairplot(df, hue='cluster', palette='Dark2')
plt.show()

# Print the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:")
print(pd.DataFrame(cluster_centers, columns=features.columns))

# Add the 'cluster' column to the original dataset
df.to_csv('clustered_dataset.csv', index=False)


# Load the dataset
# Replace 'your_dataset.csv' with the actual file path or data loading method
df = pd.read_csv('ds_salaries.csv')

# Select relevant features for clustering
features = df[['work_year', 'experience_level', 'salary_in_usd', 'remote_ratio']]

# Encode categorical variables if needed (e.g., 'experience_level' and 'work_year')
label_encoder = LabelEncoder()
features['experience_level'] = label_encoder.fit_transform(features['experience_level'])
features['work_year'] = label_encoder.fit_transform(features['work_year'])

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters (K) using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares

def fit_kmeans(num_clusters, data):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    return kmeans.inertia_

# Use joblib for parallelization
num_threads = 4  # Adjust based on your system capabilities
wcss = Parallel(n_jobs=num_threads)(delayed(fit_kmeans)(i, scaled_features) for i in range(1, 11))

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (K)
optimal_k = 3  # You can adjust this based on the visual inspection of the Elbow Method graph

# Apply K-Means clustering with the optimal K using joblib
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clusters
sns.pairplot(df, hue='cluster', palette='Dark2')
plt.show()

# Print the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:")
print(pd.DataFrame(cluster_centers, columns=features.columns))

# Add the 'cluster' column to the original dataset
df.to_csv('clustered_dataset.csv', index=False)



def fit_kmeans(num_clusters, data):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    return kmeans.inertia_

if __name__ == '__main__':
    freeze_support()
    # Load the dataset
    # Replace 'your_dataset.csv' with the actual file path or data loading method
    df = pd.read_csv('ds_salaries.csv')

    # Select relevant features for clustering
    features = df[['work_year', 'experience_level', 'salary_in_usd', 'remote_ratio']]

    # Encode categorical variables if needed (e.g., 'experience_level' and 'work_year')
    label_encoder = LabelEncoder()
    features.loc[:, 'experience_level'] = label_encoder.fit_transform(features['experience_level'].values)
    features.loc[:, 'work_year'] = label_encoder.fit_transform(features['work_year'].values)

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Determine the optimal number of clusters (K) using the Elbow Method
    wcss = []  # Within-Cluster-Sum-of-Squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.show()

    # Based on the Elbow Method, choose the optimal number of clusters (K)
    optimal_k = 3  # You can adjust this based on the visual inspection of the Elbow Method graph

    # Apply K-Means clustering with the optimal K
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # Visualize the clusters
    sns.pairplot(df, hue='cluster', palette='Dark2')
    plt.show()

    # Print the cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    print("Cluster Centers:")
    print(pd.DataFrame(cluster_centers, columns=features.columns))

    # Add the 'cluster' column to the original dataset
    df.to_csv('clustered_dataset.csv', index=False)

    # Ensure proper synchronization between threads during the update step
    # Parallelize the computation of new cluster centers using concurrent.futures
    num_threads = 4  # Adjust based on your system capabilities

    # Define a function to compute new cluster centers
    def compute_new_cluster_center(cluster, df, scaled_features):
        cluster_points = scaled_features[df['cluster'] == cluster]
        new_cluster_center = np.mean(cluster_points, axis=0)
        return cluster, new_cluster_center

    # Use concurrent.futures for parallelization
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_new_cluster_center, cluster, df, scaled_features) for cluster in range(optimal_k)]

        # Retrieve the results
        results = [future.result() for future in futures]

    # Print the updated cluster centers
    print("Updated Cluster Centers:")
    for result in results:
        cluster, new_cluster_center = result
        print(f"Cluster {cluster}:", new_cluster_center)
