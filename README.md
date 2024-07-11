# Clustering Antartic Penguin Species
Unraveling Antartic Penguin species with K-Means Clustering
![image](https://github.com/RashidTobrazune/Clustering_Antartic_Penguin_Species/assets/150378293/0c61dc4e-d5f8-4022-8951-18a1d7c31a03)

Problem Statement
You have been asked to support a team of researchers who have been collecting data about penguins in Antartica! The data is available in csv-Format as penguins.csv

Origin of this data : Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

The dataset consists of 5 columns.

Column	Description
culmen_length_mm	culmen length (mm)
culmen_depth_mm	culmen depth (mm)
flipper_length_mm	flipper length (mm)
body_mass_g	body mass (g)
sex	penguin sex
Unfortunately, they have not been able to record the species of penguin, but they know that there are at least three species that are native to the region: Adelie, Chinstrap, and Gentoo. Your task is to apply your data science skills to help them identify groups in the dataset!

Process

Here's the revised version of the code, including the use of `matplotlib` to visualize the clusters:

```python
# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Step 1: Load and preprocess the data
# Load the dataset
penguins = pd.read_csv('penguins.csv')

# Display the first few rows of the dataset
print(penguins.head())

# Drop any rows with missing values
penguins = penguins.dropna()

# Step 2: Standardize the data
# Select only the numeric columns for clustering
numeric_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Standardize the numeric columns
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins[numeric_cols])

# Step 3: Perform cluster analysis using KMeans
# Set the number of clusters (e.g., 3 clusters for the three species of penguins)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans model to the standardized data
kmeans.fit(penguins_scaled)

# Add the cluster labels to the original dataset
penguins['cluster'] = kmeans.labels_

# Step 4: Collect the average values for the clusters
# Group by the cluster labels and calculate the mean for each cluster
stat_penguins = penguins.groupby('cluster')[numeric_cols].mean()

# Step 5: Visualize the clusters
# Create a pairplot to visualize the clusters
sns.pairplot(penguins, hue='cluster', palette='Set1', diag_kind='kde')
plt.suptitle('Pairplot of Penguin Clusters', y=1.02)
plt.show()

# Plot the centroids of the clusters
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=numeric_cols)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=penguins, x='culmen_length_mm', y='culmen_depth_mm', hue='cluster', palette='Set1', style='cluster')
sns.scatterplot(data=centroids_df, x='culmen_length_mm', y='culmen_depth_mm', s=200, color='black', label='Centroids')
plt.title('Cluster Centroids and Data Points')
plt.show()

# Step 6: Output the results
print(stat_penguins)

# Save the results to a CSV file
stat_penguins.to_csv('stat_penguins.csv')
```

1.
   - Load the dataset from the `penguins.csv` file.
   - Drop any rows with missing values.

2. 
   - Select only the numeric columns for clustering.
   - Use `StandardScaler` to standardize the numeric columns.

   3. Perform cluster analysis using KMeans:
   - Set the number of clusters to 3 and fit the KMeans model to the standardized data.
   - Add the cluster labels to the original dataset.

4. Collect the average values for the clusters:
   - Group the dataset by the cluster labels and calculate the mean for each cluster.

5. Visualize the clusters:
   - Use `seaborn` to create a pairplot of the data, colored by cluster.
   - Plot the centroids of the clusters on a scatter plot to show their positions relative to the data points.

6. Output the results:
   - Print the resulting DataFrame with the average values for each cluster.
   - Save the results to a CSV file named `stat_penguins.csv`.

### Insights from Cluster Analysis of Penguin Data

<img width="643" alt="Screenshot 2024-07-11 211912" src="https://github.com/user-attachments/assets/fbc68af0-5cba-484a-a36c-2b27202f1309">
<img width="556" alt="Screenshot 2024-07-11 211931" src="https://github.com/user-attachments/assets/5910f69d-6ba6-46ac-805d-46cef357a5ca">
<img width="682" alt="rem" src="https://github.com/user-attachments/assets/41c70bb3-dc26-41f6-9c64-f31e84ce84ed">
<img width="513" alt="Screenshot 2024-07-11 211833" src="https://github.com/user-attachments/assets/edab0c45-1138-41d3-ab88-3ca99ea86918">


The pairplot visualization supports the numeric insights and highlights distinct groups of penguins with varying physical characteristics. These clusters likely represent different species, and the features such as flipper length, culmen length, and culmen depth are key differentiators. This analysis can help the researchers identify and differentiate between the penguin species in the dataset.

Given the visual separation and average values of the clusters, we can hypothesize:

- **Cluster 1 (Blue)** might represent **Chinstrap penguins**, which tend to have longer flippers and larger body sizes.
- **Cluster 2 (Green)** could represent **Adelie penguins**, which are smaller in size with shorter beaks and flippers.
- **Cluster 0 (Red)** might represent **Gentoo penguins**, which are characterized by moderate sizes in terms of culmen length, depth, and flipper length.








Given the cluster averages for the penguin dataset, we can draw some meaningful insights about the distinct groups of penguins:

| Cluster | Culmen Length (mm) | Culmen Depth (mm) | Flipper Length (mm) | Body Mass (g) |
|---------|---------------------|-------------------|---------------------|---------------|
| 0       | 47.66               | 18.75             | 196.92              | 3898.24       |
| 1       | 47.57               | 15.00             | 217.24              | 5092.44       |
| 2       | 38.31               | 18.10             | 188.55              | 3587.50       |

### Cluster Analysis

1. **Cluster 0:**
   - **Culmen Length**: 47.66 mm
   - **Culmen Depth**: 18.75 mm
   - **Flipper Length**: 196.92 mm
   - **Body Mass**: 3898.24 g

   This cluster represents penguins with a long culmen length and deep culmen depth. Their flipper length is moderate, and they have an average body mass.

2. **Cluster 1:**
   - **Culmen Length**: 47.57 mm
   - **Culmen Depth**: 15.00 mm
   - **Flipper Length**: 217.24 mm
   - **Body Mass**: 5092.44 g

   This cluster represents penguins with a similar culmen length to Cluster 0 but with a much shallower culmen depth. They have the longest flipper length and the highest body mass, suggesting they might be a different species with larger body sizes.

3. **Cluster 2:**
   - **Culmen Length**: 38.31 mm
   - **Culmen Depth**: 18.10 mm
   - **Flipper Length**: 188.55 mm
   - **Body Mass**: 3587.50 g

   This cluster represents penguins with the shortest culmen length and the smallest flipper length. Their culmen depth is similar to Cluster 0, but they have the lowest body mass.

### Possible Species Identification

While we do not have the species labels, we can hypothesize based on known species characteristics:

- **Cluster 0** might represent **Gentoo penguins**, which are known to have long beaks and moderate body sizes.
- **Cluster 1** could represent **Chinstrap penguins**, which tend to be larger with longer flippers.
- **Cluster 2** could be **Adelie penguins**, which are smaller in size with shorter beaks and flippers.

### Summary

- **Cluster 0**: Penguins with long culmens, deep culmens, moderate flippers, and average body mass.
- **Cluster 1**: Penguins with long culmens, shallow culmens, longest flippers, and highest body mass.
- **Cluster 2**: Penguins with short culmens, deep culmens, shortest flippers, and lowest body mass.

These insights suggest distinct groups of penguins with varying physical characteristics, which can help the researchers identify and differentiate between the species present in the dataset.





