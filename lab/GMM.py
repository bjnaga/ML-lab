import pandas as pd
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

iris = load_iris()
X = iris.data
feature_names = iris.feature_names

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)

cluster_labels = gmm.fit_predict(X)
soft_probabilities = gmm.predict_proba(X)

df = pd.DataFrame(X, columns=feature_names)
df['Hard_Cluster'] = cluster_labels

for i in range(3):
    df[f'Prob_Cluster_{i}'] = soft_probabilities[:, i]

print("GMM Clustering Results (First 5 Samples):")
print(df.head())

print("\nCluster Means (Centers):\n", gmm.means_)
print("\nCluster Covariances (Shape of each component):\n", gmm.covariances_)
