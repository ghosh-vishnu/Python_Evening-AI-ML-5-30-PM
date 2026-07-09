import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("customer_segmentation_1000.csv")

print(df.head())
print(df.info())

# -----------------------------
# Select Features for Clustering
# -----------------------------
X = df[["Annual_Income","Spending_Score"]]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

for k in range(1,11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(7,4))
plt.plot(range(1,11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# -----------------------------
# Train Final Model
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df.head())

# -----------------------------
# Cluster Visualization
# -----------------------------
plt.figure(figsize=(8,6))

for c in sorted(df["Cluster"].unique()):
    temp = df[df["Cluster"] == c]
    plt.scatter(
        temp["Annual_Income"],
        temp["Spending_Score"],
        label=f"Cluster {c}"
    )

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:,0],
    centers[:,1],
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()

# -----------------------------
# Predict New Customer Cluster
# -----------------------------
def predict_customer_cluster(income, spending_score):
    sample = [[income, spending_score]]
    sample_scaled = scaler.transform(sample)
    cluster = kmeans.predict(sample_scaled)[0]
    return cluster

print("Predicted Cluster:",
      predict_customer_cluster(85000,70))

# Save Result
df.to_csv("customer_segmentation_result.csv", index=False)
