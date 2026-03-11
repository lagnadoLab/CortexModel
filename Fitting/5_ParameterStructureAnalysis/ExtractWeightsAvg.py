import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Load your CSV
parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["rest", "locomotion"])
args = parser.parse_args()

if args.condition == "rest":
    df = pd.read_csv("good_fits_local_rest.csv")
elif args.condition == "locomotion":
    df = pd.read_csv("good_fits_opto_locomotion.csv")
else:
    raise ValueError(f"Unknown condition: {args.condition}")

row = df.drop(columns=["ID"]).mean()

# Extract the weights in the desired order
weights = np.array([
    row["w_0"],  # PC -> PC
    row["w_1"],  # FF -> PC
    row["w_2"],  # SS -> PC
    row["w_3"],  # PV -> PC
    row["w_4"],  # SST -> PC
    row["w_5"],  # PC -> PV
    row["w_6"],  # FF -> PV
    row["w_7"],  # SS -> PV
    row["w_8"],  # PV -> PV
    row["w_9"],  # SST -> PV
    row["w_10"], # PC -> SST
    row["w_11"], # FB -> SST
    row["w_12"], # VIP -> SST
    row["w_13"], # PC -> VIP
    row["w_14"], # SST -> VIP
    row["w_15"], # SS -> VIP
    row["w_16"], # FB -> PC
    row["w_17"], # FB -> PV
    row["w_18"], # FB -> VIP
], dtype=np.float32)

##Transfer weights to a table##
# Table size
n_rows = 7
n_cols = 4
table = np.full((n_rows, n_cols), np.nan)

positions = [
    (0, 0), # PC -> PC w_0
    (4, 0), # FF -> PC w_1
    (5, 0), # SS -> PC w_2
    (2, 0), # PV -> PC w_3
    (1, 0), # SST -> PC w_4
    (0, 2), # PC -> PV w_5
    (4, 2), # FF -> PV w_6
    (5, 2), # SS -> PV w_7
    (2, 2), # PV -> PV w_8
    (1, 2), # SST -> PV w_9
    (0, 1), # PC -> SST w_10
    (6, 1), # FB -> SST w_11
    (3, 1), # VIP -> SST w_12
    (0, 3), # PC -> VIP w_13
    (1, 3), # SST -> VIP w_14
    (5, 3), # SS -> VIP w_15
    (6, 0), # FB -> PC w_16
    (6, 2), # FB -> PV w_17
    (6, 3), # FB -> VIP w_18
]
for w, (r, c) in zip(weights, positions):
    table[r, c] = w

celltype_pre = ["PC", "SST", "PV", "VIP", "FF", "SS", "FB"]
celltype_post = ["PC", "SST", "PV", "VIP"]
df_save = pd.DataFrame(table, index=celltype_pre, columns = celltype_post)

## Clustering ##
clustering_df = df[["ID", "w_0", "w_1", "w_2", "w_3", "w_4", "w_5", "w_6",
"w_7", "w_8", "w_9", "w_10", "w_11", "w_12", "w_13", "w_14", "w_15", "w_16", "w_17", "w_18",
"k"]].copy()

param_cols = [col for col in clustering_df.columns if col not in ['ID']]
X = clustering_df[param_cols]
# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-means
inertia = []
K = range(1, 30)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

for k in range(2, 30):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"k={k}, silhouette score={score:.3f}")

# DBSCAN
db = DBSCAN(eps=4.5, min_samples=5)
labels = db.fit_predict(X_scaled)

clustering_df["Cluster"] = labels

clustering_df["Cluster"].unique()


## Saving ##
df_save.to_csv(f"ConnWeights_{args.condition}.csv", index=True)

