import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
#
# warnings.filterwarnings("ignore")


df = pd.read_csv("results/clustering_performance_.csv")

for index, row in df.iterrows():
    # print(row)
    log_name = row[0]
    encoding = row[1].split("encoding=")[1].split(",")[0]
    # print(log_name, encoding)
    df_enc = pd.read_csv(f"datasets/encodings/{encoding}/{log_name}")
    # print(df_enc)
    df_clustered_cases = pd.read_csv(f"results/cluster_configs/{row[2]}")
    # print(df_clustered_cases)

    pca = PCA(n_components=2)
    norm_data = StandardScaler().fit_transform(df_enc)
    X_new = pca.fit_transform(norm_data)
    df_pca = pd.DataFrame(X_new, columns=["PC1", "PC2"])
    df_pca.insert(0, "Clusters", df_clustered_cases["Labels"])

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        ax=ax,
        data=df_pca,
        x="PC1",
        y="PC2",
        palette="bright",
        hue="Clusters",
        alpha=0.9,
        s=100,
    )
    ax.set_xlabel(
        f"PC1 ({np.round(pca.explained_variance_ratio_[0]*100, 2)}% explained variance)"
    )
    ax.set_ylabel(
        f"PC2 ({np.round(pca.explained_variance_ratio_[1]*100, 2)}% explained variance)"
    )
    plt.tight_layout()
    # plt.savefig(f"{path}/pca.png", dpi=200)
    plt.show()
    plt.close()
    # break
