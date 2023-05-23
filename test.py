import tpot
from tpot.tpot import TPOTClustering
import pandas as pd
import pm4py

_scorers = ['sil', 'complexity']
mo = "mean_score"

log_name = "scenario5_1000_early_0.1.csv"

log = pd.read_csv(f"datasets/{log_name}")
log["time:timestamp"] = pd.to_datetime(log['time:timestamp'])
log = log.astype({'case:concept:name': 'string'})
log = log.astype({'concept:name': 'string'})

try:
    # features = pd.read_csv(f"datasets/{dataset_name}")
    # case_ids = features.iloc[:, -1]
    # features = features.iloc[: , :-1]

    print(f"\n==================== TPOT CLUSTERING TRAINING ==================== \n - Dataset: {log_name}")

    clusterer = TPOTClustering(
        generations=10,
        population_size=50,
        verbosity=2,
        config_dict=tpot.config.clustering_config_dict,
        n_jobs=1,
    )

    clusterer.fit(features=log, mo_function=mo, scorers=_scorers)

    pipeline, scores, clusters = clusterer.get_run_stats()
    print(f"Pipeline: {pipeline} Scores: {scores} Clusters: {clusters}")

except Exception as e:
    print(f"{e}")
