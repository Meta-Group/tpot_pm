import tpot
from tpot.tpot import TPOTClustering
import pandas as pd
import neptune.new as neptune

_scorers = ['sil','dbs', 'complexity']
mo = "mean_score"
dataset_name = "scenario5_1000_early_0.1_onehot.csv"
log_name = "scenario5_1000_early_0.1.csv"

log = pd.read_csv(f"datasets/{log_name}")
log["time:timestamp"] = pd.to_datetime(log['time:timestamp'])

try:
    features = pd.read_csv(f"datasets/{dataset_name}")
    case_ids = features.iloc[:,-1]
    features = features.iloc[: , :-1]
    
    print(f"\n==================== TPOT CLUSTERING TRAINING ==================== \n - Dataset: {dataset_name}")

    clusterer = TPOTClustering(
        generations=3,
        population_size=5,
        verbosity=2,
        config_dict=tpot.config.clustering_config_dict,
        n_jobs=1,
    )

    clusterer.fit(features=features, log=log, case_ids=case_ids, mo_function=mo, scorers=_scorers)

    # pipeline, scores, clusters = clusterer.get_run_stats()
    # print(f"Pipeline: {pipeline} Scores: {scores} Clusters: {clusters}")

except Exception as e:
    print(f"{e}")

