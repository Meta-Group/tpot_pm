import tpot
from tpot.tpot import TPOTClustering
import pandas as pd
# import pm4py

_scorers = ['sil', 'complexity']
mo = "mean_score"

log_name = "scenario5_1000_early_0.1.csv"
log = pd.read_csv(f"datasets/{log_name}")
log["time:timestamp"] = pd.to_datetime(log['time:timestamp'])
log = log.astype({'case:concept:name': 'string'})
log = log.astype({'concept:name': 'string'})

enc_onehot = pd.read_csv(f"datasets/encodings/one_hot/{log_name}")
enc_alignments = pd.read_csv(f"datasets/encodings/alignments/{log_name}")
enc_word2vec = pd.read_csv(f"datasets/encodings/word2vec/{log_name}")
enc_node2vec = pd.read_csv(f"datasets/encodings/node2vec/{log_name}")

encodings = {'one_hot':enc_onehot, 'alignments': enc_alignments, 'word2vec': enc_word2vec, 'node2vec': enc_node2vec}

try:

    print(f"\n==================== TPOT CLUSTERING TRAINING ==================== \n - Dataset: {log_name}")

    clusterer = TPOTClustering(
        generations=3,
        population_size=5,
        verbosity=2,
        config_dict=tpot.config.clustering_config_dict,
        n_jobs=1,
    )

    clusterer.fit(features=log, mo_function=mo, scorers=_scorers, encodings=encodings)

    pipeline, scores, clusters, case_ids, labels = clusterer.get_run_stats()
    print(f"Pipeline: {pipeline} Scores: {scores} Clusters: {clusters}")
    df = pd.DataFrame({'CaseIds': case_ids, 'Labels': labels})
    df.to_csv("case_label.csv",index=False)

except Exception as e:
    print(f"{e}")
