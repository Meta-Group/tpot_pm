# TPOT - Process Mining
A version of the [TPOT](https://github.com/EpistasisLab/tpot) for Process Mining. It optimizes both silhouette score and sequence
entropy to synthesize clustering pipelines.

> 🔍 **Also, don’t miss out on [TPOT-Clustering](https://github.com/Mcamilo/tpot-clustering)** –––––––––––––––––––––––––––––––––––––––––––––  
> a versatile and powerful library for solving a wide range of **clustering problems over tabular data**.  
> It’s designed for general-purpose use, with easy customization and high extensibility.

---

## Installation
The required packages can be installed using the following command in a terminal:

```console
 pip install -r requirements.txt
```

## Example
```python
import tpot
from tpot.tpot import TPOTClustering
import pandas as pd

_scorers = ['sil', 'complexity']
mo = "mean_score"
data_path = "sampled_helpdesk"

#log data 
log_name = "helpdesk.csv"
log = pd.read_csv(f"{data_path}/sample_{idx}_{log_name}")
# preprocessing
log["time:timestamp"] = pd.to_datetime(log['time:timestamp'])
log = log.astype({'case:concept:name': 'string'})
log = log.astype({'concept:name': 'string'})

# read the pre-encoded log data
enc_onehot = pd.read_csv(f"{data_path}/one_hot_{log_name}")
enc_alignments = pd.read_csv(f"{data_path}/alignments_{log_name}")
enc_word2vec = pd.read_csv(f"{data_path}/word2vec_{log_name}")
enc_node2vec = pd.read_csv(f"{data_path}/node2vec_{log_name}")

encodings = {'one_hot':enc_onehot, 'alignments': enc_alignments, 'word2vec': enc_word2vec, 'node2vec': enc_node2vec}

try:
    # setup the optimisation process
    clusterer = TPOTClustering(
        generations=5,
        population_size=15,
        verbosity=2,
        config_dict=tpot.config.clustering_config_dict,
        n_jobs=1,
    )
    # initialize the process
    clusterer.fit(features=log, mo_function=mo, scorers=_scorers, encodings=encodings)
    # collect the results
    pipeline, scores, clusters, case_ids, labels = clusterer.get_run_stats()
    print(f"Pipeline: {pipeline} Scores: {scores} Clusters: {clusters}")
    df = pd.DataFrame({'CaseIds': case_ids, 'Labels': labels})
    df.to_csv("case_label.csv",index=False)

except Exception as e:
    print(f"{e}")

```

## Usage
The sample example can be run using the following command:
```console
 python run test_sample.py
```
