import tpot
from tpot.tpot import TPOTClustering
import pandas as pd
import neptune.new as neptune
import requests
import json


headers = {
    "Content-Type": "application/json",
    "Access-Control-Request-Headers": "*",
    "api-key": "tURg5ipsw8mFrPwY52B0d37bzsRQLyk1UFOjFz0fkicfra1FzlcrsDwOl4ctCymr",
}

db = 'tpot'
collection = 'pm'
_scorers = ['sil', 'complexity']
mo = "mean_score"

def get_run_config():
    find_one_url = "https://eu-central-1.aws.data.mongodb-api.com/app/data-vhbni/endpoint/data/v1/action/findOne"
    payload = json.dumps(
        {
            "collection": collection,
            "database": db,
            "dataSource": "Malelab",
            "filter": {"status": "active"},
        }
    )

    response = requests.request("POST", find_one_url, headers=headers, data=payload)
    _response = response.json()
    return _response["document"]


def update_run(run, status):
    update_one_url = "https://eu-central-1.aws.data.mongodb-api.com/app/data-vhbni/endpoint/data/v1/action/updateOne"
    payload = json.dumps(
        {
            "collection": collection,
            "database": db,
            "dataSource": "Malelab",
            "filter": {
                "_id": {"$oid": run["_id"]},
            },
            "update": {"$set": {"status": status}},
        }
    )

    response = requests.request("POST", update_one_url, headers=headers, data=payload)
    print(response.text)

while 1:
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ=="
    project_name = "MaleLab/TpotPMFull"
    run_config = get_run_config()
    if not run_config:
        print("\n\n0 Active runs --- bye")
        quit()

    dataset_name = run_config["dataset"]
    gen = run_config["gen"]
    pop = run_config["pop"]
    run_id = run_config["_id"]
    run_number = run_config["status"]
    log = pd.read_csv(f"datasets/{dataset_name}")

    run = neptune.init_run(project=project_name, api_token=api_token)
    run["_id"] = run_id
    run["dataset"] = dataset_name


    enc_onehot = pd.read_csv(f"datasets/encodings/one_hot/{dataset_name}")
    enc_alignments = pd.read_csv(f"datasets/encodings/alignments/{dataset_name}")
    enc_word2vec = pd.read_csv(f"datasets/encodings/word2vec/{dataset_name}")
    enc_node2vec = pd.read_csv(f"datasets/encodings/node2vec/{dataset_name}")

    encodings = {'one_hot':enc_onehot, 'alignments': enc_alignments, 'word2vec': enc_word2vec, 'node2vec': enc_node2vec}

    try:
        print(f"\n==================== TPOT CLUSTERING PM ==================== \n Run ID: {run_id} - Dataset: {dataset_name}")
        update_run(run_config, "occupied")
        log["time:timestamp"] = pd.to_datetime(log['time:timestamp'])
        log = log.astype({'case:concept:name': 'string'})
        log = log.astype({'concept:name': 'string'})
        clusterer = TPOTClustering(
            generations=gen,
            population_size=pop,
            verbosity=2,
            config_dict=tpot.config.clustering_config_dict,
            n_jobs=1,
        )

        clusterer.fit(features=log, mo_function=mo, scorers=_scorers, encodings=encodings)

        pipeline, scores, clusters, case_ids, labels = clusterer.get_run_stats()
        print(f"Pipeline: {pipeline} Scores: {scores} Clusters: {clusters}")
        df = pd.DataFrame({'CaseIds': case_ids, 'Labels': labels})
        df.to_csv(f"{dataset_name}_{run_id}.csv",index=False)

        run["sil"] = scores['sil']
        run["complexity"] = scores['complexity']
        run["clusters"] = clusters
        run["pipeline"] = pipeline
        run["case_labels"] = f"{dataset_name}_{run_id}.csv"
        update_run(run_config, "finished")

    except Exception as e:
        run["error_msg"] = e
        print(f"{e}")
        update_run(run_config, "error")

    run.sync()
    run.stop()
