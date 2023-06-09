import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pm4py
from tpot.Complexity import (
    generate_pm4py_log,
    generate_log,
    build_graph,
    log_complexity,
)
import warnings

warnings.filterwarnings("ignore")


def cluster_performance(log_cluster, log):
    df_clustered_cases = pd.read_csv(f"results/cluster_configs/{log_cluster}")
    df_clustered_cases.rename(
        columns={"CaseIds": "case:concept:name", "Labels": "labels"}, inplace=True
    )

    fits, precs, entrs, lengths, sound, c_labels = [], [], [], [], [], []
    for group in df_clustered_cases.groupby("labels"):
        c_labels.append(group[0])
        case_ids = list(group[1]["case:concept:name"])
        case_ids = [str(x) for x in case_ids]
        lengths.append(len(case_ids))

        # anomalies
        if group[0] == -1:
            entrs.append(-1)
            fits.append(-1)
            precs.append(-1)
            continue

        sublog = log[log["case:concept:name"].isin(case_ids)].copy()
        _, entropy = log_complexity(
            build_graph(generate_log(pm4py.convert_to_event_log(sublog)))
        )
        entrs.append(np.round(entropy, 4))

        net, im, fm = pm4py.discover_petri_net_heuristics(sublog)
        try:
            woflan.apply(net, im, fm, parameters={"print_diagnostics": False})
            sound.append(True)
        except:
            sound.append(False)
        try:
            fitness_alignments = pm4py.fitness_alignments(
                sublog, net, im, fm, multi_processing=True
            )
            fits.append(np.round(fitness_alignments["log_fitness"], 4))
        except:
            fits.append(0)
        try:
            precision_alignments = pm4py.precision_alignments(
                sublog, net, im, fm, multi_processing=True
            )
            precs.append(np.round(precision_alignments, 4))
        except:
            precs.append(0)

    return pd.DataFrame(
        zip(fits, precs, entrs, sound, lengths, c_labels),
        columns=[
            "fitness",
            "precision",
            "entropy",
            "soundness",
            "weight",
            "cluster_label",
        ],
    )


df = pd.read_csv(f"results/pipeline_runs.csv")
df.dropna(inplace=True)
# print(df)

import sys
sys.setrecursionlimit(10000)

out = []
for group in df.groupby("dataset"):
    log_name = group[0]
    print(log_name)
    log = pd.read_csv(f"datasets/{log_name}")
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    log = log.astype({"case:concept:name": "string"})
    log = log.astype({"concept:name": "string"})

    # "bpi_2012_sample_15.csv"
    logs = ["bpi_2013_incidents.csv", "sepsis.csv", "bpi_2013_closed_problems.csv", "helpdesk.csv"]
    if log_name in logs:
        continue
    df_log = group[1].copy()
    # print(df_log[["sil", "complexity", "clusters"]])

    # normalizing metrics
    df_log["sil_norm"] = 1 - MinMaxScaler().fit_transform(
        np.array(df_log["sil"]).reshape(-1, 1)
    )
    df_log["complexity_norm"] = MinMaxScaler().fit_transform(
        np.array(df_log["complexity"]).reshape(-1, 1)
    )
    df_log["clusters_norm"] = MinMaxScaler().fit_transform(
        np.array(df_log["clusters"]).reshape(-1, 1)
    )

    # ranking pipelines
    df_log["rank"] = (
        df_log["sil_norm"] + df_log["complexity_norm"] + df_log["clusters_norm"]
    ) / 3
    df_log.sort_values(["rank"], inplace=True)
    # print(df_log[["dataset", "sil_norm", "complexity_norm", "clusters_norm", "rank", "pipeline"]])

    # computing performances of the original event log
    _, entropy_orig = np.round(
        log_complexity(build_graph(generate_log(pm4py.convert_to_event_log(log)))), 4
    )
    net, im, fm = pm4py.discover_petri_net_heuristics(log)
    fitness_orig = np.round(
        pm4py.fitness_alignments(log, net, im, fm, multi_processing=True)[
            "log_fitness"
        ],
        4,
    )
    precision_orig = np.round(
        pm4py.precision_alignments(log, net, im, fm, multi_processing=True), 4
    )
    print("Original log metrics")
    print(f"Fitness: {fitness_orig}")
    print(f"Precision: {precision_orig}")
    print(f"Entropy: {entropy_orig}")
    print()

    # retrieving the best pipeline performances
    df_cluster_metrics = cluster_performance(df_log.iloc[0, 9], log)
    fitness_clus = np.average(
        df_cluster_metrics["fitness"], weights=df_cluster_metrics["weight"]
    )
    precision_clus = np.average(
        df_cluster_metrics["precision"], weights=df_cluster_metrics["weight"]
    )
    entropy_clus = np.average(
        df_cluster_metrics["entropy"], weights=df_cluster_metrics["weight"]
    )

    # performance comparison between original log and clustered log
    print(f"Best pipeline for log {group[0]} is: {df_log.iloc[0, 6]}")
    print(f"Silhouette: {df_log.iloc[0, 4]}")
    print(f"Fitness: {fitness_clus}")
    print(f"Precision: {precision_clus}")
    print(f"Entropy: {entropy_clus}")
    print(f"# Clusters: {int(df_log.iloc[0, 7])}")
    print()

    print("Improvement over original log")
    print(f"Fitness: {fitness_clus - fitness_orig}")
    print(f"Precision: {precision_clus - precision_orig}")
    print(f"Entropy: {entropy_orig - entropy_clus}")
    print()

    out.append(
        [
            log_name,
            df_log.iloc[0, 6],
            df_log.iloc[0, 9],
            fitness_orig,
            precision_orig,
            entropy_orig,
            df_log.iloc[0, 4],
            fitness_clus,
            precision_clus,
            entropy_clus,
            int(df_log.iloc[0, 7]),
            (fitness_clus - fitness_orig),
            (precision_clus - precision_orig),
            (entropy_orig - entropy_clus),
        ]
    )
    # break

    pd.DataFrame(
        out,
        columns=[
            "log",
            "pipeline",
            "ref",
            "fitness_original",
            "precision_original",
            "entropy_original",
            "silhouette",
            "fitness_clus",
            "precision_clus",
            "entropy_clus",
            "#clusters",
            "fitness_improval",
            "precision_improval",
            "entropy_improval",
        ],
    ).to_csv("results/clustering_performance.csv", index=False)
