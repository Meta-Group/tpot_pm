import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import networkx as nx
from karateclub.node_embedding.neighbourhood import Node2Vec
import pm4py


def get_traces(df):
    traces, ids = [], []
    for group in df.groupby("case:concept:name"):
        events = list(group[1]["concept:name"])
        traces.append(["".join(x) for x in events])
        ids.append(group[0])
    return traces, ids


def one_hot(df):
    traces = [
        " ".join(str(x.replace(" ", "")) for x in list(group[1]["concept:name"]))
        for group in df.groupby("case:concept:name")
    ]
    return pd.DataFrame(
        CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
        .fit_transform(traces)
        .toarray()
    )


def compute_alignments(alignments):
    cost, visited_states, queued_states, traversed_arcs, fitness, lp_solved, bwc = [], [], [], [], [], [], []
    for alignment in alignments:
        if alignment is None:
            cost.append(0)
            visited_states.append(0)
            queued_states.append(0)
            traversed_arcs.append(0)
            fitness.append(0)
            lp_solved.append(0)
            bwc.append(0)
        else:
            cost.append(alignment["cost"])
            visited_states.append(alignment["visited_states"])
            queued_states.append(alignment["queued_states"])
            traversed_arcs.append(alignment["traversed_arcs"])
            fitness.append(alignment["fitness"])
            lp_solved.append(alignment["lp_solved"])
            bwc.append(alignment["bwc"])

    return pd.DataFrame(
        zip(
            cost, visited_states, queued_states, traversed_arcs, fitness, lp_solved, bwc
        )
    )


def alignments(log):
    net, im, fm = pm4py.discover_petri_net_inductive(log)
    fitness_alignments = pm4py.conformance_diagnostics_alignments(
        log, net, im, fm, multi_processing=True
    )
    return compute_alignments(fitness_alignments)


def average_feature_vector(model, traces):
    vectors_average = []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(model.wv[token])
            except KeyError:
                pass
        vectors_average.append(np.array(trace_vector).mean(axis=0))

    return pd.DataFrame(vectors_average)


def word2vec(traces):
    dimension = 8
    model = Word2Vec(vector_size=dimension, window=3, min_count=1, workers=-1)
    model.build_vocab(traces)
    model.train(traces, total_examples=len(traces), epochs=10)
    return average_feature_vector(model, traces)


def convert_traces_mapping(traces_raw, mapping):
    traces = []
    for trace in traces_raw:
        traces.append([mapping[act] for act in trace])
    return traces


def trace_feature_vector_from_nodes(embeddings, traces, dimension):
    vectors_average = []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(embeddings[token])
            except KeyError:
                pass
        if len(trace_vector) == 0:
            trace_vector.append(np.zeros(dimension))
        vectors_average.append(np.array(trace_vector).mean(axis=0))

    return pd.DataFrame(vectors_average)


def node2vec_(log, traces):
    dimension = 8
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)

    graph = nx.Graph()
    [graph.add_weighted_edges_from([(edge[0], edge[1], dfg[edge])]) for edge in dfg]

    mapping = dict(zip(graph.nodes(), [i for i in range(len(graph.nodes()))]))
    graph = nx.relabel_nodes(graph, mapping)
    traces = convert_traces_mapping(traces, mapping)

    model = Node2Vec(dimensions=dimension)
    model.fit(graph)
    return trace_feature_vector_from_nodes(model.get_embedding(), traces, dimension)
