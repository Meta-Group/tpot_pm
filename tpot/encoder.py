import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
import pm4py


def get_traces(df):
    traces, ids = [], []
    for group in df.groupby("case:concept:name"):
        events = list(group[1]["concept:name"])
        traces.append(" ".join(x for x in events))
        ids.append(group[0])
    return traces, ids

def one_hot(traces):
    corpus = CountVectorizer().fit_transform(traces)
    return Binarizer().fit_transform(corpus.toarray())

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

    return pd.DataFrame([cost, visited_states, queued_states, traversed_arcs, fitness, lp_solved, bwc])

def alignments(log):
    log = pm4py.convert_to_event_log(log)
    print(log)
    # net, im, fm = pm4py.discover_petri_net_inductive(log,activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    # fitness_alignments = pm4py.conformance_diagnostics_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    # print(fitness_alignments)
    # compute_alignments(fitness_alignments)