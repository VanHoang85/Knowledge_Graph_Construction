import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from entity_extraction import EntityTracker
from feature_extraction import PatternTracker


def pair_pattern_matrix(pattern_tracker: PatternTracker, entity_tracker: EntityTracker) -> np.ndarray:

    # pairs as rows and patterns as columns with cells as co-occurrence counts of patterns
    matrix = []
    highest_count = 0

    for pair, patterns in pattern_tracker.pairs2patterns.items():

        if patterns:
            feature_vector = [0] * len(pattern_tracker.patterns)
            entity_tracker.add_pair_idx(pair)

            for pattern in patterns:
                feature_vector[pattern_tracker.patterns.index(pattern)] += 1

            matrix.append(feature_vector)

            max_count = feature_vector[np.array(feature_vector).argmax()]
            if max_count > highest_count:
                highest_count = max_count

    print('=' * 50)
    print('Highest count of features:', highest_count)
    print('=' * 50)

    return np.array(matrix)


def cluster_pattern_matrix(clusters: AgglomerativeClustering, pp_matrix: np.ndarray, args) -> np.ndarray:

    cp_matrix = np.zeros((clusters.n_clusters_, pp_matrix.shape[1]))

    for idx, cluster_id in enumerate(clusters.labels_):
        cp_matrix[cluster_id] += pp_matrix[idx]

    if args.ranked_metric == 'tfidf':
        cp_matrix = TfidfTransformer().fit_transform(cp_matrix)

    return cp_matrix


def clustering(matrix: np.ndarray, parameters: dict) -> AgglomerativeClustering:

    clusters = AgglomerativeClustering(affinity=parameters['distance_metric'],
                                       linkage=parameters['linkage'],
                                       distance_threshold=parameters['distance_threshold'],
                                       n_clusters=parameters['n_clusters'])
    clusters.fit(matrix)

    return clusters


def build_cid2pidx(clusters: dict):

    # build a dict with keys as cluster ids and values as list of pair indexes belonging to that cluster
    cid2pidx = dict()

    for idx, cluster_id in enumerate(clusters['labels']):
        if cluster_id not in cid2pidx:
            cid2pidx[cluster_id] = [idx]
        else:
            cid2pidx[cluster_id].append(idx)

    return cid2pidx


def get_ranked_patterns(vector: np.ndarray, pattern_tracker: PatternTracker) -> Tuple[list, list, list]:

    patterns = list()
    indexes = list()
    counts = list()
    scores = list(vector)

    while len(patterns) < 10:
        highest_score_idx = scores.index(max(scores))
        counts.append(max(scores))
        scores[highest_score_idx] = 0.0

        pattern = pattern_tracker.patterns[highest_score_idx]
        patterns.append(pattern)
        indexes.append(highest_score_idx)

    return patterns, indexes, counts


def print_cluster_info(clusters: AgglomerativeClustering) -> None:

    print('\nNumber of clusters', clusters.n_clusters_)
    print('=' * 50)
