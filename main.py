import sys
import argparse
from random import randrange, shuffle
import graphviz

from read_datasets import *
from extraction import *
from clustering import *
from evaluation import *


def run_read_corpus():
    read_data(args.path_to_data_dir, args.corpus_name, args.max_sent)


def run_extraction():

    data = load_compressed_data(corpus_path)
    entity_tracker, pattern_tracker = extraction(data, args)
    matrix = pair_pattern_matrix(pattern_tracker, entity_tracker)

    print_entity_info(entity_tracker)
    print_pattern_info(pattern_tracker)

    write_compressed_data([entity_tracker, pattern_tracker, matrix], 'trackers', args.path_to_data_dir)


def run_clustering():

    _, _, matrix = load_compressed_data(trackers_path)
    clustering_parameters = {'distance_metric': args.distance_metric,
                             'linkage': args.linkage,
                             'distance_threshold': args.distance_threshold,
                             'n_clusters': None}
    clusters = clustering(matrix, clustering_parameters)
    cp_matrix = cluster_pattern_matrix(clusters, matrix, args)

    print_cluster_info(clusters)

    # save clusters as obj
    cluster_info = dict({'n_clusters': clusters.n_clusters_, 'labels': clusters.labels_})
    write_compressed_data([cluster_info, cp_matrix], 'clusters', args.path_to_data_dir)


def run_evaluation():

    clusters, cp_matrix = load_compressed_data(cluster_path)
    entity_tracker, pattern_tracker, _ = load_compressed_data(trackers_path)

    cido = get_cido_triples(entity_tracker)
    cid2pidx = build_cid2pidx(clusters)

    print_cido_info(cido)

    write_compressed_data(cido, 'cido', args.path_to_data_dir)

    if len(cido.identity_pairs) > 0:
        cluster_dict, cido_dict = build_eval_dicts(clusters, cido, entity_tracker)
        scores = bcubed_scores(cluster_dict, cido_dict)

        print('='*50)
        print('Precision, Recall, and F1 scores respectively: {}, {}, {}'.format(scores[0], scores[1], scores[2]))
    else:
        print('No automatic evaluation as no matching pair between CIDO and our dataset')

    print('='*50)
    print('Manually checking cluster quality...')

    if len(cido.identity_pairs) > 0:
        print('Check quality of pairs existing in both cido and our data')

        for pair in cido.identity_pairs:
            cid = clusters['labels'][entity_tracker.pair2idx[pair]]
            patterns, indexes, counts = get_ranked_patterns(cp_matrix[cid], pattern_tracker)

            print('Cluster id', cid)
            print('The pair:', pair)
            print('CIDO relation', cido.pair2relation[pair])
            print('Top 10 patterns found')
            print(indexes)
            print(counts)
            print(patterns)

    print('='*50)
    print('Draw randomly 30 clusters, their members / pairs and patterns belonging to the cluster....')
    print('Print only clusters having from 2 pairs....')

    valid_pairs = 0
    for cid in range(clusters['n_clusters']):
        # get list of pairs belonging to that cluster id
        pairs = [entity_tracker.idx2pair[idx] for idx in cid2pidx[cid]]

        if len(pairs) >= 2:
            patterns, indexes, counts = get_ranked_patterns(cp_matrix[cid], pattern_tracker)
            valid_pairs += 1

            print('\nCluster id', cid)
            print('Pairs belonging to the cluster...')
            print(pairs)

            print('\nTop 10 ranked patterns belonging to the clusters')
            print(indexes)
            print(counts)
            print(patterns)
            print('-'*30)
    print('Number of clusters having more than 2 pairs:', valid_pairs)


def run_visualization():

    entity_tracker, pattern_tracker, _ = load_compressed_data(trackers_path)

    if args.with_data == 'cido':
        cido = load_compressed_data(cido_path)
        visual_cido(cido)

    # Draw a graph from our dataset
    print('Randomly pick pairs from dataset and draw a graph...')
    graph = graphviz.Digraph(format='png', node_attr={'color': 'lightblue2', 'style': 'filled'})

    # get entity pairs with suitable type
    pairs = list(entity_tracker.covid_pairs)
    shuffle(pairs)

    num_nodes = 0
    for pair in pairs:
        if len(pair[0].split()) <= 2 and len(pair[1].split()) <= 2:
            graph.edge(pair[0], pair[1])
            num_nodes += 1

        if num_nodes >= args.num_nodes:
            break
    graph.view()


def visual_cido(cido):

    # Draw a graph from CIDO
    cido_graph = graphviz.Digraph(format='png', node_attr={'color': 'lightblue2', 'style': 'filled'})
    cido_nodes = set()

    while len(cido_nodes) < args.num_nodes:
        pair = list(cido.all_pairs)[randrange(len(cido.all_pairs))]
        cido_graph.edge(str(pair[0]), str(pair[1]), label=str(cido.pair2relation[pair]))
        cido_nodes.update(pair)

    print('Randomly pick pairs from CIDO and draw a graph...')
    cido_graph.view()
    sys.exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Project for Knowledge Discovery course \nKnowledge Graph Construction')
    parser.add_argument('--perform', type=str, default='extract', const='extract', nargs='?',
                        choices=['read-corpus', 'extract', 'cluster', 'evaluate', 'visual', 'all'],
                        help='Six choices: read-corpus, extract, cluster, evaluate, visual, all')
    parser.add_argument('--path_to_data_dir', type=str, default=os.path.join(os.getcwd(), 'data'))
    parser.add_argument('--corpus_name', type=str, default='covid19.vert')
    parser.add_argument('--max_sent', type=int, default=1000)
    parser.add_argument('--mark_print', type=int, default=None,
                        help='Print out features for the chosen sentence')
    parser.add_argument('--distance_metric', type=str, default='cosine', const='cosine', nargs='?',
                        choices=['cosine', 'euclidean', 'manhattan'])
    parser.add_argument('--linkage', type=str, default='average', const='average', nargs='?',
                        choices=['average', 'single', 'complete', 'ward'])
    parser.add_argument('--distance_threshold', type=float, default=0.999)
    parser.add_argument('--ranked_metric', type=str, default='count', const='count', nargs='?',
                        choices=['count', 'tfidf'])
    parser.add_argument('--with_data', type=str, default='ours', const='ours', nargs='?',
                        choices=['ours', 'cido'])
    parser.add_argument('--num_nodes', type=int, default=30,
                        help='Number of nodes to draw a graph')
    args = parser.parse_args()

    if args.perform == 'read-corpus':
        run_read_corpus() if os.path.isfile(os.path.join(args.path_to_data_dir, args.corpus_name)) \
            else print('Please give valid path and/or filename')

    elif args.perform == 'extract':
        corpus_path = os.path.join(args.path_to_data_dir, 'corpus.zipped')

        if os.path.isfile(corpus_path):
            run_extraction()
        else:
            print('Please run read-corpus first to get the zipped file of data!')
            sys.exit()

    elif args.perform == 'cluster':
        trackers_path = os.path.join(args.path_to_data_dir, 'trackers.zipped')
        run_clustering()

    elif args.perform == 'evaluate':
        cluster_path = os.path.join(args.path_to_data_dir, 'clusters.zipped')
        trackers_path = os.path.join(args.path_to_data_dir, 'trackers.zipped')

        if os.path.isfile(cluster_path):
            run_evaluation()
        else:
            print('Please get clusters and trackers files first!!!')
            sys.exit()

    elif args.perform == 'visual':
        trackers_path = os.path.join(args.path_to_data_dir, 'trackers.zipped')
        cido_path = os.path.join(args.path_to_data_dir, 'cido.zipped')

        if os.path.isfile(trackers_path):
            run_visualization()
        else:
            print('Please get clusters and trackers files first!!!')
            sys.exit()

    elif args.perform == 'all':
        run_read_corpus() if os.path.isfile(os.path.join(args.path_to_data_dir, args.corpus_name)) \
            else print('Please give valid path and/or filename')

        corpus_path = os.path.join(args.path_to_data_dir, 'corpus.zipped')
        cluster_path = os.path.join(args.path_to_data_dir, 'clusters.zipped')
        trackers_path = os.path.join(args.path_to_data_dir, 'trackers.zipped')
        cido_path = os.path.join(args.path_to_data_dir, 'cido.zipped')

        run_extraction()
        run_clustering()
        run_evaluation()
        run_visualization()
    else:
        print('Give me some proper command please.....')
