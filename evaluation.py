from typing import Tuple
from rdflib import URIRef, ConjunctiveGraph
import bcubed

from entity_extraction import EntityTracker
from read_datasets import load_cido


class CIDOTriple:

    def __init__(self):

        self.entities = set()
        self.all_entities = set()

        self.pairs = set()
        self.all_pairs = set()

        self.relations = set()
        self.all_relations = set()

        self.covid_pairs = set()
        self.all_covid_pairs = set()

        self.pair2relation = dict()

        # cido pairs which are same as pairs obtained from our data set
        self.identity_pairs = set()

    def add_entity(self, entity: str, add_all: bool):
        self.all_entities.add(entity) if add_all else self.entities.add(entity)

    def add_pair(self, pair: tuple, add_all: bool):
        self.all_pairs.add(pair) if add_all else self.pairs.add(pair)

    def add_relation(self, relation: str, add_all: bool):
        self.all_relations.add(relation) if add_all else self.relations.add(relation)

    def add_id_pair(self, pair: tuple):
        self.identity_pairs.add(pair)

    def update_pair_relation(self, pair: tuple, relation: str, add_all=False):

        self.add_entity(pair[0], add_all)
        self.add_entity(pair[1], add_all)
        self.add_pair(pair, add_all)
        self.add_relation(relation, add_all)

        if pair not in self.pair2relation:
            self.pair2relation[pair] = [relation]
        else:
            self.pair2relation[pair].append(relation)


def get_cido_triples(entity_tracker: EntityTracker) -> CIDOTriple:

    graph = load_cido()
    cido = CIDOTriple()

    for subj, pred, obj in graph:

        if type(subj) == type(pred) == type(obj) == URIRef:
            subj = get_value(subj, graph)
            pred = get_value(pred, graph)
            obj = get_value(obj, graph)

            if entity_exists(subj, entity_tracker) and entity_exists(obj, entity_tracker):
                cido.update_pair_relation((subj, obj), pred)

                if covid_term(subj) or covid_term(obj):
                    cido.covid_pairs.add((subj, obj))

                if pair_coexist((subj, obj), entity_tracker):
                    cido.add_id_pair((subj, obj))

            elif entity_exists(subj, entity_tracker):
                cido.add_entity(subj, add_all=False)

            elif entity_exists(obj, entity_tracker):
                cido.add_entity(obj, add_all=False)

            cido.update_pair_relation((subj, obj), pred, add_all=True)
            if covid_term(subj) or covid_term(obj):
                cido.all_covid_pairs.add((subj, obj))
    return cido


def get_value(node: URIRef, graph: ConjunctiveGraph) -> str:
    try:
        node = graph.label(node).value if graph.label(node) else graph.qname(node)
    except ValueError:
        node = node.title().lower()
    return str(node)


def print_cido_info(cido: CIDOTriple):

    print('=' * 50)
    print('Num entities', len(cido.entities))
    print(list(cido.entities))
    print('Num of all entities', len(cido.all_entities))
    print(list(cido.all_entities)[:30])

    print('\nNum pairs', len(cido.pairs))
    print(list(cido.pairs))
    print('Num of all pairs', len(cido.all_pairs))
    print(list(cido.all_pairs)[:30])
    print('Num of all covid pairs', len(cido.all_covid_pairs))
    print(list(cido.all_covid_pairs))

    print('\nNum of cido pairs which exist in our data', len(cido.identity_pairs))

    print('\nNum relations', len(cido.relations))
    print('Num of all relations', len(cido.all_relations))
    print('\nCido relations', list(cido.all_relations))


def entity_exists(entity: str, entity_tracker: EntityTracker) -> bool:
    return True if covid_term(entity) or entity.lower() in entity_tracker.entity2type else False


def covid_term(entity: str) -> bool:
    covid_terms = ['covid-19', 'covid', 'covid19', 'corona', 'coronavirus', 'sars-cov-2', 'coronaviruses']
    return True if entity.lower() in covid_terms else False


def pair_coexist(pair: tuple, entity_tracker: EntityTracker) -> bool:
    return True if pair in entity_tracker.entity_pairs else False


def build_eval_dicts(clusters: dict, cido: CIDOTriple,
                     entity_tracker: EntityTracker) -> Tuple[dict, dict]:

    # keys are items / pairs and values are sets of annotated categories for those items
    cdict = dict()
    gdict = dict()

    labels = clusters['labels']

    for pair in list(cido.identity_pairs):
        pair_idx = entity_tracker.pair2idx[pair]
        cluster_id = str(labels[pair_idx])

        # pair = pair[0] + '-' + pair[1]
        cdict[pair] = set(cluster_id)
        gdict[pair] = set(cido.pair2relation[pair])

    return cdict, gdict


def bcubed_scores(cdict: dict, gdict: dict) -> Tuple[float, float, float]:

    precision = bcubed.precision(cdict, gdict)
    recall = bcubed.recall(cdict, gdict)
    f1_score = bcubed.fscore(precision, recall)

    return precision, recall, f1_score
