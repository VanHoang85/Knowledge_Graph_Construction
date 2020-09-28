from typing import Set, List, Tuple
from itertools import chain, combinations
import networkx as nx
from stanza.models.common.doc import Sentence, Word


class PatternTracker:

    def __init__(self):

        # dict of form { entity_pair : [patterns] } --> pairs as keys and list of patterns / patterns as values
        self.pairs2patterns = dict()
        self.patterns = list()

    def update(self, key: Tuple[str, str], patterns: List[Set[str]]):
        self.add_pair2pattern(key, patterns)
        self.add_pattern(patterns)

    def add_pair2pattern(self, key: Tuple[str, str], patterns: List[Set[str]]):

        if key in self.pairs2patterns:
            self.pairs2patterns[key].extend(patterns)
        else:
            self.pairs2patterns[key] = patterns

    def add_pattern(self, patterns: List[Set[str]]):

        if patterns:
            for pattern in patterns:
                if pattern not in self.patterns:
                    self.patterns.append(pattern)

    def get_pair_with_no_patterns(self) -> List[Tuple[str, str]]:

        pairs = list()

        for pair, patterns in self.pairs2patterns.items():
            if not patterns:
                pairs.append(pair)

        return pairs


def extract_features(pair: Tuple[tuple, tuple], dep_path: Sentence, printing: bool) -> List[Set[str]]:

    features = list()

    # get sets of core and extra tokens from dependency path
    # each token has form: lemma-upos
    core_tokens, extra_tokens = get_feature_tokens(pair, dep_path)

    # check validity of all core and extra tokens
    # if not pass, meaning no features generated from these two sets are valid
    if check_feature_validity(core_tokens.union(extra_tokens)):

        # check validity of core tokens and add accordingly
        if check_feature_validity(core_tokens):

            # add core tokens with pos to feature list
            features.append(core_tokens)

            # add core tokens w/o pos to feature list
            sdp = remove_tag_tail(core_tokens)
            features.append(sdp)

            # get set of two NEs with position (e.g. Disease1) and add to list
            sdp_ne = {pair[0][2] + '1', pair[1][2] + '2'}
            features.append(sdp.union(sdp_ne))

        # make power set of extra tokens
        powerset = get_power_set(extra_tokens)

        # loop through each item in power set, get union with core tokens
        # check feature validity
        # get rid of pos tags and append to list if TRUE
        for extra_set in powerset:
            feature = core_tokens.union(set(extra_set))

            if check_feature_validity(feature):
                features.append(remove_tag_tail(feature))

    # printing example if True
    if printing:
        print('Dependency path', dep_path.print_dependencies())
        print('Core tokens', core_tokens)
        print('Extra tokens', extra_tokens)

        if len(features) <= 10:
            print('Features', features)
        print('-' * 30)

    return features


def check_feature_validity(feature: Set[str]) -> bool:

    closed_class = {'ADP', 'CCONJ', 'DET', 'AUX', 'NUM', 'PART', 'PRON', 'SCONJ' 'PUNCT', 'SYM', 'X'}

    # get set of POS tags in each token
    pos = {token.split('-')[1] for token in feature}

    # get the difference of set of pos with set of closed class
    # if difference empty, meaning pos set is a subset of closed class
    # if difference NOT empty, there exists at least one token in the feature with open word class
    difference = pos - closed_class

    return True if difference and len(feature) <= 10 else False


def remove_tag_tail(tokens: Set[str]) -> Set[str]:
    return {token.rsplit('-', 1)[0] for token in tokens}


def get_feature_tokens(pair: tuple, dep_path: Sentence) -> Tuple[Set[str], Set[str]]:

    # words in the dep path
    words = dep_path.words

    # each edge has form ( head word-upos-id , dep word-upos-id ) --> list of tuples
    # each deprel has form [ head word-upos-id , dep word-upos-id , deprel ] --> list of lists
    edges, deprels = get_edges(words)

    # build the graph
    graph = nx.Graph(edges)

    # look up source and target entities
    # pair[0] --> entity 1
    # pair[0][1] --> end token of the entity
    source = get_node_form(words, pair[0][1]-1)
    target = get_node_form(words, pair[1][1]-1)

    # get the shortest path --> list of tokens in the SDP
    sdp = nx.shortest_path(graph, source=source, target=target)
    sdp = set(sdp)

    # from shortest dependency path (SDP), get optional tokens
    extra_tokens = get_extra_tokens(deprels, sdp)

    # remove two entities, aka source and target tokens in sdp before return
    try:
        sdp.remove(source)
        sdp.remove(target)
    except KeyError:
        pass

    # get rid of -id attachment in token string
    # sets of tokens
    core_tokens = remove_tag_tail(sdp)
    extra_tokens = remove_tag_tail(extra_tokens)

    return core_tokens, extra_tokens


def get_node_form(words: List[Word], idx: int) -> str:

    return words[idx].lemma + '-' + words[idx].upos + '-' + str(words[idx].id)


def get_extra_tokens(deprels: List[list], sdp: Set[str]) -> Set[str]:

    relations = ['compound', 'case', 'nsubj', 'acl', 'nmod']
    extra_tokens = set()

    for deprel in deprels:
        if deprel[2] in relations:
            if deprel[0] in sdp:
                extra_tokens.add(deprel[1])
            elif deprel[1] in sdp:
                extra_tokens.add(deprel[0])

    # remove any tokens already in sdp before return
    return extra_tokens - set(sdp)


def get_edges(words: List[Word]) -> Tuple[List[tuple], List[list]]:

    edges = list()
    deprels = list()

    for word in words:
        head = words[word.head - 1].lemma if word.head > 0 else 'root'
        head_upos = words[word.head - 1].upos if word.head > 0 else 'root'

        edge = (head + '-' + head_upos + '-' + str(word.head), word.lemma + '-' + word.upos + '-' + str(word.id))
        edges.append(edge)

        dep = [edge[0], edge[1], word.deprel]
        deprels.append(dep)

    return edges, deprels


def get_power_set(extra_tokens: Set[str]) -> List[Tuple[str]]:

    extra_tokens = list(extra_tokens)
    return list(chain.from_iterable(combinations(extra_tokens, r) for r in range(1, len(extra_tokens) + 1)))
