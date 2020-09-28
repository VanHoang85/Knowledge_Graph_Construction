from typing import List, Tuple
from collections import Counter
import flair


class EntityTracker:

    def __init__(self):

        self.entity_pairs = set()  # set of all entity pairs in the data
        self.covid_pairs = set()  # set of all entity pairs related to covid
        self.occurrence_counter = Counter()

        # pair idx in the pair-pattern matrix
        self.pair2idx = dict()
        self.idx2pair = dict()

        self.entity2type = dict()
        self.type2entity = dict()

    def update(self, pair: Tuple[str, str], type1: str, type2: str):
        self.add_pair(pair)
        self.add_entity_type(pair[0], type1.upper())
        self.add_entity_type(pair[1], type2.upper())

        if type1 == 'COVID' or type2 == 'COVID':
            self.covid_pairs.add(pair)

    def add_pair(self, pair_tuple: Tuple[str, str]):
        self.entity_pairs.add(pair_tuple)
        self.occurrence_counter.update([pair_tuple])

    def add_pair_idx(self, pair: Tuple[str, str]):
        self.pair2idx[pair] = len(self.pair2idx)
        self.idx2pair[len(self.idx2pair)] = pair

    def add_entity_type(self, ne: str, ne_type: str):

        if ne not in self.entity2type:
            self.entity2type[ne] = [ne_type]
        else:
            self.entity2type[ne].append(ne_type)

        if ne_type not in self.type2entity:
            self.type2entity[ne_type] = [ne]
        else:
            self.type2entity[ne_type].append(ne)


def extract_entity(sentence: flair.data.Sentence) -> List[tuple]:

    entities = list()

    for entity in sentence.get_spans():

        # each entity has printing form:
        # Span [10,11,12]: "Fragile X Syndrome"   [− Labels: Disease (0.99)]
        # where span index indicating Token number, starting from 1
        span = [token.idx for token in entity.tokens]  # get span, which can be more than 2 numbers

        # check if start and last token in span is covid terms
        # remove if yes
        if covid_terms(span[0]-1, sentence.to_original_text().split(' ')):
            span.remove(span[0])

        if covid_terms(span[-1]-1, sentence.to_original_text().split(' ')):
            span.remove(span[-1])

        if span:
            entities.append((span[0], span[-1], entity.tag))  # add a tuple of entity span and tag

    # extend the current entity list with covid related NE if any
    entities.extend(get_covid_entity(sentence.to_original_text()))

    # sort the entity list according to their idx
    entities = sort_entities(entities)

    return entities


def covid_terms(token_idx: int, sentence: List[str]) -> bool:

    terms = ['covid-19', 'covid', 'covid19', 'corona', 'coronavirus', 'sars-cov-2', 'coronaviruses']
    return True if sentence[token_idx].lower() in terms else False


def get_covid_entity(sentence: str) -> List[tuple]:

    entity_list = list()

    # loop through the whole sentence and discover any covid related NE
    sentence = sentence.split(' ')
    for idx in range(len(sentence)):
        if covid_terms(idx, sentence):
            entity_list.append((idx+1, idx+1, 'COVID'))

    return entity_list


def sort_entities(entity_list: List[tuple]) -> List[tuple]:

    # each item in entity list has the form of a tuple (start_token, end_token, 'tag')
    # e.g. [ (1, 1, 'Disease'), (9, 11, 'Disease'), (6, 6, 'Species'), (4, 4, 'Gene') ]

    ordered_list = list()

    # loop through the list and find the item with smallest start idx
    while len(entity_list) > 0:
        smallest_idx = 0

        for idx in range(1, len(entity_list)):
            if entity_list[idx][0] < entity_list[smallest_idx][0]:
                smallest_idx = idx

        # pop the entity with smallest value/idx and append it to the ordered list
        ordered_list.append(entity_list.pop(smallest_idx))

    return ordered_list


def get_entity_text(entity_tuple: tuple, sentence: str) -> str:
    """
    From a tuple of an entity (start_token, end_token, tag), retrieve the entity text in the sentence
    Token idx starts from 1

    :param entity_tuple: an entity tuple
    :param sentence: the sentence containing the entity
    :return: the entity text WITHOUT the tag
    """

    sentence = sentence.lower().split(' ')
    return ' '.join(sentence[entity_tuple[0]-1:entity_tuple[1]]).lower().strip().replace(' : ', ':').replace(' - ', '-')


def nested_entities(pair: Tuple[tuple, tuple]) -> bool:

    start_ne1 = pair[0][0]
    end_ne1 = pair[0][1]
    start_ne2 = pair[1][0]
    end_ne2 = pair[1][1]

    if start_ne1 <= start_ne2 <= end_ne1:
        return True
    elif start_ne2 <= start_ne1 <= end_ne2:
        return True
    else:
        return False
