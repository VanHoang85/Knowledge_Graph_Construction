import stanza
from flair.models import MultiTagger

from feature_extraction import *
from entity_extraction import *
from evaluation import *


def load_models():

    # load tagger from flair
    ner_tagger = MultiTagger.load("hunflair")

    # load syntax analyzer, including dependency parser
    stanza.download('en', package='craft')
    analyzer = stanza.Pipeline('en', package='craft')

    return analyzer, ner_tagger


def ner_extract(sentence: str, ner_tagger) -> List[tuple]:

    sentence = flair.data.Sentence(sentence)
    ner_tagger.predict(sentence)  # predict NER tags

    # extract entities of current sentence
    # each entity has form of (start_idx , end_idx , tag)
    return extract_entity(sentence)


def pattern_extract(sent_entities: List[tuple], sentence: str, sent_parsed: Sentence,
                    entity_tracker: EntityTracker, pattern_tracker: PatternTracker, printing: bool) -> None:

    # get each pair
    # extract patterns / patterns
    # each feature is a set of tokens / strings
    for i in range(len(sent_entities) - 1):
        for j in range(i + 1, len(sent_entities)):
            pair = (sent_entities[i], sent_entities[j])

            if not nested_entities(pair):
                # key in lowercase
                key = (get_entity_text(pair[0], sentence), get_entity_text(pair[1], sentence))

                if printing:
                    print('The pair in focus', key)

                # if not same entities
                if key[0].lower() != key[1].lower():
                    patterns = extract_features(pair, sent_parsed, printing)

                    # dict of form { entity_pair : [patterns] } --> pairs as keys and list of patterns as values
                    # entity pair of tuple form ('ne1', 'ne2') --> e.g. ( 'Mouse', 'Fragile X Syndrome')
                    entity_tracker.update(key, pair[0][2], pair[1][2])
                    pattern_tracker.update(key, patterns)


def extraction(data: List[str], args) -> Tuple[EntityTracker, PatternTracker]:

    entity_tracker = EntityTracker()
    pattern_tracker = PatternTracker()

    analyzer, ner_tagger = load_models()

    # loop through each sentence and perform NER tagging
    # extract triple
    mark_print = args.mark_print
    num_line = 0
    for line in data:
        num_line += 1
        if num_line == 300 or num_line == 500 or num_line == 700 or num_line == 900:
            print('at line', num_line)

        # perform analysis, including tokenized, parsing
        sent_parsed = analyzer(line)

        # get tokenized sentence
        for sent_idx in range(len(sent_parsed.sentences)):

            sentence = ' '.join([token.text for token in sent_parsed.sentences[sent_idx].tokens])
            sent_entities = ner_extract(sentence, ner_tagger)

            printing = False
            if num_line == mark_print:
                printing = True

                print('=' * 50)
                print('\nOne sample of feature generation process')
                print('The sentence:', sentence)

            # extract patterns / patterns from the sentence if that sentence contains more than 2 entities
            if len(sent_entities) >= 2:
                pattern_extract(sent_entities, sentence, sent_parsed.sentences[sent_idx],
                                entity_tracker, pattern_tracker, printing)

    return entity_tracker, pattern_tracker


def print_entity_info(entity_tracker: EntityTracker) -> None:

    print('=' * 50)
    print('\nNumber of entities:', len(entity_tracker.entity2type))
    print('Number of types:', len(entity_tracker.type2entity))
    print('Number of entity pairs related to COVID:', len(entity_tracker.covid_pairs))
    # print('Ten covid pairs:', list(entity_tracker.covid_pairs)[:10])
    print('Number of pairs:', len(entity_tracker.entity_pairs))
    # print('Entities', entity_tracker.entity_pairs)
    print('Number of pairs with patterns, pair2idx:', len(entity_tracker.pair2idx))
    # print('Entities', entity_tracker.pair2idx.keys())

    print('\nPair occurrence top 10:', entity_tracker.occurrence_counter.most_common(10))
    print('='*50)


def print_pattern_info(pattern_tracker: PatternTracker) -> None:

    num_patterns = len(pattern_tracker.patterns)
    print('\nNumber of patterns:', num_patterns)
    print('\nSome patterns')
    print('...1st one:', pattern_tracker.patterns[0])
    print('...last one:', pattern_tracker.patterns[num_patterns-1])
    print('...somewhere in between:', pattern_tracker.patterns[num_patterns//2])
    print('=' * 50)
