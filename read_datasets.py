import _pickle as cPickle
import bz2
import os
import rdflib
from rdflib import ConjunctiveGraph


def load_cido() -> ConjunctiveGraph:

    path_to_cido = 'https://raw.githubusercontent.com/CIDO-ontology/cido/master/src/ontology/cido.owl'
    graph = ConjunctiveGraph()
    graph.parse(path_to_cido, format=rdflib.util.guess_format(path_to_cido))

    return graph


def read_data(path_to_data_dir: str, filename: str, max_sent: int) -> None:

    # only append lines with 3 columns
    # if 2nd column is SENT --> sentence ==> better than <s> </s>
    # note: if 1st column starts with <citation --> anything in between NOT counted </citation>
    # same: <back_matter .... </back_matter>

    corpus = list()  # list of sentences
    tags = set()

    with open(os.path.join(path_to_data_dir, filename), 'r', encoding='utf-8') as file:

        # extra info = anything in between <citation> </citation> and </back_matter> </back_matter> tags
        extra_info = False
        sentence = list()

        for line in file:
            line = line.strip().split('\t')

            if len(line) == 1:
                tags.add(line[0].strip())

                if line[0].strip() == '<citation>' or line[0].strip() == '<back_matter>':
                    extra_info = True
                elif line[0].strip() == '</citation>' or line[0].strip() == '</back_matter>':
                    extra_info = False

            elif len(line) == 3 and not extra_info:

                sentence.append(line[0].strip())

                if line[1].strip() == 'SENT':
                    if 40 > len(sentence) > 5:
                        corpus.append(' '.join(sentence))  # add complete sent to corpus as string
                    sentence = []  # new sent

            if len(corpus) >= max_sent:
                break

    # write data to file
    write_data(tags, 'tags', path_to_data_dir)
    write_compressed_data(corpus, 'corpus', path_to_data_dir)
    
    print('Number of sentences in corpus', len(corpus))


def write_data(data, filename: str, path_to_data_dir: str):

    with open(os.path.join(path_to_data_dir, filename) + '.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(data))


def write_compressed_data(data, filename: str, path_to_data_dir: str):

    with bz2.BZ2File(os.path.join(path_to_data_dir, filename) + '.zipped', 'wb') as file:
        cPickle.dump(data, file)


def load_compressed_data(path_to_data_file: str):

    with bz2.BZ2File(path_to_data_file, 'rb') as file:
        data = cPickle.load(file)

    return data
