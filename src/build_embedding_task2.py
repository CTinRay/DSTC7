import argparse
import json
import logging
import os
import pickle
import spacy
import copy
import sys
import pdb
# from IPython import embed
import traceback
import spacy
from multiprocessing import Pool
from tqdm import tqdm
from embeddings import Embeddings
from build_embedding import collect_words, oov_statistics


def main(args):
    logging.info('Collecting words from train and valid...')
    words = collect_words(args.train_path, args.valid_path)

    logging.info('Collecting words from candidate pool')
    expand_words_from_candidates(args.candidate_path, words)

    logging.info('Building embeddings...')
    embeddings = Embeddings(args.embedding_path, list(words.keys()))

    embeddings.add('speaker1')
    embeddings.add('speaker2')

    logging.info('Saving embedding to {}'.format(args.output))
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)

    if args.words is not None:
        with open(args.words, 'wb') as f:
            pickle.dump(words, f)

    logging.info('Calculating OOV statics...')
    oov, cum_sum = oov_statistics(words, embeddings.word_dict)
    logging.info('There are {} OOVS'.format(cum_sum[-1]))

    # embed()


def expand_words_from_candidates(candidate_path, words):
    nlp = spacy.load('en_core_web_sm',
                     disable=['tagger', 'ner', 'parser', 'textcat'])

    def tokenize(text):
        return [token.text
                for token in nlp(text)]

    candidates = []
    with open(candidate_path) as f:
        for line in f:
            candidates.append(line.split('\t')[1].strip())

    candidates = [tokenize(c) for c in tqdm(candidates)]

    for cand in candidates:
        for w in cand:
            if w not in words:
                words[w] = 0

            words[w] += 1


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Build embedding pickle by extracting vector from'
                    ' pretrained embeddings only for words in the data.')
    parser.add_argument('train_path', type=str,
                        help='[input] Path to the train data.')
    parser.add_argument('valid_path', type=str,
                        help='[input] Path to the dev data.')
    parser.add_argument('candidate_path', type=str,
                        help='[input] Path to the candidate pool.')
    parser.add_argument('embedding_path', type=str,
                        help='[input] Path to the embedding .vec file (such as'
                             'FastText or Glove).')
    parser.add_argument('output', type=str,
                        help='[output] Path to the output pickle file.')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--words', type=str, default=None,
                        help='If a path is specified, list of words in the'
                             'data will be dumped.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
