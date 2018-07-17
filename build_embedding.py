import argparse
import json
import logging
import os
import pickle
import spacy
import copy
import sys
import pdb
import traceback
from multiprocessing import Pool
from tqdm import tqdm
from embeddings import Embeddings


def collect_words(data_path, n_workers=16):
    logging.info('Loading valid data...')
    valid_path = os.path.join(data_path, 'ubuntu_dev_subtask1.json')
    with open(valid_path) as f:
        valid = json.load(f)
    logging.info('Tokenize words in valid...')
    valid = tokenize_data_parallel(valid, args.n_workers)

    logging.info('Loading train data...')
    train_path = os.path.join(data_path, 'ubuntu_train_subtask1.json')
    with open(train_path) as f:
        train = json.load(f)
    logging.info('Tokenize words in train...')
    train = tokenize_data_parallel(train, args.n_workers)

    logging.info('Building word list...')
    words = set()
    data = train + valid
    for sample in tqdm(data):
        utterances = (
            [message['utterance']
             for message in sample['messages-so-far']]
            + [option['utterance']
               for option in sample['options-for-correct-answers']]
            + [option['utterance']
               for option in sample['options-for-next']]
        )

        for utterance in utterances:
            for word in utterance:
                words.add(word)

    return words


def tokenize_data_parallel(data, n_workers=16):
    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(data) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(data) - 1
            else:
                batch_end = (len(data) // n_workers) * (i + 1)

            batch = data[batch_start: batch_end]
            results[i] = pool.apply_async(tokenize_data, [batch])

        pool.close()
        pool.join()

    data = []
    for result in results:
        data += result.get()

    return data


def tokenize_data(data):
    nlp = spacy.load('en_core_web_sm',
                     disable=['tagger', 'ner', 'parser', 'textcat'])

    def tokenize(text):
        return [token.text
                for token in nlp(text)]

    data = copy.deepcopy(data)
    for sample in data:
        for i, message in enumerate(sample['messages-so-far']):
            sample['messages-so-far'][i]['utterance'] = \
                tokenize(message['utterance'])
        for i, message in enumerate(sample['options-for-correct-answers']):
            sample['options-for-correct-answers'][i]['utterance'] = \
                tokenize(message['utterance'])
        for i, message in enumerate(sample['options-for-next']):
            sample['options-for-next'][i]['utterance'] = \
                tokenize(message['utterance'])

    return data


def main(args):
    word_list = collect_words(args.data_path)
    embeddings = Embeddings(args.embedding_path, word_list)
    logging.info('len of embedding.word_dict = {}, len of word_list = {}'
                 .format(len(embeddings.word_dict), len(word_list)))
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('data_path', type=str,
                        help='Path to the data.')
    parser.add_argument('embedding_path', type=str,
                        help='Path to the embedding.')
    parser.add_argument('output', type=str,
                        help='Path to the output pickle file.')
    parser.add_argument('--valid_ratio', type=float, default=0.2,
                        help='Ratio of data used as validation set.')
    parser.add_argument('--index', type=str, default=None,
                        help='JSON file that stores shuffled index.')
    parser.add_argument('--n_workers', type=int, default=16)
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
