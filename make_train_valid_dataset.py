import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from preprocessor import Preprocessor


def main(args):

    embeddings_path = os.path.join(args.data_dir,
                                   'ubuntu_embeddings.pkl')
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    preprocessor = Preprocessor(embeddings)

    train = preprocessor.get_dataset(
        os.path.join(args.data_dir, 'ubuntu_train_subtask1.json')
    )
    train_path = os.path.join(args.output_dir, 'ubuntu_train.pkl')
    with open(train_path, 'wb') as f:
        pickle.dump(train, f)

    valid = preprocessor.get_dataset(
        os.path.join(args.data_dir, 'ubuntu_dev_subtask1.json')
    )
    valid_path = os.path.join(args.output_dir, 'ubuntu_valid.pkl')
    with open(valid_path, 'wb') as f:
        pickle.dump(valid, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('data_dir', type=str,
                        help='Path to the data.')
    parser.add_argument('output_dir', type=str,
                        help='Path to the output pickle file.')
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
