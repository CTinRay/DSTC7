import argparse
import copy
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from dualrnn_predictor import DualRNNPredictor


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    with open(config['model_parameters']['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
        embeddings.add('speaker1')
        embeddings.add('speaker2')
        config['model_parameters']['embeddings'] = embeddings.embeddings

    with open(config['model_parameters']['valid'], 'rb') as f:
        config['model_parameters']['valid'] = pickle.load(f)
        config['model_parameters']['valid'].padding = embeddings.to_index('</s>')
        config['model_parameters']['valid'].n_negative = 100

    with open(config['train'], 'rb') as f:
        train = pickle.load(f)
        train.n_positive = 1
        train.n_negative = 4
        train.padding = embeddings.to_index('</s>')

    predictor = DualRNNPredictor(**config['model_parameters'])

    model_checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, 'model.pkl'),
        'loss', 1, 'all'
    )
    metrics_logger = MetricsLogger(
        os.path.join(args.model_dir, 'model.pkl')
    )

    predictor.fit_dataset(train,
                          train.collate_fn,
                          [model_checkpoint, metrics_logger])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
