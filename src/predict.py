import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import Accuracy
from dualrnn_predictor import DualRNNPredictor
from IPython import embed


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    logging.info('loading embedding...')
    with open(config['model_parameters']['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
        config['model_parameters']['embeddings'] = embeddings.embeddings

    logging.info('loading dev data...')
    with open(config['model_parameters']['valid'], 'rb') as f:
        config['model_parameters']['valid'] = pickle.load(f)
        config['model_parameters']['valid'].padding = \
            embeddings.to_index('</s>')
        config['model_parameters']['valid'].n_positive = \
            config['valid_n_positive']
        config['model_parameters']['valid'].n_negative = \
            config['valid_n_negative']
        config['model_parameters']['valid'].context_padded_len = \
            config['context_padded_len']
        config['model_parameters']['valid'].option_padded_len = \
            config['option_padded_len']
        config['model_parameters']['valid'].min_context_len = 10000

    if config['arch'] == 'DualRNN':
        from dualrnn_predictor import DualRNNPredictor
        PredictorClass = DualRNNPredictor
    elif config['arch'] == 'HierRNN':
        from hierrnn_predictor import HierRNNPredictor
        PredictorClass = HierRNNPredictor
    elif config['arch'] == 'RecurrentTransformer':
        from recurrent_transformer_predictor import RTPredictor
        PredictorClass = RTPredictor

    predictor = PredictorClass(metrics=[Accuracy()],
                               **config['model_parameters'])

    model_path = os.path.join(
        args.model_dir,
        'model.pkl.{}'.format(args.epoch))

    # model_path = '/tmp/model.pkl'
    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)
    logging.info('predicting...')
    predict = predictor.predict_dataset(
        config['model_parameters']['valid'],
        config['model_parameters']['valid'].collate_fn)

    m = Accuracy()
    m.update(predict, 0)
    print(m.get_score())
    embed()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--epoch', type=int, default=10)
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
