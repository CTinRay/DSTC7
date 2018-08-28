import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
import torch

from tqdm import tqdm
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


def main_task2(args):
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

    logging.info('loading candidates...')
    if 'candidates' not in config:
        logging.error('please specify "candidates" in config.json!')
        return

    with open(config['candidates'], 'rb') as f:
        candidates = pickle.load(f)
        candidates.padding = embeddings.to_index('</s>')
        candidates.option_padded_len = config['option_padded_len']

    config.pop('candidates')

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

    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)

    options_path = os.path.join(args.model_dir, 'subtask2_options.pkl')
    if not os.path.exists(options_path):
        logging.info('encoding candidates...')
        options_hidden = predictor.predict_dataset(
            candidates,
            candidates.collate_fn,
            128,
            predictor._encode_option_batch)

        with open(options_path, 'wb') as f:
            pickle.dump(options_hidden, f)
    else:
        logging.info('loading encoded candidates...')
        with open(options_path, 'rb') as f:
            options_hidden = pickle.load(f)

    logging.info('predicting...')
    
    predict = predictor.predict_dataset(
        config['model_parameters']['valid'],
        config['model_parameters']['valid'].collate_fn,
        128,
        predict_fn=predictor._predict_batch_option_encoded,
        options_hidden=options_hidden)

    predict = predict.cpu()
    predict_path = os.path.join(args.model_dir, 'subtask2_predicts.pkl')
    with open(predict_path, 'wb') as f:
        pickle.dump(predict, f)

    id_to_index = dict()
    for i, cand in enumerate(candidates):
        id_to_index[cand['candidate_id']] = i

    for i, pred in tqdm(enumerate(predict)):
        data_point = config['model_parameters']['valid'][i]
        correct_id = data_point['option_ids'][0]
        idx = id_to_index[correct_id]

        predict[i] = torch.cat(
            (torch.tensor(pred[idx]).unsqueeze(0), pred[:idx], pred[idx+1:]), 0)

    m = Accuracy(ats=[1, 10, 100, 1000, 10000])
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
    parser.add_argument('--task2', action='store_true',
                        help='evaluate on subtask2')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        if args.task2:
            main_task2(args)
        else:
            main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
