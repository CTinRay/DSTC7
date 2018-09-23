import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    logging.info('loading embedding...')
    with open(config['model_parameters']['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
        config['model_parameters']['embeddings'] = embeddings.embeddings

    logging.info('loading test data...')
    with open(config['test'], 'rb') as f:
        test = pickle.load(f)
        test.padding = embeddings.to_index('</s>')
        test.n_positive = 0
        test.n_negative = 100
        test.context_padded_len = config['context_padded_len']
        test.option_padded_len = config['option_padded_len']
        test.min_context_len = 10000

    if config['arch'] == 'DualRNN':
        from dualrnn_predictor import DualRNNPredictor
        PredictorClass = DualRNNPredictor
    elif config['arch'] == 'HierRNN':
        from hierrnn_predictor import HierRNNPredictor
        PredictorClass = HierRNNPredictor
    elif config['arch'] == 'RecurrentTransformer':
        from recurrent_transformer_predictor import RTPredictor
        PredictorClass = RTPredictor

    predictor = PredictorClass(metrics=[],
                               **config['model_parameters'])

    model_path = os.path.join(
        args.model_dir,
        'model.pkl.{}'.format(args.epoch))

    # model_path = '/tmp/model.pkl'
    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)
    logging.info('predicting...')
    predicts = predictor.predict_dataset(test, test.collate_fn)

    outputs = []
    for predict, sample in zip(predicts, test):
        candidate_ranking = [
            {
                'candidate_id': oid,
                'confidence': score.item()
            }
            for score, oid in zip(predict, sample['option_ids'])
        ]
        candidate_ranking = sorted(candidate_ranking,
                                   key=lambda x: -x['confidence'])
        outputs.append({
            'example_id': sample['id'],
            'candidate-ranking': candidate_ranking
        })

    output_path = os.path.join(args.model_dir,
                               'predict-{}.json'.format(args.epoch))
    logging.info('Writing output to {}'.format(output_path))
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent='    ')


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
