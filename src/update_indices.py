import argparse
import pdb
import sys
import traceback
import logging
import json
import pickle
import os
import torch
from metrics import Accuracy, F1
from dataset import DSTC7CandidateDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython import embed


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    logging.info('loading embedding...')
    with open(config['model_parameters']['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
        config['model_parameters']['embeddings'] = embeddings.embeddings

    logging.info('loading train data...')
    with open(config['train'], 'rb') as f:
        train = pickle.load(f)
        train.padding = embeddings.to_index('</s>')
        train.n_positive = config['valid_n_positive']
        train.n_negative = 0
        train.context_padded_len = config['context_padded_len']
        train.option_padded_len = config['option_padded_len']
        train.min_context_len = 10000

    if config['arch'] == 'DualRNN':
        from dualrnn_predictor import DualRNNPredictor
        PredictorClass = DualRNNPredictor
    elif config['arch'] == 'HierRNN':
        from hierrnn_predictor import HierRNNPredictor
        PredictorClass = HierRNNPredictor
    elif config['arch'] == 'UttHierRNN' or config['arch'] == 'UttBinHierRNN':
        from hierrnn_predictor import UttHierRNNPredictor
        PredictorClass = UttHierRNNPredictor
        config['model_parameters']['model_type'] = config['arch']

    predictor = PredictorClass(**config['model_parameters'])
    model_path = os.path.join(args.model_dir, 'model.pkl.{}'
                              .format(args.epoch))
    logging.info('Loading model {}...'.format(model_path))
    predictor.load(model_path)

    # f1 = F1(threshold=config['f1_threshold'],
    #         max_selected=config['f1_max_selected'])

    candidates = DSTC7CandidateDataset(
        [{'utterance': utt,
          'option_ids': oid}
         for utt, oid in zip(train.candidates, train.candidate_ids)],
        train.padding)

    candidates = DataLoader(
        candidates, shuffle=False,
        batch_size=config['model_parameters']['batch_size'],
        collate_fn=candidates.collate_fn
    )

    logging.info('Predicting {}...'.format(model_path))
    predicts = []
    labels = []
    with torch.no_grad():
        cand_encs = []
        for batch in tqdm(candidates):
            cand_encs.append(
                predictor.model.utterance_encoder(
                    predictor.embeddings(
                        batch['options'].to(predictor.device)
                    ),
                    batch['option_lens']
                )
            )
        cand_encs = torch.cat(cand_encs, 0)

        batch_size = config['model_parameters']['batch_size']
        train_loader = DataLoader(
            train, shuffle=False,
            batch_size=batch_size,
            collate_fn=train.collate_fn
            )
        train.scores = []
        for b, batch in enumerate(tqdm(train_loader)):
            context_hidden = predictor.model.context_encoder(
                predictor.embeddings(
                    batch['context'].to(predictor.device)
                ),
                batch['utterance_ends']
            )
            predict_option = predictor.model.transform(context_hidden)
            opt_scores = predictor.model.similarity(
                predict_option.unsqueeze(1),
                cand_encs.unsqueeze(0))

            score, order = opt_scores.sort(-1)
            order = order[:, -200:].tolist()
            score = score[:, -200:].tolist()
            for i in range(len(order)):
                train.rand_indices[i + b * batch_size] = order[i][::-1]
                train.scores += score

    with open(args.output, 'wb') as f:
        pickle.dump(train, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="Update indices")
    parser.add_argument('model_dir', type=str,
                        help='')
    parser.add_argument('output', type=str,
                        help='')
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

