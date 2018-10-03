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

    logging.info('loading dev data...')
    with open(config['model_parameters']['valid'], 'rb') as f:
        valid = pickle.load(f)
        valid.padding = embeddings.to_index('</s>')
        valid.n_positive = config['valid_n_positive']
        valid.n_negative = 0
        valid.context_padded_len = config['context_padded_len']
        valid.option_padded_len = config['option_padded_len']
        valid.min_context_len = 10000

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
         for utt, oid in zip(valid.candidates, valid.candidate_ids)],
        valid.padding)

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

        valid_loader = DataLoader(
            valid, shuffle=False,
            batch_size=config['model_parameters']['batch_size'],
            collate_fn=valid.collate_fn
            )
        for batch in tqdm(valid_loader):
            context_hidden = predictor.model.context_encoder(
                predictor.embeddings(
                    batch['context'].to(predictor.device)
                ),
                batch['utterance_ends']
            )
            predict_option = predictor.model.transform(context_hidden)
            opt_scores = predictor.model.similarity(
                predict_option.unsqueeze(1), # (b, 1
                cand_encs.unsqueeze(0)) # (1, n_c
            predicts.append(opt_scores)
            labels += batch['correct_candidate_index']

        predicts = torch.cat(predicts, 0)
        one_hot_labels = torch.zeros(len(labels), len(valid.candidates))
        one_hot_labels[list(range(len(labels))), labels] = 1
        accuracy = Accuracy()
        accuracy.update(predicts.cpu(), {'labels': one_hot_labels})
        print('{} = {}'.format(accuracy.name, accuracy.get_score()))
        embed()


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate task2")
    parser.add_argument('model_dir', type=str,
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

