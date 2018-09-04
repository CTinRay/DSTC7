import argparse
import pdb
import sys
import traceback
import re
import json
from tqdm import tqdm


def main(args):
    with open(args.train) as f:
        train = json.load(f)

    with open(args.valid) as f:
        valid = json.load(f)

    courses = []
    for sample in train + valid:
        courses += [c['offering'].split('-')[0]
                    for c in (sample['profile']['Courses']['Prior']
                              + sample['profile']['Courses']['Suggested'])]

    courses = list(set(courses))

    ppprocess(valid, courses)
    with open(args.valid_output, 'w') as f:
        json.dump(valid, f, indent='    ')

    ppprocess(train, courses)
    with open(args.train_output, 'w') as f:
        json.dump(train, f, indent='    ')


def ppprocess(dataset, courses):
    colleges = [re.sub('([A-Z]*)([0-9]*)', r'\1', c)
                for c in courses]
    spaced_courses = [re.sub('([A-Z]*)([0-9]*)', r'\1 \2', c).lower()
                      for c in courses]
    course_numbers = [re.sub('([A-Z]*)([0-9]*)', r' \2', c)
                      for c in courses]
    for sample in tqdm(dataset):
        for msg in (sample['messages-so-far']
                    + sample['options-for-next']
                    + sample['options-for-correct-answers']):
            for course, spaced in zip(courses, spaced_courses):
                msg['utterance'] = msg['utterance'] \
                    .lower() \
                    .replace(spaced, course)

        sample_courses = \
            [c['offering'].split('-')[0]
             for c in (sample['profile']['Courses']['Prior']
                       + sample['profile']['Courses']['Suggested'])]

        sample_numbers = [re.sub('([A-Z]*)([0-9]*)', r' \2', c)
                          for c in sample_courses]
        sample_colleges = [re.sub('([A-Z]*)([0-9]*)', r'\1', c)
                           for c in sample_courses]

        for msg in (sample['options-for-correct-answers']
                    + sample['options-for-next']
                    + sample['messages-so-far']):
            for course, number in zip(sample_courses, sample_numbers):
                msg['utterance'] = (' ' + msg['utterance']) \
                    .lower() \
                    .replace(number, ' ' + course)[1:]

            for course, college, number in zip(courses,
                                               colleges,
                                               course_numbers):
                if college not in sample_colleges:
                    continue

                msg['utterance'] = (' ' + msg['utterance']) \
                    .lower() \
                    .replace(number, ' ' + course)[1:]


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('train', type=str,
                        help='')
    parser.add_argument('valid', type=str,
                        help='')
    parser.add_argument('train_output', type=str,
                        help='')
    parser.add_argument('valid_output', type=str,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
