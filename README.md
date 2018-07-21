# DSTC7

## Usage

1. Build embeddings pickle. Example (It may take about 10 mins on a 16-cores machine):
```
python3 build_embedding.py ../data/task1/advising_train_subtask1.json ../data/task1/advising_dev_subtask1.json ../data/crawl-300d-2M.vec ../data/task1/advising_embeddings.pkl
```

If FastText `crawl-300d-2M` is used (and there is no bug in code), then OOV in Ubuntu is about 4% and 1.8% in Advising dataset.

2. Make training pickle and validation data pickle. Example (It may take about another 10 mins on a 16-cores machine):
```
python3 make_train_valid_dataset.py ../data/task1/advising_embeddings.pkl ../data/task1/advising_train_subtask1.json ../data/task1/advising_dev_subtask1.json ../data/task1/advising_train.pkl ../data/task1/advising_valid.pkl
```

3. Create a configuration file `config.json`, and put it under the directory you want to save the model (e.g. `../models/dual-rnn/config.json`). An example can be found in `src/config.json`.

4. Start training. Example
```
python3 train.py ../models/dual-rnn
```

## Current Result

### Bi-Directional DualRNN

- hidden state dimension: 128

| Dataset | Accuracy@1 | Accuracy@5 | Accuracy@10 | Accuracy@50 | epoch |
| --- | --- | --- | --- | --- |
| Ubuntu   | 0.1130226 | 0.4626962 | 0.6051210 | 0.941588 | 10 |
| Advising | 0.0200400 | 0.1302605 | 0.2565130 | 0.739478 | 11 |
