# HU2MPA

Implementation of a multi-aspect and multi-granularity automatic pronunciation assessment model using HuPER-derived phoneme posterior features.

The model performs pronunciation assessment at three levels:

- phoneme-level accuracy
- word-level accuracy, stress, and total score
- utterance-level accuracy, completeness, fluency, prosody, and total score

## Model Overview

The model uses five types of phoneme-level information:

- GOP features
- energy features
- HuPER phoneme posterior features
- self-supervised speech representations
- relative-duration features

The main modules include:

1. canonical-phoneme-guided multi-source feature fusion
2. local phoneme sequence encoding
3. dynamic phoneme relationship modeling
4. intra-word phoneme attention
5. multi-aspect word-level prediction
6. score-guided utterance-level attentive pooling

## Project Structure

```text
.
├── data/
│   └── seq_data_librispeech/
├── exp/
├── figures/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── run.bat
│   └── traintest.py
├── extract_huper_features.py
├── requirements.txt
└── README.md
```

## Environment

The code was prepared with Python 3.10.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy==1.26.4
torch==2.3.1
torchaudio==2.3.1
transformers==4.41.2
tqdm==4.66.4
scipy==1.13.1
```

For CUDA 11.8, PyTorch can be installed with:

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

## Dataset

The experiments use the Speechocean762 pronunciation assessment corpus.

The original speech recordings are not distributed in this repository. After preparing the dataset and extracting the features, place the processed files in:

```text
data/seq_data_librispeech/
```

The expected files are:

```text
tr_feat.npy
te_feat.npy

tr_energy_feat.npy
te_energy_feat.npy

tr_huper_feat.npy
te_huper_feat.npy

tr_w2v_300m_feat_v2.npy
te_w2v_300m_feat_v2.npy

tr_dur_feat.npy
te_dur_feat.npy

tr_label_phn.npy
te_label_phn.npy

tr_label_word.npy
te_label_word.npy

tr_label_utt.npy
te_label_utt.npy

tr_word_id.npy
te_word_id.npy
```

The feature dimensions used in the full model are:

| Feature                 | Dimension |
| ----------------------- | --------: |
| GOP                     |        84 |
| Energy                  |         7 |
| HuPER posterior feature |         5 |
| SSL representation      |      1024 |
| Relative duration       |         1 |

The maximum phoneme sequence length is 50.

## HuPER Feature Extraction

HuPER phoneme posterior features can be extracted with:

```bash
python extract_huper_features.py \
    --audio-root PATH_TO_AUDIO \
    --phone-text PATH_TO_PHONE_TEXT \
    --train-order PATH_TO_TRAIN_ORDER \
    --test-order PATH_TO_TEST_ORDER \
    --output-dir data/seq_data_librispeech
```

On Windows:

```bat
python extract_huper_features.py ^
    --audio-root "PATH_TO_AUDIO" ^
    --phone-text "PATH_TO_PHONE_TEXT" ^
    --train-order "PATH_TO_TRAIN_ORDER" ^
    --test-order "PATH_TO_TEST_ORDER" ^
    --output-dir "data\seq_data_librispeech"
```

The script produces:

```text
tr_huper_feat.npy
te_huper_feat.npy
```

Each phoneme is represented by five posterior statistics:

1. canonical phoneme posterior probability
2. top-1 posterior probability
3. posterior margin
4. normalized posterior entropy
5. canonical phoneme match rate

The output shape of each dataset split is:

```text
[number of utterances, 50, 5]
```

## Training

The full model uses GOP, energy, HuPER, SSL, and relative-duration features.

Run one experiment with:

```bash
python src/traintest.py \
    --data-dir data/seq_data_librispeech \
    --exp-dir exp/full_model \
    --seed 0 \
    --epochs 100 \
    --batch-size 25 \
    --lr 1e-3 \
    --embed-dim 24 \
    --depth 3 \
    --num-heads 1
```

The main experimental settings are:

| Setting             |         Value |
| ------------------- | ------------: |
| Epochs              |           100 |
| Batch size          |            25 |
| Learning rate       |          1e-3 |
| Embedding dimension |            24 |
| Random seeds        | 0, 1, 2, 3, 4 |

## Five-Seed Experiment

On Windows, run:

```bat
src\run.bat
```

The script trains the model with five random seeds:

```text
0
1
2
3
4
```

The output directories are created under:

```text
exp/
```

For example:

```text
exp/full_model_duration_seed0
exp/full_model_duration_seed1
exp/full_model_duration_seed2
exp/full_model_duration_seed3
exp/full_model_duration_seed4
```

## Output

Each experiment directory contains the training results and the best model checkpoint.

A typical output structure is:

```text
exp/full_model_duration_seed0/
├── result.csv
├── models/
│   └── best_model.pth
└── predictions/
```

`result.csv` records phoneme-, word-, and utterance-level evaluation results for each epoch.

The evaluation metrics include:

- mean squared error for phoneme-level prediction
- Pearson correlation coefficient for phoneme-, word-, and utterance-level prediction

## Notes

- Padding phoneme positions use negative labels.
- Word-level scores are predicted at phoneme positions and aggregated according to word IDs during evaluation.
- Word- and utterance-level labels are normalized to the same score range as phoneme-level labels.
- Large feature files and model checkpoints are not tracked by Git.
- Experiment outputs are stored locally under the `exp` directory.