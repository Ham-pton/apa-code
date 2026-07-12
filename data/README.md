# Data

This directory contains the processed features used by the pronunciation assessment model.

## Provided Files

The repository provides the extracted HuPER phoneme-posterior features:

```text
seq_data_librispeech/
├── tr_huper_feat.npy
└── te_huper_feat.npy
```

The two files correspond to the training and test splits of Speechocean762.

Their shapes are:

```text
tr_huper_feat.npy: [2500, 50, 5]
te_huper_feat.npy: [2500, 50, 5]
```

Each phoneme is represented by five posterior statistics:

1. canonical-phoneme posterior probability
2. top-1 posterior probability
3. posterior margin
4. normalized posterior entropy
5. canonical-phoneme match rate

Padding positions are filled with `-1`.

## Full Data Structure

To train the full model, place the remaining processed features and labels in:

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

The original speech recordings and the remaining processed features are not included in this repository.

HuPER features can be regenerated with:

```bash
python extract_huper_features.py --help
```