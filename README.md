# Mapping Political Ideologies on Social Media Using Text Mining and Deep Learning

## Overview
This repository contains the code, datasets, and supplementary materials for the paper
"Mapping Political Ideologies on Social Media Using Text Mining and Deep Learning"
submitted to ICCRaids 2026.

The system maps social media posts onto a two-dimensional political compass using
transformer-based classifiers for the economic (left-right) and social
(authoritarian-libertarian) axes.

## Repository Structure
```
├── Datasets/
│   ├── x-axis/
│   │   ├── economic_axis_train.ipynb          # Training pipeline for economic axis
│   │   └── SenatorTweet_Translated_Slang.csv  # Constructed economic axis dataset
│   └── y-axis/
│       ├── proxy.ipynb                  # Social-axis proxy labeling script
│       ├── translate_injectSlang.ipynb  # Translation + slang + code-switching pipeline
│       ├── social_axis_train.ipynb      # Training pipeline for social axis
│       ├── MITweet_Auth_Lib.csv         # Labeled social axis dataset
│       └── MITweet_Translated_Slang.csv # Constructed social axis dataset
└── Results/
    ├── x-axis/                          # Economic axis training outputs
    └── y-axis/                          # Social axis training outputs
```

## Dataset Construction Pipeline
The dataset was constructed in three stages:
1. **Translation** — Helsinki-NLP OPUS-MT (en→id)
2. **Slang injection** — 80+ term dictionary, 50% replacement rate
3. **Code-switching** — 25% English word reinsertion

## Social Axis Labeling
Labels were derived algorithmically from five MITweet facets (I5, I6, I7, I11, I12).
See `Datasets/y-axis/proxy.ipynb` for full labeling logic and a sample of labeled instances.

## Models
Four transformer architectures were evaluated per axis:
- mBERT (`bert-base-multilingual-cased`)
- IndoBERT (`indolem/indobert-base-uncased`)
- IndoRoBERTa (`flax-community/indonesian-roberta-base`)
- XLM-RoBERTa (`xlm-roberta-base`)

Pre-trained model weights are not included due to file size.
To reproduce, run the training notebooks with the provided datasets.

## Requirements
```
torch
transformers
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
scipy
```

## Reproducibility
- Random seed: 123
- Train/test split: 80:20 stratified
- All hyperparameters are documented in the training notebooks

## Citation
[To be added upon acceptance]

## License
This repository is released for academic research purposes only.
The source datasets (Senator Tweets, MITweet) are subject to their respective licenses.
