# Vocal Percussion Transcription

This repository currently holds the official implementation of "Deep Embeddings for Robust User-Based Amateur Vocal Percussion Transcription" (under review). The repository will also hold future studies on real-time vocal percussion transcription.

## 0 - Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The AVP-LVT dataset can be accessed thorugh this link: (link). Once downloaded, place its contents in the "data/external" directory.

## 1 - Spectrogram Representations

The first step to reproduce this study is to generate the spectrogram reperesentations that are later fed to the networks. These are 64x48 log Mel spectrograms computed with a frame size of 23 ms and a hop size of 8 ms.

To build the spectrogram representations, run this command:

```spec
python src/data/generate_interim_datasets.py
```

## Training and/or Embedding Vectors

To train the models and save embeddings predicted from evaluation data, run this command:

```train
python train.py --input-data <path_to_data>
```

Due to the long training time required (32 hours on a typical GPU approx.), we also provide the final learnt embedding vectors that are used as input features in the evaluation section. These can be downloaded thorugh the following link: (link)

## Evaluation

To evaluate the performance of input embedding vectors, run:

```evalknn
python offline_evaluate_processed_knn.py
```

for KNN classification or

```evalalt
python offline_evaluate_processed_alternative_classifiers.py
```

for classification with three alternative classifiers (logistic regression, random forest, and extreme gradient boosting).

## Pre-trained Models

Final pretrained models for each of the seven embedding learning methods can be downloaded here: (link)

## Results

Our models achieve the following performances on the AVP-LVT dataset:

| Method                   | Participant-wise Accuracy | Boxeme-wise Accuracy |
| ------------------------ |-------------------------- | -------------------- |
| GMM-HMM                  |           .725            |         .734         |
| Timbre                   |           .840            |         .835         |
| Feature Selection (max.) |        .827 ± .012        |      .795 ± .011     |
| Instrument Original      |        .812 ± .012        |      .774 ± .014     |
| Instrument Reduced       |        .779 ± .019        |      .738 ± .031     |
| Syllable Original        |        .899 ± .005        |      .874 ± .008     |
| Syllable Reduced         |        .883 ± .005        |      .852 ± .012     |
| Phoneme Original         |        .876 ± .014        |      .840 ± .018     |
| Phoneme Reduced          |        .874 ± .013        |      .838 ± .019     |
| Boxeme Original          |        .861 ± .016        |      .832 ± .018     |




