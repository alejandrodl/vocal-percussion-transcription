Deep Embeddings for Robust Vocal Percussion Transcription
=========================================================

This is the code repository for the ICASSP 2021 paper 
*Deep Embeddings for Robust User-Based Amateur Vocal Percussion Transcription*
by Alejandro Delgado, Emir Demirel, Vinod Subramanian, Charalampos Saitis, and Mark Sandler.

Contents
--------

- `src` – the main codebase with scripts for processing data, models, and results (usage details [below](#Usage))
- `data` – datasets used throughout the study.
- `models` – folder that hosts trained models.
- `results` – folder that hosts information relative to final accuracy results.

Setup
-----

To install requirements:

```sh
pip install -r requirements.txt
```

Usage
-----

Before running any commands, please [download](link_to_be_created_soon) the AVP-LVT dataset and, once downloaded, place its contents in the `data/external` directory.

### Spectrogram Generation

The first step is to generate the spectrogram reperesentations that are later fed to the networks. These are 64x48 log Mel spectrograms computed with a frame size of 23 ms and a hop size of 8 ms.

To build these spectrogram representations, which will be saved in the `data/interim` directory, run this command:

```sh
python src/data/generate_interim_datasets.py
```

### Training

To train the models and save embeddings predicted from evaluation data, run this command:

```sh
python train.py --input-data <path_to_data>
```

### Processed Embeddings

Due to the long training time required (32 hours on a typical GPU approx.), we also provide the final learnt embedding vectors that are used as input features in the evaluation section. These are organised in folders and may be directly [downloaded](link_to_be_created_soon) and placed in `data/processed` to jump to evaluation.

### Evaluation

To evaluate the performance of input embedding vectors, run:

```sh
python offline_evaluate_processed_knn.py
```

for KNN classification or

```sh
python offline_evaluate_processed_alternative_classifiers.py
```

for classification with three alternative classifiers (logistic regression, random forest, and extreme gradient boosting).

### Pre-trained Models

Final pretrained models for each of the seven embedding learning methods can be downloaded here: (link)

Results
-------

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

Acknowledgments
---------------
This work has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No. 765068.




