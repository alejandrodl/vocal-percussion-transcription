Deep Embeddings for Robust User-Based Amateur Vocal Percussion Transcription
============================================================================

This is the code repository for the ICASSP 2022 paper (under review) 
*Deep Embeddings for Robust User-Based Amateur Vocal Percussion Transcription*
by Alejandro Delgado, Emir Demirel, Vinod Subramanian, Charalampos Saitis, and Mark Sandler.

Contents
--------

- `src` – the main codebase with scripts for processing data, models, and results (details in sections below).
- `data` – datasets and processed data used throughout the study.
- `models` – folder that hosts already trained models.
- `results` – folder that hosts information relative to final accuracy results.

Requirements
------------

To install requirements:

```sh
pip install -r requirements.txt
```

Processed Files
---------------

Training deep learning models and feature selection algorithms is time-consuming, taking approximately 36 hours (GPU) and 48 hours (CPU) respectively. To avoid this, we provide the final learnt embeddings and feature importances arrays that are used in the evaluation section. These are organised in folders and may be directly [downloaded](link_to_be_created_soon) and placed in `data/processed` to jump directly to the evaluation section [below](#Evaluation).

Data
----

Once the AVP-LVT dataset is [downloaded](link_to_be_created_soon), place its contents in the `data/external` directory.

The first step is to generate the spectrogram reperesentations that are later fed to the networks. These are 64x48 log Mel spectrograms computed with a frame size of 23 ms and a hop size of 8 ms. Also, several engineered (hand-crafted) feature vectors need to be extracted for the baseline methods using the same frame-wise parameters as for the spectrogram.

To build spectrogram representations, which will be saved in the `data/interim` directory, run this command:

```sh
python src/data/generate_interim_datasets.py
```

To extract engineered features, also saved in the `data/interim` directory, run this command:

```sh
python src/data/extract_engineered_features_mfcc_env.py
```

to extract "MFCCs + Envelope" features or

```sh
python src/data/extract_engineered_features_all.py
```

to extract 258-dimensional feature vectors to feed feature selection algorithms.

Training
--------

To train deep learning models and save embeddings predicted from evaluation data, run this command:

```sh
python src/models/train_deep.py
```

To train feature selection methods and save feature importances, run this command:

```sh
python src/models/train_selection.py
```

Evaluation
----------

To evaluate the performance of learnt embeddings and selected features, which should be stored in `data/processed` by now, run:

```sh
python src/results/eval_knn.py
```

for KNN classification or

```sh
python src/results/eval_alt.py
```

for classification with three alternative classifiers (logistic regression, random forest, and extreme gradient boosting).

Results
-------

Our learnt embeddings and engineered features achieve the following performances on the AVP-LVT dataset with a KNN classifier:

| Method              | Participant-wise Accuracy| Boxeme-wise Accuracy |
| --------------------|------------------------- | -------------------- |
| GMM-HMM             |           .725           |         .734         |
| Timbre              |           .840           |         .835         |
| Feature Selection   |        .827 ± .012       |      .795 ± .011     |
| Instrument Original |        .812 ± .012       |      .774 ± .014     |
| Instrument Reduced  |        .779 ± .019       |      .738 ± .031     |
| Syllable Original   |        .899 ± .005       |      .874 ± .008     |
| Syllable Reduced    |        .883 ± .005       |      .852 ± .012     |
| Phoneme Original    |        .876 ± .014       |      .840 ± .018     |
| Phoneme Reduced     |        .874 ± .013       |      .838 ± .019     |
| Boxeme Original     |        .861 ± .016       |      .832 ± .018     |

Pretrained Models
-----------------

Weights relative to the final pretrained models for each of the seven embedding learning methods can be downloaded here: (link)

We recommend using the `cnn_syllable_level_original.h5` for feature extraction, as it yields the best performance in the table [above](#Results).

TODO List
---------

- [x] Add full table with results
- Add data and paper links
- Finish tidying up code
- Write routines for personal use

Acknowledgments
---------------
This work has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No. 765068.




