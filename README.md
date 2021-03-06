Deep Embeddings for Robust User-Based Amateur Vocal Percussion Transcription
============================================================================

This is the code repository for the SMC 2022 paper
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

If you are a Mac user, you may need to install [Essentia](https://essentia.upf.edu/installing.html) using Homebrew.


Data
----

Once the AVP-LVT dataset is [downloaded](https://zenodo.org/record/5578744#.YW7Wl9nML0o) and built following the instructions inside, place its contents in the `data/external` directory.

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


GMM-HMM
-------

To reproduce the results for the GMM-HMM model, all routines relative to data engineering, model training, and model evaluation are included in `src/models/GMM-HMM`. The folder includes its own README file with instructions.

To train and test the GMM-HMM model, simply run the following command on a terminal:

```sh
audio_path=../../data/external
run.sh $audio_path
```


Results
-------

Our learnt embeddings and engineered features achieve the following performances on the AVP-LVT dataset with a KNN classifier:

| Method              | Participant-wise Accuracy| Boxeme-wise Accuracy |
| --------------------|------------------------- | -------------------- |
| GMM-HMM             |           .725           |         .734         |
| Timbre              |           .840           |         .835         |
| Feature Selection   |        .827 ± .030       |      .795 ± .011     |
| Instrument Original |        .812 ± .037       |      .774 ± .038     |
| Instrument Reduced  |        .779 ± .034       |      .738 ± .033     |
| Syllable Original   |        .899 ± .025       |      .874 ± .029     |
| Syllable Reduced    |        .883 ± .030       |      .852 ± .031     |
| Phoneme Original    |        .876 ± .028       |      .840 ± .029     |
| Phoneme Reduced     |        .874 ± .030       |      .838 ± .032     |
| Boxeme Original     |        .861 ± .030       |      .832 ± .031     |


Pretrained Models
-----------------

Weights relative to the final pretrained models for each of the seven embedding learning methods can be downloaded here: (link)

We recommend using the `cnn_syllable_level_original.h5` for feature extraction, as it yields the best performance in the table [above](#Results).


Saliency Maps
-------------

Saliency maps of 30 input spectrograms of different instrument classes can be computed by running:

```sh
python src/models/compute_saliency_maps.py
```

The resulting maps are saved in JPEG format in the folder `data/processed/spatial_abs_grad_bottom`. We provide the 30 already computed maps in the subfolders of such directory.


TODO List
---------

- [x] Add full table with results
- [x] Add links
- Finish tidying up code
- Write routines for personal use


Acknowledgments
---------------
This work has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No. 765068.



