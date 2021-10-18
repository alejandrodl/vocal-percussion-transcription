Deep Embeddings for Robust User-Based Amateur Vocal Percussion Transcription
============================================================================

This is the code repository for the ICASSP 2022 paper (under review) 
*Deep Embeddings for Robust User-Based Amateur Vocal Percussion Transcription*
by Alejandro Delgado, Emir Demirel, Vinod Subramanian, Charalampos Saitis, and Mark Sandler.

Contents
--------

- `src` – the main codebase with scripts for processing data, models, and results (usage details [below](#Usage))
- `data` – datasets used throughout the study (download details [below](#Usage))
- `models` – folder that hosts trained models (download details [below](#Pretrained Models))
- `results` – folder that hosts information relative to final accuracy results.

Setup
-----

To install requirements:

```sh
pip install -r requirements.txt
```

Processed Files
---------------

Training deep learning models and feature selection is time-consuming, taking approximately 36 hours (GPU) and 48 hours (CPU) respectively. To avoid this inconvenience, we provide the final learnt embeddings and feature importances that are used in the evaluation section. These are organised in folders and may be directly [downloaded](link_to_be_created_soon) and placed in `data/processed` to jump to the evaluation section below.

Data
----

Before running any commands, please [download](link_to_be_created_soon) the AVP-LVT dataset and, once downloaded, place its contents in the `data/external` directory.

### Representations

The first step is to generate the spectrogram reperesentations that are later fed to the networks. These are 64x48 log Mel spectrograms computed with a frame size of 23 ms and a hop size of 8 ms. Also, for the baseline methods, 

To build these spectrogram representations, which will be saved in the `data/interim` directory, run this command:

```sh
python generate_interim_datasets.py
```

To extract engineered features, also saved in the `data/interim` directory, run this command:

```sh
python generate_interim_datasets.py
```

to extract "MFCCs + Envelope" features or

```sh
python generate_interim_datasets.py
```

to extract 258-dimensional feature vectors to feed feature selection algorithms.

Training
--------

To train deep learning models and save embeddings predicted from evaluation data, run this command:

```sh
python train_deep.py
```

To train feature selection methods and save feature importances, run this command:

```sh
python train_selection.py
```

Evaluation
----------

To evaluate the performance of learnt embeddings and selected features stored in `data/processed`, run:

```sh
python eval_knn.py
```

for KNN classification or

```sh
python eval_alt.py
```

for classification with three alternative classifiers (logistic regression, random forest, and extreme gradient boosting).

Results
-------

Our models achieve the following performances on the AVP-LVT dataset:

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
| E2E CNN             |        .896 ± .008       |      .877 ± .010     |

Note: the last model, E2E CNN, is trained end-to-end (no embeddings nor KNN) on single participants' data exclusively (12x12 spectrograms + 15x data augmentation). The main drawback of this method is its long training time, which lasts for 4 minutes approx. on a typical CPU. For more details, see (link).

Pretrained Models
-----------------

Final pretrained models for each of the seven embedding learning methods can be downloaded here: (link)

Acknowledgments
---------------
This work has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No. 765068.




