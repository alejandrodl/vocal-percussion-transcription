A GMM-HMM based approach for vocal percussive sound detection
============================================================================

This model is an adaptation of the traditional GMM-HMM based automatic speech recognition to the task of vocal percussive sound transcription. 
  - According to this approach, there are 3 main building blocks of the transcriber: Language, Pronunciation and Acoustic Models.
  - The language model can be perceived as statistical grammar approximation. In our approach, we consider percussive instrument types as word tokens and build a 4-gram language model.
  - The pronunciation dictionary is generated based on the phoneme - to - instrument type annotations provided within the training set. This is used to convert phoneme probabilities to instrument type probabilities.
  - The acoustic model learns a mapping between acoustic features (13-band MFCCs) and phonemes. This is based on the traditional triphone GMM-HMM approach which is optimized using the Expectation-Maximization algorithm.
  - The acoustic, pronunciation and language models are composed into a single decoding graph using Weighted Finite State Transducers (WFST).
  - The above steps are the standard Kaldi recipe for building a GMM-HMM based speech recognizer.


Contents
--------

- `conf` – Configuration files
- `data` – Data information formatted in Kaldi style for easy processing
- `exp` – This folder is generated after running the training script which stores the relevant info for each training step.

Requirements
------------

This package specifically requires Kaldi installation. For a detailed info please visit:

```
https://github.com/kaldi-asr/kaldi
```

OR

For installation with Docker, please refer to the example in   

```
https://github.com/emirdemirel/ASA_ICASSP2021/Dockerfile
```


Data
----

Once the AVP-LVT dataset is [downloaded](https://zenodo.org/record/5578744#.YW7Wl9nML0o) and built following the instructions inside, place its contents in the `data/external` directory. (Same procedure as main page)


Training
--------

To train and test this model, simply run the following on a terminal.


```sh
audio_path=../data/external
run.sh $audio_path
```


Results
-------

The scoring files are stored at

```
exp/${model}/decode_test/scoring_kaldi
```

To retrieve utterance-wise results, please refer to
```
exp/${model}/decode_test/scoring_kaldi/wer_details/per_utt
```

To retrieve participant-wise results, please refer to

```
exp/${model}/decode_test/scoring_kaldi/wer_details/per_spk
```

where model can be ```mono```, ```tri1```, ```tri2b``` or ```tri3b``` (The last model referred to as in the paper).



Acknowledgments
---------------
This work has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No. 765068.


