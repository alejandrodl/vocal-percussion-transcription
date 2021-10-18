# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Alejandro Delgado
# License: Apache 2.0

import setuptools

setuptools.setup(
    name='vocal-percussion-transcription',
    author='Alejandro Delgado',
    url='https://github.com/alejandrodl/vocal-percussion-transcription',
    packages=setuptools.find_packages(),
    install_requires=[
        'bidict',
        'confugue>=0.1,<1',
        'librosa>=0.8,<1',
        'matplotlib',
        'numpy',
        'scikit_learn',
        'SoundFile',
        'tensorflow>=2,<3',
        'torch>=1.5,<2',
    ],
    python_requires='>=3.6',
)