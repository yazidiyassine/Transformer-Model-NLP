# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:11:58 2023

@author: Yassine Yazidi
"""

# ===========================================================================================
# A transformer model is a type of neural network architecture used for natural             =
#  language processing tasks such as language translation, text summarization,              =
#  and question answering. It was introduced in the 2017 paper "Attention Is All You Need"  =
#  by Google researchers. The transformer uses self-attention mechanisms to weigh           =
#  the importance of different parts of the input when generating the output,               =
#  allowing it to effectively handle input sequences of varying lengths                     =
#  and to parallelize computations across the sequence. This has made transformer           =
#  models the go-to choice for many NLP tasks, and they have been shown to achieve          =
#  state-of-the-art results on a wide range of benchmarks.                                  =
# ===========================================================================================


import tensorflow as tf
# print(tf.__version__)
import pathlib

# Downloading dataset
text_file = tf.keras.utils.get_file(
    fname='fra-eng.zip',
    origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
    extract=True,
    )

# Showing where the file is located now
text_file = pathlib.Path(text_file).parent / "fra.txt"
print(text_file)
