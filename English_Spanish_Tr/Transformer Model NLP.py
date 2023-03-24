# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 21:57:30 2023

@author: Yassine Yazidi
"""

""" Objectives
    * Vectorize text using the Keras TextVectorization layer.
    * Implement a TransformerEncoder layer, a TransformerDecoder layer, 
        and a PositionalEmbedding layer.
    * Prepare data for training a sequence-to-sequence model.
    * Use the trained model to generate translations of never-seen-before
        input sentences (sequence-to-sequence inference).
"""


""" Obtaining the data"""

"""This section imports necessary libraries and modules, sets up the data download
and extraction from a URL using get_file() function from Keras, and extracts 
the downloaded file. It then sets text_file to the path of the extracted file"""

import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

text_file = tf.keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"
# print(text_file)

""" Parsing the data """

with open(text_file, encoding="utf-8") as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((eng, spa))


for _ in range(5):
    print(random.choice(text_pairs))

""" Splitting the sentence pairs into a training set, validation set, test set"""
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

''' Vectorizing the text data '''
#  to turn the original strings into integer sequences where each integer 
# represents the index of a word in a vocabulary.

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

#  custom standardization function that converts input strings 
# to lowercase and removes punctuation marks
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
# Two instances of TextVectorization : one for English text and one for Spanish text.

# The max_tokens parameter sets the maximum number of words to keep in the vocabulary
eng_vectorization = TextVectorization(
    max_tokens = vocab_size, output_mode="int", output_sequence_length = sequence_length,
    )

spa_vectorization = TextVectorization(
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length=sequence_length + 1,
    standardize = custom_standardization
    )

train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]

# The adapt method is called on both instances, which analyzes 
# the training data and builds the vocabulary based on the most common words.
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)

"""The format_dataset function takes in two arguments, eng and spa, which
represent English and Spanish texts respectively. It first applies 
the eng_vectorization and spa_vectorization TextVectorization instances 
to the corresponding input texts. It then returns a tuple of two elements. 
The first element is a dictionary with two keys, "encoder_inputs" 
and "decoder_inputs", which map to the English and Spanish integer sequences 
respectively. The Spanish integer sequence is sliced to exclude the last integer,
which corresponds to the end-of-sequence token. The second element of 
the tuple is the target Spanish integer sequence, which is sliced to exclude 
the first integer, which corresponds to the start-of-sequence token."""

def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    
    return ({"encoder_inputs": eng, "decoder_inputs": spa[:, :-1],}, spa[:, 1:])

"""The make_dataset function takes in a list of pairs of English 
and Spanish texts, which represent the training or validation dataset.
It first separates the English and Spanish texts into two lists,
then creates a TensorFlow dataset using the tf.data.Dataset.from_tensor_slices method.
It then applies batching to the dataset using the batch method, and applies the format_dataset 
function to each batch using the map method. Finally, it shuffles the dataset, 
caches it for faster processing, and prefetches a small number of elements for performance optimization."""

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")
    