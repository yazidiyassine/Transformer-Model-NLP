# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 21:57:30 2023

@author: Yassine Yazidi
"""

""" Objectifs
    * Vectorize text using the Keras TextVectorization layer.
    * Implement a TransformerEncoder layer, a TransformerDecoder layer, 
        and a PositionalEmbedding layer.
    * Prepare data for training a sequence-to-sequence model.
    * Use the trained model to generate translations of never-seen-before
        input sentences (sequence-to-sequence inference).
"""


""" Obtaining the data"""

import pathlib
import random
import tensorflow as tf
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
