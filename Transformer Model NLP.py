# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:11:58 2023

@author: Yassine Yazidi
"""

# ===========================================================================================
#  A transformer model is a type of neural network architecture used for natural            =
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

# =============================================================================
"""Obtaining Data"""
# =============================================================================
# Downloading dataset

text_file = tf.keras.utils.get_file(
    fname="fra-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "fra.txt"

# =============================================================================
"""Text Normalization"""

# This is a Python function called "normalization" that takes a line of text as
#  its input. The function uses several steps to normalize and format the input text.

# First, the line is stripped of any leading or trailing whitespace and converted 
# to lowercase using the strip() and lower() methods.

# Then, the line is normalized using the unicodedata.normalize() method with 
# the argument "NFKC". This normalization method is used to ensure that certain
#  Unicode characters are represented in a consistent way across different platforms.

# Next, the function uses four regular expressions (regex) to make specific 
# changes to the text. re.sub() method is used to substitute the matched pattern 
# with the desired replacement. The first two regex are used to add spaces around 
# certain characters that are not letters or numbers. The last two regex are used
#  to add spaces around certain words that are preceded or followed by 
#  non-letter/non-number characters.

# Finally, the line is split into two parts at the tab character using the split() method.
#  The second part is then wrapped in "[start]" and "[end]" before being returned along
#  with the first part.

# The following code is reading a text file and normalizing the lines in it. 
# It then selects 5 random pairs from the normalized text and prints them. 
# Then it saves the text pairs to a binary file called "text_pairs.pickle"
#  using the python pickle module. The code then reads the "text_pairs.pickle" file
#  and loads the text pairs into the variable "text_pairs".

# Next, the code tokenizes the English and French sentences and counting the tokens in each.
#  It also finds the maximum length of the sentences in tokens and prints the results.

# Finally, the code plots a histogram of the sentence length (in tokens) for both English 
# and French sentences. The y-axis is in log scale, and the histograms are plotted in red
#  and blue, respectively. The code also plots the maximum sentence length for each 
#  language using the same color as the corresponding histogram. The code also adds 
#  a title and a legend to the plot and shows it.

# It is also noted that sentence length fits Benford's law and can be plotted with the log scale.
# =============================================================================

import pickle
import random
import re
import unicodedata

 
def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)
    eng, fra = line.split("\t")
    fra = "[start] " + fra + " [end]"
    return eng, fra
 
# normalize each line and separate into English and French
with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]
 
# print some samples
for _ in range(5):
    print(random.choice(text_pairs))
 
with open("text_pairs.pickle", "wb") as fp:
    pickle.dump(text_pairs, fp)

with open("text_pairs.pickle", "rb") as fp:
    text_pairs = pickle.load(fp)
 
# count tokens
eng_tokens, fra_tokens = set(), set()
eng_maxlen, fra_maxlen = 0, 0
for eng, fra in text_pairs:
    eng_tok, fra_tok = eng.split(), fra.split()
    eng_maxlen = max(eng_maxlen, len(eng_tok))
    fra_maxlen = max(fra_maxlen, len(fra_tok))
    eng_tokens.update(eng_tok)
    fra_tokens.update(fra_tok)
print(f"Total English tokens: {len(eng_tokens)}")
print(f"Total French tokens: {len(fra_tokens)}")
print(f"Max English length: {eng_maxlen}")
print(f"Max French length: {fra_maxlen}")
print(f"{len(text_pairs)} total pairs")

import matplotlib.pyplot as plt

#histogram of sentence length intokens
en_lengths = [len(eng.split()) for eng, fra in text_pairs]
fr_lengths = [len(fra.split()) for eng, fra in text_pairs]

plt.hist(en_lengths, label="en", color="red", alpha=0.33)
plt.hist(fr_lengths, label="fr", color="blue", alpha=0.33)
plt.yscale("log") #Sentence length fits Benford's law
plt.ylim(plt.ylim()) # Make y-axis consistent for both plots
plt.plot([max(en_lengths), max(en_lengths)], plt.ylim(), color="red")
plt.plot([max(fr_lengths), max(fr_lengths)], plt.ylim(), color="blue")
plt.legend()
plt.title("Examples count vs Token length")
plt.show()

# =============================================================================
"""Vectorization and making Dataset"""
# =============================================================================
from tensorflow.keras.layers import TextVectorization

# Loads the normalized sentence pairs from a binary file called "text_pairs.pickle"
# using the python pickle module.
with open("text_pairs.pickle", "rb") as fp:
    text_pairs = pickle.load(fp)
 
# Shuffles the sentence pairs and splits them into training, validation and test sets.
random.shuffle(text_pairs)
n_val = int(0.15*len(text_pairs))
n_train = len(text_pairs) - 2*n_val
train_pairs = text_pairs[:n_train]
val_pairs = text_pairs[n_train:n_train+n_val]
test_pairs = text_pairs[n_train+n_val:]
 
# Parameter determined after analyzing the input data
vocab_size_en = 10000
vocab_size_fr = 20000
seq_length = 20
 
# Creates a vectorization layer called TextVectorization 
# from the TensorFlow library, which is used to convert 
# text into numerical representations.
eng_vectorizer = TextVectorization(
    max_tokens=vocab_size_en,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length,
)
fra_vectorizer = TextVectorization(
    max_tokens=vocab_size_fr,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length + 1
)
 
# Adapts the vectorization layer to the training dataset.
train_eng_texts = [pair[0] for pair in train_pairs]
train_fra_texts = [pair[1] for pair in train_pairs]
eng_vectorizer.adapt(train_eng_texts)
fra_vectorizer.adapt(train_fra_texts)
 
# Saves the training, validation and test sets and the vectorization
# layer's configuration and weights to a file called "vectorize.pickle"
with open("vectorize.pickle", "wb") as fp:
    data = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
        "engvec_config":  eng_vectorizer.get_config(),
        "engvec_weights": eng_vectorizer.get_weights(),
        "fravec_config":  fra_vectorizer.get_config(),
        "fravec_weights": fra_vectorizer.get_weights(),
    }
    pickle.dump(data, fp)
    
    
 
# Loads the text data and the vectorization layer's weights from 
# the "vectorize.pickle" file
with open("vectorize.pickle", "rb") as fp:
    data = pickle.load(fp)
 
train_pairs = data["train"]
val_pairs = data["val"]
test_pairs = data["test"]
 
eng_vectorizer = TextVectorization.from_config(data["engvec_config"])
eng_vectorizer.set_weights(data["engvec_weights"])
fra_vectorizer = TextVectorization.from_config(data["fravec_config"])
fra_vectorizer.set_weights(data["fravec_weights"])
 

# Defines a function called format_dataset that takes in a pair 
# of English and French sentences and converts them into inputs
 # and target for training the model.
def format_dataset(eng, fra):
    """Take an English and a French sentence pair, convert into input and target.
    The input is a dict with keys `encoder_inputs` and `decoder_inputs`, each
    is a vector, corresponding to English and French sentences respectively.
    The target is also vector of the French sentence, advanced by 1 token. All
    vector are in the same length.
 
    The output will be used for training the transformer model. In the model we
    will create, the input tensors are named `encoder_inputs` and `decoder_inputs`
    which should be matched to the keys in the dictionary for the source part
    """
    eng = eng_vectorizer(eng)
    fra = fra_vectorizer(fra)
    source = {"encoder_inputs": eng,
              "decoder_inputs": fra[:, :-1]}
    target = fra[:, 1:]
    return (source, target)

# Defines a function called make_dataset that creates a TensorFlow Dataset
# object for the sentence pairs, shuffles them, applies the format_dataset 
# function and batches them for training.
# def make_dataset(pairs, batch_size=64):
#     """Create TensorFlow Dataset for the sentence pairs"""
#     # aggregate sentences using zip(*pairs)
#     eng_texts, fra_texts = zip(*pairs)
#     # convert them into list, and then create tensors
    
#     dataset = tf.data.Dataset.from_tensor_slices((list(eng_texts), list(fra_texts)))
#     return dataset.shuffle(2048) \
#                   .batch(batch_size).map(format_dataset).prefetch(16).cache()

def make_dataset(pairs, batch_size=64):
    eng_texts, fra_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    fra_texts = list(fra_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, fra_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

# Creates train_ds and val_ds datasets
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

 
# Tests the dataset by printing the shape and the first element of inputs and targets
for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["encoder_inputs"][0]: {inputs["encoder_inputs"][0]}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"][0]: {inputs["decoder_inputs"][0]}')
    print(f"targets.shape: {targets.shape}")
    print(f"targets[0]: {targets[0]}")
# print(tf.__version__)
# =============================================================================
"""Positional Encoding Matrix"""
# """Attention Is All You Need""" is a research paper published in 2017 by 
# Google Brain which proposed a new neural network architecture for natural 
# language processing tasks called the Transformer. This architecture uses
# a self-attention mechanism, which allows the model to weigh the importance
# of different parts of the input when making a prediction, rather than 
# using a fixed-length context like previous models. This architecture showed 
# state-of-the-art performance on a variety of NLP tasks and is widely used 
# in many state-of-the-art models such as BERT and GPT-3.


"""
In the Transformer architecture, the input is represented as a sequence 
of tokens (e.g. words in a sentence), and the model processes these tokens 
in parallel rather than sequentially. However, the order of the tokens 
in the sequence is still important information that needs to be taken 
into account by the model. To solve this problem, the Transformer uses 
a concept called positional embeddings.

A positional embedding is a fixed-size vector that is added to the token 
representation at each position in the input sequence. These vectors 
are designed to encode information about the position of the token in 
the sequence, so that the model can take into account the order of the tokens. 
The vectors are learned during the training process along with the other
 model parameters.

In practice, the positional embeddings are added to the token representations 
before they are fed into the self-attention mechanism. This allows the attention
 mechanism to take into account the relative position of the tokens when
 calculating the attention weights.

Overall, positional embedding is a technique used to incorporate the order
 of the tokens in the input, since the Transformer is a parallel processing
 architecture that doesn't have a notion of the order of the tokens in the input
 sequence.
"""
# =============================================================================

import numpy as np

# =============================================================================
# """ reates a positional encoding matrix using the numpy library.
# The matrix is used to represent the position of elements in 
# a sequence in a continuous and differentiable way. 
# The function pos_enc_matrix takes in three 
# parameters: L, which is the input dimension or length of the sequence,
#  d, which is the output dimension or depth of the matrix, and n, 
#  which is a constant used for the sinusoidal functions that create
#  the matrix. The function asserts that d is an even number and then
#  creates the matrix using sin and cos functions, with the arguments
#  being based on the values of k and i, which are the position and
#  depth of the element respectively."""
# =============================================================================
 
def pos_enc_matrix(L, d, n=10000):
    """Create positional encoding matrix
       Args:
           L: Input dimension (length)
           d: Output dimension (depth), even only
           n: Constant for the sinusoidal functions
           
      Returns:
          numpy matrix of floats of dimension L-by-d. At element (k, 2i) the value
          is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
     """
    assert d % 2 == 0, "Output dimension needs to be an even integer"
    d2 = d//2
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1) # L-column vector
    i = np.arange(d2).reshape(1, -1) # d-row vector
    denom = np.power(n, -i/d2) # n**(-2*i/d)
    args = k * denom
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P;

# Plot the positional encoding matrix
pos_matrix = pos_enc_matrix(L=2048, d=512)
assert pos_matrix.shape == (2048, 512)
plt.pcolormesh(pos_matrix, cmap='RdBu')
plt.xlabel('Depth')
plt.ylabel('Position')
plt.colorbar()
plt.show()

with open("posenc-2048-512.pickle", "wb") as fp:
    pickle.dump(pos_matrix, fp)
    
with open("posenc-2048-512.pickle", "rb") as fp:
    pos_matrix = pickle.load(fp)

assert pos_matrix.shape == (2048,512)
#plot the poistional encoding matrix
plt.pcolormesh(np.hstack([pos_matrix[:, ::2], pos_matrix[:, 1::2]]), cmap='RdBu')
plt.xlabel('Depth')
plt.ylabel('Position')
plt.colorbar()
plt.show()


# Plot two curves from different position
plt.plot(pos_matrix[100], alpha=0.66, color="red", label="position 100")
plt.legend()
plt.show()

# Show the dot product between different normalized positional vectors
pos_matrix /= np.linalg.norm(pos_matrix, axis=1, keepdims=True)
p = pos_matrix[789] # All vectors compare to vector at position 789
dots = pos_matrix @ p
plt.plot(dots)
plt.ylim([0, 1])
plt.show()