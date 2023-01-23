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

# =============================================================================
"""Obtaining Data"""
# =============================================================================
# Downloading dataset
text_file = tf.keras.utils.get_file(
    fname='fra-eng.zip',
    origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
    extract=True,
    )

# Showing where the file is located now
text_file = pathlib.Path(text_file).parent / "fra.txt"
# print(text_file)

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
# =============================================================================

import pickle
import random
import re
import unicodedata

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    line = re.sub(r"^([^ \w])(?!\s)",r"\1 ", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(?!\s)(^ \w)$", r" \1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)
    eng, fra = line.split("\t")
    fra = "[start]" + fra + "[end]"
    return eng, fra

#normalize each line and separate into English and French
with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]
    
#print some samples

for _ in range(5):
    print(random.choice(text_pairs))

with open("text_pairs.pickle", "wb") as fp:
    pickle.dump(text_pairs, fp)

with open("text_pairs.pickle", "rb") as fp:
    text_pairs = pickle.load(fp)
    
#Count tokens
eng_tokens, fra_tokens = set(), set()
eng_maxlen, fra_maxlen = 0, 0

for eng, fra in text_pairs:
    eng_tok, fra_tok = eng.split(), fra.split()
    eng_maxlen = max(eng_maxlen, len(eng_tok))
    fra_maxlen = max(fra_maxlen, len(fra_tok))
    eng_tokens.update(eng_tok)
    fra_tokens.update(fra_tok)
    
print(f"Total English tokens : {len(eng_tokens)}")
print(f"Total French tokens : {len(fra_tokens)}")
print(f"Max English lengths : {eng_maxlen}")
print(f"Max French lengths : {fra_maxlen}")
print(f"{len(text_pairs)} total pairs")