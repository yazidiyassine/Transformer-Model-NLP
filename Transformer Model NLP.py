# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:11:58 2023

@author: Yassine Yazidi
"""

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
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import positional_encoding_matrix as pem



# =============================================================================
"""Positional Encoding Layer"""
"""
A positional encoding layer is a technique used in natural language processing (NLP)
tasks such as machine translation, language modeling, and text generation, among others.
It is used to incorporate the position of a word in a sentence into the model's understanding
of the sentence.

In a transformer-based model, the input is passed through an embedding layer that converts
the input into a continuous space, however, this does not account for the position of the words
in the sentence. The positional encoding layer addresses this limitation by adding a fixed encoding
to each word's embedding that represents its position in the sentence. This allows the model 
to understand the relative position of words in the sentence and improve its performance on certain 
NLP tasks.

There are several methods for creating the positional encoding vectors, common method is using
sine and cosine functions to calculate the encoding, which creates unique encodings for each 
position.
"""
# =============================================================================
 
def pos_enc_matrix(L, d, n=10000):
    """Create positional encoding matrix
 
    Args:
        L: Input dimension (length)
        d: Output dimension (depth), even only
        n: Constant for the sinusoidal functions
 
    Returns:
        numpy matrix of floats of dimension L-by-d. At element (k,2i) the value
        is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
    """
    assert d % 2 == 0, "Output dimension needs to be an even integer"
    d2 = d//2
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)     # L-column vector
    i = np.arange(d2).reshape(1, -1)    # d-row vector
    denom = np.power(n, -i/d2)          # n**(-2*i/d)
    args = k * denom                    # (L,d) matrix
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P
 
class PositionalEmbedding(tf.keras.layers.Layer):
    """Positional embedding layer. Assume tokenized input, transform into
    embedding and returns positional-encoded output."""
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        """
        Args:
            sequence_length: Input sequence length
            vocab_size: Input vocab size, for setting up embedding matrix
            embed_dim: Embedding vector size, for setting up embedding matrix
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim     # d_model in paper
        # token embedding layer: Convert integer token to D-dim float vector
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        # positional embedding layer: a matrix of hard-coded sine values
        matrix = pos_enc_matrix(sequence_length, embed_dim)
        self.position_embeddings = tf.constant(matrix, dtype="float32")
 
    def call(self, inputs):
        """Input tokens convert into embedding vectors then superimposed
        with position vectors"""
        embedded_tokens = self.token_embeddings(inputs)
        return embedded_tokens + self.position_embeddings
 
    # this layer is using an Embedding layer, which can take a mask
    # see https://www.tensorflow.org/guide/keras/masking_and_padding#passing_mask_tensors_directly_to_layers
    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)
 
    def get_config(self):
        # to make save and load a model using custom layer possible
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

vocab_size_en = 10000
seq_length = 20
import Text_Vectorization as tv
# test the dataset
for inputs, targets in tv.train_ds.take(1):
    print(inputs["encoder_inputs"])
    embed_en = PositionalEmbedding(seq_length, vocab_size_en, embed_dim=512)
    en_emb = embed_en(inputs["encoder_inputs"])
    print(en_emb.shape)
    print(en_emb._keras_mask)