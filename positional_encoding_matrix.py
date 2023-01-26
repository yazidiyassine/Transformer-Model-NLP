# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:35:08 2023

@author: Yassine Yazidi
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

# =============================================================================
# """ creates a positional encoding matrix using the numpy library.
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
assert pos_matrix.shape == (2048, 512)
# Plot the positional encoding matrix, alternative way
plt.pcolormesh(np.hstack([pos_matrix[:, ::2], pos_matrix[:, 1::2]]), cmap='RdBu')
plt.xlabel('Depth')
plt.ylabel('Position')
plt.colorbar()
plt.show()

with open("posenc-2048-512.pickle", "rb") as fp:
    pos_matrix = pickle.load(fp)
assert pos_matrix.shape == (2048, 512)
# Plot two curves from different position
plt.plot(pos_matrix[100], alpha=0.66, color="red", label="position 100")
plt.legend()
plt.show()


with open("posenc-2048-512.pickle", "rb") as fp:
    pos_matrix = pickle.load(fp)
assert pos_matrix.shape == (2048, 512)
# Show the dot product between different normalized positional vectors
pos_matrix /= np.linalg.norm(pos_matrix, axis=1, keepdims=True)
p = pos_matrix[789]  # all vectors compare to vector at position 789
dots = pos_matrix @ p
plt.plot(dots)
plt.ylim([0, 1])
plt.show()