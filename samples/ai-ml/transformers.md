# The Transformer Architecture

## Overview

The Transformer is a deep learning model architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google Brain. Unlike earlier sequence models based on recurrent neural networks (RNNs) or convolutional neural networks (CNNs), the Transformer relies entirely on self-attention mechanisms to draw global dependencies between input and output. This design allows for significantly more parallelisation during training, enabling models to be trained on far larger datasets.

The Transformer has become the dominant architecture for natural language processing (NLP) tasks and has since been adapted for computer vision, speech recognition, protein structure prediction, and many other domains.

## Self-Attention Mechanism

The core innovation of the Transformer is the self-attention mechanism, also called scaled dot-product attention. For each position in a sequence, self-attention computes a weighted sum of all other positions in the same sequence. The weights are determined by the compatibility between the query at the current position and the keys at all other positions.

Formally, given queries Q, keys K, and values V (all derived from the same input), attention is computed as:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

where d_k is the dimensionality of the key vectors. The division by sqrt(d_k) prevents the dot products from growing too large in magnitude, which would push the softmax into regions of very small gradients.

## Multi-Head Attention

Rather than performing a single attention function, the Transformer uses multi-head attention. The queries, keys, and values are linearly projected h times with different learned projection matrices. Attention is applied to each of these projections in parallel, and the h resulting outputs are concatenated and projected again.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging over all heads inhibits this capability.

The number of attention heads in GPT-3 is 96, and each head has a dimensionality of 128, giving a total model dimension of 12,288.

## Positional Encoding

Because the Transformer contains no recurrence and no convolution, it has no inherent sense of order in a sequence. To give the model information about the position of each token, positional encodings are added to the input embeddings at the bottom of both the encoder and decoder stacks.

The original Transformer paper uses sine and cosine functions of different frequencies:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

This encoding allows the model to easily learn to attend by relative positions, since for any fixed offset k, PE(pos+k) can be represented as a linear function of PE(pos).

## Encoder–Decoder Structure

The original Transformer uses an encoder–decoder structure suited for sequence-to-sequence tasks such as machine translation.

The encoder maps an input sequence of symbol representations to a sequence of continuous representations. It is composed of N identical layers (N=6 in the original paper), each containing two sub-layers: multi-head self-attention and a position-wise fully connected feed-forward network. Residual connections and layer normalisation are applied around each sub-layer.

The decoder generates an output sequence one token at a time. It has an additional third sub-layer that performs multi-head attention over the encoder output (cross-attention). The decoder's self-attention is masked to prevent positions from attending to subsequent positions, preserving the autoregressive property during training.

## Feed-Forward Networks

Each layer of the Transformer contains a position-wise feed-forward network applied identically and independently to each position. It consists of two linear transformations with a ReLU activation in between:

    FFN(x) = max(0, xW_1 + b_1) * W_2 + b_2

The inner dimensionality of the feed-forward layer (d_ff) is typically larger than the model dimension (d_model). In the base Transformer, d_model = 512 and d_ff = 2048.

## Encoder-Only Models: BERT

BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder part of the Transformer. It is pre-trained on two tasks: Masked Language Modelling (MLM), where random tokens are masked and the model must predict them, and Next Sentence Prediction (NSP), where the model predicts whether two sentences appear consecutively in the original document.

BERT's bidirectional attention — the ability to attend to both left and right context simultaneously — makes it particularly powerful for classification, named entity recognition, and question answering tasks.

## Decoder-Only Models: GPT

The GPT (Generative Pre-trained Transformer) family uses only the decoder stack with causal (left-to-right) masking. These models are trained to predict the next token given all previous tokens. GPT models excel at open-ended text generation, code completion, and instruction following. GPT-4, released in March 2023, is a large multimodal model capable of accepting both image and text inputs.

## Efficiency Improvements

Standard self-attention has quadratic complexity O(n^2) in sequence length, because every pair of positions must attend to every other. For long sequences this becomes computationally prohibitive. Several alternatives have been proposed:

- **Sparse attention** (e.g., Longformer, BigBird): restricts attention to a local window plus a set of global tokens, reducing complexity to O(n).
- **Linear attention** (e.g., Performer): approximates the softmax kernel using random feature maps, achieving O(n) complexity.
- **Flash Attention**: an IO-aware exact attention algorithm that avoids materialising the large attention matrix in HBM (high-bandwidth memory), dramatically reducing memory usage and increasing speed without approximation.

## Key Hyperparameters

| Parameter | Base Transformer | GPT-3 |
|---|---|---|
| Layers (N) | 6 | 96 |
| Model dimension (d_model) | 512 | 12,288 |
| Attention heads (h) | 8 | 96 |
| Feed-forward dimension (d_ff) | 2,048 | 49,152 |
| Parameters | ~65M | ~175B |
