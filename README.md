**Original transformer architecture as per the "Attention is all you need" paper**

This repo contains an implemention from scratch for the orignal transformer architecture.
Here is a illustration of the marvellous transformer architecture:

![TransformerArch](https://github.com/shanchy/Transformer-Original/blob/master/images/transformers_zoomed.png?raw=true)

Understanding the transformer architecture boils down to understanding these essential components:
- The multi-head self-attention block
- The Positional encoding
- The position-wise feed forward network
- The encoder block
- The decoder block
- Padding masks and Look-ahead masks

And how they all combine and relate to each other to construct the full transformer. 

There are numerous articles online explaining how each of these components work. However, it may still be difficult to fully comprehend what is happening in the transformer without properly understand what tensors are going in and out of each layer/component and how the tensor sizes change.

This repo aims to provide that information. Detailed explanation of how the tensor sizes change in each step are provided within the functions, along with elaboration of some of the more complicated concepts.
