## Original transformer architecture as per the "Attention is all you need" paper

This repo contains an implemention from scratch for the orignal transformer architecture.
Here is an illustration of the marvellous transformer architecture:

![TransformerArch](https://github.com/shanchy/Transformer-Original/blob/master/images/transformers_zoomed.png?raw=true)

Understanding the transformer architecture boils down to familiarity with the following essential components:
- The multi-head self-attention block
- The Positional encoding
- The position-wise feed forward network
- The encoder block
- The decoder block
- Padding masks and Look-ahead masks

And how they all combine and relate to each other to construct the full transformer. 

There are numerous articles online explaining how each of these components work. However, it may still be difficult to fully comprehend what is happening in the transformer without properly understanding what tensors are going in and out of each layer/component and how the tensor sizes change.

This repo aims to provide that information, and the entirety of it can be found in the single jupyter notebook "Transformers-standard.ipynb". Inside the notebook, you can find the following :

1) A full implementation of the transformer, broken down into the essential components listed above
2) Detailed explanation of how the tensor sizes change in each step are provided within the functions, along with elaboration of some of the more complicated concepts. 
3) Training & testing code for a simple sequence reversal task 
4) Training & testing code for machine translation from german to english, using the Multi30k dataset

**The notebook has been fully tested using these package versions : python=3.8.13, torch=2.2.2, torchtext=0.17.2, torchdata=0.7.1**

### Additional notes :
- For the sequence reversal task, run at least 15 epochs. The transformer is tiny, with just under 200k parameters, but performs quite well with just 15 epochs. Sample input & output are shown below
  
``` python
Input   =  tensor([[56, 71, 23, 44, 91, 92, 93, 55, 56, 57, 83, 71, 33]]) 
Output  =  tensor([[33, 71, 83, 57, 56, 55, 93, 92, 91, 44, 23, 71, 56]])
```
- For the machine translation task, run at least 50 epochs. Any lower, the model will produce gibberish output. The transformer has about 15 million parameters, which is considered EXTREMELY SMALL compared to LLMs. Yet, it manages to produce decent results with 60 epochs. Here are 3 sample outputs

Sample 1 :
```python
German : 
ein Frau in schwarzer Jacke, der auf einem weißen Pferd reitet.

English : 
a woman in a black jacket riding a white horse
```

Sample 2:
```python
German : 
Fünf Wanderer, von denen einer in Richtung Kamera blickt und die anderen von der Kamera weg, gehen durch ein steiniges Flussbett.

English : 
five hikers , one facing towards the camera and the others facing away from it , are walking through a rocky riverbed .
```

Sample 3:
```python
German : 
Ein Mann isst eine Orange, während er mit seinem Sohn spricht.

English : 
a man eats corn while talking to his son .
```
The model does well on sample 1. Sample 2 is taken from the train set, just presented here to illustrate how well the model was able to fit to a long sentence. For sample 3, it wrongly translates orange to corn. Overall, the model's translation ability is very limited and it does make a lot of mistakes for unseen sentences. However, taking into consideration the relatively tiny size of the model and a limited dataset size of 30k, the translation outputs of the model are impressive.
