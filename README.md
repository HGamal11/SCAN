# SCAN: Sequence-Character Aware Network for Text Recogntion

This is the implementation of the paper ["SCAN: Sequence-Character Aware Network for Text Recogntion
"](https://www.scitepress.org/Papers/2021/103211/103211.pdf). SCAN starts by locating and recognizing the characters, and then generates the word using a sequence-based approach. It comprises two modules: a semantic-segmentation-based character prediction, and an encoder-decoder network for word generation. The training is done over two stages. In the first stage, we adopt a multi-task training technique
with both character-level and word-level losses and trainable loss weighting. In the second stage, the characterlevel
loss is removed, enabling the use of data with only word-level annotations.

&nbsp;

<p align="center">
<img src="https://github.com/HGamal11/SCAN/blob/master/Sample.png" width="60%">
<\p>
&nbsp;  
<p align="center">
<img src="https://github.com/HGamal11/SCAN/blob/master/scan.png" width="80%">
<\p>  


## Requirements
- Python 3.6
- numpy
- opencv-python
- keras 2.2.4
- tensorflow 1.14
- keras-self-attention

