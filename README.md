# Contrast-CAT: Contrasting Activations for Enhanced Interpretability in Transformer-based Text Classifiers, UAI'25
Sungmin Han, Jeonghyun Lee, Sangkyun Lee

Korea University (AIRLAB)

----

Transformers have profoundly influenced AI research, but explaining their decisions remains challenging – even for relatively simpler tasks such as classification – which hinders trust and safe deployment in real-world applications. Although activation-based attribution methods effectively explain transformer-based text classification models, our findings reveal that these methods can be undermined by class-irrelevant features within activations, leading to less reliable interpretations.
To address this limitation, we propose ContrastCAT, a novel activation contrast-based attribution method that refines token-level attributions by filtering out class-irrelevant features. By contrasting the activations of an input sequence with reference activations, Contrast-CAT generates clearer and more faithful attribution maps. 
Experimental results across various datasets and models confirm that Contrast-CAT consistently outperforms stateof-the-art methods. Notably, under the MoRF setting, it achieves average improvements of ×1.30 in AOPC and ×2.25 in LOdds over the most competing methods, demonstrating its effectiveness in enhancing interpretability for transformer-based text classification.


### Environment
  * Python v3.7.4
  * PyTorch v1.9.1
  * Hugging Face Hub v0.14.1


### 📊 Datasets

We used five publicly available NLP datasets for text classification tasks:

- **Amazon Polarity**  
  _Zhang et al._
  [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf). NeurIPS, 2015.

- **Yelp Polarity**  
  _Xiang Zhang et al._
  Same as Amazon Polarity (subset of the same paper).

- **SST-2 (Stanford Sentiment Treebank)**  
  _Socher et al._  
  [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170). EMNLP, 2013.

- **IMDB Reviews**  
  _Maas et al._  
  [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015). ACL-HLT, 2011.

- **AG News**  
  _Del Corso et al._
  [Ranking a Stream of News](https://dl.acm.org/doi/10.1145/1060745.1060781). WWW, 2005.
