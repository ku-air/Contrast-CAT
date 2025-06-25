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


### Datasets
We used five publicly available NLP datasets for text classification tasks:
  * Amazon and Yelp Polarity : Xiang Zhang, Junbo Zhao, and Yann LeCun. Character Level convolutional networks for text classification. In NIPS, 2015.
  * SST2 : Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, 2013.
  * IMDB : Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In ACL-HLT, 2011.
  * AgNews : Gianna M. Del Corso, Antonio Gullí, and Francesco Romani. Ranking a stream of news. In WWW, 2005.
