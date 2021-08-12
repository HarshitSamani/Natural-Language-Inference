<br />
<p align="center">
  </a>
  <h3 align="center">Natural Language Inference (NLI)</h3>
  <p align="center">
    Natural Language Inference (NLI) using SNLI dataset.
    <br />
    <br />
    <br />
  </p>
</p>

> tags : natural language inferencing, natural language processing, NLI, SNLI, MNLI, deep learning, tensorflow


<!-- ABOUT THE PROJECT -->
## About The Project

This project is an implementation of the task of Natural Language Inference (NLI). In this task, we are given two sentences called premise and hypothesis. We are supposed to determine whether the ”hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) provided that the ”premise” is true. For this project, we have used the [Stanford Natural Language Inference](https://nlp.stanford.edu/projects/snli/) (SNLI) dataset. From the dataset, we use the files `snli_1.0_train.jsonl` for training the model and `snli_1.0_test.jsonl` for testing the model. From each entry in these files, we consider the fields corresponding to `gold_label`, `sentence1` and `sentence2`. `sentence1` serves as the premise and `sentence2` serves as the hypothesis and `gold_label` serves as the relationship label. The following models were implemented and the performance was evaluated. 

* Logistic regression classifier using TF-IDF features
* Deep model classifiers for text such as GRU, LSTM and SumEmbeddings  
* BERT-based classifiers (Only implementation code is available. Unable to train and test due to scarcity of resources)


## Models overview

#### Logistic regression

Logistic regression model was trained using TF-IDF (Term Frequency-Inverse Document Frequency) features obtained using *sklearn* python library. The feature vector used to train the model is obtained by concatenating the TF-IDF vectors of premise and hypothesis. The model is trained (fit) using L-BFGS (Limited memory - Broyden–Fletcher–Goldfarb–Shanno algorithm) solver with a maximum iteration limit of 1000. The trained (fit) model is saved at `./models/LR.pickle` for future uses and testing. The model attains an accuracy of **63.38%** and the results of prediction are written to a file at `./tfidf.txt`.

#### Deep learning models

The first step towards implementing a deep model for text is to convert each atomic discrete entity in the input (words or characters) into real vectors from $ \mathbf{R}^{d} $ so that their semantics are captured meaningfully. For this purpose, GloVe embedding has been used in this project. Different pretrained GloVe embeddings are available and the embedding chosen for this project is the one with 6 billion tokens trained over Wikipedia corpus. Once the training is completed the model is stored at `./model/` directory as a h5 file.

##### BiGRU (Bidirectional Gated Reccurent Unit)

The model attains an accuracy of **78.58%** and the output text file and plots are saves at `./results/BiGRU/`.

##### BiLSTM (Bidirectional Long  Short Term Memory)

The model attains an accuracy of **76.38%** and the output text file and plots are saves at `./results/BiLSTM/`.

##### SumEmbeddings

A SumEmbedding lambda layer, which sums up all the embedding vectors in the sentence is used in this model. The model attains an accuracy of **80.38%** and the output text file and plots are saves at `./results/SumEmbeddings/`. Since this is the best performing model the output text files are stored at `./deep_model.txt` as well

##### BERT (Code-only)

An experimental Huggingface transformer based BERT is also implemented in the project. All the codes work and the model trains and tests, but the training process is computationally very expensive. Even after using Google Colab TPU and dividing the data into smaller parts, the training was unable to be completed due to Google usage time restrictions. Hence, the performance analysis or plots are not attached. With minor modifications to code, we can use any transformers based approach supported by Huggingface. Due to the scarcity of computing resources, I have not tried using other transformer based approaches.



<!-- RESULTS -->

## Results

|      **Model**      | **Accuracy** |
| :-----------------: | :----------: |
| Logistic regression |    63.38%    |
|        BiGRU        |    78.58%    |
|       BiLSTM        |    76.38%    |
|    SumEmbeddings    |    80.38%    |



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Harshit Samani  - harshit.samani26@gmail.com

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* SNLI Dataset

  > [Samuel R. Bowman](https://www.nyu.edu/projects/bowman/),  [Gabor Angeli](http://cs.stanford.edu/~angeli/),  [Christopher Potts](http://www.stanford.edu/~cgpotts/),  and [Christopher D. Manning](http://nlp.stanford.edu/~manning/).  2015.  A large annotated corpus for learning natural language inference.  In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

* GloVe Embeddings

  >Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.  [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
