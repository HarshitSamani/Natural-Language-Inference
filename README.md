# Natural-Language-Inference
In this repository, we deal with the task of implementing Natural Language Inferencing (NLI) using the SNLI dataset. Different methods such as BiLSTM, BiGRU, Attention models and Logistic Regression are experimented.
## About the Project
This project is an implementation of the task of Natural Language Inference (NLI). In this task, we are given two sentences called premise and hypothesis. We are supposed to determine whether the ”hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) provided that the ”premise” is true. For this project, we have used the [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) dataset. From each entry in these files, we consider the fields corresponding to gold_label, sentence1 and sentence2. sentence1 serves as the premise and sentence2 serves as the hypothesis and gold_label serves as the relationship label. 

The following models were implemented and the performance was evaluated.
1.  Logistic regression classifier using TF-IDF features
2.  Deep model classifiers for text such as GRU, LSTM
3.  Attention models

GloVe embedding from Stanford has been used throughout this project to embed the words to vectors. We have used 300d GloVe embedding with 840 billion tokens. Same can be downloaded from [here](https://nlp.stanford.edu/data/glove.840B.300d.zip).

## Models overview

### Bi-LSTM (Bidirectional Long Short Term Memory)

