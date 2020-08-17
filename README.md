# SCB_occupation_classifier
Python code develped at SCB for text preprocessing and label classification with NN model

With the occupation_text_preprocessing.py, text data can be explored and the word length statistics can be calculated. This file can also clean the text, tokenize the text string, clean the unwated characters, delete stop words, stemming the word. The cleaned text can be saved in csv files for classification.

With the occupation_modelengineering.py, the cleaned text data and the labels are processed for model training. The data are divided into training and and testing data sets.

The processing including Tfidf coding, chi2 feature selecting. After the text have been transformed into vector, the data are used for tuning process for chossing the best hyper-parameters.

In the last, for model is trained by the training data and evealuated by the test data.

The feature selctor and the model can be saved and reloaded for new data prediction.

The code has been inspired by https://developers.google.com/machine-learning/guides/text-classification
