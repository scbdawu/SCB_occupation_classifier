import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import initializers

import pandas as pd
import numpy as np
import random
import statistics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from pickle import dump, load

from sklearn.pipeline import make_pipeline
from sklearn import svm
"""
Author:
(SCB) WU, Dan
This file contains functions for feature engineering, modelling and plotting
"""


def plot_confusion_matrix(cm, class_names):
    # Arguments:
    #    #cm: confusion matrix
    #    #class_names: classes of the true name and the prediction name
    # Return:
    #   #a matplotlob figure showing cm
    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(46)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # predicted of the class/total predicted of the class
    cm = np.around(np.array(cm).astype('float') /
                   np.array(cm).sum(axis=1)[:, np.newaxis],
                   decimals=1)
    threshold = np.array(cm).max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j,
                 i,
                 cm[i, j],
                 fontsize=7,
                 horizontalalignment="center",
                 color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def ngram_vectorizer(train_text, train_label, val_text):
    # Transform text into n-gram vectors
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # Arguments:
    # train_text: text corpus for training
    # train_label: labels
    # val_text: evaluation text corpus
    # Return:
    #   x_train, x_val: vectorizing matrix from the input
    keyargs = {
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',
        'min_df': 2
    }
    vectorizer = TfidfVectorizer(**keyargs)

    x_train = vectorizer.fit_transform(train_text)
    x_val = vectorizer.transform(val_text)
    #the minimun feature for ngram model is 4000
    selector = SelectKBest(chi2, k=min(4000, x_train.shape[1]))
    selector.fit(x_train, train_label)
    # code below study the features selected,check mean and draw feature scores
    # pscores = selector.pvalues_
    #scores divide the max scores
    #scores /= scores.max()
    # print('---scores----')
    # Further analyzing the scores and features
    # sorted_pscores = sorted(pscores)
    # print(statistics.mean(sorted_pscores))
    # dfscores = pd.DataFrame(pd.Series(selector.scores_))
    # dfcolumns = pd.DataFrame(pd.Series(vectorizer.get_feature_names()))
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['feature', 'score']
    # featureScores.sort_values(by=['score'], inplace=True, ascending=False)
    # print(featureScores.head(150))
    # print(featureScores.score.describe())
    # featureScores.head(150).to_csv(
    #    r'path to the file\features.csv', index=False)
    #plt.savefig(
    #    r'path to the feature-score graph\features_contribution.png')
    # save the selector into pkl file for future usage
    # selector_path = r'path to the selector\selector.pkl'
    # dump(selector, open(selector_path), 'wb')
    x_train = selector.transform(x_train).astype('float64')
    x_val = selector.transform(x_val).astype('float64')
    return x_train.toarray(), x_val.toarray()


def classifier_model(layers,
                     units,
                     dropout_rate,
                     input_shape,
                     num_classes,
                     embedding=False):
    # a multi-layer perceptron model
    # Arguments:
    #layers: int, number of 'dense layers in the model
    #units: int, output dimension of the layers
    #dropout_rate: float, perentage of input to drop at Dropout layers
    #input_shaple: tuple, input shape
    #num_classes: int, number of classes
    #Return:
    #  An MLP model instance
    model = models.Sequential()
    #if we want to train embedding first, add one embedding layer
    if embedding is True:
        model.add(Embedding(input_dim=input_shape, output_dim=units))
        model.add(GlobalAveragePooling1D())
    else:
        model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    #first and other layers
    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
    #last layer: units is the class number and activation function is softmax
    model.add(Dense(units=46, activation='softmax'))
    return model


def train_ngram_model(data,
                      learning_rate=0.0003,
                      epochs=50,
                      batch_size=128,
                      layers=2,
                      units=4000,
                      dropout_rate=0.1):
    # total number of the classes
    number_of_classes = 46
    # access to the data
    (train_text, train_label), (val_text, val_label) = data
    x_train, x_val = ngram_vectorizer(train_text, train_label, val_text)

    model = classifier_model(layers=layers,
                             units=units,
                             dropout_rate=dropout_rate,
                             input_shape=x_train.shape[1:],
                             num_classes=number_of_classes)
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]

    history = model.fit(x_train,
                        train_label,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, val_label),
                        verbose=2,
                        batch_size=batch_size)
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))

    model.save(r'path to the model\imdb_mlp_4000model.h5')
    #if the fuction is called by tune_ngram_model, return the value below
    return history['val_accuracy'][-1], history['val_loss'][-1]
    #else return the history for plot the picture
    #return history


def tune_ngram_model(data):
    # THe function draw the tuning result of hyper-parameter
    # in order to choose the best parameters for the model
    # Argument:
    # data: is training_data, labels and evaluating data, labels
    # Return:
    # the picture of the tuning result of hyper-parameters
    num_layers = [1, 2, 3]
    # need to be adjusted by the input data shape!
    num_units = [10000, 7000, 6000, 5500, 4000]
    params = {'layers': [], 'units': [], 'accuracy': []}

    for layers in num_layers:
        for units in num_units:
            params['layers'].append(layers)
            params['units'].append(units)

            accuracy, _ = train_ngram_model(data=data,
                                            layers=layers,
                                            units=units)
            print(('Accuracy: {accuracy}, Parameters: (layers={layers}, '
                   'units={units})').format(accuracy=accuracy,
                                            layers=layers,
                                            units=units))
            params['accuracy'].append(accuracy)
    _plot_parameters(params)


def _plot_parameters(params):
    # The function is called by the tune_ngram_mode to draw the result
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(params['layers'],
                    params['units'],
                    params['accuracy'],
                    cmap=cm.coolwarm,
                    antialiased=False)
    plt.show()


def loadModel_predict(x_test, test_label):
    # load a model ready-trained and predict the classification result for x
    # Argument:
    #   #x_test: clean data of text description
    #   #test_label: class label
    # Return:
    # the prediction result in confusion matrix format
    # the selector has been saved in function ngram_vectorizer
    path = 'file path where the feature selector model is saved'
    selector = load(open(path, 'rb'))
    x_test_transformed = selector.transform(x_test)
    model = keras.models.load_model(path)
    prediction_result = model.predict(x_test_transformed,
                                      verbose=1,
                                      use_multiprocessing=True)
    class_result = np.argmax(prediction_result, axis=1)
    cm = tf.math.confusion_matrix(labels=test_label,
                                  predictions=class_result,
                                  num_classes=46)
    return cm


def get_evaluation(cm, tot):
    # the function calculates recall, precision, accuracy for all groups and the average
    # with the predicted result cm
    # Arguments:
    #   #cm: confusion matrix,
    #   # tot: sample number
    # Return:
    # A dictionary of the metrics for each class and the average of total

    true_pos = np.diag(cm)
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)
    precision = np.around(true_pos / col_sum, decimals=3)
    recall = np.around(true_pos / row_sum, decimals=3)
    #now calculate the accuracy: TP+TN/TP+TN+FP+FN
    true_neg = tot - row_sum - col_sum + true_pos
    acc = np.around((true_pos + true_neg) / tot * 100, decimals=3)
    #F1 score : 2*(recall*precision)/Recall+Precision)
    f1 = np.around(2 * (recall * precision) / (recall + precision), decimals=3)
    avg = [
        np.average(recall),
        np.average(precision),
        np.average(acc),
        np.average(f1)
    ]
    res = {'recall': recall, 'precision': precision, 'acc': acc, 'f1': f1}
    return (res, avg)


def standard_ssyk(ssyk):
    # The function returns an integer, tensorflow require the classes is coded in 0, 1, 2, ...n
    # therefore our occupation code need to be transformed into tensorflow format
    # ssyk code need to be adjusted into the format required by tensorflow
    # Arguments:
    #  ssyk: a ssyk2digits code in string format
    # Return:
    #  a standard ssyk code in integer, enabling runing in tensorflow fit
    path = 'path to the file containing original occupation code and the standardard code',
    df = pd.read_csv(path, sep=';', dtype={'ssyk': str, 'standard': str})
    # the columns is adjusted to the standard file
    df.columns = ['ssyk', 'description', 'standard']
    # compare the input code with the standard table
    result_df = df.loc[df['ssyk'] == str(ssyk)]
    if result_df.shape[0] == 1:
        result = int(result_df.standard.values[0])
    else:
        print('Please check the ssyk value')
        result = -1
    return result


if __name__ == '__main__':
    # step 1: load clean data
    train_df = pd.read_csv(r'path to the clean data\clean_text.csv',
                           dtype={
                               'text': str,
                               'ssyk': str
                           })
    train_df.fillna('', inplace=True)
    #print(train_df.ssyk.value_counts())
    #print('----train_df----')
    #print(train_df.head())
    # generate the occupation standard code for tensorflow
    train_df['ssyk_standard'] = train_df['ssyk'].apply(
        lambda x: standard_ssyk(x))

    # step 2: split the data into train and text dataset
    train_text, test_text, train_label, test_label = train_test_split(
        train_df.text.values,
        train_df.ssyk_standard.values,
        test_size=0.33,
        random_state=30)

    #step 3: use tune_ngram_model first, then comment this line, select the best hyperparameters in train_ngram_model
    #data = (train_text, np.array(train_label)), (test_text,
    #                                             np.array(test_label))
    #tune_ngram_model(data)

    #step 4: use the best hyperparameters in train_ngram_model, comments step3
    #tf.debugging.set_log_device_placement(True)
    #train_result = train_ngram_model(data=((train_text, np.array(train_label)),
    #                                       (test_text, np.array(test_label))))
    #print(train_result)

    #step 5: predict new data with the classifier model and feature selector saved
    #get prediction result on evaluating data, test_label is used only for comparison with the prediction values
    #result = loadModel_predict(df.text.values.tolist(),
    #                           df.ssyk_standard.values.tolist())

    # step 6: examine the metrics of each class
    #print(get_evaluation(result, test_text.shape[0]))
    #print(result)
    #plot_confusion_matrix(result, ssyk_df['standard'].values.tolist())

    # together with step 4, the training history is drawn into graph
    # #Draw learning history
    #print(result.keys())
    # # plot the history data and understand the model
    #plt.plot(result['accuracy'])
    #plt.plot(result['val_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
