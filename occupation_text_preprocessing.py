import pandas as pd
import collections
import re
import statistics
import numpy as np
import matplotlib as plt
import sqlalchemy as sa
from sklearn.utils import resample
import nltk.tokenize as tokenize
"""
Author:
(SCB) Wu, Dan
# This file contains functions for data exploration, text cleaning and preprocessing
# In main() function, other functions can be called in different order for data processing.
"""


def get_file_data(path=r'path to the file\raw_text.csv')
    # The function access to data in csv files
    # Arguments:
    #   # path -  the path to csv file
    # Return:
    #   #a tuple (advertisment titles, labels)
    text_list = []
    label_list = []
    df_sample = pd.read_csv(path, sep=',')
    # the column name is text
    text_list = df_sample.text.values.tolist()
    # the column name is ssyk
    label_list = df_sample.ssyk.values.tolist()

    return (text_list, label_list)


def get_db_data(en, sample_size=800):
    # Access to data in DB, where the total dataset is saved, we can draw different samples from the DB
    # Arguments:
    #   en - the connection string to the database
    #   sample_size - the size of each category in the sample
    # Return:
    # # a tuple (advertisment titles, labels)
    sql = 'sql select statement'
    df = pd.DataFrame()
    df_sample = pd.DataFrame()
    # when datasize is big use chunksize retrieving data
    # concate each chunk into one dataframe
    for chunk in pd.read_sql(sql, en, chunksize=70000):
        df = pd.concat([df, chunk])
    # extract the first two digits of the profession code
    df['ssyk2'] = df['SSYK'].apply(lambda x: str(x)[:2])
    #print(df.head())
    # in each profession group ramdomly select sample with a chosen size
    ssyk_category = df.ssyk2.unique()
    #for 1000 samples random state is 0
    for kod in ssyk_category:
        #print(f'sample on ssyk {kod}')
        piece = df.loc[df['ssyk2'] == kod]
        piece_sample = resample(piece,
                                    n_samples=sample_size,
                                    random_state=1)
        #print(piece_sample.shape)
        # concate sample data into one dataframe
        df_sample = pd.concat([df_sample, piece_sample])
    # column names are PLATSRUBRIK and ssyk2
    text_list = df_sample.PLATSRUBRIK.values.tolist()
    label_list = df_sample.ssyk2.values.tolist()

    #print(f' text list {text_list[100]}; labels are {label_list[100]}')

    return (text_list, label_list)


def preprocess_text(string_list, stemming=True):
    # The function clean the text e.g. delete stop words and so on
    # Arguments:
    # string_list: the strings are the titles of advertisements
    # stemming: if True go through stemming, otherwize not
    # Return:
    # clean text as string list without stopwords and not wanted characters
    from nltk.corpus import stopwords
    #use both swedish and english stopwords, since the text contain small portion of english text
    swed_stpwrds = stopwords.words("swedish")
    eng_stpwrds = stopwords.words("english")
    # some location words are defined as stop words
    self_defined = [
        'stockholm', 'göteborg', 'malmö', 'kommun', 'jobba', 'tjänst', 'hos',
        'jobb', 'arbeta', 'uppsala', 'ab', 'vill', 'helsingborg', 'västerås',
        'företag', 'söker', 'sökes', 'örebro', 'linköping', 'lund', 'sveriges'
    ]
    stpwrds = swed_stpwrds + eng_stpwrds + self_defined

    new_list = []
    for text in string_list:
        #extract only the correct characters in the string
        new_text = re.sub(r'[^a-zåäö]',
                          ' ',
                          str(text).lower(),
                          flags=re.IGNORECASE)
        #transform the text into word list
        new_text_list = tokenize.wordpunct_tokenize(new_text)
        #delte sotpwords
        new_text_list = [
            word for word in filter(lambda x: x.lower() not in stpwrds,
                                    new_text_list)
        ]
        if stemming:
            from nltk.stem.snowball import SwedishStemmer
            stemmer = SwedishStemmer()
            new_text_list = [stemmer.stem(x) for x in new_text_list]
        new_list.append(' '.join(new_text_list))
    return new_list


def get_categorystatistics(string_list, label_list):
    # Arguments:
    # string_list: list of advertisements' titles
    # label_list: list of labels
    # Return:
    #  the statistics of words' count of each category and total

    df = pd.DataFrame({'text': string_list, 'ssyk': label_list})
    #method 1: start
    #calculate words in each title and put the length of words in a column
    df['text_length'] = df['text'].apply(
        lambda x: len(tokenize.wordpunct_tokenize(str(x))))
    mean_statistics = df.groupby(by='ssyk')['text_length'].mean()
    print(mean_statistics.index)
    print(mean_statistics.values)
    median_statistics = df.groupby(by='ssyk')['text_length'].median()
    print(median_statistics.index)
    print(median_statistics.values)
    min_statistics = df.groupby(by='ssyk')['text_length'].min()
    print(min_statistics.index)
    print(min_statistics.values)
    max_statistics = df.groupby(by='ssyk')['text_length'].max()
    print(max_statistics.index)
    print(max_statistics.values)
    total_mean = df['text_length'].mean()
    print(total_mean)
    total_median = df['text_length'].median()
    total_min = df['text_length'].min()
    total_max = df['text_length'].max()
    # statistics of each label (category)
    result = pd.DataFrame({
        'ssyk': mean_statistics.index.tolist(),
        'mean': mean_statistics.values.tolist(),
        'median': median_statistics.values.tolist(),
        'min': min_statistics.values.tolist(),
        'max': max_statistics.values.tolist()
    })
    #statistics of total
    result = result.append(
        {
            'ssyk': 'total',
            'mean': total_mean,
            'median': total_median,
            'min': total_min,
            'max': total_max
        },
        ignore_index=True)
    print(result.tail())
    return result


def get_wordCounts(string_list, number_words=50):
    # The function appends all the strings in the list into one list and tokenize the list
    # return a counter of the most common words
    # Arguments:
    # # string_list: text of advertisments' title
    # Return:
    # Counter of the most common wirds of the string list

    from collections import Counter
    whole_list = []
    for substring in string_list:
        #check if substring is nan, if not, add the tokens into list
        if substring == substring:
            sublist = tokenize.wordpunct_tokenize(substring)
            for word in sublist:
                whole_list.append(word.upper())
    print(f'text list is {len(string_list)}')
    print(f'total words are {len(whole_list)}')
    result = Counter(whole_list).most_common(number_words)
    return result


if __name__ == '__main__':

    # step 1: retreive data from DB
    en = sa.create_engine(
        'DB connection string')
    (text, labels) = get_db_data(en, sample_size=5000)
    #print(text[1], labels[1])
    # step2: save data into csv file
    df = pd.DataFrame({
        'text': text,
        'ssyk': labels
    })
    df.to_csv(r'path to data file\raw_data5000.csv',
              index=False)
    
    #step 3 explore the raw date
    rawtext_statistics = get_categorystatistics(text, labels)
    #rawtext_statistics.to_csv(
    #    r'path to file\rawdata_statistics_5000.csv',
    #    index=False)

    # step 4: clean the data
    cleantext = preprocess_text(text)
    #print(len(cleantext))
    cleantext_df = pd.DataFrame({'text': cleantext, 'ssyk': labels})
    cleantext_df.to_csv(
        r'path to the file\clean_text5000.csv',
        index=False)


    # the distribution of the text length in the text corpus
    #print(cleantext_df.words_counts.value_counts())
    #words_counts_group = cleantext_df.groupby(
    #    by='words_counts')['text'].count()
    #path = r'path to the file\rawtext_words_distribution.csv
    #most_common = get_wordCounts(cleantext_df.text.values.tolist(),
    #                             number_words=100)
    #words = []
    #counts = []
    #for (w, c) in most_common:
    #    words.append(w)
    #    counts.append(c)
    #words_counts_df = pd.DataFrame({'words': words, 'counts': counts})
    #words_counts_df.to_csv(
    #    r'E:\TSTDAWU\results\Essnet2020_report\common100words_sample5000.csv',
    #    index=False)
    #print(words_counts_df.head())
    #print(cleantext_df['words_counts'].min())
    #df_notitle = cleantext_df.loc[cleantext_df['words_counts'] == 0]
    #print(df_notitle.shape)
    #print(df_notitle)

    #print(rawtext_statistics)
    #print(cleantext_statistics)