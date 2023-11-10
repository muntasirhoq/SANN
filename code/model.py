import pandas as pd
import json
import numpy as np
import argparse
import random
from config import Config
from parserBFS import parser_main
from anytree.exporter import DotExporter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from sklearn import metrics
from time import time

np.random.seed(4)

def read_data_GA(data_path):
    data_df = pd.read_csv(data_path)
    codes = data_df['Code'].tolist()
    Y = data_df['Score'].tolist()
    
    # train val split
    TEST_SIZE = 0.2
    codes_train, codes_test, Y_train, Y_test = train_test_split(codes, Y, 
                                                                         test_size=TEST_SIZE, random_state=4)
    
    # split for GA
    TEST_SIZE = 0.25
    codes_train, codes_test, Y_train, Y_test = train_test_split(codes_train, Y_train, 
                                                                         test_size=TEST_SIZE, random_state=4)
    
    return codes_test, Y_test
    
def read_data(main_df):
    raw_ast = main_df['raw_ast'].tolist()
    Y = main_df['Score'].tolist()
    
    problems = [paths.split('@@$@@') for paths in raw_ast]
    
    raw_paths = []
    
    for p in problems:
        path = [paths.split('$$@$$') for paths in p]
        raw_paths.append(path)
    
    # train val split
    TEST_SIZE = 0.2
    raw_paths_train, raw_paths_test, Y_train, Y_test = train_test_split(raw_paths, Y, 
                                                                         test_size=TEST_SIZE, random_state=4)
    
    raw_paths_train_path = []
    raw_paths_test_path = []
    
    for problem in raw_paths_train:
        problem_con = []
        for subtree in problem:
            s = " ".join(subtree)
            problem_con.append(s)
        raw_paths_train_path.append(problem_con)
            
    for problem in raw_paths_test:
        problem_con = []
        for subtree in problem:
            s = " ".join(subtree)
            problem_con.append(s)
        raw_paths_test_path.append(problem_con)
    
    return raw_paths_train, raw_paths_train_path, raw_paths_test, raw_paths_test_path, Y_train, Y_test
    
from keras_preprocessing.sequence import pad_sequences

def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary
    """
    ixtoword = {}
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword


def encode_paths(raw_paths, wordtoix, MAX_SUBTREE_LENGTH):
    config = Config()
    encoded_paths = []
    encoded_paths_padded = []
    for problem in raw_paths:
        encoded_path = []
        for subtree in problem:
            encoded_subtree = []
            for p in subtree:
                if p in wordtoix:
                    encoded_subtree.append(wordtoix[p])
                else:
                    encoded_subtree.append(wordtoix['UNK'])
            encoded_path.append(encoded_subtree)
        encoded_path_padded = pad_sequences(encoded_path, maxlen=MAX_SUBTREE_LENGTH, padding='pre', truncating='post').tolist()
        if len(problem) < MAX_SUBTREE_LENGTH:
            [encoded_path_padded.append([0]*MAX_SUBTREE_LENGTH) for i in range(MAX_SUBTREE_LENGTH - len(problem))]
        else:
            encoded_path_padded = encoded_path_padded[:MAX_SUBTREE_LENGTH]
        encoded_paths.append(encoded_path)
        encoded_paths_padded.append(encoded_path_padded)
    return encoded_paths, encoded_paths_padded
                    

def preprocess_raw_paths_wordtoken(raw_paths, MAX_SUBTREE_LENGTH):
    path_hist = {}
    for problem in raw_paths:
        for subtree in problem:
            for p in subtree:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1
    node_count = len(path_hist)
    valid_nodes = [path for path, count in path_hist.items()]
    
    wordtoix, ixtoword = create_word_index_table(valid_nodes)
    
    encoded_paths, encoded_paths_padded = encode_paths(raw_paths, wordtoix, MAX_SUBTREE_LENGTH)
    return encoded_paths, encoded_paths_padded, valid_nodes, wordtoix, ixtoword
    
    
from keras_preprocessing.sequence import pad_sequences


def encode_paths_path(raw_paths, wordtoix, MAX_SUBTREE_LENGTH):
    config = Config()
    encoded_paths = []
    encoded_paths_padded = []
    for problem in raw_paths:
        encoded_path = []
        encoded_path_padded = []
        for subtree in problem:
            if subtree in wordtoix:
                encoded_path.append(wordtoix[subtree])
                encoded_path_padded.append(wordtoix[subtree])
            else:
                encoded_path.append(wordtoix['UNK'])
                encoded_path_padded.append(wordtoix['UNK'])
        #encoded_path_padded = pad_sequences(encoded_path, maxlen=MAX_SUBTREE_LENGTH, padding='pre', truncating='post').tolist()
        if len(problem) < MAX_SUBTREE_LENGTH:
            [encoded_path_padded.append(0) for i in range(MAX_SUBTREE_LENGTH - len(problem))]
        else:
            encoded_path_padded = encoded_path_padded[:MAX_SUBTREE_LENGTH]
        encoded_paths.append(encoded_path)
        encoded_paths_padded.append(encoded_path_padded)
    return encoded_paths, encoded_paths_padded
                    

def preprocess_raw_paths_pathtoken(raw_paths, MAX_SUBTREE_LENGTH):
    path_hist = {}
    for problem in raw_paths:
        for p in problem:
            if not p in path_hist:
                path_hist[p] = 1
            else:
                path_hist[p] += 1
    node_count = len(path_hist)
    valid_nodes = [path for path, count in path_hist.items()]
    
    wordtoix, ixtoword = create_word_index_table(valid_nodes)
    
    encoded_paths, encoded_paths_padded = encode_paths_path(raw_paths, wordtoix, MAX_SUBTREE_LENGTH)
    return encoded_paths, encoded_paths_padded, valid_nodes, wordtoix, ixtoword
    
    
from gensim.models import Word2Vec

def embedding_paths(all_subtrees, vocab, embedding_size, wordtoix):
    VOCABULARY_SIZE = len(vocab) + 2
    embedding_weights = np.zeros((VOCABULARY_SIZE, embedding_size))
    #word2vec = Word2Vec(all_subtrees, size=embedding_size, window=5, workers=3, min_count=1)
    #for word, index in wordtoix.items():
    #  try:
    #    embedding_weights[index, :] = word2vec[word]
    #  except KeyError:
    #    pass
    return embedding_weights
    
    
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, concatenate, Embedding, Concatenate, Dropout, TimeDistributed, Softmax
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN, Conv1D
from tensorflow import keras
import tensorflow as tf
from keras import callbacks
import tensorflow.keras.backend as K
from config import Config

def create_model(embedding_weights, node_vocab_size, path_vocab_size, MAX_SUBTREE_LENGTH):
    config = Config()
    node_input = Input((MAX_SUBTREE_LENGTH,MAX_SUBTREE_LENGTH), dtype=tf.int32)
    path_input = Input((MAX_SUBTREE_LENGTH,), dtype=tf.int32)
    
    #embedding layer
    nodes_embedded = Embedding(node_vocab_size+2, config.embedding_size, trainable = True, name='node_embedding')(node_input)
    path_embedded = Embedding(path_vocab_size+2, config.embedding_size, 
                              trainable = True, name='path_embedding')(path_input) #(b,max_subtree,embedsize)
    
    # path embeddings from node embeddings
    nodes_embedded_merged = K.sum(nodes_embedded, axis=2) #(b,max_subtree,embedsize)
    
    # Attention Layer for node embeddings
    #node_attention_vectors = Dense(1,)(nodes_embedded)
    #node_attention_weights = Softmax()(node_attention_vectors)
    #nodes_embedded_merged = K.sum(nodes_embedded * node_attention_weights, axis=2)
    
    node_path_merged = concatenate([nodes_embedded_merged, path_embedded])
    
    subtree_vectors = TimeDistributed(Dense(config.embedding_size*2, use_bias=False, activation='tanh'))(node_path_merged)
    
    
    # Attention Layer
    attention_vectors = Dense(1,)(subtree_vectors)
    attention_weights = Softmax(axis=1)(attention_vectors)
    
    # Generating code vectors
    code_vectors = K.sum(subtree_vectors * attention_weights, axis=1)
    
    # Prediction layer
    output_class = Dense(config.num_classes, use_bias=False, activation='softmax')(code_vectors)
    
    model = Model(inputs=[node_input, path_input], outputs=output_class)
    return model
    
def train_evaluate(ga_individual_solution):   
    config = Config()
    data_path = "data.csv"
    codes, Y = read_data_GA(data_path)
    
    # Decode GA solution to integer for window_size and num_units
    max_depth_bits = BitArray(ga_individual_solution[0:3]) 
    max_depth = max_depth_bits.uint
    max_subtree_length_bits = BitArray(ga_individual_solution[3:]) 
    max_subtree_length = max_subtree_length_bits.uint
    print("************************************")
    print('Max depth: ', max_depth)
    print('Max subtree: ', max_subtree_length)
    print("************************************")
    
    # Return fitness score of 0 if max_depth is zero
    if max_depth == 0:
        return 0, 
    
    if max_subtree_length<64:
        return 0,
    
    # parse codes according to new max_depth
    main_df = parser_main(codes, Y, max_depth)
    
    # Segment the train_data based on new max_depth; split into train and validation (80/20)
    raw_paths_train, raw_paths_train_path, raw_paths_test, raw_paths_test_path, Y_train, Y_test = read_data(main_df)
    
    print("----Encoding start----")
    
    #encoding node
    encoded_paths_train, X_train, node_vocab, wordtoix, ixtoword = preprocess_raw_paths_wordtoken(raw_paths_train, 
                                                                                                  max_subtree_length)
    encoded_paths_test, X_test = encode_paths(raw_paths_test, wordtoix, max_subtree_length)
    
    #encoding path
    encoded_paths_train_path, X_train_path, path_vocab, pathtoix, ixtopath = preprocess_raw_paths_pathtoken(
        raw_paths_train_path, max_subtree_length)
    encoded_paths_test_path, X_test_path = encode_paths_path(raw_paths_test_path, pathtoix, max_subtree_length)
    print("----Encoding finished----")
    
    print("----Embedding start----")
    
    # word2vec weights calculation
    all_subtrees_train = []
    subtree_length_count = []
    for problem in raw_paths_train:
        for subtree in problem:
            all_subtrees_train.append(subtree)
            subtree_length_count.append(len(subtree))

    
    """
    Need to correctly implement word2vec
    """
    embedding_weights = embedding_paths(all_subtrees_train, node_vocab, config.embedding_size, wordtoix)
    print("----Embedding finished----")
    
    X_train = np.array(X_train)
    X_train_path = np.array(X_train_path)
    X_test = np.array(X_test)
    X_test_path = np.array(X_test_path)
    #print(Y_train.count(0), Y_train.count(1), Y_test.count(0), Y_test.count(1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    print("Train shape: Node: ", X_train.shape, " Path: ", X_train_path.shape, " Y: ", Y_train.shape)
    print("Test shape: Node: ", X_test.shape, " Path: ", X_test_path.shape, " Y: ", Y_test.shape)
    
    model = create_model(embedding_weights, len(node_vocab), len(path_vocab), max_subtree_length)
    #compile model
    model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['acc'])
    # check summary of model
    #model.summary()
    
    # Early stopping
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 10, 
                                        restore_best_weights = True)
    
    # train model
    model.fit(x=[X_train,X_train_path], y=Y_train,batch_size=config.batch_size,epochs=config.epoch, 
              validation_data=([X_test,X_test_path], Y_test), callbacks =[earlystopping])
    
    # Performance
    # This version of the code does not have test set split. Using the validation data. Please update accordingly
    predicted = model.predict(x=[X_test,X_test_path])
    #print(predicted[0], Y_test[0])
    predicted = np.where(predicted > 0.5, 1, 0)
    #print(predicted[0], Y_test[0])
    accuracy = metrics.accuracy_score(Y_test, predicted)
    print("Acuracy: ", accuracy)
    return round(accuracy, 4),
    
def train_test_evaluate(MAX_DEPTH, MAX_SUBTREE_LENGTH):   
    config = Config()
    data_path = "data.csv"
    main_df = pd.read_csv(data_path)
    codes = main_df["Code"].tolist()
    Y = main_df["Score"].tolist()
    
    max_depth = MAX_DEPTH
    
    # parse codes according to new max_depth
    main_df = parser_main(codes, Y, max_depth)
    
    # Segment the train_data based on new max_depth; split into train and validation (80/20)
    raw_paths_train, raw_paths_train_path, raw_paths_test, raw_paths_test_path, Y_train, Y_test = read_data(main_df)
    
    print("----Encoding start----")
    
    #encoding node
    encoded_paths_train, X_train, node_vocab, wordtoix, ixtoword = preprocess_raw_paths_wordtoken(raw_paths_train, 
                                                                                                  MAX_SUBTREE_LENGTH)
    encoded_paths_test, X_test = encode_paths(raw_paths_test, wordtoix, MAX_SUBTREE_LENGTH)
    
    #encoding path
    encoded_paths_train_path, X_train_path, path_vocab, pathtoix, ixtopath = preprocess_raw_paths_pathtoken(
        raw_paths_train_path, MAX_SUBTREE_LENGTH)
    encoded_paths_test_path, X_test_path = encode_paths_path(raw_paths_test_path, pathtoix, MAX_SUBTREE_LENGTH)
    print("----Encoding finished----")
    
    print("----Embedding start----")
    
    # word2vec weights calculation
    all_subtrees_train = []
    subtree_length_count = []
    for problem in raw_paths_train:
        for subtree in problem:
            all_subtrees_train.append(subtree)
            subtree_length_count.append(len(subtree))
            
    print("Maximum number of subtree: ", max(len(x) for x in raw_paths_train))
    print("Maximum subtree length: ", max(subtree_length_count))
    
    """
    Need to correctly implement word2vec
    """
    embedding_weights = embedding_paths(all_subtrees_train, node_vocab, config.embedding_size, wordtoix)
    print("----Embedding finished----")
    
    X_train = np.array(X_train)
    X_train_path = np.array(X_train_path)
    X_test = np.array(X_test)
    X_test_path = np.array(X_test_path)
    #print(Y_train.count(0), Y_train.count(1), Y_test.count(0), Y_test.count(1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    print("Train shape: Node: ", X_train.shape, " Path: ", X_train_path.shape, " Y: ", Y_train.shape)
    print("Test shape: Node: ", X_test.shape, " Path: ", X_test_path.shape, " Y: ", Y_test.shape)
    
    model = create_model(embedding_weights, len(node_vocab), len(path_vocab), MAX_SUBTREE_LENGTH)
    #compile model
    model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['acc'])
    # check summary of model
    model.summary()
    
    # Early stopping
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 20, 
                                        restore_best_weights = True)
    
    # train model
    model.fit(x=[X_train,X_train_path], y=Y_train,batch_size=config.batch_size,epochs=config.epoch, 
              validation_data=([X_test,X_test_path], Y_test), callbacks =[earlystopping])


    # Add test code here
    
 
def main():
    
    population_size = 10
    num_generations = 5
    gene_length = 10
    start = time()
    # As we are trying to minimize the RMSE score, that's why using -1.0. 
    # In case, when you want to maximize accuracy for instance, use 1.0
    creator.create('FitnessMax', base.Fitness, weights = (1.0,))
    creator.create('Individual', list , fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, 
    n = gene_length)
    toolbox.register('population', tools.initRepeat, list , toolbox.individual)

    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', train_evaluate)

    population = toolbox.population(n = population_size)
    r = algorithms.eaSimple(population, toolbox, cxpb = 0.6, mutpb = 0.2,
    ngen = num_generations, verbose = False)
    
    # Print top N solutions - (1st only, for now)
    best_individuals = tools.selBest(population,k = 1)
    best_window_size = None
    best_num_units = None

    for bi in best_individuals:
        max_depth_bits = BitArray(bi[0:3])
        best_max_depth = max_depth_bits.uint
        max_subtree_length_bits = BitArray(bi[3:])
        best_max_subtree_length = max_subtree_length_bits.uint
        print('Best Max depth: ', best_max_depth)
        print('Best Max subtree length: ', best_max_subtree_length)
        
    start1 = time()
    print("************************************")
    print("************************************")
    print("Final Training")
    print("************************************")
    print("************************************")
    train_test_evaluate(best_max_depth, best_max_subtree_length)
    
    
if __name__ == "__main__":
    
    main()
    
    
