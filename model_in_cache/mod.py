import streamlit as st 
import logging
import tensorflow as tf 
from modules.nmt import load_dataset
from built.nmt_model import model_nmt
from built.siamense import built_siamense_model, TripletLoss
import os
import numpy as np
import pandas as pd 
from transformers import (pipeline, AutoModelForTokenClassification, AutoTokenizer,
                          AutoModelForQuestionAnswering, Trainer, TrainingArguments)

from transformers import BertForTokenClassification, BertTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


@st.cache_data()
def read_glove_vecs(glove_file = './data/glove.6B.50d.txt'):
    with open(glove_file, 'rb') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            #word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        #index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            #index_to_words[i] = w
            i = i + 1
    return words_to_index #, index_to_words, word_to_vec_map

@st.cache_resource()
def load_all_models(machine_vocab = None):
    loaded_sentence_vectorizer  = read_tokenizer()

    all_models = read_models(machine_vocab=machine_vocab)
    train_data = read_senquences()
    loaded_sentence_vectorizer.adapt(train_data)
    
    del train_data 

    return all_models, loaded_sentence_vectorizer 

def read_tokenizer(path='./models/sentence_vectorizer_config.json'):
    import json

    with open(path, 'r') as config_file:
        loaded_config = json.load(config_file)
        loaded_sentence_vectorizer = tf.keras.layers.TextVectorization.from_config(loaded_config)

    return loaded_sentence_vectorizer 

def load_data(file_path):
    import numpy as np 

    with open(file_path,'rb') as file:
        data = np.array([line.strip() for line in file.readlines()])
    return data

def read_senquences():
    path = "./data/sentences.txt"
    train_sentences = load_data(file_path=path)

    return train_sentences

def read_models(list_of_models : list = ["NMT.keras", "NER.keras", 'siamense.keras', 
                'emojify.h5', 'HF_QA.HF1', 'HF_QA_FT.HF', 
                "HF_SA.HF1"], machine_vocab = None): #"HF_NER.HF",
    Models = {}

    for names in list_of_models:
        print(names)
        if   names == 'NER.keras':
            with tf.keras.utils.custom_object_scope({"loss_package>masked_loss" : masked_loss,
                                                     'accuracy_package>masked_accuracy' : masked_accuracy
                                                     }):
                # Chargez le modèle
                model = tf.keras.models.load_model(f"./models/{names}")
        elif names == "NMT.keras":
            model = model_nmt(machine_vocab=machine_vocab).built()
            model.load_weights("./models/NMT_W.h5")
        elif names == 'siamense.keras':
            with tf.keras.utils.custom_object_scope({"tripletloss_package>TripletLoss" : TripletLoss }):
                model = tf.keras.models.load_model(f"./models/siamense/siamense.tf", compile=True)
            model = load_siamense_weights(model=model)   
        elif names == 'emojify.h5':
            model = tf.keras.models.load_model(f"./models/{names}", compile=True)    
        elif names == 'HF_SA.HF':
            """
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            Models['HF_SA_T'] = tokenizer
            model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            model.save_pretrained("./models/HF_SA/")
            """
            model       = DistilBertForSequenceClassification.from_pretrained("./models/HF_SA/")
            tokenizer   = DistilBertTokenizer.from_pretrained("./models/HF_SA/")
            Models['HF_SA_T'] = tokenizer

        elif names == 'HF_NER.HF1':
            tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
            model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
            Models['HF_NER_T'] = tokenizer

        elif names == 'HF_QA_FT.HF':
            model       = AutoModelForQuestionAnswering.from_pretrained('./models/HF_QA_FT/')
            tokenizer   = AutoTokenizer.from_pretrained('./models/HF_QA_FT/')
            Models['HF_QA_T'] = tokenizer

        else:
            """
            model = pipeline(task="question-answering", model = "distilbert-base-cased-distilled-squad")
            model.save_pretrained("./models/HF_QA/")
            """
            model       = AutoModelForQuestionAnswering.from_pretrained("./models/HF_QA/")
            tokenizer   = AutoTokenizer.from_pretrained("./models/HF_QA/")
            model = pipeline(task="question-answering", model = model, tokenizer=tokenizer)
            

        Models[names.split('.')[0]] = model

        del model 

    return Models

def load_siamense_weights(model):
    loaded_weights = [np.load(f'./models/siamense_weights/weight_{i}.npy', allow_pickle=True) for i in range(5)]
    model.set_weights(loaded_weights)

    return model

@st.cache_data()
def read_data(m : int) -> tuple[list, dict, dict, dict]:
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m) 

    return dataset, human_vocab, machine_vocab, inv_machine_vocab 

@st.cache_data()
def read_question_siamense():
    data        = pd.read_csv("./data/questions.csv")
    N_train     = 300000
    data_train  = data[:N_train]
    del (data)

    td_index    = data_train['is_duplicate'] == 1
    td_index    = [i for i, x in enumerate(td_index) if x]

    Q1_train    = np.array(data_train['question1'][td_index])
    Q2_train    = np.array(data_train['question2'][td_index])

    cut_off     = int(len(Q1_train) * 0.8)
    train_Q1, train_Q2  = Q1_train[:cut_off], Q2_train[:cut_off]

    return (train_Q1, train_Q2), (Q1_train, Q2_train)

def log_softmax(x):
    return tf.nn.log_softmax(x)

def masked_loss(y_true, y_pred):
    """
    Calculate the masked sparse categorical cross-entropy loss.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.
    
    Returns:
    loss (tensor): Calculated loss.
    """
    
    ### START CODE HERE ### 
    
    # Calculate the loss for each item in the batch. Remember to pass the right arguments, as discussed above!
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1, from_logits=True)
    # Use the previous defined function to compute the loss
    loss = loss_fn(y_true, y_pred)
    
    ### END CODE HERE ### 

    return  loss 

def masked_accuracy(y_true, y_pred):
    """
    Calculate masked accuracy for predicted labels.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    accuracy (tensor): Masked accuracy.

    """
    
    ### START CODE HERE ### 
    
    # Calculate the loss for each item in the batch.
    # You must always cast the tensors to the same type in order to use them in training. Since you will make divisions, it is safe to use tf.float32 data type.
    y_true = tf.cast(y_true, tf.float32) 
    # Create the mask, i.e., the values that will be ignored
    mask = tf.math.not_equal(y_true, -1)
    mask = tf.cast(mask, tf.float32) 
    # Perform argmax to get the predicted values
    y_pred_class = tf.math.argmax(y_pred, axis=-1)
    y_pred_class = tf.cast(y_pred_class, tf.float32) 
    # Compare the true values with the predicted ones
    matches_true_pred  = tf.equal(y_true, y_pred_class)
    matches_true_pred = tf.cast(matches_true_pred , tf.float32) 
    # Multiply the acc tensor with the masks
    matches_true_pred *= mask
    # Compute masked accuracy (quotient between the total matches and the total valid values, i.e., the amount of non-masked values)
    masked_acc = tf.reduce_sum(matches_true_pred)/tf.reduce_sum(mask)
    
    ### END CODE HERE ### 

    return masked_acc