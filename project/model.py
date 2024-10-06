import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from .utils import model_config

total_words = model_config['total_words']
max_sequence_length = model_config['max_sequence_length']

def build_model_1():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 512, input_length = max_sequence_length - 1),
        tf.keras.layers.LSTM(units = 128, return_sequences = True),
        tf.keras.layers.LSTM(units = 128, return_sequences = True),
        tf.keras.layers.LSTM(units = 128),
        tf.keras.layers.Dense(total_words, activation = 'softmax')
    ])


    model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(1e-3), 
                metrics = ['accuracy'])

    model.trainable = False
    model.load_weights("project/tokenization/checkpoint_model_word_1.h5")
    
    return model

def build_model_2():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 128, input_length = max_sequence_length - 1),
        tf.keras.layers.LSTM(units = 128, return_sequences = True),
        tf.keras.layers.LSTM(units = 128),
        tf.keras.layers.Dense(total_words, activation = 'softmax')
    ])


    model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(1e-3), 
                metrics = ['accuracy'])
    
    model.trainable = False
    model.load_weights("project/tokenization/checkpoint_model_word_2.h5")

    return model