import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def numerical_preprocessing(df, input):
    '''
    Preprocessing CSV Numeric Inputs
    '''
    input_x = {}
    for name, column in df.items():
        if column.dtype == object:
            column.dtype = tf.string
        else:
            column.dtype == tf.float32
        input[name] = tf.keras.Input(shape = (1,), name = name, column = column)

    num_inputs = {name: input for name, input in input_x.items() if input.dtype == tf.float32}
    x = tf.keras.layers.Concatenate(list(num_inputs.values()))
    normalization_layer = tf.keras.layers.Normalization()
    normalization_layer.adapt(np.array(df[num_inputs.keys()]))
    all_num_inputs = normalization_layer(x)
    preprocessed_inputs = [all_num_inputs]
    return all_num_inputs, preprocessed_inputs

def string_preprocessing(df, input_x, preprocessed_inputs):
    '''
    Preprocessing CSV String Inputs
    '''
    for name, input in input_x.items():
        if input.dtype != tf.string:
            continue
        lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary = np.unique(df[name]))
        one_hot = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens = lookup.vocab_size())

        x = tf.strings.lower(name)
        x = tf.strings.strip(x)
        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)
        return preprocessed_inputs

def generate_input_model(inputs, outputs):
    preprocessing_layer = tf.keras.Model(inputs = inputs, outputs = outputs)
    return preprocessing_layer

def get_data_dicts(data):
    data_dict = {name: np.array(value) for name, value in data.items()}
    two_sample_dict = {name: values[1:3,] for name, values in data_dict.items()}
    fitted_dict = generate_input_model(two_sample_dict)
    return data_dict, two_sample_dict, fitted_dict


    

