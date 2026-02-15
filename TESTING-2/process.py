import sys
import json
import random
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

global responses, lemmatizer, tokenizer, le
global interpreter, input_details, output_details
input_shape = 11


def load_response():
    global responses
    responses = {}
    with open('dataset/Dataset3.json') as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']


def preparation():
    global lemmatizer, tokenizer, le
    global interpreter, input_details, output_details

    load_response()

    tokenizer = pickle.load(open('model/tokenizers.pkl', 'rb'))
    le = pickle.load(open('model/le.pkl', 'rb'))

    interpreter = tf.lite.Interpreter(model_path="model/inimodelygy.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    lemmatizer = WordNetLemmatizer()

    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def remove_punctuation(text):
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    return [text]


def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector.astype(np.float32)


def predict(vector):
    interpreter.set_tensor(input_details[0]['index'], vector)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag


def generate_response(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer
