
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import AutoTokenizer

OUT_DIM = 2
DATA_PATH = "/Users/thomastesselaar/Downloads/MTHE493PreProcessing"

files = ["100.txt","1016.txt","1030.txt","10039.txt","10615.txt","10616.txt","1079.txt","1080.txt","1090.txt",
         "10010.txt","10069.txt","10072.txt","10075.txt","10318.txt","10357.txt","10451.txt","102.txt","103.txt",
         "105.txt","107.txt","1015.txt","1017.txt","1022.txt","1023.txt","1024.txt","1026.txt","101.txt","106.txt",
         "108.txt","109.txt","1013.txt","1014.txt","1021.txt","1027.txt","1029.txt","1031.txt"]
flabels = [1,1,1,1,1,1,1,1729,1707,
           1,1,1,1,1,1,1,1894,1873,
           1818,1874,1847,1891,1851,1853,1892,1892,1992,1919,
           1905,1912,1901,1907,1914,1915,1916,1913]

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = text.strip()  # Remove leading and trailing spaces
    return text


class Gutenberg:
    def get_data(self, **kwargs):
        data = self.load_data()
        return self.preprocess(data, **kwargs)


    def load_data(self) -> pd.DataFrame:
        texts = []
        labels = []
        for i, fname in enumerate(files):
            book = open(f"{DATA_PATH}/{fname}", encoding='utf-8')
            text = book.read()
            # if flabels[i] > 1800:
            paragraphs = [clean_text(x) for x in text.split('\n\n') if len(x)>300]
            texts += paragraphs
            labels += [0 if flabels[i]<1800 else 1] * len(paragraphs)
                # labels += [0 if flabels[i]>1900 else 1] * len(paragraphs)

        return pd.DataFrame({'text':texts, 'label':labels})
    

    def preprocess(self, data: pd.DataFrame, num_tokens: int = 1000, pad: bool = True, normalize: bool = True, 
                   maxlen: int = 500, tokenizer_name: str = 'tf', **kwargs):
        train_msg_raw, test_msg_raw, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2)

        # learn and tokenize tokens
        if tokenizer_name.lower() == tf:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_tokens, oov_token='<OOV>')
            tokenizer.fit_on_texts(train_msg_raw)
            train_msg_seq = tokenizer.texts_to_sequences(train_msg_raw)
            test_msg_seq = tokenizer.texts_to_sequences(test_msg_raw)
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            train_msg_seq = tokenizer(train_msg_raw.to_list(), truncation=True, padding=True, max_length=512)["input_ids"]
            test_msg_seq = tokenizer(test_msg_raw.to_list(), truncation=True, padding=True, max_length=512)["input_ids"]

        # add padding
        if pad and tokenizer_name.lower() != 'bert':
            train_msg = tf.keras.preprocessing.sequence.pad_sequences(train_msg_seq, padding='post', 
                                                                        truncating='post', maxlen=maxlen)
            test_msg = tf.keras.preprocessing.sequence.pad_sequences(test_msg_seq, padding='post', 
                                                                    truncating='post', maxlen=maxlen)
        else:
            train_msg, test_msg = np.array(train_msg_seq), np.array(test_msg_seq)

        # TODO normalize by number of tokens
        # normalize
        if normalize:
            train_msg = train_msg / float(num_tokens)
            test_msg = test_msg / float(num_tokens)

        # One-hot encoding of labels
        train_labels = tf.one_hot(train_labels, OUT_DIM)
        test_labels = tf.one_hot(test_labels, OUT_DIM)

        return train_msg, test_msg, train_labels, test_labels