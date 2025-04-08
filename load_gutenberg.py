
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import AutoTokenizer

OUT_DIM = 2
DATA_PATH = "./data"

#These files come from data folder in repository. Could be rewritten to include more files or expanded to a larger dataset
files = ["100.txt","1016.txt","1030.txt","10039.txt","10615.txt","10616.txt","1079.txt","1080.txt","1090.txt",
         "10010.txt","10069.txt","10072.txt","10075.txt","10318.txt","10357.txt","10451.txt","102.txt","103.txt",
         "105.txt","107.txt","1015.txt","1017.txt","1022.txt","1023.txt","1024.txt","1026.txt","101.txt","106.txt",
         "108.txt","109.txt","1013.txt","1014.txt","1021.txt","1027.txt","1029.txt","1031.txt"]
flabels = [1623,1600,1600,1600,1600,1600,1700,1729,1707,
           1700,1775,1764,1798,1784,1784,1774,1894,1873,
           1818,1874,1847,1891,1851,1853,1892,1892,1992,1919,
           1905,1912,1901,1907,1914,1915,1916,1913]

#Function to strip text, ensures compatability with tokenizer
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = text.strip()  # Remove leading and trailing spaces
    return text


class Gutenberg:
    def get_data(self, **kwargs):
        data = self.load_data()
        return self.preprocess(data, **kwargs)

# This function reads each book, groups into paragraphs, and strips of white space 
# and unreadable characters, and creates text with year labels
    def load_data(self, min_words: int = 300) -> pd.DataFrame:
        texts = []
        labels = []
        for i, fname in enumerate(files):
            book = open(f"{DATA_PATH}/{fname}", encoding='utf-8')
            text = book.read()
            # Binary label is if book is before or after 1800
            paragraphs = [clean_text(x) for x in text.split('\n\n') if len(x)>min_words]
            texts += paragraphs
            # labels += [0 if flabels[i]<1800 else 1 if flabels[i]<1900 else 2] * len(paragraphs)
            # labels += [flabels[i]//100 - 16] * len(paragraphs)
            labels += [0 if flabels[i]>1800 else 1] * len(paragraphs)

        return pd.DataFrame({'text':texts, 'label':labels})
    

    def preprocess(self, data: pd.DataFrame, num_tokens: int = 2048, pad: bool = True, normalize: bool = True, 
                   maxlen: int = 500, tokenizer_name: str = 'tf', verbose: bool = False, **kwargs):
        train_msg_raw, test_msg_raw, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2)

        # learn and tokenize tokens
        if tokenizer_name.lower() == 'tf':
                 #About 5% of tokens are mapped to <OOV>
            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_tokens, oov_token='<OOV>')
            tokenizer.fit_on_texts(train_msg_raw)
            train_msg_seq = tokenizer.texts_to_sequences(train_msg_raw)
            test_msg_seq = tokenizer.texts_to_sequences(test_msg_raw)
        else:
                 #Bert tokenizer not reccomended since the size of the vocab is unsuitable for embeddings when training
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            train_msg_seq = tokenizer(train_msg_raw.to_list(), truncation=True, padding=True, max_length=maxlen)["input_ids"]
            test_msg_seq = tokenizer(test_msg_raw.to_list(), truncation=True, padding=True, max_length=maxlen)["input_ids"]

        # add padding, this ensures that sequences are all equal to 500, compatibility with training
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

        if verbose:
            oov_count = sum(token == 1 for seq in train_msg for token in seq)
            total_tokens = sum(len(seq) for seq in train_msg)
            oov_percentage = (oov_count / total_tokens) * 100 if total_tokens > 0 else 0

            print(f"Total Tokens: {total_tokens}")
            print(f"OOV Tokens: {oov_count}")
            print(f"OOV Percentage: {oov_percentage:.2f}%")

        # One-hot encoding of labels
        train_labels = tf.one_hot(train_labels, OUT_DIM)
        test_labels = tf.one_hot(test_labels, OUT_DIM)

        return train_msg, test_msg, train_labels, test_labels
    
    
if __name__ == "__main__":
    from vib import VIB
    gutenberg = Gutenberg()
    for n in [50, 100, 200, 300, 400, 500]:
        print(f"Number of words: {n}")
        data = gutenberg.load_data(min_words=n)
        print(f"Number of samples: {len(data)}")
        acc = []
        for i in range(5):
            train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.preprocess(data, num_tokens=2048, normalize=False)
            _data = {'train_data': train_msg_pad, 'test_data': test_msg_pad, 
                    'train_labels': train_labels, 'test_labels': test_labels}
        
            # Instantiate and train the model
            vib = VIB(encoder_args={'num_layers':2, 'num_units':[128,64]})
            res = vib.train(_data, epochs=8, batch_size=50, beta=10**-4, alpha=1.0, verbose=False)
            acc.append(res['Test acc'].iloc[-1])
            print(f"Trial {i+1}, Accuracy: {res['Test acc'].iloc[-1]:.4f}")

        print(f"Average accuracy: {np.mean(acc):.4f}")

    # for n in [256, 512, 1024, 2048, 4096, 8192]:
    #     print(f"Number of tokens: {n}")
    #     data = gutenberg.get_data(num_tokens=n, verbose=True, normalize=False)
    #     # print(f"Number of samples: {len(data)}")
