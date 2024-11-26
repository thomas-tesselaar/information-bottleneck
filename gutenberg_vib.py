
import math
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp


OUT_DIM = 2
DATA_PATH = "/Users/thomastesselaar/Downloads/MTHE493PreProcessing"

# ======================
# ==== Loading Data ====
# ======================
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


data = pd.DataFrame({'text':texts, 'label':labels})
train_msg, test_msg, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2)

# learn tokens
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_msg)

# tokenize texts
train_msg_seq = tokenizer.texts_to_sequences(train_msg)
test_msg_seq = tokenizer.texts_to_sequences(test_msg)

# add padding
maxlen = 500
train_msg_pad = tf.keras.preprocessing.sequence.pad_sequences(train_msg_seq, padding='post', 
                                                              truncating='post', maxlen=maxlen)
test_msg_pad = tf.keras.preprocessing.sequence.pad_sequences(test_msg_seq, padding='post', 
                                                             truncating='post', maxlen=maxlen)
# TODO normalize by number of tokens
# normalize
train_msg_pad = train_msg_pad / 1000.
test_msg_pad = test_msg_pad / 1000.

# One-hot encoding of labels
train_labels = tf.one_hot(train_labels, OUT_DIM)
test_labels = tf.one_hot(test_labels, OUT_DIM)

# Probability distributions
ds = tfp.distributions

# ============================
# ==== Defining functions ====
# ============================
# TODO these are constant across implementations
#      they should be moved to another file

# define Encoder
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.first_hidden_layer = tf.keras.layers.Dense(128, activation='relu')
        self.second_hidden_layer = tf.keras.layers.Dense(64, activation='relu')
        self.third_hidden_layer = tf.keras.layers.Dense(32, activation='relu')
        self.fourth_hidden_layer = tf.keras.layers.Dense(16, activation='relu')
        self.fifth_hidden_layer = tf.keras.layers.Dense(8, activation='relu')
        # self.sixth_hidden_layer = tf.keras.layers.Dense(8, activation='relu')
        self.output_layer = tf.keras.layers.Dense(4)  # 2 for mu and 2 for rho
    
    def call(self, data):
        x = self.first_hidden_layer(2 * data - 1)
        x = self.second_hidden_layer(x)
        x = self.third_hidden_layer(x)
        x = self.fourth_hidden_layer(x)
        x = self.fifth_hidden_layer(x)
        # x = self.sixth_hidden_layer(x)
        output = self.output_layer(x)

        mu, rho = output[:, :2], output[:, 2:]
        encoding = ds.Normal(loc=mu, scale=tf.nn.softplus(rho - 5.0))
        return encoding

# define Decoder 
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense = tf.keras.layers.Dense(OUT_DIM)

    def call(self, encoding_sample):
        return self.dense(encoding_sample)


# betas = [10**x for x in [-7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]]
# beta_accuracy = []
# for BETA in betas:
# Instantiate the encoder and decoder
encoder = Encoder()
decoder = Decoder()

# Set the prior distribution
prior = ds.Normal(0.0, 1.0)

# Define the loss functions and metrics
BETA = 10**-3

def compute_loss(labels, encoding, logits):
    class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) / math.log(2)
    info_loss = tf.reduce_mean(tfp.distributions.kl_divergence(encoding, prior)) / math.log(2)
    total_loss = class_loss + BETA * info_loss
    return total_loss, class_loss, info_loss

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Training step
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        encoding = encoder(images)
        sample = encoding.sample()
        logits = decoder(sample)
        total_loss, class_loss, info_loss = compute_loss(labels, encoding, logits)
    
    # Get the trainable variables from both encoder and decoder models
    gradients = tape.gradient(total_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return total_loss, class_loss, info_loss

# Evaluate function
def evaluate(data, labels):
    encoding = encoder(data)
    sample = encoding.sample()
    logits = decoder(sample)
    
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Monte Carlo average accuracy over 12 samples
    many_encodings = encoding.sample(12)
    many_logits = decoder(many_encodings)
    avg_output = tf.reduce_mean(tf.nn.softmax(many_logits), axis=0)
    avg_correct_prediction = tf.equal(tf.argmax(avg_output, axis=1), tf.argmax(labels, axis=1))
    avg_accuracy = tf.reduce_mean(tf.cast(avg_correct_prediction, tf.float32))
    
    IZY_bound = math.log(OUT_DIM, 2) - compute_loss(labels, encoding, logits)[1]
    IZX_bound = compute_loss(labels, encoding, logits)[2]
    
    return IZY_bound.numpy(), IZX_bound.numpy(), accuracy.numpy(), avg_accuracy.numpy()

# Training loop
epochs = 15
batch_size = 50
steps_per_batch = len(train_msg_pad) // batch_size

# Keep track of history
# Train set
IZY_array = []
IZX_array = []
acc_array = []
avg_acc_array = []

# Test set 
IZY_array2 = []
IZX_array2 = []
acc_array2= []
avg_acc_array2= []


for epoch in range(epochs):
    for step in range(steps_per_batch):
        batch_images = train_msg_pad[step * batch_size:(step + 1) * batch_size]
        batch_labels = train_labels[step * batch_size:(step + 1) * batch_size]
        train_step(batch_images, batch_labels)

    IZY, IZX, acc, avg_acc = evaluate(train_msg_pad, train_labels)
    IZY_array.append(IZY)
    IZX_array.append(IZX)
    acc_array.append(acc)
    avg_acc_array.append(avg_acc)

    IZY2, IZX2, acc2, avg_acc2 = evaluate(test_msg_pad, test_labels)
    IZY_array2.append(IZY2)
    IZX_array2.append(IZX2)
    acc_array2.append(acc2)
    avg_acc_array2.append(avg_acc2)
    print(f"Epoch {epoch + 1}, Accuracy: {acc2:.4f}, Avg Accuracy: {avg_acc2:.4f}, IZY: {IZY2:.2f}, IZX: {IZX2:.2f}")

# Save and restore model can use tf.train.Checkpoint or Keras model save, depending on specific requirements

res = pd.DataFrame({
    'Epoch':np.arange(epochs)+1,
    'TrainIZY':IZY_array,
    'TrainIZX':IZX_array,
    'TrainAcc':acc_array,
    'TrainAvgAcc':avg_acc_array,
    'TestIZY':IZY_array2,
    'TestIZX':IZX_array2,
    'TestAcc':acc_array2,
    'TestAvgAcc':avg_acc_array2,
})

res.to_csv('model_results.csv', index=False)

# beta_accuracy.append(res.iloc[-1,-2])

# pd.DataFrame({'beta':betas,'accuracy':beta_accuracy}).to_csv('beta_accuracy.csv', index=False)
