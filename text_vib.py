
import math
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp


# Load data
data = pd.read_csv('spam_data.csv')

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = text.strip()  # Remove leading and trailing spaces
    return text

data['Message'] = data['Message'].apply(clean_text)
data['label'] = np.where(data['Category']=='ham', 0, 1)
train_msg, test_msg, train_labels, test_labels = train_test_split(data['Message'], data['label'], test_size=0.2)

# learn tokens
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_msg)

# tokenize texts
train_msg_seq = tokenizer.texts_to_sequences(train_msg)
test_msg_seq = tokenizer.texts_to_sequences(test_msg)

# add padding
maxlen = 50
train_msg_pad = tf.keras.preprocessing.sequence.pad_sequences(train_msg_seq, padding='post', 
                                                              truncating='post', maxlen=maxlen)
test_msg_pad = tf.keras.preprocessing.sequence.pad_sequences(test_msg_seq, padding='post', 
                                                             truncating='post', maxlen=maxlen)
# TODO normalize by number of tokens
# normalize
train_msg_pad = train_msg_pad / 1000.
test_msg_pad = test_msg_pad / 1000.


# One-hot encoding of labels
train_labels = tf.one_hot(train_labels, 2)
test_labels = tf.one_hot(test_labels, 2)

# Probability distributions
ds = tfp.distributions

# define Encoder
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.first_hidden_layer = tf.keras.layers.Dense(32, activation='relu')
        self.second_hidden_layer = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(4)  # 2 for mu and 2 for rho
    
    def call(self, data):
        x = self.first_hidden_layer(2 * data - 1)
        x = self.second_hidden_layer(x)
        output = self.output_layer(x)

        mu, rho = output[:, :2], output[:, 2:]
        encoding = ds.Normal(loc=mu, scale=tf.nn.softplus(rho - 5.0))
        return encoding

# define Decoder 
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense = tf.keras.layers.Dense(2)  # 2 for the 2 output classes

    def call(self, encoding_sample):
        return self.dense(encoding_sample)

# Instantiate the encoder and decoder
encoder = Encoder()
decoder = Decoder()

# Set the prior distribution
prior = ds.Normal(0.0, 1.0)

# Define the loss functions and metrics
BETA = 10**(-0.5)

def compute_loss(images, labels, encoding, logits):
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
        total_loss, class_loss, info_loss = compute_loss(images, labels, encoding, logits)
    
    # Get the trainable variables from both encoder and decoder models
    gradients = tape.gradient(total_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return total_loss, class_loss, info_loss

# Evaluate function
def evaluate(data, labels):
    encoding = encoder(data)
    sample = encoding.sample()
    logits = decoder(sample)
    # print(type(encoding))
    # print(encoding)
    # print(sample)
    # print(logits)
    
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Monte Carlo average accuracy over 12 samples
    many_encodings = encoding.sample(12)
    many_logits = decoder(many_encodings)
    avg_output = tf.reduce_mean(tf.nn.softmax(many_logits), axis=0)
    avg_correct_prediction = tf.equal(tf.argmax(avg_output, axis=1), tf.argmax(labels, axis=1))
    avg_accuracy = tf.reduce_mean(tf.cast(avg_correct_prediction, tf.float32))
    
    # print(data)
    # print(labels)
    # print(encoding)
    # print(logits)
    IZY_bound = math.log(2, 2) - compute_loss(data, labels, encoding, logits)[1]
    IZX_bound = compute_loss(data, labels, encoding, logits)[2]
    
    return IZY_bound.numpy(), IZX_bound.numpy(), accuracy.numpy(), avg_accuracy.numpy()

# Training loop
epochs = 10
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


