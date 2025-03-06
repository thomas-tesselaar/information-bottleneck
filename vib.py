import math
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from load_gutenberg import Gutenberg

# Probability distributions
ds = tfp.distributions

# ============================
# ==== Defining functions ====
# ============================
class Encoder(tf.keras.Model):
    def __init__(self, num_layers: int = 2, num_units: Union[int, list[int]] = 128, 
                 latent_dim: int = 2):
        """
        Parameters
        ----------
        num_layers : int
            Number of hidden layers in the encoder (not including the output layer)
        num_units : Union[int, List[int]]
            Number of neurons in each hidden layer. If an integer is passed, the same 
            number of units is used for all layers
        """
        super(Encoder, self).__init__()

        # set parameters
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.embed = tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=500)
        self.flatten = tf.keras.layers.Flatten()
        self._layers = [tf.keras.layers.Dense(num_units[i], activation='relu') 
                            for i in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(latent_dim*2) # mu and rho for each latent dimension
    
    def call(self, data):
        # x = self._layers[0](2 * data - 1)
        x = self.flatten(self.embed(data))
        for i in range(0, self.num_layers):
            x = self._layers[i](x)
        output = self.output_layer(x)

        mu, rho = output[:, :self.latent_dim], output[:, self.latent_dim:]
        encoding = ds.Normal(loc=mu, scale=tf.nn.softplus(rho - 5.0))
        return encoding


class Decoder(tf.keras.Model):
    def __init__(self, out_dim: int = 2, num_layers: int = 0, 
                 num_units: Union[int, list[int]] = 16):
        super(Decoder, self).__init__()

        # set parameters
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        self.num_layers = num_layers
        self.out_dim = out_dim

        self._layers = [tf.keras.layers.Dense(num_units[i], activation='relu') 
                            for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(out_dim)

    def call(self, encoding_sample):
        for i in range(self.num_layers):
            encoding_sample = self._layers[i](encoding_sample)
        return self.dense(encoding_sample)


def merge_dicts(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


class VIB:
    def __init__(self, encoder_args: dict = {}, decoder_args: dict = {}):
        self.encoder = Encoder(**encoder_args)
        self.decoder = Decoder(**decoder_args)
        self.prior = ds.Normal(0.0, 1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def train(self, data:dict[str, np.array], epochs: int = 25, 
              batch_size: int = 50, beta: float = 10**-3, alpha: float = 1.0):
        """
        Train the VIB model
        
        Parameters
        ----------
        data : dict[str, np.array]
            Dictionary containing the training data and test data
            keys: 'train_data', 'test_data', train_labels', 'test_labels'
        labels : np.array
            One-hot encoded labels
        epochs : int
            Number of epochs to train the model
        alpha : float
            Order of the Renyi divergence
        """
        steps_per_batch = len(data['train_data']) // batch_size

        # Keep track of history
        train_results = {'Train IZY':[], 'Train IZX':[], 'Train acc':[], 'Train avg_acc':[]}
        test_results = {'Test IZY':[], 'Test IZX':[], 'Test acc':[], 'Test avg_acc':[]}

        for epoch in range(epochs):
            for step in range(steps_per_batch):
                batch_images = data['train_data'][step * batch_size:(step + 1) * batch_size]
                batch_labels = data['train_labels'][step * batch_size:(step + 1) * batch_size]
                self.train_step(batch_images, batch_labels, beta, alpha)

            # Evaluate the model on the training data
            IZY, IZX, acc, _acc = self.evaluate(data['train_data'], 
                                                    data['train_labels'], beta, alpha)
            train_results['Train IZY'].append(IZY); train_results['Train IZX'].append(IZX)
            train_results['Train acc'].append(acc); train_results['Train avg_acc'].append(_acc)

            # Evaluate the model on the testing data
            IZY, IZX, acc, _acc = self.evaluate(data['test_data'], 
                                                    data['test_labels'], beta, alpha)
            test_results['Test IZY'].append(IZY); test_results['Test IZX'].append(IZX)
            test_results['Test acc'].append(acc); test_results['Test avg_acc'].append(_acc)

            print(f"Epoch {epoch + 1}, Accuracy: {acc:.4f}, Avg Accuracy: {_acc:.4f}, IZY: {IZY:.2f}, IZX: {IZX:.2f}")
        
        res = merge_dicts({'Epochs':np.arange(epochs)+1}, train_results, test_results)
        return pd.DataFrame(res)

    def compute_loss(self, labels, encoding, logits, beta: float, alpha: float = 1):
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) / math.log(2)
        info_loss = tf.reduce_mean(tfp.distributions.kl_divergence(encoding, self.prior)) / math.log(2)
        total_loss = class_loss + beta * info_loss
        return total_loss, class_loss, info_loss
    
    @tf.function
    def train_step(self, images, labels, beta: float, alpha: float = 1):
        with tf.GradientTape() as tape:
            encoding = self.encoder(images)
            sample = encoding.sample()
            logits = self.decoder(sample)
            total_loss, class_loss, info_loss = self.compute_loss(labels, encoding, logits, beta, alpha)
        
        # Get the trainable variables from both encoder and decoder models
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
        return total_loss, class_loss, info_loss
    
    def evaluate(self, data, labels, beta: float, alpha: float = 1):
        encoding = self.encoder(data)
        sample = encoding.sample()
        logits = self.decoder(sample)
        
        correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # Monte Carlo average accuracy over 12 samples
        many_encodings = encoding.sample(12)
        many_logits = self.decoder(many_encodings)
        avg_output = tf.reduce_mean(tf.nn.softmax(many_logits), axis=0)
        avg_correct_prediction = tf.equal(tf.argmax(avg_output, axis=1), tf.argmax(labels, axis=1))
        avg_accuracy = tf.reduce_mean(tf.cast(avg_correct_prediction, tf.float32))
        
        IZY_bound = math.log(self.decoder.out_dim, 2) - self.compute_loss(labels, encoding, logits, beta, alpha)[1]
        IZX_bound = self.compute_loss(labels, encoding, logits, beta, alpha)[2]
        
        return IZY_bound.numpy(), IZX_bound.numpy(), accuracy.numpy(), avg_accuracy.numpy()


if __name__ == '__main__':
    # load the data
    gutenberg = Gutenberg(normalize=False)
    train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.get_data()
    data = {'train_data': train_msg_pad, 'test_data': test_msg_pad, 
            'train_labels': train_labels, 'test_labels': test_labels}

    # Instantiate the model
    vib = VIB(encoder_args={'num_layers':3, 'num_units':[128,64,32]})
    
    # Train the model
    res = vib.train(data, epochs=25, batch_size=50, beta=10**-4, alpha=1.0)
    print(res)