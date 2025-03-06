
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from load_gutenberg import Gutenberg


OUT_DIM = 2

gutenberg = Gutenberg()
train_msg_pad, test_msg_pad, train_labels, test_labels = gutenberg.get_data()


# Probability distributions
ds = tfp.distributions

# ============================
# ==== Defining functions ====
# ============================
# TODO these are constant across implementations
#      they should be generalized to some abstract vib class

# define Encoder
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embed = tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=500)
        self.flatten = tf.keras.layers.Flatten()
        self.first_hidden_layer = tf.keras.layers.Dense(128, activation='relu')
        self.second_hidden_layer = tf.keras.layers.Dense(64, activation='relu')
        # self.third_hidden_layer = tf.keras.layers.Dense(32, activation='relu')
        # self.fourth_hidden_layer = tf.keras.layers.Dense(16, activation='relu')
        # self.fifth_hidden_layer = tf.keras.layers.Dense(8, activation='relu')
        # self.sixth_hidden_layer = tf.keras.layers.Dense(8, activation='relu')
        self.output_layer = tf.keras.layers.Dense(4)  # 2 for mu and 2 for rho
    
    def call(self, data):
        x = self.flatten(self.embed(data)) # * data - 1
        x = self.first_hidden_layer(x)
        x = self.second_hidden_layer(x)
        # x = self.third_hidden_layer(x)
        # x = self.fourth_hidden_layer(x)
        # x = self.fifth_hidden_layer(x)
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


#betas = [10**x for x in [-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]]
betas = [10**-4]
beta_train_accuracy = []
beta_test_accuracy = []
try:
    for BETA in betas:
        tmp_train_acc = []
        tmp_test_acc = []

        for i in range(5):
            # Instantiate the encoder and decoder
            encoder = Encoder()
            decoder = Decoder()

            # Set the prior distribution
            prior = ds.Normal(0.0, 1.0)

            # Define the loss functions and metrics
            # BETA = 10**-3

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

            tmp_train_acc = res.iloc[-1,-6]
            tmp_test_acc = res.iloc[-1,-2]

        beta_train_accuracy.append(np.mean(tmp_train_acc))
        beta_test_accuracy.append(np.mean(tmp_test_acc))

        # print(tmp_train_acc)
        # print(tmp_test_acc)

    pd.DataFrame(
        {'beta':betas,'train accuracy':beta_train_accuracy,'test accuracy':beta_test_accuracy}
    ).to_csv('beta_accuracy.csv', index=False)
except Exception as e:
    print(e)
    print(BETA)
    print(betas)
    print(beta_train_accuracy)
    print(beta_test_accuracy)
