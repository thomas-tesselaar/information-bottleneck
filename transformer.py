
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import math

from datasets import Dataset
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from load_gutenberg import Gutenberg


OUT_DIM = 2

gutenberg = Gutenberg()
train_msg, test_msg, train_labels, test_labels = gutenberg.get_data(maxlen=500, tokenizer_name='bert', normalize=False)
train_labels, test_labels = train_labels.numpy(), test_labels.numpy()

# print(type(train_msg))
# print(type(train_labels))
# print(type(train_msg[0]))
# print(train_labels[0])

train_dataset = Dataset.from_dict({
    "input_ids": train_msg,  # Your tokenized data
    "attention_mask": np.where(train_msg != 0, 1, 0),  # Create a mask
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_msg,
    "attention_mask": np.where(test_msg != 0, 1, 0),
    "labels": test_labels
})



# train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# predictions = trainer.predict(test_msg)


