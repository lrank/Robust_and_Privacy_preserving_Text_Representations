# coding=utf
"""
@author: Yantong Lai
@date: 11/18/2019
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets

import json
import random
SEED = 12345


# Dataset path
TrustPilot_processed_dataset_path = "../../dataset/TrustPilot_processed/"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################################
#         Hyper-parameters         #
####################################
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROUPOUT = 0.5
NUM_EPOCHS = 20


####################################
#          Preparing Data          #
####################################
# 1. data.Field()
TEXT = data.Field(include_lengths=True, pad_token='<pad>', unk_token='<unk>')
TAG_LABEL = data.LabelField()
AGE_LABEL = data.LabelField()
GENDER_LABEL = data.LabelField()

# 2. data.TabularDataset
train_data, test_data = data.TabularDataset.splits(path=TrustPilot_processed_dataset_path,
                                                   train="train.csv",
                                                   test="test.csv",
                                                   fields=[('text', TEXT), ('tag_label', TAG_LABEL),
                                                           ('age_label', AGE_LABEL), ('gender_label', GENDER_LABEL)],
                                                   format="csv")

# 3. Split train_data to train_data, valid_data
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print("Number of train_data = {}".format(len(train_data)))
print("Number of valid_data = {}".format(len(valid_data)))
print("Number of test_data = {}\n".format(len(test_data)))

# 4. data.BucketIterator
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                               batch_size=BATCH_SIZE,
                                                               device=device,
                                                               sort_key=lambda x: len(x.text))

# 5. Build vocab
TEXT.build_vocab(train_data)
TAG_LABEL.build_vocab(train_data)
AGE_LABEL.build_vocab(train_data)
GENDER_LABEL.build_vocab(train_data)


####################################
#          Build the Model         #
####################################
class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNN, self).__init__()

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. RNN layer
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        # 3. Linear layer
        self.fc = nn.Linear(in_features=hidden_dim * 2,
                            out_features=output_dim)

    def forward(self, text, text_lengths):

        # 1. Embedding
        # text = [sent len, batch size]
        embedded = self.embedding(text)

        # 2. Pack sequence
        # embedded = [sent len, batch size, embed size]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)

        # 3. RNN
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # 4. Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num_directions]
        # output over padding tokens are zero tensors

        # hidden = [num_layers * num_directions, batch size, hid dim]
        # cell = [num_layers * num_directions, batch_size, hid dim]

        # 5. Concat the final forward (hidden[-2, :, :]) and backward (hidden[-1, :, :])
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hid dim * num_directions]

        return self.fc(hidden)

# Parameters
INPUT_DIM = len(TEXT.vocab)
TAG_OUTPUT_DIM = len(TAG_LABEL.vocab)
AGE_OUTPUT_DIM = len(AGE_LABEL.vocab)
GENDER_OUTPUT_DIM = len(GENDER_LABEL.vocab)

# Create an RNN instance
model = RNN(vocab_size=INPUT_DIM,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=TAG_OUTPUT_DIM,
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROUPOUT)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("The model has {} trainable parameters".format(count_parameters(model)))


####################################
#          Train the Model         #
####################################
# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)


########## Train ##########
total_step = len(train_iter)

for epoch in range(NUM_EPOCHS):
    total_loss = []
    train_total_correct = 0
    for i, batch in enumerate(train_iter):

        text, text_lengths = batch.text
        y = batch.tag_label
        # print("y = {}".format(y))

        # Forward pass
        y_pred = model(text, text_lengths)
        # print("y_pred = {}".format(y_pred))
        loss = criterion(y_pred, y)

        # _, pred = torch.max(output.data, 1)
        pred = torch.argmax(y_pred.data, dim=1)
        train_total_correct += (pred == y).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("total_loss = {}".format(total_loss))
    print("total_accuracy = {:.4f}%\n".format(100 * train_total_correct / len(train_data)))


########## Evaluation ##########
model.eval()
total_correct = 0
avg_loss = 0.0
for i, batch in enumerate(valid_iter):
    text, text_lengths = batch.text

    y = batch.tag_label

    # Forward pass
    y_pred = model(text, text_lengths)
    loss = criterion(y_pred, y)
    avg_loss += loss.item()

    # _, pred = torch.max(output.data, 1)
    pred = torch.argmax(y_pred.data, dim=1)
    total_correct += (pred == y).sum().item()

avg_loss = avg_loss / len(valid_data)
print("Test Avg. Loss: {:.4f}, Accuracy: {:.4f}%"
      .format(avg_loss, 100 * total_correct / len(valid_data)))


