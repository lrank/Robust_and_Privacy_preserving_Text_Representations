# coding=utf8
"""
@author: Yantong Lai
@date: 12/18/2019
"""

import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import spacy
spacy.load("en_core_web_sm")

import random

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Dataset path
dataset_path = "../../dataset/WWW2015_processed/"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################################
#         Hyper-parameters         #
####################################
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 50
N_FILTERS = 128
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 5
DROPOUT = 0.5
NUM_EPOCHS = 200


####################################
#          Preparing Data          #
####################################
# 1. data.Field()
TEXT = data.Field('spacy')
RATING_LABEL = data.LabelField()
GENDER_LABEL = data.LabelField()
AGE_LABEL = data.LabelField()
LOCALTION_LABEL = data.LabelField()

# 2. data.TabularDataset
train_data, valid_data, test_data = data.TabularDataset.splits(path=dataset_path,
                                                               train="train.csv",
                                                               validation="valid.csv",
                                                               test="test.csv",
                                                               fields=[('text', TEXT), ('rating', RATING_LABEL), ('gender', GENDER_LABEL),
                                                                       ('age', AGE_LABEL), ('location', LOCALTION_LABEL)],
                                                               format="csv")

print("Number of train_data = {}".format(len(train_data)))
print("Number of valid_data = {}".format(len(valid_data)))
print("Number of test_data = {}".format(len(test_data)))
print("vars(train_data[0]) = {}\n".format(vars(train_data[0])))

# 3. data.BucketIterator
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                               batch_size=BATCH_SIZE,
                                                               device=device,
                                                               sort_key=lambda x: len(x.text))

# 4. Build vocab
# TEXT.build_vocab(train_data)
# unk_init=torch.Tensor.normal_)
# LABELS.build_vocab(train_data)
# print("vars(train_data[0]) = ", vars(train_data[0]))

# 4.1 (Optional) If build vocab with pre-trained word embedding vectors
TEXT.build_vocab(train_data, vectors="glove.6B.100d")
RATING_LABEL.build_vocab(train_data)
GENDER_LABEL.build_vocab(train_data)
AGE_LABEL.build_vocab(train_data)
LOCALTION_LABEL.build_vocab(train_data)

print("After embedding, vars(train_data[0]) = ", vars(train_data[0]))


####################################
#          Build the Model         #
####################################
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # torch.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

# Parameters
INPUT_DIM = len(TEXT.vocab)

# Create an instance
model = TextCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)


####################################
#          Train the Model         #
####################################
# criterion = nn.BCEWithLogitsLoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)

########## Train and Validation ##########
total_step = len(train_iter)
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = []
    train_total_correct = 0

    for i, batch in enumerate(train_iter):

        text = batch.text
        y = batch.rating

        # Forward pass
        # y_pred = model(text).squeeze(1).float()
        y_pred = model(text).squeeze(1)

        loss = criterion(y_pred, y)

        pred = torch.argmax(y_pred.data, dim=1)
        train_total_correct += (pred == y).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("Training total_loss = {}".format(total_loss))
    print("Training total_accuracy = {:.4f}%".format(100 * train_total_correct / len(train_data)))

    # Validation
    model.eval()
    total_correct = 0
    total_loss = 0.0
    for i, batch in enumerate(valid_iter):

        text = batch.text
        y = batch.rating

        y_pred = model(text).squeeze(1)

        loss = criterion(y_pred, y)

        pred = torch.argmax(y_pred.data, dim=1)
        total_correct += (pred == y).sum().item()
        total_loss += loss.item()

    avg_loss = total_loss / len(valid_data)
    print("Validation Avg. Loss: {:.4f}, Accuracy: {:.4f}%\n".format(avg_loss, 100 * total_correct / len(valid_data)))


########## Evaluation ##########
model.eval()
total_correct = 0
total_loss = 0.0

for i, batch in enumerate(test_iter):
    text = batch.text
    y = batch.rating

    # Forward pass
    # y_pred = model(text).squeeze(1).float()
    y_pred = model(text).squeeze(1)

    loss = criterion(y_pred, y)

    pred = torch.argmax(y_pred.data, dim=1)
    total_correct += (pred == y).sum().item()

    total_loss += loss.item()

    if (i + 1) % 10 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

avg_loss = total_loss / len(test_data)
print("Test Avg. Loss: {:.4f}, Accuracy: {:.4f}%".format(avg_loss, 100 * total_correct / len(test_data)))

