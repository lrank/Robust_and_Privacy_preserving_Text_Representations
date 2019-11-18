# coding = utf8
"""
@author: Yantong Lai
@date: 2019.11.4
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

# ! Mark Pytorch/postag as root directory in Pycharm !
from utils import load_json


train_csv_path = "../../dataset/TrustPilot_processed/train_data.csv"
test_csv_path = "../../dataset/TrustPilot_processed/test_data.csv"
train_json_path = "../../dataset/TrustPilot_processed/train_data.json"
test_json_path = "../../dataset/TrustPilot_processed/test_data.json"

symbol_list = [",", ".", "-", "/", "[", "]", "?", "<", ">", "{", "}", "|", "\\", ":", ";", "'", "!", "@", "#", "$", "%",
               "_", "(", ")"]

########## Parameters ##########
EMBEDDING_DIM = 12
HIDDEN_DIM = 300
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64


def prepare_sequence(seq, to_ix):
    """
    It is a function to change index to tensor.
    :param seq:
    :param to_ix:
    :return:
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def build_word_tag_tuple(dataset_path):

    json_data = load_json(dataset_path)

    word_tag_list = []
    for item in json_data:
        word = item['words']
        tag = item['tags']
        word_tag_list += list(zip(word, tag))

    return word_tag_list


training_data = build_word_tag_tuple(train_json_path)
testing_data = build_word_tag_tuple(test_json_path)
print("training_data = {}".format(training_data))


def build_word_tag_to_idx(training_data):
    """
    It is a function to build word_to_idx and tag_to_idx
    :param training_data: training_data
    :return: word_to_idx, tag_to_idx
    """
    word_to_idx = {}
    tag_to_idx = {}
    for word, tag in training_data:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)

    return word_to_idx, tag_to_idx


word_to_idx, tag_to_idx = build_word_tag_to_idx(training_data)
print("word_to_idx = {}".format(word_to_idx))
print("tag_to_idx = {}\n".format(tag_to_idx))


########## Model ##########
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, tagset_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.5)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(in_features=hidden_dim * 2, out_features=tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


########## Train ##########
model = LSTMTagger(embedding_dim=EMBEDDING_DIM,
                   hidden_dim=HIDDEN_DIM,
                   vocab_size=len(word_to_idx),
                   tagset_size=len(tag_to_idx))

# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Before training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_idx)
    tag_scores = model(inputs)
    print("Before training, tag_scores = {}".format(tag_scores))
    print("Before training, tag_scores.size() = {}".format(tag_scores.size()))
    # (1, 12)

# Training
total_step = len(training_data)
for epoch in range(NUM_EPOCHS):
    model.train()
    train_total_correct = 0
    for i, (sentence, tags) in enumerate(training_data):
        total_loss = []
        model.zero_grad()
        sentence_in = torch.tensor([word_to_idx[sentence]])
        targets = torch.tensor([tag_to_idx[tags]])

        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        # print("loss = {}".format(loss))
        total_loss.append(loss.item())

        # _, pred = torch.max(output.data, 1)
        pred = torch.argmax(tag_scores.data, dim=1)
        train_total_correct += (pred == targets).sum().item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("total_loss = {}".format(total_loss))
    print("total_accuracy = {:.4f}%\n".format(100 * train_total_correct / len(training_data)))


# Test
with torch.no_grad():
    correct = 0
    total = 0
    for sentence, tags in testing_data:
        sentence_in = torch.tensor([word_to_idx[sentence]])
        targets = torch.tensor([tag_to_idx[tags]])

        outputs = model(sentence_in)
        # _, predicted = torch.max(outputs.data, 1)
        predicted = torch.argmax(outputs.data, 1)
        total += tags.size(0)
        correct += (predicted == tags).sum().item()

    print('Test Accuracy of the model on the testing data: {} %'.format(100 * correct / total))
