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
LAMBDA = 1e-3
EMBEDDING_DIM = 50
HIDDEN_DIM = 300
N_FILTERS = 200
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
                                                               fields=[('text', TEXT), ('rating', RATING_LABEL),
                                                                       ('gender', GENDER_LABEL),
                                                                       ('age', AGE_LABEL),
                                                                       ('location', LOCALTION_LABEL)],
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


####################################
#      Build the Discriminator     #
####################################
class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        out = self.fc1(input)
        # print("First, out.size() = ", out.size())

        out = self.relu(out)
        # print("Second, out.size() = ", out.size())

        out = self.fc2(out)

        out = out.view(input.shape[0], -1)

        out = F.log_softmax(out, dim=1)

        return out


# Parameters
INPUT_DIM = len(TEXT.vocab)

# Create an instance
model = TextCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

# Create an Age Discriminator instance
discriminator_age = Discriminator(input_size=EMBEDDING_DIM,
                                  hidden_size=HIDDEN_DIM,
                                  num_classes=2)

# Create a Gender Discriminator instance
discriminator_gender = Discriminator(input_size=EMBEDDING_DIM,
                                     hidden_size=HIDDEN_DIM,
                                     num_classes=2)

# Create a Location Discriminator instance
discriminator_location = Discriminator(input_size=EMBEDDING_DIM,
                                       hidden_size=HIDDEN_DIM,
                                       num_classes=5)

####################################
#          Train the Model         #
####################################
# criterion = nn.BCEWithLogitsLoss().to(device)
criterion_rating = nn.CrossEntropyLoss().to(device)
criterion_age = nn.CrossEntropyLoss().to(device)
criterion_gender = nn.CrossEntropyLoss().to(device)
criterion_location = nn.CrossEntropyLoss().to(device)

# Optimizer
optimizer_rating = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer_age = optim.Adam(discriminator_age.parameters(), lr=LEARNING_RATE)
optimizer_gender = optim.Adam(discriminator_gender.parameters(), lr=LEARNING_RATE)
optimizer_location = optim.Adam(discriminator_location.parameters(), lr=LEARNING_RATE)

optimizer = optim.Adam(list(model.parameters()) + list(discriminator_age.parameters()) +
                       list(discriminator_gender.parameters()) + list(discriminator_location.parameters()),
                       lr=LEARNING_RATE)

model = model.to(device)
discriminator_age = discriminator_age.to(device)
discriminator_gender = discriminator_gender.to(device)
discriminator_location = discriminator_location.to(device)

########## Train and Validation ##########
total_step = len(train_iter)
for epoch in range(NUM_EPOCHS):
    model.train()
    discriminator_age.train()
    discriminator_gender.train()
    discriminator_location.train()

    total_loss = []
    train_total_correct = 0
    train_age_correct = 0
    train_gender_correct = 0
    train_location_correct = 0
    for i, batch in enumerate(train_iter):

        text = batch.text
        y = batch.rating
        y_gender = batch.gender
        y_age = batch.age
        y_location = batch.location

        optimizer_rating.zero_grad()
        optimizer_age.zero_grad()
        optimizer_gender.zero_grad()
        optimizer_location.zero_grad()

        # Forward pass
        # y_pred = model(text).squeeze(1).float()
        y_pred = model(text).squeeze(1)

        h = model.embedding(text).permute(1, 0, 2)

        y_age_pred = discriminator_age(h).squeeze()
        y_gender_pred = discriminator_gender(h).squeeze()
        y_location_pred = discriminator_location(h).squeeze()

        rating_loss = criterion_rating(y_pred, y)
        gender_loss = criterion_gender(y_gender_pred, y_gender)
        age_loss = criterion_age(y_age_pred, y_age)
        location_loss = criterion_location(y_location_pred, y_location)

        discriminator_loss = gender_loss + age_loss + location_loss
        loss = rating_loss - LAMBDA * discriminator_loss

        rating_pred = torch.argmax(y_pred.data, dim=1)
        train_total_correct += (rating_pred == y).sum().item()

        age_pred = torch.argmax(y_age_pred.data, dim=1)
        train_age_correct += (age_pred == y_age).sum().item()

        gender_pred = torch.argmax(y_gender_pred.data, dim=1)
        train_gender_correct += (gender_pred == y_gender).sum().item()

        location_pred = torch.argmax(y_location_pred.data, dim=1)
        train_location_correct += (location_pred == y_location).sum().item()

        # Backward and optimize
        loss.backward()
        # discriminator_loss.backward()

        optimizer_rating.step()
        optimizer_age.step()
        optimizer_gender.step()
        optimizer_location.step()

        total_loss.append(loss.item())

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("Training total_loss = {}".format(total_loss))
    print("Training total_rating_accuracy = {:.4f}%".format(100 * train_total_correct / len(train_data)))
    print("Training total_age_accuracy = {:.4f}%".format(100 * train_age_correct / len(train_data)))
    print("Training total_gender_accuracy = {:.4f}%".format(100 * train_gender_correct / len(train_data)))
    print("Training total_location_accuracy = {:.4f}%".format(100 * train_location_correct / len(train_data)))

    # Validation
    model.eval()
    total_valid_correct = 0
    total_valid_loss = 0.0
    valid_age_correct = 0
    valid_gender_correct = 0
    valid_location_correct = 0

    for i, batch in enumerate(valid_iter):
        text = batch.text
        y = batch.rating
        y_gender = batch.gender
        y_age = batch.age
        y_location = batch.location

        y_pred = model(text).squeeze(1)
        h = model.embedding(text).permute(1, 0, 2)

        y_age_pred = discriminator_age(h).squeeze()
        y_gender_pred = discriminator_gender(h).squeeze()
        y_location_pred = discriminator_location(h).squeeze()

        rating_loss = criterion_rating(y_pred, y)
        gender_loss = criterion_gender(y_gender_pred, y_gender)
        age_loss = criterion_age(y_age_pred, y_age)
        location_loss = criterion_location(y_location_pred, y_location)

        rating_pred = torch.argmax(y_pred.data, dim=1)
        total_valid_correct += (rating_pred == y).sum().item()
        total_valid_loss += loss.item()

        age_pred = torch.argmax(y_age_pred.data, dim=1)
        valid_age_correct += (age_pred == y_age).sum().item()

        gender_pred = torch.argmax(y_gender_pred.data, dim=1)
        valid_gender_correct += (gender_pred == y_gender).sum().item()

        location_pred = torch.argmax(y_location_pred.data, dim=1)
        valid_location_correct += (location_pred == y_location).sum().item()

    avg_loss = total_valid_loss / len(valid_data)
    print("Validation total_loss = {}".format(total_valid_loss))
    print(
        "Validation Avg. Loss: {:.4f}, Accuracy: {:.4f}%".format(avg_loss, 100 * total_valid_correct / len(valid_data)))
    print("Validation total_rating_accuracy = {:.4f}%".format(100 * total_valid_correct / len(valid_data)))
    print("Validation total_age_accuracy = {:.4f}%".format(100 * valid_age_correct / len(valid_data)))
    print("Validation total_gender_accuracy = {:.4f}%".format(100 * valid_gender_correct / len(valid_data)))
    print("Validation total_location_accuracy = {:.4f}%".format(100 * valid_location_correct / len(valid_data)))

####################################
#            Evaluation            #
####################################
model.eval()
total_correct = 0
total_loss = 0.0

for i, batch in enumerate(test_iter):
    text = batch.text
    y = batch.rating

    # Forward pass
    # y_pred = model(text).squeeze(1).float()
    y_pred = model(text).squeeze(1)

    loss = criterion_rating(y_pred, y)

    pred = torch.argmax(y_pred.data, dim=1)
    total_correct += (pred == y).sum().item()

    total_loss += loss.item()

    if (i + 1) % 10 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

avg_loss = total_loss / len(test_data)
print("Test Avg. Loss: {:.4f}, Accuracy: {:.4f}%\n".format(avg_loss, 100 * total_correct / len(test_data)))


