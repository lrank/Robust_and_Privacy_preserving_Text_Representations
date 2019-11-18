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


train_path = "../../dataset/TrustPilot/en.O45-UKC1_WORST-F.data.TT.tagged.gold"

symbol_list = [",", ".", "-", "/", "[", "]", "?", "<", ">", "{", "}", "|", "\\", ":", ";", "'", "!", "@", "#", "$", "%",
               "_", "(", ")"]

########## Parameters ##########
EMBEDDING_DIM = 12
HIDDEN_DIM = 6
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
LAMBDA = 1e-3


def prepare_sequence(seq, to_ix):
    """
    It is a function to change index to tensor.
    :param seq:
    :param to_ix:
    :return:
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def data_cleaning(lines):

    clean_lines = []
    for item in lines:
        if item == "\n":
            continue
        elif item[0] in symbol_list:
            continue
        else:
            clean_lines.append(item)
    return clean_lines


def load_TrustPilot(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Data Cleaning
    clean_lines = data_cleaning(lines)

    # Load Data
    words_list = []
    tags_list = []

    for item in clean_lines:
        if item == "\n":
            continue
        else:
            word = item.split("\t")[0]
            words_list.append(word)
            tag = item.split("\t")[1]
            tags_list.append(tag.strip("\n"))

    training_data = list(zip(words_list, tags_list))
    return training_data

training_data = load_TrustPilot(train_path)
print("training_data = {}\n".format(training_data))


def get_gender_attribute(filename):
    """
    It is a function to get attributes
    F: 0
    M: 1
    :param filename: filename
    :return: gender_list
    """
    gender_list = []
    if "-F" in filename:
        gender_list.append(0)
    if "-M" in filename:
        gender_list.append(1)

    return gender_list


def get_age_attribute(filename):
    """
    It is a function to get age attributes
    O45: 0
    U35: 1
    :param filename: filename
    :return: age_list
    """
    age_list = []
    if "O45" in filename:
        age_list.append(0)
    if "U35" in filename:
        age_list.append(1)

    return age_list

gender_list = get_gender_attribute(train_path)
age_list = get_age_attribute(train_path)

gender_labels = torch.tensor([item for item in gender_list])
print("gender_labels = {}".format(gender_labels))
age_labels = torch.tensor([item for item in age_list])
print("age_labels = {}".format(age_labels))


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

        # Word embedding
        self.word_embeddings = nn.Embedding(vocab_size, tagset_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # embedding_dim = tagset_size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.5)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(in_features=hidden_dim * 2, out_features=tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


########## Train ##########
model = LSTMTagger(embedding_dim=EMBEDDING_DIM,
                   hidden_dim=HIDDEN_DIM,
                   vocab_size=len(word_to_idx),
                   tagset_size=len(tag_to_idx))

discriminator = Discriminator(input_size=EMBEDDING_DIM,
                              hidden_size=HIDDEN_DIM,
                              num_classes=2)


# loss_function = nn.NLLLoss()
# loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(model.parameters()) + list(discriminator.parameters()), lr=LEARNING_RATE)
# print("model.parameters() = {}".format(model.parameters()))
# print("list(model.parameters()) = {}".format(list(model.parameters())))
# print("discriminator.parameters() = {}".format(discriminator.parameters()))
# print("list(discriminator.parameters()) = {}".format(list(discriminator.parameters())))

# def reset_grad():
#     model_optimizer.zero_grad()
#     discriminator.zero_grad()


# print("model.word_embeddings = {}".format(model.word_embeddings))
# model.word_embeddings =  Embedding(391, 12)
# Embedding(VOCAB_DIM, EMBEDDING_DIM)



# Before training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_idx)
    tag_scores = model(inputs)
    print("Before training, tag_scores = {}".format(tag_scores))
    print("Before training, tag_scores.size() = {}\n".format(tag_scores.size()))
    # (1, 12)



discriminator_criterion = nn.CrossEntropyLoss()
model_criterion = nn.CrossEntropyLoss()




# Training
total_step = len(training_data)
for epoch in range(NUM_EPOCHS):
    model.train()
    total_label_accuracy = 0

    for i, (sentence, tags) in enumerate(training_data):
        total_loss = []

        # First, zero gradients
        model.zero_grad()
        discriminator.zero_grad()

        # X_train
        sentence_in = torch.tensor([word_to_idx[sentence]])
        # print("sentence_in = {}".format(sentence_in))
        # print("sentence_in.size() = {}".format(sentence_in.size()))

        # y_train
        targets = torch.tensor([tag_to_idx[tags]])
        # print("targets = {}".format(targets))
        # print("targets.size() = {}".format(targets.size()))

        # hidden representation h
        h = model.word_embeddings(sentence_in)
        # print("h = {}".format(h))
        # print("h.size() = {}".format(h.size()))
        # h.size() = torch.Size([1, 12])

        # gender_label
        gender_label = torch.tensor([0], dtype=torch.long)
        # print("gender_label = {}".format(gender_label))
        # print("gender_label.size() = {}".format(gender_label.size()))

        # gender_pred
        gender_pred = discriminator(h)
        # print("gender_pred = {}".format(gender_pred))
        # print("gender_pred.size() = {}".format(gender_pred.size()))
        # gender_pred = tensor([-0.5426, -0.8704], grad_fn=<SqueezeBackward0>)
        # gender_pred.size() = torch.Size([2])

        # gender_pred = tensor([[-0.7543, -0.6355]], grad_fn=<LogSoftmaxBackward>)
        # gender_pred.size() = torch.Size([1, 2])

        # y_pred
        tag_scores = model(sentence_in)
        # print("tag_scores = {}".format(tag_scores))
        # print("tag_socres.size() = {}".format(tag_scores.size()))
        # tag_scores = tensor([[-2.3341, -1.7593, -1.4718, -2.6736, -4.2360, -2.3703, -3.0320, -2.2331,
        #                       -2.1798, -3.6190, -4.0845, -4.3900]], grad_fn= < LogSoftmaxBackward >)
        # tag_socres.size() = torch.Size([1, 12])


        discriminator_loss = discriminator_criterion(gender_pred, gender_label)
        # print("dicriminator_loss = {}".format(discriminator_loss))

        model_loss = model_criterion(tag_scores, targets)
        # print("model_loss = {}".format(model_loss))

        loss = model_loss - LAMBDA * discriminator_loss
        # print("loss = {}".format(loss))


        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

        total_label_accuracy += (tag_scores.max(1)[1] == targets).float().mean().item()

    print("average_loss = {}".format(total_loss[0] / len(training_data)))
    print("average_label_accuracy = {}\n".format(total_label_accuracy / len(training_data)))
