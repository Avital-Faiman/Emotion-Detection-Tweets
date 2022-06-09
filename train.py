import pickle
import re
import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import torch.nn as nn
from spacy.lang.en import English
import string
from sklearn.metrics import confusion_matrix


def line_replacements(line):
    line = line.lower()
    line = re.sub("www.[^ ]*", "", line)
    line = re.sub("https?[^ ]*", "", line)
    line = re.sub("@\w+", "", line)
    line = re.sub("[^ ]*.com[^ ]*", "",line)
    return line


def create_dic_of_words(train_lines, dict_of_words, counter):
    nlp = English()
    for content in train_lines:
        content_split = [x.text for x in nlp(line_replacements(content))]
        # content_split = content.split()
        for word in content_split:
            lower_word = word.lower()
            if lower_word in dict_of_words.keys():
                continue
            else:
                dict_of_words[lower_word] = counter
                counter += 1
    return counter, dict_of_words


def find_best_line_length(train_lines):
    from statistics import median_high
    all_length = {}
    for content in train_lines:
        content_length = len(content.split())
        if content_length in all_length.keys():
            continue
        else:
            all_length[content_length] = 1
    all_length = dict(sorted(all_length.items()))
    return max(all_length.keys())


def create_vec_remove(line, dict_of_words, median_line):
    new_line = []
    for i in range(median_line):
        new_line.append(line[i])
    return new_line


def create_vec_padding(line, dict_of_words, median_line):
    new_line = []
    for i in range(len(line)):
        new_line.append(line[i])
    for i in range(median_line - len(line)):
        new_line.append(0)
    return new_line


def create_vector_per_line(train_lines, dict_of_words, median_line):
    new_train_lines = []
    nlp = English()
    for content in train_lines:
        new_content = []
        # content_splited = content.split()
        content_split = [x.text for x in nlp(line_replacements(content))]
        for word in content_split:
            lower_word = word.lower()
            if lower_word in dict_of_words.keys():
                numeric_word = dict_of_words[lower_word]
                new_content.append(numeric_word)
            else:
                continue
        if len(new_content) > median_line:
            new_line = create_vec_remove(new_content, dict_of_words, median_line)
            new_train_lines.append(new_line)
        elif len(new_content) < median_line:
            new_line = create_vec_padding(new_content, dict_of_words, median_line)
            new_train_lines.append(new_line)
        else:
            new_train_lines.append(new_content)
    return new_train_lines


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers, dropout):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, nlayers, dropout=dropout)

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, num_class):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=n_layers, batch_first=True)

        self.drop = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(n_layers * hidden_size)
        self.dense = nn.Linear(n_layers * hidden_size, num_class)

    def dense_parameters(self):
        return list(self.lstm.parameters()) + list(self.dense.parameters())

    def forward(self, encoded_text, lengths):
        batch_size = lengths
        # embedding
        embedded = self.embedding(encoded_text)
        # packed_embeded = nn.utils.rnn.pack_padded_sequence(embedded)
        # lstm
        _, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.permute([1, 0, 2]).contiguous().view(batch_size, -1)
        hidden = self.drop(hidden)
        hidden = self.batch_norm(hidden)
        hidden = self.dense(hidden)
        return hidden


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading CSV...")
    test = pd.read_csv("testEmotions.csv", encoding="UTF-8")
    train = pd.read_csv("trainEmotions.csv", encoding="UTF-8")

    cleanup_nums = {"emotion": {"happiness": 1, "sadness": 2, "neutral": 0}}
    test = test.replace(cleanup_nums)
    train = train.replace(cleanup_nums)

    #prerering the data and create data loader
    all_letters = string.ascii_letters + " .,;'"

    n_letters = len(all_letters)

    category_lines = {0: [], 1: [], 2: []}
    for content, emotion in zip(train['content'].values, train['emotion'].values):
        category_lines[emotion].append(content)

    train_lines = train['content'].values
    test_lines = test['content'].values

    dict_of_words = {}
    counter = 1
    current_counter, dict_of_words = create_dic_of_words(train_lines, dict_of_words, counter)
    median_line = find_best_line_length(train_lines)

    new_train_lines = create_vector_per_line(train_lines, dict_of_words, median_line)
    new_test_lines = create_vector_per_line(test_lines, dict_of_words, median_line)

    tensor_sample_train = torch.tensor(new_train_lines)
    tensor_sample_test = torch.tensor(new_test_lines)

    tensor_label_train = torch.tensor(train['emotion'].values)
    tensor_label_test = torch.tensor(test['emotion'].values)

    # save dict of words
    dict_words = open("dict.pkl", "wb")
    pickle.dump(dict_of_words, dict_words)
    dict_words.close()

    INPUT_DIM = current_counter
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3

    train_data = []
    for i in range(len(tensor_sample_train)):
        train_data.append([tensor_sample_train[i], tensor_label_train[i]])

    test_data = []
    for i in range(len(tensor_sample_test)):
        test_data.append([tensor_sample_test[i], tensor_label_test[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=100)

    # training parameters
    n_epoch = 20
    lr = 0.001
    n_layers = 9

    model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, n_layers, 3)

    losses = {"train": [], "validation": []}
    accuracies = {"train": [], "validation": []}

    criterion = nn.CrossEntropyLoss()
    optimizer_sparse = optim.SparseAdam(model.embedding.parameters(), lr=lr)
    optimizer_dense = optim.Adam(model.dense_parameters(), lr=lr)

    model = model.to(device)

    for n in range(n_epoch):
        epoch_loss = []
        epoch_acc = []
        for encoded_text, labels in trainloader:
            lengths = encoded_text.shape[0]
            model = model.train()
            optimizer_dense.zero_grad()
            optimizer_sparse.zero_grad()

            encoded_text, lengths, labels = encoded_text.to(device), lengths, labels.to(device)

            y_pred = model(encoded_text, lengths)
            softmax = torch.nn.Softmax(dim=1)
            y_pred1 = softmax(y_pred)
            loss = criterion(y_pred1, labels)
            loss.backward()
            optimizer_sparse.step()
            optimizer_dense.step()
            epoch_loss.append(loss.item())
            acc = accuracy_score(labels.detach().cpu(), y_pred1.argmax(1).detach().cpu())
            epoch_acc.append(acc)


        avg_loss = (sum(epoch_loss) / len(epoch_loss))
        avg_acc = (sum(epoch_acc) / len(epoch_acc))
        print(f"epoch:{n} train_loss: {avg_loss:.4f}; train_acc: {avg_acc:.4f}")
        losses["train"].append(avg_loss)
        accuracies["train"].append(avg_acc)

        epoch_loss = []
        epoch_acc = []
        with torch.no_grad():
            for encoded_text, labels in testloader:
                lengths = encoded_text.shape[0]
                encoded_text, lengths, labels = encoded_text.to(device), lengths, labels.to(device)
                y_pred = model(encoded_text, lengths)
                softmax = torch.nn.Softmax(dim=1)
                y_pred1 = softmax(y_pred)
                loss = criterion(y_pred1, labels)
                epoch_loss.append(loss.item())
                acc = accuracy_score(labels.detach().cpu(), y_pred1.argmax(1).detach().cpu())
                epoch_acc.append(acc)

            avg_loss = (sum(epoch_loss) / len(epoch_loss))
            avg_acc = (sum(epoch_acc) / len(epoch_acc))
            print(f"epoch:{n} validation_loss: {avg_loss:.4f}; validation_acc: {avg_acc:.4f}")
            losses["validation"].append(avg_loss)
            accuracies["validation"].append(avg_acc)
            if avg_acc >= 0.45:
                break

    print("max test accuracy:", max(accuracies["validation"]))
    print("max train accuracy:", max(accuracies["train"]))
    print("avg test accuracy:", sum(accuracies["validation"]) / len(accuracies["validation"]))
    print("avg train accuracy:", sum(accuracies["train"]) / len(accuracies["train"]))


    #create confusion matrix
    y_pred_test, predictions_test, true_test= [], [], []

    with torch.no_grad():
        for encoded_text, labels in testloader:
            lengths = encoded_text.shape[0]
            encoded_text, lengths, labels = encoded_text.to(device), lengths, labels.to(device)
            y_pred = model(encoded_text, lengths)
            softmax = torch.nn.Softmax(dim=1)
            y_pred1 = softmax(y_pred)
            y_pred_test += y_pred1
            ps = torch.exp(y_pred)
            top_p, top_class = ps.topk(1, dim=1)
            top_class1 = top_class.view(-1)
            predictions_test.append(top_class1.tolist())
            true_test.append(labels.tolist())

    y_pred_train, predictions_train, true_train = [], [], []

    with torch.no_grad():
        for encoded_text, labels in trainloader:
            lengths = encoded_text.shape[0]
            encoded_text, lengths, labels = encoded_text.to(device), lengths, labels.to(device)
            y_pred = model(encoded_text, lengths)
            softmax = torch.nn.Softmax(dim=1)
            y_pred1 = softmax(y_pred)
            y_pred_test += y_pred1
            ps = torch.exp(y_pred)
            top_p, top_class = ps.topk(1, dim=1)
            top_class1 = top_class.view(-1)
            predictions_train.append(top_class1.tolist())
            true_train.append(labels.tolist())

    final_true_test, final_pred_test =[],[]
    for prediction,true_label in zip(predictions_test,true_test):
        for pred, label in zip(prediction,true_label):
            final_true_test.append(label)
            final_pred_test.append(pred)

    final_true_train, final_pred_train = [], []
    for prediction, true_label in zip(predictions_train, true_train):
        for pred, label in zip(prediction, true_label):
            final_true_train.append(label)
            final_pred_train.append(pred)


    # confusion matrix-
    confusion_mat_train = confusion_matrix(final_true_train, final_pred_train)
    confusion_mat_test = confusion_matrix(final_true_test, final_pred_test)

    #save all needed data
    np.save('accuracies_validations.npy', accuracies["validation"])
    np.save('accuracies_train.npy', accuracies["train"])
    np.save('losses_train.npy', losses["train"])
    np.save('losses_test.npy', losses["validation"])
    np.save('confusion_mat_train', confusion_mat_train)
    np.save('confusion_mat_test', confusion_mat_test)
    torch.save(model, 'model.pkl')


if __name__ == '__main__':
    main()
