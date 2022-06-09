import os
import argparse
from collections import namedtuple
import numpy as np
import pandas as pd
import joblib
import torch
from PIL import Image
import torch.nn as nn
import pickle
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from spacy.lang.en import English
from train import create_vector_per_line


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
    model = torch.load('model.pkl')
    print(model)
    dict_of_words1 = open("dict.pkl", "rb")
    dict_of_words = pickle.load(dict_of_words1)

    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_file', type=str, help='Input path file, containing tweets')
    args = parser.parse_args()

    # Reading input file
    print("Loading CSV...")
    test = pd.read_csv(args.input_file, encoding="UTF-8")

    # Encoding the emotions as numbers
    cleanup_nums = {"emotion": {"happiness": 1, "sadness": 2, "neutral": 0}}
    test = test.replace(cleanup_nums)
    # cleaning and transforming the data
    test_lines = test['content'].values
    new_test_lines = create_vector_per_line(test_lines, dict_of_words, 32)
    tensor_sample_test = torch.tensor(new_test_lines)
    tensor_label_test = torch.tensor(test['emotion'].values)
    # create test loader
    test_data =[]
    for i in range(len(tensor_sample_test)):
        test_data.append([tensor_sample_test[i], tensor_label_test[i]])
    testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=100)
    lr = 0.001
    batch_size = 512

    y_pred_test, predictions_df1 = [],[]

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
            predictions_df1.append(top_class1.tolist())


    dict_of_all= {"emotion":[], "content":[]}
    for batch_prediction in predictions_df1:
        for prediction in batch_prediction:
            if prediction == 0:
                dict_of_all["emotion"].append('neutral')
            elif prediction == 1:
                dict_of_all["emotion"].append('happiness')
            else:
                dict_of_all["emotion"].append('sadness')

    for line in test_lines:
        dict_of_all["content"].append(line)


        #create predictions csv
    prediction_df = pd.DataFrame(dict_of_all)
    prediction_df.to_csv("prediction.csv", index=False, header=True)
    dict_of_words1.close()


if __name__ == '__main__':
    main()

