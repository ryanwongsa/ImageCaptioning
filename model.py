import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision.models.resnet import model_urls
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
        resnet = models.resnet50(pretrained=True)
#         print(resnet)
        modules = list(resnet.children())[:-1]
#         print("--------------------------------------------------")
#         print(modules)
        self.resnet = nn.Sequential(*modules)
#         print("--------------------------------------------------")
#         print(self.resnet)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        # making it a variable to indicate it changes
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        print(captions.shape[1])
        embeddings = self.embed(captions)
        print(embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        print(embeddings.shape)
        output, (hn, cn) = self.lstm(embeddings)
        print(output.shape, hn.shape, cn.shape)
        outputs = self.linear(output[:,1:,:])
        print(outputs.shape)
        return outputs

    def sample(self, inputs):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass