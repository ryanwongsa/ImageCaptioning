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
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        # making it a variable to indicate it changes
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        output, (hn, cn) = self.lstm(embeddings)
        outputs = self.linear(output[:,:-1,:])
        return outputs

    def sample(self, inputs, states=None):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        for i in range(20): 
            if i == 0:
                output, (hiddens, states) = self.lstm(inputs)
            else:
                output, (hiddens, states) = self.lstm(outputs,(hiddens,states)) 
            outputs = self.linear(output)  
            prediction = torch.argmax(outputs, dim=2)
            sampled_ids.append(prediction[0])
            outputs = self.embed(prediction)
        sampled_ids = torch.stack(sampled_ids, 1)     
        return sampled_ids