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
#         print("caption shape:",captions.shape)
        embeddings = self.embed(captions)
        print("embedding shape:",embeddings.shape)
        print("features shape:",features.shape, "--- unsqueezed features shape:", features.unsqueeze(1).shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        print("combined features and embedding shape:",embeddings.shape)
        output, (hn, cn) = self.lstm(embeddings)
#         print(output.shape, hn.shape, cn.shape)
        outputs = self.linear(output[:,:-1,:])
#         print(outputs.shape)
        return outputs

    def sample(self, inputs, states=None):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
#         print("inputs:", inputs.shape)
        for i in range(20): 
            if i == 0:
                output, (hiddens, states) = self.lstm(inputs) 
                print("0:",output.shape, hiddens.shape, states.shape)
#                 outputs = hiddens
#                 outputs = self.embed(output) 
#                 print(i,":",output.shape, hiddens.shape, states.shape)
#             embedding start
                outputs = self.embed(torch.tensor([[0]]))
#                 print(output.shape)
#                 print(output.data)
            else:
                print(i,":",outputs.shape, hiddens.shape, states.shape)
                output, (hiddens, states) = self.lstm(outputs,(hiddens,states)) 
#                 print("LSTM Output:",output.shape)
                outputs = self.linear(output)  
#                 print("Predicted Output",outputs)
    #             print("outputs:", outputs.shape)
                prediction = torch.argmax(outputs, dim=2)
                print(prediction[0])
                sampled_ids.append(prediction[0])
                outputs = self.embed(prediction)                  
#                 print("Embedding Output:", outputs)
        sampled_ids = torch.stack(sampled_ids, 1)     
        return sampled_ids