import torch
import torch.nn as nn

from model import DNAbert

'''bert Fusion 模型'''
class FusionBERT(nn.Module):
    def __init__(self, config):
        super(FusionBERT,self).__init__()
        self.config = config

        self.config.kmer = self.config.kmers[0]
        self.bertone = DNAbert.BERT(self.config)

        self.config.kmer = self.config.kmers[1]
        self.berttwo = DNAbert.BERT(self.config)

        self.Ws = torch.randn(1, 768).cuda()
        self.Wh = torch.randn(1, 768).cuda()

        self.classification = nn.Sequential(
            nn.Linear(768, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, seqs):
        # print(seqs)
        representationX = self.bertone(seqs)
        representationY = self.berttwo(seqs)

        # print(representationX)
        # print(representationY)
        F = torch.sigmoid(self.Ws * representationX + self.Wh * representationX)
        # print(F)
        representation = F * representationX + (1 - F) * representationY
        # print(representation)

        output = self.classification(representation)

        return output, representation
