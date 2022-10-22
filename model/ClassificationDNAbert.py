import torch
import torch.nn as nn

from model import DNAbert

'''Classification DNA bert model'''
class ClassificationBERT(nn.Module):
    def __init__(self, config):
        super(ClassificationBERT,self).__init__()
        self.config = config

        # 加载预训练模型参数
        self.BERT = DNAbert.BERT(config)

        self.classification = nn.Sequential(
            nn.Linear(768, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, seqs):
        representation = self.BERT(seqs)

        output = self.classification(representation)

        return output, representation