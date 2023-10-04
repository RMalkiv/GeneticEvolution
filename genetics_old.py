import torch
from torch import nn
import numpy as np



class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class FloatHead(nn.Module):
    def __init__(self):
        super(FloatHead, self).__init__()
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        return self.linear(x)


dna_transcriptor = DNATranscriptor()
aminoacid_tokenizer = AminoacidTokenizer()
model = GeneEncoder()
model.eval()

is_cool_gene = Head()
is_float_gene = FloatHead()


def encode_gene(gene):
    return aminoacid_tokenizer(dna_transcriptor(gene))


def inference_gene(gene):
    with torch.no_grad():
        model.eval()
        gene_tokenzied = encode_gene(gene)
        out, pooling = model(gene_tokenzied['protein'], gene_tokenzied['position'], gene_tokenzied['s-codon'])
        return float(is_cool_gene(pooling)), float(is_float_gene(pooling))

