import torch
from torch import nn
import numpy as np

AMINOACIDS = sorted(list('FLIMVPTAYHQNKDECWRSG'))
NUCLEOTIDES = list('ATGC')
AMINOACIDS_MAPPING = {
    'AAA': 'K',    'AAT': 'N',    'AAG': 'K',    'AAC': 'N',
    'ATA': 'I',    'ATT': 'I',    'ATG': 'M',    'ATC': 'I',
    'AGA': 'R',    'AGT': 'S',    'AGG': 'R',    'AGC': 'S',
    'ACA': 'T',    'ACT': 'T',    'ACG': 'T',    'ACC': 'T',
    'TAA': '$',    'TAT': 'Y',    'TAG': '$',    'TAC': 'Y',
    'TTA': 'L',    'TTT': 'F',    'TTG': 'L',    'TTC': 'F',
    'TGA': '$',    'TGT': 'C',    'TGG': 'W',    'TGC': 'C',
    'TCA': 'S',    'TCT': 'S',    'TCG': 'S',    'TCC': 'S',
    'GAA': 'E',    'GAT': 'D',    'GAG': 'E',    'GAC': 'D',
    'GTA': 'V',    'GTT': 'V',    'GTG': 'V',    'GTC': 'V',
    'GGA': 'G',    'GGT': 'G',    'GGG': 'G',    'GGC': 'G',
    'GCA': 'A',    'GCT': 'A',    'GCG': 'A',    'GCC': 'A',
    'CAA': 'H',    'CAT': 'H',    'CAG': 'H',    'CAC': 'H',
    'CTA': 'L',    'CTT': 'L',    'CTG': 'L',    'CTC': 'L',
    'CGA': 'R',    'CGT': 'R',    'CGG': 'R',    'CGC': 'R',
    'CCA': 'P',    'CCT': 'P',    'CCG': 'P',    'CCC': 'P'
}


class DNATranscriptor:
    def __init__(self, mapping=AMINOACIDS_MAPPING):
        self.mapping = mapping

    def transcript(self, dna):
        triplet_list = self.cut_triplets(dna)
        return ''.join([self.mapping[triplet] for triplet in triplet_list])

    def cut_triplets(self, dna):
        triplet_list = []
        step_count = len(dna) // 3
        for i in range(step_count):
            triplet_list.append(dna[3 * i:3 * i + 3])
        return triplet_list

    def __call__(self, dna):
        return self.transcript(dna)


class AminoacidTokenizer:
    def __init__(self):
        items = np.unique(list(zip(*AMINOACIDS_MAPPING.items()))[1]).tolist()
        self.mapping = dict(zip(items, range(len(items))))

    def tokenize(self, x):
        gene = torch.tensor(list(map(lambda x: self.mapping[x], x)))
        position = torch.arange(len(gene))
        s_codon = torch.roll(torch.cumsum((gene == 0), dim=0), 1)
        s_codon[0] = 0
        return {'protein': gene.view(1, -1), 'position': position.view(1, -1), 's-codon': s_codon.view(1, -1)}

    def __call__(self, x):
        return self.tokenize(x)


class GeneEncoder(nn.Module):
    def __init__(self):
        super(GeneEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=128)
        self.position_embedding = nn.Embedding(num_embeddings=128, embedding_dim=128, padding_idx=0)
        self.stop_codon_embedding = nn.Embedding(num_embeddings=128, embedding_dim=128)
        block = nn.TransformerEncoderLayer(d_model=128,
                                           nhead=8,
                                           dim_feedforward=64,
                                           dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 6)

    def forward(self, gene, position, stop_codon_alignment):
        gene_encoding = self.embedding(gene)
        position_encoding = self.position_embedding(position)
        stop_codon_encoding = self.stop_codon_embedding(stop_codon_alignment)

        embeddings = (gene_encoding + stop_codon_encoding) * position_encoding
        outputs = self.encoder(embeddings)
        pooling = outputs.mean(dim=1)
        return outputs, pooling


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

