import genetics.utils
import numpy as np
import torch

class DNATranscriptor:
    def __init__(self, mapping=genetics.utils.constants.AMINOACIDS_MAPPING):
        self.mapping = mapping

    def transcript(self, dna):
        triplet_list = self.cut_triplets(dna)
        return ''.join([self.mapping[triplet] for triplet in triplet_list])

    @staticmethod
    def cut_triplets(dna):
        return [dna[i:i + 3] for i in range(0, len(dna), 3)]

    def __call__(self, dna):
        return self.transcript(dna)


class AminoacidTokenizer:
    def __init__(self):
        items = np.unique(list(zip(*genetics.utils.constants.AMINOACIDS_MAPPING.items()))[1]).tolist()
        self.mapping = dict(zip(items, range(len(items))))

    def tokenize(self, x):
        gene = torch.tensor([self.mapping[aa] for aa in x])
        position = torch.arange(len(gene))
        s_codon = torch.roll(torch.cumsum((gene == 0), dim=0), 1)
        s_codon[0] = 0
        return {'protein': gene.view(1, -1), 'position': position.view(1, -1), 's-codon': s_codon.view(1, -1)}

    def __call__(self, x):
        return self.tokenize(x)


dna_transcriptor = DNATranscriptor()
