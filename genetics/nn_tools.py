import torch
from torch import nn
import genetics.utils


class GeneEncoder(nn.Module):
    def __init__(self):
        super(GeneEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=128)
        self.position_embedding = nn.Embedding(num_embeddings=128, embedding_dim=128, padding_idx=0)

        block = nn.TransformerEncoderLayer(d_model=128,
                                           nhead=8,
                                           dim_feedforward=128,
                                           dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 6)

    def forward(self, gene, position, return_outputs=False):
        gene_encoding = self.embedding(gene)
        position_encoding = self.position_embedding(position)

        embeddings = gene_encoding * position_encoding
        outputs = self.encoder(embeddings)
        pooling = outputs.mean(dim=1)
        if return_outputs:
            return outputs, pooling
        else:
            return pooling


class GenomeEncoder(nn.Module):
    def __init__(self):
        super(GenomeEncoder, self).__init__()
        self.position_embedding = nn.Embedding(num_embeddings=128, embedding_dim=128, padding_idx=0)

        block = nn.TransformerEncoderLayer(d_model=128,
                                           nhead=8,
                                           dim_feedforward=128,
                                           dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 6)

    def forward(self, genes, positions, return_outputs=False):
        position_encoding = self.position_embedding(positions)

        embeddings = genes + position_encoding
        outputs = self.encoder(embeddings)
        pooling = outputs.mean(dim=1)
        if return_outputs:
            return outputs, pooling
        else:
            return pooling


class OrganismPropertiesEncoder(nn.Module):
    def __init__(self):
        super(OrganismPropertiesEncoder, self).__init__()

        self.gene_encoder = GeneEncoder()
        self.genome_encoder = GenomeEncoder()




    # def get_genome_embedding(self, genome):
    #     genes = genetics.utils.cut_gene(genome)
    #     genes_embeddings = [self.gene_encoder(gene) for gene in genes]
    #     return genes_embeddings

    def forward(self, genes, stop_codon_alignment):
        embedded_genes = [self.gene_encoder(gene, positions) for gene, positions in genes]
        embedded_genome = self.genome_encoder(torch.cat(embedded_genes, stop_codon_alignment), dim=-1)
        return embedded_genome