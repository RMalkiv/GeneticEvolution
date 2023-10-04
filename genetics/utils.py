class Constants:
    def __init__(self):

        self.AMINOACIDS = sorted(list('FLIMVPTAYHQNKDECWRSG'))
        self.NUCLEOTIDES = list('ATGC')
        self.AMINOACIDS_MAPPING = {
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
            'CCA': 'P',    'CCT': 'P',    'CCG': 'P',    'CCC': 'P'}

constants = Constants()


def cut_gene(genome):
    genes = genome.split('$')
    genes = [gene + '$' for gene in genes]
    return genes
