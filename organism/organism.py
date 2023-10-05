import organism.utils
import genetics.basic_tools


class Organism:
    def __init__(self, dna):


        self.dna = organism.utils.generate_dna() if dna is None else dna
        self.genome = genetics.basic_tools.dna_transcriptor(self.dna)

    @staticmethod
    def get_genome():
        pass

