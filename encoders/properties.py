from encoders.utils import BasicPropertyEncoder

class PropertiesEncoder:
    def __init__(self, input_dim):
        self.base_hp = BasicPropertyEncoder(input_dim, is_relu=True)
        self.base_size = BasicPropertyEncoder(input_dim, is_relu=True)
        self.base_latency = BasicPropertyEncoder(input_dim, is_relu=True)

        self.can_photosynthesisA_1_2 = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.5)
        self.can_photosynthesisA_2_2 = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.5)
        self.can_photosynthesisB_1_1 = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.5)
        self.can_photosynthesisB_blocker = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.5)

        self.can_chemosynthesisA_1_2 = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.5)
        self.can_chemosynthesisA_2_2 = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.5)
        self.can_chemosynthesisB_1_1 = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.5)

        self.can_inhale = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.4)
        self.can_inhale_blocker = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.8)
        self.can_exhale = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.4)
        self.can_exhale_blocker = BasicPropertyEncoder(input_dim, is_sigmoid=True, threshold=0.8)

    def _get_phososynthesis(self, embedding):
        a = self.can_photosynthesisA_1_2(embedding)
        b = self.can_photosynthesisA_2_2(embedding)
        c = self.can_photosynthesisB_1_1(embedding)
        d = self.can_photosynthesisB_blocker(embedding)
        return int((a and b) or (c and (not d)))

    def _get_chemosynthesis(self, embedding):
        a = self.can_chemosynthesisA_1_2(embedding)
        b = self.can_chemosynthesisA_2_2(embedding)
        return int(a and b)

    def _get_inhale(self, embedding):
        a = self.can_inhale_blocker(embedding)
        b = self.can_inhale(embedding)
        return int((not a) and b)

    def _get_exhale(self, embedding):
        a = self.can_exhale_blocker(embedding)
        b = self.can_exhale(embedding)
        return int((not a) and b)

    def _get_floats(self, embedding):
        base_hp = self.base_hp(embedding)
        base_size = self.base_size(embedding)
        base_latency = self.base_latency(embedding) * 10
        return {'base_hp': base_hp, 'base_size': base_size, 'base_latency': base_latency}

    def get_properties(self, embedding):
        output_properties = self._get_floats(embedding=embedding)
        output_properties['can_inhale'] = self._get_inhale(embedding=embedding)
        output_properties['can_exhale'] = self._get_exhale(embedding=embedding)
        output_properties['can_photosynthesis'] = self._get_phososynthesis(embedding=embedding)
        output_properties['can_chemosynthesis'] = self._get_chemosynthesis(embedding=embedding)

        return output_properties

