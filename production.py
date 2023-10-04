from resource import ResourceType

class Production:
    def __init__(self):
        self.requirements = ResourceType()

        self.output = ResourceType()

    def produce(self, input_resources):
        multiplier = (input_resources // self.requirements).min()
        result_resources = input_resources - (self.requirements - self.output) * multiplier
        return result_resources