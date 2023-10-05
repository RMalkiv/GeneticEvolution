import torch
from torch import nn
from resource.resource import ResourceType


class BasicProduction:
    def __init__(self, requirements: ResourceType, output: ResourceType):
        self.requirements = requirements

        self.output = output

    def produce(self, input_resources: ResourceType) -> ResourceType:
        multiplier = (input_resources // self.requirements).min()
        result_resources = input_resources - (self.requirements - self.output) * multiplier
        return result_resources


# class BasicProductionEncoder(nn.Module):
#     def __init__(self):
#         super(BasicProductionEncoder, self).__init__()
#         self.
#         self.linear_priority =
#
#     def forward(self, x):
#         return x


class BasicPropertyEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 is_sigmoid=False,
                 is_relu=False,
                 threshold=0.5):
        super(BasicPropertyEncoder, self).__init__()
        self.is_sigmoid = is_sigmoid
        self.is_relu = is_relu
        self.threshold = threshold
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x_ = self.linear(x)
        if self.is_sigmoid:
            x_ = torch.sigmoid(x_)
            x_ = float((x_ > self.threshold).int())
        elif self.is_relu:
            x_ = nn.ReLU(x_)

        return float(x_)


class BasicProductionEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 is_sigmoid=False,
                 is_relu=False,
                 threshold=0.5):
        super(BasicProductionEncoder, self).__init__()
        self.is_sigmoid = is_sigmoid
        self.is_relu = is_relu
        self.threshold = threshold
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x_ = self.linear(x)
        if self.is_sigmoid:
            x_ = torch.sigmoid(x_)
        elif self.is_relu:
            x_ = nn.ReLU(x_)

        return x_ > self.threshold