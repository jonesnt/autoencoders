from abc import ABC, abstractmethod
from torch import nn

class AutoencoderBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, x) -> tuple:
        pass
