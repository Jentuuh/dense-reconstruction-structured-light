from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Encoder(ABC):
    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.n = 0

    @abstractmethod
    def get_encoding_pattern(self, depth) -> np.array:
        pass


class Decoder(ABC):
    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.n = 0

    @abstractmethod
    def set_frame(self, depth: int, frame: np.array):
        pass

    @abstractmethod
    def decode_frames(self) -> Tuple[np.array, np.array, np.array]:
        pass
