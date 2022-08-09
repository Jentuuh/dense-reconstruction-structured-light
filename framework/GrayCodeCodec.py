import math
import cv2
import numpy as np
from typing import Tuple

from codec import Encoder, Decoder


def binary_to_gray(num: int) -> int:
    """
    The purpose of this function is to convert an unsigned
    binary number to reflected binary Gray code.

    The operator >> is shift right. The operator ^ is exclusive or.
    Source: http://en.wikipedia.org/wiki/Gray_code
    """
    return (num >> 1) ^ num


def get_bit(decimal: int, n: int) -> int:
    """
    Function takes the decimal number
    Function takes the Nth bit (1 to 31)
    Return the value of Nth bit from decimal
    Source: http://icfun.blogspot.com/2009/04/get-n-th-bit-value-of-any-integer.html
    """
    constant = 1 << (n - 1)
    return 1 if decimal & constant else 0


class GrayCodeEncoder(Encoder):
    def __init__(self, rows: int, cols: int, depth: int):
        super().__init__(cols, rows)
        self.n = depth * 4  # number of frames to be computed
        self.patterns = []

        n_horiz_bits = math.ceil(math.log2(float(self.cols)))
        n_vert_bits = math.ceil(math.log2(float(self.rows)))

        # compute horizontal encoding patterns
        for i in range(depth):
            pattern = np.zeros((1, cols, 3), np.uint8)
            # loop through columns in first row
            for j in range(cols):
                gray = binary_to_gray(j)
                # amplitude of channels
                amp = get_bit(gray, n_horiz_bits - i)
                pattern[0, j] = [255.0 * amp, 255.0 * amp, 255.0 * amp]
            self.patterns.append(pattern)
            self.patterns.append(cv2.bitwise_not(pattern))

        # compute vertical encoding patterns
        for i in range(depth):
            pattern = np.zeros((rows, 1, 3), np.uint8)
            # Loop through rows in first column
            for j in range(rows):
                gray = binary_to_gray(j)
                # Amplitude of channels
                amp = get_bit(gray, n_vert_bits - i)
                pattern[j, 0] = [255.0 * amp, 255.0 * amp, 255.0 * amp]
            self.patterns.append(pattern)
            self.patterns.append(cv2.bitwise_not(pattern))

    def get_encoding_pattern(self, depth) -> np.array:
        return self.patterns[depth]


class GrayCodeDecoder(Decoder):
    def __init__(self, cols: int, rows: int, depth: int):
        super().__init__(cols, rows)
        self.n = depth * 4
        self.frames = self.n * [None]

    def decode_frames(self) -> Tuple[np.array, np.array, np.array]:
        # TODO: implement decoder
        pass

    def set_frame(self, depth: int, frame: np.array):
        self.frames[depth] = frame
