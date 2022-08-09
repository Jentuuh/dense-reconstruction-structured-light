import math
from cv2 import phase

import numpy as np
from codec import Encoder, Decoder
from typing import Tuple


N_PHASES = 16

# pitch = frequency
def compute_phase_vector(length: int, phase: float, pitch: float) -> np.array:
    phase_vector = np.zeros((length, 1, 3), np.uint8)
    # print(phase_vector)
    for i in range(length):
        # 0.5 and 1 + = to map result between 0:1 from -1:1

        # math.cos(2 * math.pi * i / pitch - phase) 
        # i/pitch = which fraction of circle we are on currently
        # -phase = phase shift
        amp = 0.5 * (1 + math.cos(2 * math.pi * i / pitch - phase))
        phase_vector[i, 0] = [255.0 * amp, 255.0 * amp, 255.0 * amp]

    
    return phase_vector

# 1/f = T
# b/T 
# cols / (1/f)
class PhaseShift2x3Encoder(Encoder):
    def __init__(self, cols: int, rows: int):
        super().__init__(cols, rows)
        self.n = 12
        self.patterns = []

        # phases in radians
        phases = [2.0 * math.pi / 3.0 * i for i in range(3)]
        # horizontal encoding patterns
        self.patterns.extend([compute_phase_vector(cols, phase, float(cols)/N_PHASES).transpose((1, 0, 2)) for phase in phases])

        # print(self.patterns)

        # phase cue patterns
        self.patterns.extend([compute_phase_vector(cols, phase, cols).transpose((1, 0, 2)) for phase in phases])

        # vertical encoding patterns
        self.patterns.extend([compute_phase_vector(rows, phase, float(rows)/N_PHASES) for phase in phases])

        # vertical phase cue patterns
        self.patterns.extend([compute_phase_vector(rows, phase, rows) for phase in phases])

        # print(np.array(self.patterns).size)
        

    def get_encoding_pattern(self, depth) -> np.array:
        return self.patterns[depth]


class PhaseShift2x3Decoder(Decoder):
    def __init__(self, cols: int, rows: int):
        super().__init__(cols, rows)
        self.n = 12
        self.frames = self.n * [None]

    def decode_frames(self) -> Tuple[np.array, np.array, np.array]:
        # TODO
        pass

    def set_frame(self, depth: int, frame: np.array):
        self.frames[depth] = frame