import numpy as np
from lauscher.abstract import Transformation
from lauscher.audiowaves import MonoAudioWave
from lauscher.spike_train import SpikeTrain
from lauscher.transformations import RmsNormalizer, HanningWindow, \
    BasilarMembrane, HairCell, BushyCell


class Wave2Spike(Transformation):
    def __init__(self,
                 num_channels: int,
                 rng: np.random.Generator):
        self.num_channels = num_channels
        self.rng = rng

    def __call__(self, wave: MonoAudioWave) -> SpikeTrain:
        # noinspection PyTypeChecker
        return wave \
            .transform(RmsNormalizer(0.3)) \
            .transform(HanningWindow(rampup_time=30e-3, rampdown_time=30e-3)) \
            .transform(BasilarMembrane(channels=self.num_channels)) \
            .transform(HairCell()) \
            .transform(BushyCell(self.rng))
