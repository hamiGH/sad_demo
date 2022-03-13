from vad import Vad
import torchaudio
import numpy as np
import os


class SpeechProcessingUnit:

    def __init__(self, fs):
        self.fs = fs

    def __vad(self, signal, fs, gpu_number):
        """
        :param signal:  numpy array of signal data
        :param fs: sample rate
        :param gpu_number: gpu device number
        :return: {"vad_out": vad output(numpy array), "fs": sample rate of vad(int)}
        """
        # convert to mono
        if len(signal.shape) > 1:
            signal = signal[:, 0]

        # apply vad on input speech
        vad_unit = Vad(gpu_number)
        vad_prob, vad_fs = vad_unit.run(signal, fs)
        # result = {"vad_out": vad_out.tolist(), 'fs': vad_fs}
        return vad_prob, vad_fs

    @staticmethod
    def vad2annotation(vad_prob, vad_fs=100, threshold=0.5):
        # applying the threshold
        vad_prob = np.array((vad_prob+(0.5-threshold)).round())
        # extracting speech index
        vad_prob = np.where(vad_prob == 1)[0]
        vad_prob = sorted(set(vad_prob))
        # extracting speech groups
        gaps = [[s, e] for s, e in zip(vad_prob, vad_prob[1:]) if s + 1 < e]
        edges = iter(vad_prob[:1] + sum(gaps, []) + vad_prob[-1:])
        lists = list(zip(edges, edges))
        # converting to annotation (sad style)
        annotation = []
        for pair in lists:
            annotation.append({"begin": pair[0] / vad_fs, "end": (pair[1] + 1) / vad_fs})
        return annotation

    def apply_vad(self, file_path, threshold, gpu_number):
        # load audio
        signal, fs = torchaudio.load(file_path)
        os.remove(file_path)
        signal = signal[0]

        # adjust fs
        adjust_fs = torchaudio.transforms.Resample(fs, self.fs)
        signal = adjust_fs(signal)
        signal = signal.numpy()

        vad_prob, vad_fs = self.__vad(signal, fs, gpu_number)

        vad_annotation = self.vad2annotation(vad_prob, vad_fs, threshold)
        return vad_annotation
