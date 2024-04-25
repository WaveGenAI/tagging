from logging import Logger

import numpy as np
import torch

from model.bart import BartCaptionModel
from utils.audio_utils import load_audio, STR_CH_FIRST

class Tagger:

    def __init__(self):
        self.model = None
        self.device = None
        self.logger = Logger("Tagger")
    def load_model(self):
        """
          Function to load the pre-trained model.

          :return: None
        """
        model: BartCaptionModel = BartCaptionModel.from_pretrained("Ostixe360/lp-music-caps")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model.to(self.device, dtype=torch.float32)

        model.eval()
        self.model = model



    def get_audio(self, audio_paths, duration=10, target_sr=16000, max_batch=4):
        """
          Function to load and process audio files.

          :param audio_paths: List of paths to the audio files
          :param duration: Duration of the audio to be processed in seconds
          :param target_sr: Target sample rate for the audio
          :param max_batch: Maximum batch size for processing
          :return: Batched audio tensor and a map of audio paths to their indices in the batch
        """
        n_samples = int(duration * target_sr)
        batched_audio = []
        batched_audio_map = []
        audios = []
        audio_map = {}
        i = 0
        for path in audio_paths:
            audio, _ = load_audio(
                path=path,
                ch_format=STR_CH_FIRST,
                sample_rate=target_sr,
                downmix_to_mono=True,
            )
            if len(audio.shape) == 2:
                audio = audio.mean(0, False)  # to mono
            input_size = int(n_samples)
            if audio.shape[-1] < input_size:  # pad sequence
                pad = np.zeros(input_size)
                pad[: audio.shape[-1]] = audio
                audio = pad
            ceil = int(audio.shape[-1] // n_samples)
            audio_split = np.split(audio[:ceil * n_samples], ceil)
            audio = torch.from_numpy(np.stack(audio_split).astype('float32'))
            i += len(audio_split)
            is_truncated = False
            if i >= max_batch:
                i = 0
                if len(audios) == 0:
                    self.logger.warning(f"Audio file {path} is too long to be processed (max batch size {max_batch} actual batch: {len(audio_split)}). truncating it.")
                    a = audio[:max_batch]
                    audios.append(a)
                    audio_map[path] = max_batch
                    is_truncated = True
                    # continue    # for skipping the audio file
                audio_cat = torch.cat(audios, 0)
                batched_audio.append(audio_cat)
                batched_audio_map.append(audio_map)
                audios = []
                audio_map = {}
            if not is_truncated:
                audio_map[path] = i = len(audio_split)
                audios.append(audio)


        if len(audios) == 0:
            raise ValueError("No audio files to process.")
        elif len(audios[0]) >= max_batch:
            self.logger.warning(
                f"Last audio file {path} is too long to be processed (max batch size {max_batch} actual batch: {len(audios[0])}). Truncating it.")
            audios[0] = audios[0][:max_batch]
            audio_map[path] = max_batch
        audio = torch.cat(audios, 0)
        batched_audio.append(audio)
        batched_audio_map.append(audio_map)

        return batched_audio, batched_audio_map


    def tag(self, audio_paths: list):
        """
          Function to generate tags for the given audio files.

          :param audio_paths: List of paths to the audio files
          :return: Dictionary mapping audio paths to their generated tags
        """
        response = {}
        batched_audio_tensor, batched_audio_map = self.get_audio(audio_paths=audio_paths)
        for audio_tensor, audio_map in zip(batched_audio_tensor, batched_audio_map):
            if self.device is not None:
                audio_tensor = audio_tensor.to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    samples=audio_tensor,
                    num_beams=5,
                )
            i = 0
            for path, stop in audio_map.items():
                chunk = 0
                inference = ""
                for text in output[i:stop]:
                    time = f"[{chunk * 10}:00-{(chunk + 1) * 10}:00]"
                    inference += f"{time}\n{text} \n \n"
                    chunk += 1
                response[path] = inference
                i = stop
        return response


if __name__ == "__main__":
    audio_paths = ["./orchestra.wav", "./electronic.mp3", ]
    tagger = Tagger()
    tagger.load_model()
    res = tagger.tag(audio_paths)
    for path, description in res.items():
        print(path)
        print(description)

    print("Done!")
