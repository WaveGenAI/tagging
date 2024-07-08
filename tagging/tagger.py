import random
from logging import Logger

import numpy as np
import torch
from .metatags import CLAPTagger
from tagging.model.bart import BartCaptionModel
from tagging.utils.audio_utils import STR_CH_FIRST, load_audio


class Tagger:

    def __init__(self, prob_threshold=0.5):
        """Initializes the Tagger class.

        Args:
            prob_threshold (float, optional): the sensibility of the tagger for CLAP. Defaults to 0.5.
        """

        self.model = None
        self.device = None
        self.logger = Logger("Tagger")

        self.clap_tagger = CLAPTagger(prob_threshold=prob_threshold)

    def load_model(self):
        """
        Function to load the pre-trained model.

        :return: None
        """
        model: BartCaptionModel = BartCaptionModel.from_pretrained(
            "Ostixe360/lp-music-caps"
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model.to(self.device, dtype=torch.float32)

        model.eval()
        self.model = model

    def unload_model(self):
        """
        Function to unload the pre-trained model.

        :return: None
        """
        del self.model
        torch.cuda.empty_cache()
        self.model = None
        self.device = None

    def get_audio(
        self, audio_paths, duration=10, target_sr=16000, max_batch=50, n_samples=4
    ):
        """
        Function to load and process audio files.

        :param audio_paths: List of paths to the audio files
        :param duration: Duration of the audio to be processed in seconds
        :param target_sr: Target sample rate for the audio
        :param max_batch: Maximum batch size for processing
        :return: Batched audio tensor and a map of audio paths to their indices in the batch
        """
        t_samples = int(duration * target_sr)
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
            input_size = int(t_samples)
            if audio.shape[-1] < input_size:  # pad sequence
                pad = np.zeros(input_size)
                pad[: audio.shape[-1]] = audio
                audio = pad
            ceil = int(audio.shape[-1] // t_samples)
            audio_split = np.split(audio[: ceil * t_samples], ceil)
            indices = sorted(random.sample(range(len(audio_split)), n_samples))
            audio_split = [audio_split[i] for i in indices]
            audio = torch.from_numpy(np.stack(audio_split).astype("float32"))
            i += len(audio_split)
            if i >= max_batch:
                if len(audios) == 0:
                    self.logger.warning(
                        f"Audio file {path} is too long to be processed (max batch size {max_batch} actual batch: {len(audio_split)}). truncating it."
                    )
                    a = audio[:max_batch]
                    audios.append(a)
                    audio_map[path] = max_batch
                    i = 0
                    # continue # for skipping the audio file
                else:
                    i = len(audio_split)
                audio_cat = torch.cat(audios, 0)
                batched_audio.append(audio_cat)
                batched_audio_map.append(audio_map)
                audios = []
                audio_map = {}
                if (
                    i != 0
                ):  # if i == 0, then the previous audio was too long and has been processed
                    audio_map[path] = i
                    audios.append(audio)
            else:
                audio_map[path] = i
                audios.append(audio)
        if len(audios) != 0:
            if len(audios[0]) >= max_batch:
                self.logger.warning(
                    f"Last audio file {path} is too long to be processed (max batch size {max_batch} actual batch: {len(audios[0])}). Truncating it."
                )
                audios[0] = audios[0][:max_batch]
                audio_map[path] = max_batch
            audio = torch.cat(audios, 0)
            batched_audio.append(audio)
            batched_audio_map.append(audio_map)

        return batched_audio, batched_audio_map

    def tag(self, audio_paths: list, max_batch: int = 50):
        """
        Function to generate tags for the given audio files.

        :param audio_paths: List of paths to the audio files
        :return: Dictionary mapping audio paths to their generated tags
        """
        response = {}
        if self.model is None:
            self.logger.warning("Model not loaded. Loading the model.")
            self.load_model()

        batched_audio_tensor, batched_audio_map = self.get_audio(
            audio_paths=audio_paths, max_batch=max_batch
        )
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
                response[path] = [inference, self.clap_tagger.tag(path)]
                i = stop
        return response


if __name__ == "__main__":
    audio_paths = [
        "./orchestra.wav",
        "./electronic.mp3",
    ]
    tagger = Tagger(0.03)
    tagger.load_model()
    res = tagger.tag(audio_paths, max_batch=50)
    for path, description in res.items():
        print(path)
        print(description)

    print("Done!")
