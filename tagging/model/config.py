from transformers import PretrainedConfig


class MusicCapsConfig(PretrainedConfig):
    def __init__(self, n_mels=128, num_of_conv=6, sr=16000, duration=10, max_length=128, label_smoothing=0.1,
                 bart_type="facebook/bart-base", audio_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.num_of_conv = num_of_conv
        self.sr = sr
        self.duration = duration
        self.max_length = max_length
        self.label_smoothing = label_smoothing
        self.bart_type = bart_type
        self.audio_dim = audio_dim
