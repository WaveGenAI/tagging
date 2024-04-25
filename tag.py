from tagger import Tagger

tagger = Tagger()
tagger.load_model()
audio_paths = ["./orchestra.wav", "./electronic.mp3"]
tags = tagger.tag(audio_paths)
for path, description in tags.items():
    print(path)
    print(description)
