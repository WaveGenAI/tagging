from tagging.tagger import Tagger

tagger = Tagger()
tagger.load_model()
audio_paths = ["./orchestra.wav", "./electronic.mp3"]
max_batch = 50
tags = tagger.tag(audio_paths, max_batch=max_batch)
for path, description in tags.items():
    print(path)
    print(description)
