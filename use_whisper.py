import whisper

# We're using the `base` size model. Check out
# https://github.com/openai/whisper#available-models-and-languages
# for more robust models.
model = whisper.load_model("base")

result = model.transcribe("audio.mp3",fp16=False, language='English')

print(result["text"][:300])
