import whisper


def get_tokenizer(model, language):
    multilingual = not model.endswith(".en")
    return whisper.tokenizer.get_tokenizer(
        multilingual,
        language=language,
        task="transcribe",
        num_languages=100 if model == "large-v3" else 99,
    )
