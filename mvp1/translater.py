from transformers import pipeline
from loguru import logger

class Translater:
    def __init__(self):
        logger.info("Loading translation models...")
        self.translate_pipe_ru_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
        self.translate_pipe_en_to_ru = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
        logger.success("Models loaded!")

    from_ru_to_en = lambda self, text: self.translate_pipe_ru_to_en(text)[0]["translation"]
    from_en_to_ru = lambda self, text: self.translate_pipe_en_to_ru(text)[0]["translation"]