import json
import os
import subprocess
from datetime import datetime

from vosk import KaldiRecognizer, Model


class Speech_Recognition:



    def __init__(self):

        self.model_path = 'models/vosk-model-small-ru-0.22'
        self.sample_rate = 16000
        self.ffmpeg_path = 'models/ffmpeg/bin/ffmpeg'

        model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(model, self.sample_rate)
        self.recognizer.SetWords(True)


    def audio_to_text(self, audio_file_name=None) -> str:

        # Конвертация аудио в wav
        process = subprocess.Popen(
            [self.ffmpeg_path,
             "-loglevel", "verbose",
             "-i", audio_file_name,
             "-ar", str(self.sample_rate),
             "-ac", "1",
             "-f", "s16le",
             "-"
             ],
            stdout=subprocess.PIPE
                                   )

        # Чтение данных кусками и распознование
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if self.recognizer.AcceptWaveform(data):
                pass

        # Возвращаем распознанный текст
        result_json = self.recognizer.FinalResult()
        result_dict = json.loads(result_json)
        return result_dict["text"]

