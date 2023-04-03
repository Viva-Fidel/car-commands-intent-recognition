import logging
import pyaudio
import tensorflow as tf
import json
import wave

from transformers import BertTokenizer, TFBertForSequenceClassification

from vosk import KaldiRecognizer, Model

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
loaded_model = TFBertForSequenceClassification.from_pretrained('saved_model')

model = Model('models/vosk-model-small-ru-0.22')

prediction_classes = {0: "включение_дворников",
                   1: "включение_кондиционера",
                   2: "включение_радио",
                   3: "выключение_дворников",
                   4: "выключение_кондиционера",
                   5: "выключение_радио",
                   6: "закрытие_двери",
                   7: "открытие_двери",
                   8: "повысить_температуру_кондиционера",
                   9: "поиск_маршрута",
                   10: "понизить_температуру_кондиционера"

}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Записываем аудио с микрофона
def record_audio(duration, file_path):

    print("Начните говорить")

    chunk_size = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    frames_per_buffer=chunk_size,
                    input=True)

    frames = []

    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    p.terminate()

    # Сохраняем аудио в wav
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Попытка конвертации аудио в текст и вывод сообщения пользователю
    try:
        text = transform_audio_to_text(file_path)
        intent = classify_intent(text)
        return (f"Ваше сообщение: {text}, Выполняемое действие: {intent}")
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        return("Вас плохо слышно, отправьте аудисообщение заново")


# Конвертации аудио в текст
def transform_audio_to_text(local_path):

    with wave.open(local_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        recognizer = KaldiRecognizer(model, sample_rate)

        while True:
            data = wav_file.readframes(4000)
            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result['text']
                return text


# Получаем намерение
def classify_intent(text):
    input_encoding = tokenizer.encode_plus(text, truncation=True, padding=True, return_tensors="tf")
    logits = loaded_model(input_encoding['input_ids'], input_encoding['attention_mask'])[0]
    probabilities = tf.nn.softmax(logits, axis=-1)
    prediction = tf.argmax(probabilities, axis=-1).numpy()[0]
    confidence = probabilities[0][prediction].numpy()

    # Проверяем насколько система уверена
    if confidence > 0.5:
        return prediction_classes[prediction]
    else:
        return "Система не смогла опознать команду, отправьте аудиосообщение ещё раз"


audio_results = record_audio(5, 'my_audio.wav')
print(audio_results)
