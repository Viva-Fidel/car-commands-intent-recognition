import logging
import tensorflow as tf
import telegram.ext.filters as filters

from telegram import Update
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler, MessageHandler

from transformers import BertTokenizer, TFBertForSequenceClassification

from speech import Speech_Recognition

speech_recognition = Speech_Recognition()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
loaded_model = TFBertForSequenceClassification.from_pretrained('saved_model')

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

# Вывод сообщения при выборе команды /start
async def start(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Привет, отправь аудиосообщение, чтобы управлять авто. Бот понимает команды, которые связаны с: поиском маршрута, открытием/закрытием дверей, включением/выключением радио, кондиционера, дворников и управлением температуры в салоне. Для того, чтобы увидеть примеры команд набери /help")

# Вывод сообщения при выборе команды /help
async def help(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Нужно всего лишь отправить аудиосообщение. Пример аудио сообщений: проложи маршрут домой, повысь температуру кондиционера, закрой переднюю дверь, включи радио")

# Получение и обработка аудио
async def handle_voice_message(update: Update, context: CallbackContext):
    audio_file = await context.bot.get_file(update.message.voice.file_id)

    # Скачивание аудиофайла
    local_path = await audio_file.download_to_drive()
    logging.info(f"Audio file saved to {local_path}")

    # Попытка конвертации аудио в текст и вывод сообщения пользователю
    try:
        text = await transform_audio_to_text(local_path)
        intent = await classify_intent(text)
        await update.message.reply_text(f"Ваше сообщение: {text}")
        await update.message.reply_text(f"Выполняемое действие: {intent}")
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        await update.message.reply_text("Вас плохо слышно, отправьте аудисообщение заново")

# Конвертации аудио в текст
async def transform_audio_to_text(local_path):
    logging.info(f"Transcribing audio file at {local_path}")
    text = speech_recognition.audio_to_text(local_path)
    return text

# Получаем намерение
async def classify_intent(text):
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


if __name__ == '__main__':
    application = ApplicationBuilder().token('').build()  # Insert your bot token here

    start_handler = CommandHandler('start', start)
    help_handler = CommandHandler('help', help)
    application.add_handler(start_handler)
    application.add_handler(help_handler)

    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.run_polling()

