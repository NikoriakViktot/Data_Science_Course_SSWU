
'''
Приклад системи розпізнавання голосу / природньої мови (Український текст) з конверцією повідомлення у текст:
https://reintech.io/blog/how-to-create-a-voice-recognition-system-with-python

необхідне оточення:
pip install SpeechRecognition
pip install PyAudio

Сценарій роботи:
1. Запис голосового повідомлення;
2. Перетворення запису до тексту;
3. Реалізація діалогу.

Застосування:
text mining голосових повідомлень;
елементарний голосовий bot (robot).

Package                      Version
---------------------------- -----------
PyAudio                      0.2.13
SpeechRecognition            3.10.0


'''

import speech_recognition as sr
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Створення екземпляру класу Recognizer
recognizer = sr.Recognizer()

# 2. Запис голосового повідомлення
def capture_voice_input():
    with sr.Microphone() as source:
        print("Говоріть...")
        audio = recognizer.listen(source)
    return audio


# 3. Перетворення голосового повідомлення на текст
def convert_voice_to_text(audio):
    try:
        text = recognizer.recognize_google(audio, language="uk-UA")
        print("Ви сказали: " + text)
    except sr.UnknownValueError:
        text = ""
        print("Вибачте, я Вас не розумію.")
    except sr.RequestError as e:
        text = ""
        print("Error; {0}".format(e))
    return text

# 4. Обробка голосових команд
def process_voice_command(text):
    if "вітаю" in text.lower():
        print("Привіт! Чим я можу Вам допомогти?")
    elif "бувай" in text.lower():
        print("До побачення! Гарного дня!")
        return True
    else:
        print("Команда не зрозумілау. Будь ласка спробуйте ще раз.")
    return False

# 5. ------------------- переклад  ----------------------------
def translator_text(text_uk):
    text_en = GoogleTranslator(source='auto', target='en').translate(text_uk)
    print('text_uk: ', text_uk)
    print('text_en: ', text_en, '\n')
    return text_en

# 6. ---------- визначення тональності текста -----------------
def analysis_tonality_vaderSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    print('Результат vaderSentiment: ')
    print("{:-<65} {}".format(text, str(vs)), '\n')
    return

# 7. Головні виклики
def main():
    end_program = False
    while not end_program:
        audio = capture_voice_input()
        text = convert_voice_to_text(audio)
        analysis_tonality_vaderSentiment(translator_text(text))
        end_program = process_voice_command(text)


if __name__ == "__main__":
    main()