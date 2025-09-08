import numpy as np
import pickle, json, nltk, re, random
from pyvi import ViTokenizer, ViPosTagger, ViUtils
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model
from datetime import date, datetime, time, timedelta

lemmatizer = WordNetLemmatizer()

model = load_model(f"chatbot_model.h5")
intents = json.loads(open(f"intents.json", encoding='utf-8').read())
words = pickle.load(open(f"words.pkl", "rb"))
classes = pickle.load(open(f"classes.pkl", "rb"))
data_file = open(f"intents.json", encoding='utf-8').read()
intents = json.loads(data_file)

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(ViTokenizer.tokenize(sentence))
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def rounding_text_hour(text):

    mask1 = any(word in text for word in ['tiếng', 'phút', 'giờ', 'minute', 'minutes'])
    mask2 = any(word in text for word in ['giờ làm việc', 'giờ hành chính', 'giờ đi làm', 'giờ vào làm', 'mấy giờ', "giờ ngủ trưa",
                                          'giờ có mặt', 'giờ tan sở', 'giờ ra về', 'giờ giải lao', 'giờ nghỉ trưa', "giờ nghỉ ngơi"])
    if mask1 & ~mask2:
        text = text.replace('giờ', 'tiếng').replace('hour', 'tiếng').replace('minute', 'phút').replace('mins', 'phút')
        h = re.findall('[0-9]+tiếng', text.replace(" ", ""))
        m = re.findall('[0-9]+phút', text.replace(" ", ""))
        if ((len(h) > 0) & (len(m) > 0)):
            h = h[0].replace("tiếng", "")
            m = m[0].replace("phút", "")
            if (int(m) < 0) or (int(m) > 60):
                txt = "số giờ (hoặc phút) làm việc không hợp lệ"
            elif int(m) < 60:
                txt = f"dưới {int(h) + 1} tiếng nhưng đã trên {int(h)} giờ làm việc"
            else:
                txt = f"dưới {int(h) + (m // 60) + 1} tiếng nhưng đã trên {int(h)} giờ làm việc"
        elif (len(h) == 0):
            txt = "dưới 1 tiếng làm việc"
        elif (len(m) == 0):
            txt = f"dưới {re.findall('[0-9]', h[0])[0]} giờ làm việc"
        else: 
            txt = text
        return txt    
    else: 
        return text

print(f"{100*'-'}\nWelcome to Retrieval-base Chatbot, developed by Mr. Đỗ Văn Nhân\n{100*'-'}")

while True:
    print(100*"=")
    sentence = input("You: ")
    sentence = rounding_text_hour(sentence)
    if sentence == "quit":
        break
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    ints = return_list
    print(f"{100*'-'}\nlist of probability from possible-class(es) detected: {ints}\n{100*'-'}")

    if len(return_list) > 0:
        tag = ints[0]["intent"]
        list_of_intents = intents["intents"]

        for i in list_of_intents:
            if i["tag"] == tag:
                print(f"{100*'-'}\nkeyword detected to class: {tag}\n{100*'-'}")
                result = random.choice(i["response"])
                print(f"BOT: {result}")
                if i["tag"] == "daytime_today":
                    today_date = (datetime.now() + timedelta(hours=7)).strftime('%b-%d, %Y \t %H:%M:%S GMT+07')
                    print(f"\tToday is {today_date }")
    else: 
        print("xin giải thích rõ ràng hơn vì có thể bot không hiểu hoặc chưa được học!")