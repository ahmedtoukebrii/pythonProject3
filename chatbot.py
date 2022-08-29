import random
import json
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
#words = pickle.loads(open('words.pkl', 'rb'))
#classes = pickle.loads(open('classes.pkl', 'rb'))
#words = joblib.load(open('words.pkl', 'rb'))
#classes = joblib.load(open('classes.pkl', 'rb'))
words=[]
classes=[]
documents=[]
ignore_letters=['?','0','.',',']
for intent in intents['intents']:
    for pattern in intent['patterns']:
              word_list= nltk.word_tokenize(pattern)
              words.extend(word_list)
              documents.append((word_list,intent['tag']))
              if intent['tag'] not in classes:
                  classes.append(intent['tag'])
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters ]
words = sorted(set(words))
classes=sorted(set(classes))
pickle.dump(words,open('words.pk1', 'wb'))
pickle.dump(classes,open('classes.pk1', 'wb'))

model = load_model('chatbotmodel.h5')
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r]for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
         return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! BOT is running!" )

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)