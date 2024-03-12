import random
import tensorflow as tf
from tensorflow.keras.optimizers import SGD # from tensorflow.keras.optimizers.legacy import SGD # 
from pyvi import ViTokenizer, ViPosTagger, ViUtils
from tensorflow.python.keras.layers import Dense, Dropout # from keras.layers import Dense, Dropout
from tensorflow.python.keras.models import load_model # from keras.models import load_model
from tensorflow.python.keras.models import Sequential # from keras.models import Sequential
import numpy as np
import pickle, os
import json, warnings
import nltk, re
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

print("Firstly, download some nltk packages")
lemmatizer = WordNetLemmatizer()
try:
    nltk.download('omw-1.4')
    nltk.download("punkt")
    nltk.download("wordnet")
except :
    print("You already download these packages")

words = []
classes = []
documents = []
ignore_words = ["?", "!"]
current_fpath = os.getcwd()
print("current file-path", current_fpath)
data_file = open(f"{current_fpath}/intents.json", encoding='utf-8').read()
intents = json.loads(data_file)
print("Load intents.json done")

# words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # take each word and tokenize it
        patt = ViTokenizer.tokenize(pattern)
        w = nltk.word_tokenize(patt)
        words.extend(w)
        # adding documents
        documents.append((w, intent["tag"]))

        # adding classes to our class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
print("Split class and intents with respect to each topic", "100*'=")
pickle.dump(words, open(f"words.pkl", "wb"))
pickle.dump(classes, open(f"classes.pkl", "wb"))

# training initializer
# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype='object')
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# actual training
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(512, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation="softmax"))
print(model.summary())

from keras import callbacks 

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer='SGD', metrics=["accuracy"])

earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
callbacks =[earlystopping]
print("Congratulation, your model successfully trained!")

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1, callbacks = callbacks)
model.save(f"chatbot_model.h5", hist)
print("model created")
print("Finish")