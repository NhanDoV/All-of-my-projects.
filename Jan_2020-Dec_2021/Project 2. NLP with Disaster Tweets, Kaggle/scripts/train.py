import re
import pickle
import string
import numpy as np
import pandas as pd
from joblib import dump
from wordcloud import STOPWORDS
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from spellchecker import SpellChecker    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
#====================================================
data_dir = {}
df = pd.read_csv(data_dir)
#====================================================
def enhanced_data(df_train, df_test):
    """ Returns
        -----------------
            This function is used to check the following table:
                ===================================================================================================
                *------------------------------------------------------------------------------------------*
                | Cases                                                | Examples (discriptions)           | 
                *------------------------------------------------------|-----------------------------------* 
                | number of hastags                                    | #memories                         |
                *------------------------------------------------------------------------------------------*
                | number of url_link and unique url in email, messages | http//:google.com                 |
                | or tweets / etc                                      | www.google.com                    |
                *------------------------------------------------------------------------------------------*
                | mention someone else`                                | @David                            |
                *------------------------------------------------------------------------------------------*
                | hour of day / day of week / or any mentioned-        | 2020-12-12                        |
                | timestamp when the email` or messages / tweets       | 21 Jun 2020                       |
                | was send / posted`                                   | etc                               |
                *------------------------------------------------------------------------------------------*
                | number of emojicon                                   | ":)", ":v", "=))", etc            |
                |                                                      | "\U000024C2-\U0001F251", etc                      
                *------------------------------------------------------------------------------------------*
                | number of capitalized words`                         | AbBa MoHameD                      |
                *------------------------------------------------------------------------------------------*
                | sum of all the character-lengths of word`            | len(word_splited)                 |
                *------------------------------------------------------------------------------------------*
                | number of words containing letters and numbers       | "128abc9*", "29Jun1998", etc.     |
                *------------------------------------------------------------------------------------------*
                | number of words containing only numbers or letters   | "12300 people...", etc.           |             
                *------------------------------------------------------------------------------------------*
                | max ratio of digit characters to all characters of   | max([len`(digit(word))            |
                | each word                                            |  / len(word) for word in words])  |
                *------------------------------------------------------------------------------------------*
                | max the charecter-lengths of all words.              | max([len(word) for word in words])|
                *------------------------------------------------------------------------------------------*
                | number of words in email, messages or tweets / etc.  | len(word.split())                 |
                *------------------------------------------------------------------------------------------*
                | max length of word                                   | max([len(w) for w in words])      |
                *------------------------------------------------------------------------------------------*
                | average length of word                               | mean([len(w) for w in words])     |
                *------------------------------------------------------------------------------------------*
                | number of punctuation                                |                                   | 
                *------------------------------------------------------------------------------------------*
    """      
    # word_count
    df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
    df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))

    # unique_word_count
    df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
    df_test['unique_word_count'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

    # stop_word_count
    df_train['stop_word_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df_test['stop_word_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    # url_count
    df_train['url_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    df_test['url_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

    # average_word_length
    df_train['avg_word_length'] = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df_test['avg_word_length'] = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # char_count
    df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
    df_test['char_count'] = df_test['text'].apply(lambda x: len(str(x)))

    # punctuation_count
    df_train['punctuation_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df_test['punctuation_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # hashtag_count
    df_train['hashtag_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
    df_test['hashtag_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

    # mention_count
    df_train['mention_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
    df_test['mention_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

    return df_train, df_test
#====================================================
def process_text(str_input):
    """
        Text pre-processing
    """
    ## 1. Remove url_link
    remove_url = re.compile(r'https?://\S+|www\.\S+').sub(r'', str_input)
    
    ## 2. Remove html_link
    remove_html = re.compile(r'<.*?>').sub(r'', remove_url)
    
    ## 3. Remove Emojis
    remove_emo = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE).sub(r'', remove_html)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", remove_emo).lower().split()    
        
    ## 4. spell_correction
    spell = SpellChecker()
    words = [spell.correction(word) for word in words[:50]]

    return words
#====================================================
df.loc[:, 'correct_text'] = df.loc[:, 'text'].apply(lambda x: process_text(x))
df['Text_length'] = df['text'].str.len()
df['Numb_words'] = df['text'].str.split().map(lambda x: len(x))
df = df.set_index('id')
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english') 
X = tfidf_vectorizer.fit_transform(df['text']) 
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    stratify = y, 
                                                    test_size=0.3, 
                                                    random_state=42)
#==============================
svm = svm.SVC()
grid_params = [{
                'kernel':['linear', 'rbf', 'poly'],
                'C': [0.1, 1, 5]
                 }]
clf_svm = GridSearchCV(estimator=svm, 
                       param_grid = grid_params, 
                       cv = 8, 
                       verbose = 0)

clf_svm.fit(X_train, y_train)
pred_train = clf_svm.predict(X_train)
pred_test = clf_svm.predict(X_test)

print(accuracy_score(y_train, pred_train))
print(accuracy_score(y_test, pred_test))

print(confusion_matrix(y_train, pred_train))
print(confusion_matrix(y_test, pred_test))
#==============================
estimator = clf_svm.best_estimator_
dump(estimator, "svm.joblib")
pickle.dump(tfidf_vectorizer, open("tfidf.pickle", "wb"))