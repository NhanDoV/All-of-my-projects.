import re
import pickle
from joblib import load
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from spellchecker import SpellChecker  

data_dir_submit = {}
data_test = pd.read_csv(data_dir_submit)
X_test = data_test['text']
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
model = load("svm.joblib")
xvect = pickle.load("tfidf.pickle")
X = xvect.transform(X_test) 
preds = model.predict(X)
data_test['pred'] = preds

data_test.to_csv("submit.csv", index=False)