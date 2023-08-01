import streamlit as st
import re
from pymorphy2 import MorphAnalyzer
from functools import lru_cache
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import numpy as np
from catboost import Pool
import nltk
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
model = CatBoostClassifier()
model.load_model('classifire_model_MVP.cbm')
model_reg = CatBoostRegressor()
model_reg.load_model('regressor_model_MVP.cbm')
nltk.download('stopwords')
st.markdown(
    """
    <style>
    [data-testid = "stAppViewContainer"]{
        background-color: #e5e5f7;
        opacity: 1;
        background-image: radial-gradient(circle at center center, #5f8c4a, #e5e5f7), repeating-radial-gradient(circle at center center, #5f8c4a, #5f8c4a, 14px, transparent 28px, transparent 14px);
        background-blend-mode: multiply;
    }
   </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: white;'>Green Atom</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Check the status of your comment </h2>", unsafe_allow_html=True)

start_text = "The Boys is one of the best Superhero shows I've ever seen. While Season 1 was the best season of the series, Season 2 and 3 were also both very good and absolutely worth watching. Season 3 was fantastic, Jensen Ackles was the perfect actor to add to this already incredible show! This show continues to amaze as it's not afraid to try new things and is a show that is definitely for adults. It has no problems being offensive and making you feel squeamish. You don't even have to be a fan of superhero shows to enjoy this. It's violent, funny, thrilling, etc.. Everything you want in a good superhero show. Season 4 just added Jeffrey Dean Morgan to the cast. Another great addition to an already incredible cast. I don't know what else I can say except I absolutely love this series and can't wait for more to come!"
text = st.text_area('Enter your review text and click Ctrl+Enter',start_text)
print(text)

data = pd.DataFrame({'text': [text]})
m = MorphAnalyzer()
regex = re.compile("[А-Яа-яA-z]+")
mystopwords = stopwords.words('english')


def words_only(text, regex=regex):
    try:
        return regex.findall(text.lower())
    except:
        return []


@lru_cache(maxsize=128)
def lemmatize_word(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form


def lemmatize_text(text):
    return [lemmatize_word(w) for w in text]


def remove_stopwords(lemmas, stopwords=mystopwords):
    return [w for w in lemmas if not w in stopwords and len(w) > 3]


def clean_text(text):
    tokens = words_only(text)
    lemmas = lemmatize_text(tokens)

    return ' '.join(remove_stopwords(lemmas))


from multiprocessing import Pool as PoolSklearn

with PoolSklearn(4) as p:
    lemmas = list(tqdm(p.imap(clean_text, data['text']), total=len(data)))

data['text_lemmas'] = lemmas

data['sym_len'] = data.text_lemmas.apply(len)
data['word_len'] = data.text_lemmas.apply(lambda x: len(x.split()))
data['sym_len'] = np.log(data['sym_len'])
data['word_len'] = np.log(data['word_len'])

test_pool = Pool(
    data,
    text_features=['text', 'text_lemmas'],
)

y_pred = model.predict(test_pool)
y_pred_reg = model_reg.predict(test_pool)
arr = np.round(y_pred_reg, 0).astype(int)
arr_1 = []
for i in range(len(arr)):
  if arr[i]<=0:
    arr_1.append(1)
  elif arr[i]>10:
    arr_1.append(10)
  else:
    arr_1.append(arr[i])

ans = ''
if y_pred[0] == 1:
    ans = 'positive'
else:
    ans = 'negative'
st.write('Estimated revocation status :', ans)
st.write('Estimated comment rating :', arr_1[0])
