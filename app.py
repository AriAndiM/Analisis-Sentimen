import streamlit as st
import streamlit.components.v1 as html
import pandas as pd
import numpy as np
import pickle
import string
import string
import io
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from streamlit_option_menu import option_menu
from PIL import Image

with st.container():
    st.markdown('<h1 style = "text-align: center;"> Analisis Sentimen - Anteraja </h1>', unsafe_allow_html = True)
    st.markdown('<div style ="text-align: justify;"> Analisis sentimen adalah proses menganalisis teks digital untuk menentukan apakah nada emosional pesan tersebut positif, negatif, atau netral. Anteraja adalah jasa pengiriman online yang mengedepankan nilai SATRIA. Sigap, Aman, Terpercaya, Ramah, Integritas, dan Amanah. Aplikasi ini digunakan untuk mengetahui analisis sentimen komentar tentang anteraja. </div>', unsafe_allow_html = True)

    logo = Image.open('anteraja.png')
    st.image(logo, caption='')

    df = pd.read_csv('https://raw.githubusercontent.com/AriAndiM/dataset/main/anteraja-balanced.csv')
    # hasil_tfidf1
    
    # mengambil data
    x = df['Filter By Length']
    # x
    # mengambil label
    y = df['Label'].values
    # y
    
    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    # st.write(y_train.shape)
    # st.write(y_test.shape)

    with open('nb_pickle.pkl','rb') as r:
        nbp = pickle.load(r)

    # pred = nbp.predict(x_test)
    # st.write('Akurasi',round(accuracy_score(y_test, pred)*100,2),'%')

    teks_inputan = st.text_input('Masukkan Teks')
    cek = st.button('cek', type='primary')
    if cek:
        # preprocesing
        # remove punctuation, emot
        def remove_punctuation(text):
            # untuk mengganti username menjadi spasi
            teks_inputan=re.sub('@[^\s]+', ' ', text)
            # untuk mengganti link dengan spasi
            teks_inputan = re.sub(r'http\S*', ' ', teks_inputan)
            # menghapus tanda baca seperti #,^ dll
            teks_inputan=teks_inputan.translate(str.maketrans(' ',' ',string.punctuation))
            # mengapus semua karakter lain selain a-z atau A-Z seperti emot
            teks_inputan=re.sub('[^a-zA-Z]',' ',teks_inputan)
            # mengganti newline dengan spasi
            teks_inputan=re.sub("\n"," ",teks_inputan)
            # menghapus kata yang hanya memiliki 1 huruf saja
            teks_inputan=re.sub(r"\b[a-zA-z]\b"," ",teks_inputan)
            # menghilangkan spasi berlebihan pada kalimat
            teks_inputan=' '.join(teks_inputan.split())
            return teks_inputan
        
        # lower case
        lower = remove_punctuation(teks_inputan).lower()
        st.write(lower)

        # proses tokenize berdasarkan spasi menggunakan WhitespaceTokenizer() dari hasil lower tweet
        tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize(lower)
        st.write(tokenize)

        # stemming
        # create stemmer
        Fact = StemmerFactory()
        Stemmer = Fact.create_stemmer()

        def steming(x):
            # proses stemming
            b = []
            for kata in x:
                # proses stemming setiap kata
                a = Stemmer.stem(kata)
                # memasukkan kata yang sudah di stemming ke array
                b.append(a)
            # menggabungkan array dengan pemisal ','
            return ','.join(b)
        
        data_steming = steming(tokenize)
        st.write(data_steming)
        # Stopword Removal            

        # Initialize the stopwords
        nltk.download('stopwords')
        stoplist = stopwords.words('indonesian')
        def remove_stopwords(x):
            a = []
            # memisahkan antar kata berdasarkan spasi
            for i in x.split(','):
                # dilakukan pengecekan pada stoplist
                if i not in stoplist:
                    # memasukkan kata yang tidak ada di stoplist ke array
                    a.append(i)
            return ','.join(a)
            
        stopword = remove_stopwords(data_steming)
        st.write(stopword)

        # Filter Token By Length
        def filter_by_length(kata):
            a = []
            # memisahkan antar kata berdasarkan koma
            for i in kata.split(','):
                # dilakukan pengecekan jumlah huruf dalam kata
                if len(i) >= 4 and len(i) <= 25:
                    # memasukkan kata yang memiliki jumlah huruf >=4 dan <= 25 ke list
                    a.append(i)
            return ','.join(a)
        
        
        #pickle
        import pickle
        # wb - write binary
        with open('tfidf_pickle.pkl','rb') as f:
            vectorizer = pickle.load(f)

        # x_train, x_test, y_train, y_test = train_test_split(dt, label, test_size=0.4, shuffle=False)
        with open('nb_pickle.pkl','rb') as r:
            nbp = pickle.load(r)

        filter = filter_by_length(stopword)
        st.write(filter)
        # transform = vectorizer.transform([" ".join(a)])
        transform = vectorizer.transform([filter])
        st.write(transform)
        
        # pred = nbp.predict(np.asarray(transform.todense()))
        pred = nbp.predict(transform)
        st.write(pred)
        
            
