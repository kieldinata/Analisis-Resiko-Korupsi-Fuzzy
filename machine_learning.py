import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import joblib

def train_model(kamus_path='kamus_urgensi.csv'):
    kamus = pd.read_csv(kamus_path)
    vectorizer = TfidfVectorizer(vocabulary=kamus['keyword'].tolist())
    X = vectorizer.fit_transform(kamus['keyword'])
    y = kamus['skor']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'model_urgensi.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    return True

def predict_urgensi(nama_paket_list):
    model = joblib.load('model_urgensi.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    X_input = vectorizer.transform(nama_paket_list)
    prediksi = model.predict(X_input)
    return [max(0, min(1, val)) for val in prediksi]