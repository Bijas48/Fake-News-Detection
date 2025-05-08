import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("Detektor Berita Palsu")
st.write("Aplikasi ini menggunakan model Logistic Regression untuk mendeteksi berita palsu.")
st.write("Masukkan judul berita yang ingin Anda periksa di bawah ini :")

news_input = st.text_area("Judul Berita:", "")

if st.button("Check Berita"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("Berita ini Asli! ")
        else:
            st.error("Berita ini Palsu! ")
    else:
        st.warning("Masukan beberapa kalimat tambahan...")