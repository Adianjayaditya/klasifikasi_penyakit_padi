import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Aplikasi Klasifikasi Penyakit Padi",
    layout="wide"
)

model = load_model('model_cnn.h5')


st.title("Selamat datang di :red[Aplikasi Klasifikasi Penyakit Padi]")
st.write("---")
st.write("Aplikasi ini digunakan untuk mengklasifikasikan penyakit padi menggunakan metode Convolutional Neural Network (CNN). Penyakit yang dapat diklasifikasikan ialah bacterial leaf blight', 'brown spot', 'healthy', 'leaf blast', 'leaf scald', 'narrow brown spot', 'neck blast', 'rice hispa', 'sheath blight', 'tungro")
st.write("Untuk dapat melakukan prediksi, silahkan upload gambar tanaman padi.")
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

class_names = ['bacterial leaf blight', 'brown spot', 'healthy', 'leaf blast', 'leaf scald', 'narrow brown spot', 'neck blast', 'rice hispa', 'sheath blight', 'tungro']

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image)
    img = load_img(file, target_size=(256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    st.write(f'Penyakit padi: {class_names[predicted_class[0]]}')
