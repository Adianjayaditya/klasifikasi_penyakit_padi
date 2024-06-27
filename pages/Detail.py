import streamlit as st
import pandas as pd

st.title("Model CNN")
st.write("---")

st.subheader("Akurasi dan loss model")
col1, col2 = st.columns(2)
with col1:
   st.image('https://i.ibb.co.com/K2P0fsq/accu.png', caption='Akurasi model saat proses training')
with col2:
   st.image('https://i.ibb.co.com/6b9dc28/loss.png', caption='Loss model saat proses training')

st.subheader("Arsitektur model")
col1, col2 = st.columns(2)
with col1:
   st.image('https://i.ibb.co.com/NLbLwTH/summary.png', caption='Arsitektur model CNN')
