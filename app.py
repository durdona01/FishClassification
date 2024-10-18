import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Baliqning to'rt turga ajratib aniqlovchi model. \n 4 turdagi(Fish(Goldfish, Shark, Rays and skates, Seahorse)) baliqlarni klassifikatsiya qila oladi")

# rasmni joylash 
file = st.file_uploader('Rasm yuklash', type=['png', 'svg', 'jpeg', 'gif', 'jfif' ])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner('fishes_model.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:1f}%")

    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
