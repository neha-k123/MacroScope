import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
import datetime

st.set_page_config(page_title="Sources and Feedback", layout="wide")


import streamlit as st

st.markdown("<h2 style='text-align: center;'>Works Cited</h2>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 16px'>
    <p><a href="https://fred.stlouisfed.org/" target="_blank">FRED Economic Data (Federal Reserve)</a></p>
    <p><a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance (yfinance API)</a></p>
    <p><a href="https://www.reddit.com/r/trumptweets2/" target="_blank">r/TrumpTweets2 Subreddit</a></p>
    <p><a href="https://github.com/cjhutto/vaderSentiment" target="_blank">VADER Sentiment Analysis</a></p>
    <p><a href="https://github.com/tesseract-ocr/tesseract" target="_blank">Tesseract OCR Engine</a></p>
    <p><a href="https://www.investopedia.com/terms/y/yieldcurve.asp" target="_blank">Yield Curve</a></p>
    <p><a href="https://matplotlib.org/" target="_blank">Matplotlib – Python Plotting Library</a></p>
    <p><a href="https://plotly.com/python/" target="_blank">Plotly – Interactive Data Visualization for Python</a></p>
    <p><a href="https://streamlit.io/" target="_blank">Streamlit – Frontend Framework for Data Apps</a></p>
    <p><a href="https://scikit-learn.org/stable/" target="_blank">scikit-learn – Machine Learning in Python</a></p>
    <p><a href="https://pypi.org/project/yfinance/" target="_blank">yfinance – Yahoo Finance Python API</a></p>
    <p><a href="https://praw.readthedocs.io/en/latest/" target="_blank">PRAW – Python Reddit API Wrapper</a></p>
    <p><a href="https://pandas.pydata.org/" target="_blank">pandas – Data Analysis Library</a></p>
    <p><a href="https://numpy.org/" target="_blank">NumPy – Numerical Python Library</a></p>
    <p><a href="https://wordcloud.readthedocs.io/en/latest/" target="_blank">WordCloud – Python Word Cloud Generator</a></p>

</div>
""", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center;'>Provide Feedback!</h2>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
  <p> Provide app feedback! </p>
  <iframe src="https://docs.google.com/forms/d/e/1FAIpQLScX0nxNgFFLpPQ1Rf38rPiwiLGGy4bODVSieLDgZOgq0SuKjg/viewform?embedded=true" width="640" height="419" frameborder="0" marginheight="0" marginwidth="0">Loading…
  </iframe>
</div>
""", unsafe_allow_html=True)




