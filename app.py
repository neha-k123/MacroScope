import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
import datetime

st.set_page_config(page_title="Dashboard Landing Page", layout="wide")

# Custom CSS for centering content
st.markdown("""
    <style>
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;
        text-align: center;
    }
    .content-box {
        max-width: 700px;
    }
    </style>
""", unsafe_allow_html=True)

# Centered Content
st.markdown("""
    <div class="centered-container">
        <div class="content-box">
            <h1 style="font-size: 40px;">MacroScope Quant Dashboard</h1>
            <p style="font-size: 24px;">Welcome, I'm Neha!</p>
            <p style="font-size: 20px;">
                This is a quant dashboard showcasing 2025 macro-economic indicators. <br>
                Use the sidebar to explore various pages. <br>
                <br>
                Some of the pages take a bit to load as it's loading in models, please be patient.
                Enjoy!
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

