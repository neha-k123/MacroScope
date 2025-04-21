import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
import datetime

st.set_page_config(page_title="US Yield Curve Dashboard", layout="wide")

st.title("MacroScope Recession Dashboard")

st.markdown("""
Welcome to **MacroScope**, your interactive recession risk tracker.

Use the sidebar to explore:
- Recession risk modeling
- Feature correlation
- Model performance

""")


# === FRED Setup ===
fred = Fred(api_key='b80e439e493aa86fbb9f080c3c2600f9')

# === Functions ===

def create_yield_curve(series_id): # Creates the 10Y - 2Y Treasury Spread
    data = fred.get_series(series_id)
    df = pd.DataFrame(data, columns=[series_id])
    df.index = pd.to_datetime(df.index)
    return df

def create_recessions(series_id): # Scrapes data for past recessions
    recessions = fred.get_series(series_id)
    recessions = recessions.to_frame(name='Recession')
    recessions.index = pd.to_datetime(recessions.index)
    return recessions

def create_unemployment(series_id): # with SAHM indicator, https://fred.stlouisfed.org/series/SAHMREALTIME#:~:text=Sahm%20Recession%20Indicator%20signals%20the,from%20the%20previous%2012%20months.
    unemployment = fred.get_series('UNRATE')
    unemployment = unemployment.to_frame(name='Unemployment')
    unemployment.index = pd.to_datetime(unemployment.index)
    # Compute 3 month moving average (the data is already monthly)
    unemployment['MA3'] = unemployment['Unemployment'].rolling(window=3).mean()
    
    unemployment['Min_MA3_Last_12'] = unemployment['MA3'].rolling(window=12).min().shift(1)
    # SAHM Signal is defined as the rate of change of the 3-month moving average of the unemployment rate, relative to the minimum value of the 3-month moving average over the last 12 months.
    unemployment['Sahm_Signal'] = unemployment['MA3'] - unemployment['Min_MA3_Last_12']
    unemployment['Sahm_Recession'] = unemployment['Sahm_Signal'] >= 0.5
    sahm = unemployment[['Sahm_Recession']].copy()
    sahm['Shift'] = sahm['Sahm_Recession'].shift(1)

    sahm_starts = sahm[(sahm['Sahm_Recession'] == True) & (sahm['Shift'] == False)].index
    sahm_ends = sahm[(sahm['Sahm_Recession'] == False) & (sahm['Shift'] == True)].index

    # In case the last period is still active, append the last date as end
    if len(sahm_starts) > len(sahm_ends):
        sahm_ends = sahm_ends.append(pd.Index([sahm.index[-1]]))

    sahm_bands = list(zip(sahm_starts, sahm_ends))
    return unemployment,sahm_bands


# Returns a list of tuples with the start and end dates of recession periods
def get_recession_periods(recessions):
    recessions['Shift'] = recessions['Recession'].shift(1)
    starts = recessions[(recessions['Recession'] == 1) & (recessions['Shift'] == 0)].index
    ends = recessions[(recessions['Recession'] == 0) & (recessions['Shift'] == 1)].index
    return list(zip(starts, ends))


# returns the plot to be drawn on the Streamlit page
def plot(df, recession_periods, unemployment_df,sahm_bands, show_bands=True ,show_unemployment_ma3=True,show_sahm_bands=True):
    fig = px.line(
        df,
        x=df.index,
        y='T10Y2Y',
        title='10Y - 2Y Treasury Spread Over Time',
        labels={'x': 'Date', 'T10Y2Y': 'Yield Spread (%)'}
    )

    fig.update_layout(
        title={'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Date',
        yaxis_title='10Y - 2Y Spread (%)',
        yaxis2=dict(title='Unemployment (%)', overlaying='y', side='right'),
        template='plotly_white',
        height=500
    )
    
    if unemployment_df is not None:

        if show_unemployment_ma3:
            fig.add_trace(go.Scatter(
                x=unemployment_df.index, y=unemployment_df['MA3'],
                name='Unemployment MA3',
                yaxis='y2',
                line=dict(color='orange', dash='dash'),
                hovertemplate='Date: %{x}<br>MA3: %{y:.2f}%<extra></extra>'
            ))
        if show_sahm_bands:    
            for start, end in sahm_bands:
                fig.add_vrect(
                    x0=start.to_pydatetime(),
                    x1=end.to_pydatetime(),
                    fillcolor="red",
                    opacity=0.15,
                    layer="below",
                    line_width=0,

                )

            
    if show_bands:
        for start, end in recession_periods:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0
            )
            
    return fig

# === Streamlit App ===


st.title("US Yield Curve & Recession Monitor")

# Data loading: yield curve, recession data, 50d MA unemployment
with st.spinner("Loading data from FRED..."):
    yield_curve = create_yield_curve('T10Y2Y')
    recession_data = create_recessions('JHDUSRGDPBR')
    recession_periods = get_recession_periods(recession_data)
    unemployment,sahm_bands = create_unemployment('UNRATE')

# Toggle for shading
st.sidebar.title("Toggles")
show_bands = st.sidebar.checkbox("Recession Bands", value=True)
show_unemployment_ma3 = st.sidebar.checkbox("3-month Unemployment Moving Average", value=True)
show_sahm_bands = st.sidebar.checkbox(" Sahm Rule Bands", value=True)



# Plot
fig = plot(
    yield_curve,
    recession_periods,
    unemployment_df=unemployment,
    sahm_bands=sahm_bands,
    show_bands=show_bands,
    show_unemployment_ma3=show_unemployment_ma3,
    show_sahm_bands=show_sahm_bands,
)
st.plotly_chart(fig, use_container_width=True)


