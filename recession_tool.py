from fredapi import Fred
import pandas as pd
import plotly.express as px

fred = Fred(api_key='b80e439e493aa86fbb9f080c3c2600f9')


def create_yield_curve(series_id): #given a series id, scrapes FRED and returns a dataframe with requested data
    # https://fred.stlouisfed.org/series/T10Y2Y
    data = fred.get_series(series_id)

    df = pd.DataFrame(data, columns=[series_id])
    df.index = pd.to_datetime(df.index)
    #print(df)
    return(df)

def create_recessions(series_id):
    recessions = fred.get_series(series_id)
    recessions = recessions.to_frame(name='Recession')
    recessions.index = pd.to_datetime(recessions.index)
    return(recessions)

 #given a dataframe, uses plotly package to generate a figure on the HTML site. 
def plot(df, recession_periods): #df = yield curve df to plot, recession_periods = list of Timestamps for scraped recession periods
    fig = px.line(
        d