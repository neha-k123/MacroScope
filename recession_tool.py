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
        df,
        x=df.index,
        y='T10Y2Y',
        title='10Y - 2Y Treasury Spread Over Time',
        labels={'x': 'Date', 'T10Y2Y': 'Yield Spread (%)'}
    )

    fig.update_layout(
        title={
            'text': "10Y - 2Y Treasury Spread Over Time",
            'x': 0.5,  # Centers the title
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='10Y - 2Y Spread (%)',
        template='plotly_white',
        height=500
    )
    
    for start, end in recession_periods:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="gray", opacity=0.2,
            layer="below", line_width=0
        )

    #fig.show()
    fig.write_html("yield_curve.html", auto_open=True)
    
    
    
    
def main():  
    series_id = 'T10Y2Y' # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (T10Y2Y)       
                                                                                                                                                                                                                  
    df = create_yield_curve(series_id)    
    #plot(df)
    
    # US Recessions as defined by FRED
    series_id = 'JHDUSRGDPBR'
    recessions = create_recessions(series_id) # This returns a dataframe where 1 = recession, 0 = no recession
    #print(recessions.tail())
    
    # We create a list of start/end dates for each recession period
    recessions['Shift'] = recessions['Recession'].shift(1)
    starts = recessions[(recessions['Recession'] == 1) & (recessions['Shift'] == 0)].index
    ends = recessions[(recessions['Recession'] == 0) & (recessions['Shift'] == 1)].index

    recession_periods = list(zip(starts, ends))
    print(recession_periods)
    
    plot(df,recession_periods)
    
main()
