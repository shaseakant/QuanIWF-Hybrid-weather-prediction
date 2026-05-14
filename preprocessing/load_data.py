import pandas as pd

def load_weather_data(path="data/weather.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    return df
