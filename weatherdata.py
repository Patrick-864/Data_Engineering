from meteostat import Point, Daily
from datetime import datetime
import pandas as pd

# Define the location: Chicago
chicago = Point(41.8781, -87.6298)

# Match the date range of Fitbit data
start = datetime(2016, 3, 12)
end = datetime(2016, 4, 12)

# Fetch daily weather data
data = Daily(chicago, start, end)
data = data.fetch()

# Format the dataframe
data.reset_index(inplace=True)
data = data[['time', 'tavg', 'prcp']]  # avg temp (°C), precipitation (mm)
data.columns = ['date', 'temp', 'precip']  # rename for your dashboard code

# Save to CSV
data.to_csv("weather_chicago.csv", index=False)

print("weather_chicago.csv created.")
