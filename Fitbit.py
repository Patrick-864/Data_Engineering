# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
#TESTTTTTT
#Makes the code more reusable 
FILENAME = "daily_acivity.csv"

df = pd.read_csv(FILENAME)

def printUniqueUsers():
  unique_users = df['Id'].nunique()
  print(f"Number of unique users: {unique_users}")


def totalDistance():
  total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()
  total_distance_per_user.plot(kind='bar', figsize=(10,5), title='Total Distance per User')
  plt.xlabel('User Id')
  plt.ylabel('Total Distance')
  plt.show()

# Function to display calories burnt over a date range for a specific user
def plot_calories_burnt(user_id, start_date=None, end_date=None):
    user_data = df[df['Id'] == user_id]
    user_data['ActivityDate'] = pd.to_datetime(user_data['ActivityDate'])
    
    if start_date and end_date:
        mask = (user_data['ActivityDate'] >= start_date) & (user_data['ActivityDate'] <= end_date)
        user_data = user_data[mask]
    
    plt.figure(figsize=(10, 5))
    plt.plot(user_data['ActivityDate'], user_data['Calories'], marker='o', linestyle='-')
    plt.title(f'Calories Burnt for User {user_id}')
    plt.xlabel('Date')
    plt.ylabel('Calories Burnt')
    plt.xticks(rotation=45)
    plt.show()

def workoutPerDay(): 
  # Convert date and plot frequency of workouts per weekday
  df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
  df['Weekday'] = df['ActivityDate'].dt.day_name()
  sns.countplot(x='Weekday', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
  plt.xticks(rotation=45)
  plt.title('Workout Frequency by Weekday')
  plt.xlabel('Day of Week')
  plt.ylabel('Count')
  plt.show()
  

# Linear regression model for Calories burnt vs. Total Steps
df['Id'] = df['Id'].astype(str)  # Ensure Id is a categorical variable
model = smf.ols('Calories ~ TotalSteps + Id', data=df).fit()
print(model.summary())

# Scatterplot with regression line
def plot_regression(user_id):
    user_data = df[df['Id'] == user_id]
    sns.lmplot(x='TotalSteps', y='Calories', data=user_data)
    plt.title(f'Calories vs. Steps for User {user_id}')
    plt.xlabel('Total Steps')
    plt.ylabel('Calories Burnt')
    plt.show()

#Addtional Part, made for the creative Part does not have to be definitive
def plot_sunburst():
    df_filtered = df[df['TotalSteps'] > 0]  # Remove rows where TotalSteps is zero
    df_filtered['Weekday'] = df_filtered['ActivityDate'].dt.day_name()
    
    if df_filtered.empty:
        print("No valid data available for the Sunburst chart.")
        return

    fig = px.sunburst(df_filtered, path=['Weekday', 'Id'], values='TotalSteps', 
                      title='Daily Activity Breakdown by User and Weekday', 
                      color='TotalSteps', color_continuous_scale='Blues')
    fig.show()


if __name__ == "__main__":
  plot_regression(1624580081)
  printUniqueUsers()
  totalDistance()
  plot_sunburst()

