# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px

#Makes the code adaptable 
FILENAME = 'daily_acivity.csv'

df = pd.read_csv(FILENAME)
df['Id'] = df['Id'].astype(str)
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

def printUniqueUsers():
  unique_users = df['Id'].nunique()
  print(f"Number of unique users: {unique_users}")


def totalDistance():
  total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()
  total_distance_per_user.plot(kind='bar', figsize=(10,5), title='Total Distance per User')
  plt.xlabel('User Id')
  plt.ylabel('Total Distance')
  plt.show()

# Function to display calories burnt over a certain date range for a specific user
def plot_calories_burnt(user_id, start_date=None, end_date=None):
    # Ensure the user_id is a string to match the DataFrame's 'Id' type
    user_id = str(user_id)
    user_data = df[df['Id'] == user_id].copy()

    # Make sure ActivityDate is datetime (if not already converted)
    user_data['ActivityDate'] = pd.to_datetime(user_data['ActivityDate'])
    
    # If start_date and end_date are provided, convert them to datetime and filter
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (user_data['ActivityDate'] >= start_date) & (user_data['ActivityDate'] <= end_date)
        user_data = user_data[mask]
    
    if user_data.empty:
        print(f"No data available for user {user_id} with the specified date range.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(user_data['ActivityDate'], user_data['Calories'], marker='o', linestyle='-')
    plt.title(f'Calories Burnt for User {user_id}')
    plt.xlabel('Date')
    plt.ylabel('Calories Burnt')
    plt.xticks(rotation=45)
    plt.tight_layout()
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
  

#Calories burnt vs. Total Steps
  """
  Fit a linear regression model: Calories = Beta_0 + Beta_1 * TotalSteps
  for the given user, and print the model summary plus an interpretation of Beta_1.
  """
def linear_regression_for_user(user_id):

    user_id = str(user_id)
    user_data = df[df['Id'] == user_id]
    
    if user_data.empty:
        print(f"No data found for user {user_id}")
        return
    
    # Fit the OLS regression using statsmodels
    model = smf.ols('Calories ~ TotalSteps', data=user_data).fit()
    print(model.summary())
    
    # Interpret Beta_1
    beta_1 = model.params['TotalSteps']
    print(f"\nInterpretation of Beta_1: For user {user_id}, "
          f"each additional step is associated with an increase of {beta_1:.2f} calories burned.")
    
"""
  Display a scatterplot + regression line of Calories vs. TotalSteps for a given user.
  Also prints the interpretation of Beta_1 from the fitted model.
"""
def plot_regression(user_id):

    user_id = str(user_id)
    user_data = df[df['Id'] == user_id]
    
    if user_data.empty:
        print(f"No data found for user {user_id}")
        return
    
    # Fit OLS regression
    model = smf.ols('Calories ~ TotalSteps', data=user_data).fit()
    # Predict Calories based on the model
    user_data = user_data.assign(Predicted=model.predict(user_data['TotalSteps']))
    
    # Scatterplot
    plt.figure(figsize=(10, 6))
    plt.scatter(user_data['TotalSteps'], user_data['Calories'], label='Data points')
    # Regression line
    plt.plot(user_data['TotalSteps'], user_data['Predicted'], color='red', label='Regression line')
    
    plt.title(f'Calories vs. Total Steps for User {user_id}')
    plt.xlabel('Total Steps')
    plt.ylabel('Calories')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print interpretation
    beta_1 = model.params['TotalSteps']
    print(f"\nFor user {user_id}, Beta_1 = {beta_1:.2f}.\n"
          f"Interpretation: Each additional step increases calories burned by ~{beta_1:.2f}.")

def plot_sunburst():
    """
    An additional optional plot (using plotly) that shows a sunburst chart 
    of total steps by weekday and user.
    """
    df_filtered = df[df['TotalSteps'] > 0].copy()
    df_filtered['Weekday'] = df_filtered['ActivityDate'].dt.day_name()
    
    if df_filtered.empty:
        print("No valid data available for the Sunburst chart.")
        return

    fig = px.sunburst(df_filtered, path=['Weekday', 'Id'], values='TotalSteps', 
                      title='Daily Activity Breakdown by User and Weekday', 
                      color='TotalSteps', color_continuous_scale='Blues')
    fig.show()

def runAll(id):
  printUniqueUsers()
  totalDistance()
  plot_calories_burnt(user_id=id, start_date='4/3/2016', end_date='4/4/2016')

  workoutPerDay()
  linear_regression_for_user(id)
  plot_regression(id)
  plot_sunburst()

if __name__ == "__main__":
  runAll(1624580081)
