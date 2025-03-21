# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sqlite3
import logging

#Makes the code adaptable 
FILENAME = 'daily_acivity.csv'


#Initializes the CSV file
def load_csv(filename='daily_acivity.csv'):
    try:
        df = pd.read_csv(filename)
        df['Id'] = df['Id'].astype(str)
        df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
        logging.info("CSV file loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        return None


#Initiliazes the connection to the data base
def db_init(dataBase_path="fitbit_database.db"):
    try:
        conn = sqlite3.connect(dataBase_path)
        return conn
    except Exception as e:
        print("Error connecting to database:", e)
        raise



def printUniqueUsers(df):
  unique_users = df['Id'].nunique()
  print(f"Number of unique users: {unique_users}")


def totalDistance(df):
  total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()
  total_distance_per_user.plot(kind='bar', figsize=(10,5), title='Total Distance per User')
  plt.xlabel('User Id')
  plt.ylabel('Total Distance')
  plt.show()

# Function to display calories burnt over a certain date range for a specific user
def plot_calories_burnt(df, user_id, start_date=None, end_date=None):
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


def workoutPerDay(df): 
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
def linear_regression_for_user(df,user_id):

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
def plot_regression(df, user_id):

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



def run_first_part(df,id):
  printUniqueUsers(df)
  totalDistance(df)
  plot_calories_burnt(df,user_id=id, start_date='4/3/2016', end_date='4/4/2016')
  workoutPerDay(df)
  linear_regression_for_user(df,id)
  plot_regression(df,id)


if __name__ == "__main__":
  df = load_csv()
  run_first_part(df,1624580081)


db_path =  "fitbit_database.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()



def sleep_duration():
    """Computes total sleep duration per user and visualizes it."""
    query = "SELECT Id, SUM(value) as total_sleep FROM minute_sleep GROUP BY Id"

    with db_init() as conn:
        df = pd.read_sql(query, conn)

    df["Id"] = df["Id"].astype(str)

    # Plot sleep duration per user
    plt.figure(figsize=(10, 5))
    sns.histplot(df["total_sleep"], bins=20, kde=True)
    plt.xlabel("Total Sleep Duration (minutes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Sleep Duration")
    plt.show()

    return df


#this function classifies the users based on the frquency of their activity"
# the function returns a dataframe where the users are either classified ass heavy, moderate or light user.
#The dataframe has 2 cols, 1 with the id the other one with the class of activity
def classify_users():
    query = """ SELECT Id, COUNT(*) as activity_count FROM daily_activity GROUP BY Id"""
    cursor.execute(query)
    results = cursor.fetchall()
    
    classified_users = []
    for row in results:
        user_id, activity_count = row
        if activity_count <= 10:
            user_class = "Light user"
        elif 11 <= activity_count <= 15:
            user_class = "Moderate user"
        else:
            user_class = "Heavy user"
        classified_users.append((user_id, user_class))
    
    return pd.DataFrame(classified_users, columns=["Id", "Class"])

#This function analyses the correleation between the sleep duration and the amount of activity
#The function returns nothing, and displays a plot after it is executed
def sleep_vs_activity():
    query = """
        SELECT m.Id, SUM(m.value) as total_sleep, 
               SUM(d.VeryActiveMinutes + d.FairlyActiveMinutes + d.LightlyActiveMinutes) as total_active 
        FROM minute_sleep m 
        JOIN daily_activity d ON m.Id = d.Id
        GROUP BY m.Id
    """
    cursor.execute(query)
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=["Id", "total_sleep", "total_active"]).dropna()
    
    if df.empty:
        print("No data available for sleep vs. activity analysis.")
        return

    model = smf.ols('total_sleep ~ total_active', data=df).fit()
    print(model.summary())
    
    sns.regplot(x=df["total_active"], y=df["total_sleep"])
    plt.xlabel("Total Active Minutes")
    plt.ylabel("Total Sleep (minutes)")
    plt.title("Sleep Duration vs. Active Minutes")
    plt.show()


classified_users_df = classify_users()

# Display classified users
print("Classified Users:")
print(classified_users_df.head())


def heart_rate_vs_intensity(user_id):
    """Compares heart rate with exercise intensity for a user."""
    query = f"""
        SELECT h.Id, h.Time, h.Value as HeartRate, i.TotalIntensity
        FROM heart_rate h
        JOIN hourly_intensity i ON h.Id = i.Id AND h.Time = i.ActivityHour
        WHERE h.Id = '{user_id}'
    """

    with db_init() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("No data found for this user.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["HeartRate"], label="Heart Rate", color="red")
    plt.plot(df["Time"], df["TotalIntensity"], label="Exercise Intensity", color="blue")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Heart Rate / Intensity")
    plt.title(f"Heart Rate vs. Exercise Intensity for User {user_id}")
    plt.xticks(rotation=45)
    plt.show()


def sedentary_vs_sleep():
    query = """
        SELECT d.Id, d.SedentaryMinutes, SUM(m.value) as total_sleep
        FROM daily_activity d
        JOIN minute_sleep m ON d.Id = m.Id
        GROUP BY d.Id
    """
    cursor.execute(query)
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=["Id", "SedentaryMinutes", "total_sleep"]).dropna()
    
    if df.empty:
        print("No data available for sedentary vs. sleep analysis.")
        return
    
    model = smf.ols("total_sleep ~ SedentaryMinutes", data=df).fit()
    print(model.summary())
    
    sns.regplot(x=df["SedentaryMinutes"], y=df["total_sleep"])
    plt.xlabel("Sedentary Minutes")
    plt.ylabel("Total Sleep (minutes)")
    plt.title("Sedentary Activity vs. Sleep Duration")
    plt.show()

# 4. Breaking down the day into 4-hour blocks
def activity_by_time_blocks():
    query = """
        SELECT Id, ActivityHour, Calories 
        FROM hourly_calories
    """
    cursor.execute(query)
    results = cursor.fetchall()
    df_block = pd.DataFrame(results, columns=["Id", "ActivityHour", "Calories"]).dropna()
    
    df_block["ActivityHour"] = pd.to_datetime(df_block["ActivityHour"], errors="coerce")
    
    if df_block["ActivityHour"].isna().all():
        print("Failed to parse ActivityHour. Check data format.")
        return
    
    df_block["hour"] = df_block["ActivityHour"].dt.hour
    bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']
    df_block["time_block"] = pd.cut(df_block["hour"], bins=bins, labels=labels, include_lowest=True)
    df_grouped = df_block.groupby("time_block")["Calories"].sum()
    
    df_grouped.plot(kind="bar", figsize=(10, 5))
    plt.title("Calories Burned Breakdown by Time Block")
    plt.xlabel("Time Block")
    plt.ylabel("Total Calories")
    plt.show()


def steps_by_time_blocks():
    query = "SELECT Id, ActivityHour, StepTotal FROM hourly_steps"
    cursor.execute(query)
    results = cursor.fetchall()
    if not results:
        print("No data available from hourly_steps.")
        return
    df_steps = pd.DataFrame(results, columns=["Id", "ActivityHour", "Steps"])
    df_steps['ActivityHour'] = pd.to_datetime(df_steps['ActivityHour'], errors='coerce')
    if df_steps['ActivityHour'].isna().all():
        print("Failed to parse Time in hourly_steps.")
        return
    df_steps['Hour'] = df_steps['ActivityHour'].dt.hour
    bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']
    df_steps['time_block'] = pd.cut(df_steps['Hour'], bins=bins, labels=labels, include_lowest=True)
    df_grouped = df_steps.groupby("time_block")["Steps"].mean().reset_index()
    df_grouped.plot(kind="bar", x="time_block", y="Steps", figsize=(10, 5), title="Average Steps by 4-hour Block")
    plt.xlabel("Time Block")
    plt.ylabel("Average Steps")
    plt.show()


def tempDebugInfo():
    cursor.execute("PRAGMA table_info(hourly_steps)")
    columns_info = cursor.fetchall()


    print(columns_info)
    cursor.execute("SELECT DISTINCT Id FROM heart_rate")
    userInfo = cursor.fetchall();
    print(userInfo)
    

# TEST PART 4
#part 4
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

def fill_missing_weight_values(weight_log):
    """
    Function to fill missing values in WeightKg column.
    - Uses forward and backward fill within each participant (Id)
    - Fills remaining missing values with the participant's mean
    """
    print("\nOriginal missing values in weight_log:")
    print(weight_log.isnull().sum())
    
    # Replace empty strings and spaces with NaN
    weight_log.replace(["", " "], pd.NA, inplace=True)
    
    # Step 1: Forward and Backward Fill within each participant (Id)
    weight_log['WeightKg'] = weight_log.groupby('Id')['WeightKg'].fillna(method='ffill').fillna(method='bfill')
    
    # Step 2: Fill remaining missing values with the participant's mean
    weight_log['WeightKg'] = weight_log.groupby('Id')['WeightKg'].transform(lambda x: x.fillna(x.mean()))
    
    print("\nMissing values in weight_log after filling:")
    print(weight_log.isnull().sum())
    
    return weight_log

# Connect to the Fitbit database
db_path = 'fitbit_database.db'  # Update this path if necessary
conn = sqlite3.connect(db_path)

# Load only necessary columns from tables
daily_activity = pd.read_sql("SELECT Id, Calories AS CaloriesBurned, TotalSteps FROM daily_activity", conn)
heart_rate = pd.read_sql("SELECT Id, Value AS HeartRate FROM heart_rate", conn)
weight_log = pd.read_sql("SELECT Id, WeightKg FROM weight_log", conn)

# Convert Id to a common type (string) to avoid mismatches
daily_activity['Id'] = daily_activity['Id'].astype(str).str.strip()
heart_rate['Id'] = heart_rate['Id'].astype(str).str.strip()
weight_log['Id'] = weight_log['Id'].astype(str).str.strip()

# Fill missing values in WeightKg
weight_log = fill_missing_weight_values(weight_log)

# Create two separate merged DataFrames
merged_weight_activity = daily_activity.merge(weight_log, on='Id', how='inner')
merged_heart_activity = daily_activity.merge(heart_rate, on='Id', how='inner')

# Print table heads to inspect the data
print("\nHead of merged_weight_activity:")
print(merged_weight_activity.head())
print("\nHead of merged_heart_activity:")
print(merged_heart_activity.head())

# Compute summary statistics per individual
summary_weight_activity = merged_weight_activity.groupby('Id').agg({
    'CaloriesBurned': 'mean',
    'TotalSteps': 'mean',
    'WeightKg': 'mean'
})

summary_heart_activity = merged_heart_activity.groupby('Id').agg({
    'CaloriesBurned': 'mean',
    'TotalSteps': 'mean',
    'HeartRate': 'mean'
})

print("\nSummary Statistics for Weight and Activity:")
print(summary_weight_activity.head())
print("\nSummary Statistics for Heart Rate and Activity:")
print(summary_heart_activity.head())

# Ensure there is data for visualization
if summary_weight_activity.dropna().shape[0] > 0:
    # Visualization: Weight vs. Calories Burned
    plt.figure(figsize=(10, 6))
    plt.scatter(summary_weight_activity['WeightKg'], summary_weight_activity['CaloriesBurned'], alpha=0.7, color='green')
    plt.xlabel('Weight (Kg)')
    plt.ylabel('Average Calories Burned')
    plt.title('Weight vs. Calories Burned per Individual')
    plt.grid()
    plt.show()

if summary_heart_activity.dropna().shape[0] > 0:
    # Visualization: Heart Rate vs. Total Steps
    plt.figure(figsize=(10, 6))
    plt.scatter(summary_heart_activity['HeartRate'], summary_heart_activity['TotalSteps'], alpha=0.7, color='blue')
    plt.xlabel('Heart Rate')
    plt.ylabel('Total Steps')
    plt.title('Heart Rate vs. Total Steps per Individual')
    plt.grid()
    plt.show()

    # Visualization: Heart Rate vs. Calories Burned
    plt.figure(figsize=(10, 6))
    plt.scatter(summary_heart_activity['HeartRate'], summary_heart_activity['CaloriesBurned'], alpha=0.7, color='red')
    plt.xlabel('Heart Rate')
    plt.ylabel('Average Calories Burned')
    plt.title('Heart Rate vs. Calories Burned per Individual')
    plt.grid()
    plt.show()
else:
    print("\nNot enough valid data to generate plots.")

# Close database connection
conn.close()
