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
import streamlit as st

#Makes the code adaptable 
FILENAME = 'daily_activity.csv'


#Initializes the CSV file
def load_csv(filename='daily_activity.csv'):
    try:
        df = pd.read_csv(filename)
        df['Id'] = df['Id'].astype(str)
        df['ActivityDate'] = pd.to_datetime(df['ActivityDate'], errors='coerce')
        df = df.dropna(subset=['ActivityDate'])
        df['ActivityDate'] = df['ActivityDate'].astype('datetime64[ns]')  # Force dtype
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return None


#Initiliazes the connection to the data base
def db_init(dataBase_path="fitbit_database.db"):
    try:
        conn = sqlite3.connect(dataBase_path)
        return conn
    except Exception as e:
        print("Error connecting to database:", e)
        raise

def close_conn():
    db_init().close();

def printUniqueUsers(df):
  unique_users = df['Id'].nunique()
  print(f"Number of unique users: {unique_users}")

conn = db_init()
def totalDistance(df):
    total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()
    plt.figure(figsize=(10,5))
    plt.bar(total_distance_per_user.index, total_distance_per_user.values)
    plt.xlabel('User Id')
    plt.ylabel('Total Distance')
    plt.title('Total Distance per User')
    st.pyplot(plt.gcf())  

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
    st.pyplot(plt.gcf())


def workoutPerDay(df): 
  # Convert date and plot frequency of workouts per weekday
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

    # Count number of users who worked out each day
    users_per_day = df.groupby('ActivityDate')['Id'].nunique().reset_index()
    users_per_day['Weekday'] = users_per_day['ActivityDate'].dt.day_name()
    weekday_avg = users_per_day.groupby('Weekday')['Id'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

    plt.figure(figsize=(10, 6))
    sns.barplot(x=weekday_avg.index, y=weekday_avg.values, palette='coolwarm')
    plt.title('Average Number of Users Active per Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Average Number of Users')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
  

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
    st.pyplot(plt.gcf())
    
    # Print interpretation
    beta_1 = model.params['TotalSteps']
    print(f"\nFor user {user_id}, Beta_1 = {beta_1:.2f}.\n"
          f"Interpretation: Each additional step increases calories burned by ~{beta_1:.2f}.")







#this function classifies the users based on the frquency of their activity"
# the function returns a dataframe where the users are either classified ass heavy, moderate or light user.
#The dataframe has 2 cols, 1 with the id the other one with the class of activity
def classify_users(conn):
    query = """SELECT Id, COUNT(*) as activity_count FROM daily_activity GROUP BY Id"""
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    classified_users = []
    for user_id, activity_count in results:
        if activity_count <= 10:
            user_class = "Light user"
        elif 11 <= activity_count <= 15:
            user_class = "Moderate user"
        else:
            user_class = "Heavy user"
        classified_users.append((user_id, user_class))

    return pd.DataFrame(classified_users, columns=["Id", "Class"])
    
#Displays a pie chart 
def plot_user_class_distribution(conn):
    df_classified = classify_users(conn)
    class_counts = df_classified['Class'].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%',
           startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title('User Classification by Activity Level')
    st.pyplot(plt.gcf())



def sleep_duration(conn):
    """Computes total sleep duration per user and visualizes it."""
    query = "SELECT Id, SUM(value) as total_sleep FROM minute_sleep GROUP BY Id"

    
    df = pd.read_sql(query, conn)

    df["Id"] = df["Id"].astype(str)

    # Plot sleep duration per user
    plt.figure(figsize=(10, 5))
    sns.histplot(df["total_sleep"], bins=20, kde=True)
    plt.xlabel("Total Sleep Duration (minutes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Sleep Duration")
    st.pyplot(plt.gcf())

    return df




#This function analyses the correleation between the sleep duration and the amount of activity
#The function returns nothing, and displays a plot after it is executed
def sleep_vs_activity(conn):
    query = """
        SELECT m.Id, SUM(m.value) as total_sleep, 
               SUM(d.VeryActiveMinutes + d.FairlyActiveMinutes + d.LightlyActiveMinutes) as total_active 
        FROM minute_sleep m 
        JOIN daily_activity d ON m.Id = d.Id
        GROUP BY m.Id
    """
    df = pd.read_sql(query, conn).dropna()

    if df.empty:
        st.warning("No data available for sleep vs. activity analysis.")
        return

    model = smf.ols('total_sleep ~ total_active', data=df).fit()
    st.text(model.summary())

    sns.regplot(x=df["total_active"], y=df["total_sleep"])
    plt.xlabel("Total Active Minutes")
    plt.ylabel("Total Sleep (minutes)")
    plt.title("Sleep Duration vs. Active Minutes")
    st.pyplot(plt.gcf())


def sedentary_vs_sleep(conn):
    query = """
        SELECT d.Id, d.SedentaryMinutes, SUM(m.value) as total_sleep
        FROM daily_activity d
        JOIN minute_sleep m ON d.Id = m.Id
        GROUP BY d.Id
    """
    df = pd.read_sql(query, conn).dropna()

    if df.empty:
        st.warning("No data available for sedentary vs. sleep analysis.")
        return

    # model = smf.ols("total_sleep ~ SedentaryMinutes", data=df).fit()
    # st.text(model.summary())

    sns.regplot(x=df["SedentaryMinutes"], y=df["total_sleep"])
    plt.xlabel("Sedentary Minutes")
    plt.ylabel("Total Sleep (minutes)")
    plt.title("Sedentary Activity vs. Sleep Duration")
    st.pyplot(plt.gcf())




# 4. Breaking down the day into 4-hour blocks
def activity_by_time_blocks(conn):
    query = "SELECT Id, ActivityHour, Calories FROM hourly_calories"
    df_block = pd.read_sql(query, conn).dropna()

    df_block["ActivityHour"] = pd.to_datetime(df_block["ActivityHour"], errors="coerce")
    df_block["hour"] = df_block["ActivityHour"].dt.hour
    df_block["time_block"] = pd.cut(df_block["hour"], bins=[0, 4, 8, 12, 16, 20, 24],
                                    labels=['0-4', '4-8', '8-12', '12-16', '16-20', '20-24'], include_lowest=True)

    df_grouped = df_block.groupby("time_block")["Calories"].sum()
    df_grouped.plot(kind="bar", figsize=(10, 5))
    plt.title("Calories Burned Breakdown by Time Block")
    plt.xlabel("Time Block")
    plt.ylabel("Total Calories")
    st.pyplot(plt.gcf())



def steps_by_time_blocks(conn):
    query = "SELECT Id, ActivityHour, StepTotal FROM hourly_steps"
    df = pd.read_sql(query, conn)

    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"])
    df["Hour"] = df["ActivityHour"].dt.hour
    df["TimeBlock"] = pd.cut(df["Hour"], bins=[0, 4, 8, 12, 16, 20, 24],
                             labels=["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"], include_lowest=True)

    avg_steps = df.groupby("TimeBlock")["StepTotal"].mean()
    avg_steps.plot(kind="bar", figsize=(8, 5), title="Average Steps per 4-hour Block")
    plt.xlabel("Time Block")
    plt.ylabel("Average Steps")
    st.pyplot(plt.gcf())


    
def heart_rate_vs_intensity(user_id, conn):
    query = f"""
        SELECT h.Id, h.Time, h.Value as HeartRate, i.TotalIntensity
        FROM heart_rate h
        JOIN hourly_intensity i ON h.Id = i.Id AND h.Time = i.ActivityHour
        WHERE h.Id = '{user_id}'
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        st.warning("No data found for this user.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["HeartRate"], label="Heart Rate", color="red")
    plt.plot(df["Time"], df["TotalIntensity"], label="Exercise Intensity", color="blue")
    plt.legend()
    plt.xticks([])  # This line removes the x-axis labels
    plt.xlabel("Time")
    plt.ylabel("Heart Rate / Intensity")
    plt.title(f"Heart Rate vs. Exercise Intensity for User {user_id}")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

def weather_vs_activity(weather_df, activity_df):
    activity_df['ActivityDate'] = pd.to_datetime(activity_df['ActivityDate'], errors='coerce')
    weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
    merged = pd.merge(activity_df, weather_df, left_on='ActivityDate', right_on='date')

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    sns.regplot(data=merged, x='temp', y='TotalSteps', ax=axs[0], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    axs[0].set_title('Effect of Temperature on Steps')
    axs[0].set_xlabel('Temperature (°C)')
    axs[0].set_ylabel('Total Steps')
    axs[0].grid(True)

    sns.regplot(data=merged, x='precip', y='Calories', ax=axs[1], scatter_kws={'alpha':0.5}, line_kws={'color': 'green'})
    axs[1].set_title('Effect of Precipitation on Calories Burned')
    axs[1].set_xlabel('Precipitation (mm)')
    axs[1].set_ylabel('Calories Burned')
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)


#part 4
def fill_missing_weight_values(weight_log):
    
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




# Load only necessary columns from tables
def get_user_dataframes(conn):
    daily_activity = pd.read_sql("SELECT * FROM daily_activity", conn)
    heart_rate = pd.read_sql("SELECT * FROM heart_rate", conn)
    weight_log = pd.read_sql("SELECT * FROM weight_log", conn)
    hourly_calories = pd.read_sql("SELECT * FROM hourly_calories", conn)
    hourly_steps = pd.read_sql("SELECT * FROM hourly_steps", conn)
    minute_sleep = pd.read_sql("SELECT * FROM minute_sleep", conn)

    daily_activity.rename(columns={'Calories': 'CaloriesBurned'}, inplace=True)
    heart_rate.rename(columns={'Value': 'HeartRate'}, inplace=True)

    for df in [daily_activity, heart_rate, weight_log, hourly_calories, hourly_steps, minute_sleep]:
        df['Id'] = df['Id'].astype(float).astype(int).astype(str)

    weight_log['Date'] = pd.to_datetime(weight_log['Date'], errors='coerce')
    daily_activity['ActivityDate'] = pd.to_datetime(daily_activity['ActivityDate'], errors='coerce')
    heart_rate['Time'] = pd.to_datetime(heart_rate['Time'], errors='coerce')
    hourly_calories['ActivityHour'] = pd.to_datetime(hourly_calories['ActivityHour'], errors='coerce')
    hourly_steps['ActivityHour'] = pd.to_datetime(hourly_steps['ActivityHour'], errors='coerce')
    minute_sleep['date'] = pd.to_datetime(minute_sleep['date'], errors='coerce')

    return {
        "daily_activity": daily_activity,
        "heart_rate": heart_rate,
        "weight_log": weight_log,
        "hourly_calories": hourly_calories,
        "hourly_steps": hourly_steps,
        "minute_sleep": minute_sleep
    }

def get_individual_data(user_id, dfs, start_date=None, end_date=None, time_of_day=None):
    user_id = str(int(float(user_id))).strip()

    weight_data = dfs['weight_log'][dfs['weight_log']['Id'] == user_id]
    daily_data = dfs['daily_activity'][dfs['daily_activity']['Id'] == user_id]
    heart_data = dfs['heart_rate'][dfs['heart_rate']['Id'] == user_id]
    hourly_cals = dfs['hourly_calories'][dfs['hourly_calories']['Id'] == user_id]
    hourly_steps_data = dfs['hourly_steps'][dfs['hourly_steps']['Id'] == user_id]
    sleep_data = dfs['minute_sleep'][dfs['minute_sleep']['Id'] == user_id]

    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        weight_data = weight_data[(weight_data['Date'] >= start_date) & (weight_data['Date'] <= end_date)]
        daily_data = daily_data[(daily_data['ActivityDate'] >= start_date) & (daily_data['ActivityDate'] <= end_date)]
        heart_data = heart_data[(heart_data['Time'] >= start_date) & (heart_data['Time'] <= end_date)]
        hourly_cals = hourly_cals[(hourly_cals['ActivityHour'] >= start_date) & (hourly_cals['ActivityHour'] <= end_date)]
        hourly_steps_data = hourly_steps_data[(hourly_steps_data['ActivityHour'] >= start_date) & (hourly_steps_data['ActivityHour'] <= end_date)]
        sleep_data = sleep_data[(sleep_data['date'] >= start_date) & (sleep_data['date'] <= end_date)]

    if time_of_day:
        heart_data = heart_data[heart_data['Time'].dt.hour.between(time_of_day[0], time_of_day[1])]
        hourly_cals = hourly_cals[hourly_cals['ActivityHour'].dt.hour.between(time_of_day[0], time_of_day[1])]
        hourly_steps_data = hourly_steps_data[hourly_steps_data['ActivityHour'].dt.hour.between(time_of_day[0], time_of_day[1])]

    return {
        "weight": weight_data,
        "daily_activity": daily_data,
        "heart_rate": heart_data,
        "hourly_calories": hourly_cals,
        "hourly_steps": hourly_steps_data,
        "minute_sleep": sleep_data
    }


def get_summary_stats(user_data):
    stats = {}
    if not user_data['daily_activity'].empty:
        stats['Total Steps'] = user_data['daily_activity']['TotalSteps'].sum()
        stats['Total Calories Burned'] = user_data['daily_activity']['CaloriesBurned'].sum()
        plt.figure(figsize=(10, 4))
        user_data['daily_activity'].sort_values('ActivityDate').set_index('ActivityDate')['TotalSteps'].plot()
        plt.title('Steps Over Time')
        plt.ylabel('Steps')
        plt.xlabel('Date')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt.gcf())

    if not user_data['heart_rate'].empty:
        stats['Average Heart Rate'] = user_data['heart_rate']['HeartRate'].mean()
        plt.figure(figsize=(8, 4))
        user_data['heart_rate']['HeartRate'].plot.hist(bins=30, alpha=0.7)
        plt.title('Heart Rate Distribution')
        plt.xlabel('Heart Rate')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt.gcf())

    if not user_data['weight'].empty:
        w = user_data['weight'].sort_values('Date')
        stats['Average Weight'] = w['WeightKg'].mean()

    if not user_data['minute_sleep'].empty:
        sleep = user_data['minute_sleep'].copy()
        sleep['date'] = pd.to_datetime(sleep['date']).dt.date
        daily_sleep = sleep.groupby('date')['value'].sum() / 60
        stats['Avg Sleep Duration (hrs)'] = daily_sleep.mean()

        plt.figure(figsize=(10, 4))
        daily_sleep.sort_index().plot(marker='o', linestyle='-', color='purple')
        plt.title('Sleep Duration Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sleep Duration (hours)')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt.gcf())

        if not user_data['daily_activity'].empty:
            daily_activity = user_data['daily_activity'].copy()
            daily_activity['date'] = daily_activity['ActivityDate'].dt.date
            merged = pd.merge(daily_activity, daily_sleep.rename('SleepDuration'), left_on='date', right_index=True, how='inner')

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Total Steps', color='blue')
            ax1.plot(merged['date'], merged['TotalSteps'], color='blue', label='Steps')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Sleep Duration (hrs)', color='purple')
            ax2.plot(merged['date'], merged['SleepDuration'], color='purple', label='Sleep Duration')
            ax2.tick_params(axis='y', labelcolor='purple')

            plt.title('Daily Steps and Sleep Duration Over Time')
            fig.tight_layout()
            plt.grid(True)
            st.pyplot(fig)

    return stats

conn.close()