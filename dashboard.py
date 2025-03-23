import streamlit as st
#from Fitbit import load_csv, totalDistance, plot_calories_burnt, workoutPerDay, plot_regression, sleep_vs_activity
from Fitbit import *
from datetime import datetime

# Decorate the load_csv function with caching to avoid reloads on every interaction
@st.cache_data
def load_data():
    return load_csv('daily_activity.csv')

df = load_data()
weather_df = pd.read_csv("weather_chicago.csv")
conn = db_init()

if df is None or df.empty:
    st.error("Failed to load data. Please check the data file and path.")
else:
    st.title('Fitbit Dashboard')

# Sidebar for user ID selection and date filtering
user_id = st.sidebar.selectbox('Select User ID:', df['Id'].unique())
min_date, max_date = df['ActivityDate'].min(), df['ActivityDate'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
# single_user = get_individual_data();
selected_user = st.sidebar.selectbox("Select a User ID", user_id)
tabs = st.sidebar.radio("Navigation", [
    "General Statistics",
    "Individual Overview",
    "Time-based Analysis",
    "Sleep Analysis",
    "Weather"
])

if tabs == "General Statistics":
    st.header("General Insights")
    if not df.empty:
        st.write("Graphical Summary:")
        totalDistance(df)  # Assuming this is a graphical summary of total distance per user
        st.write("Numerical Summary:")
        #st.write(df.describe())  # Provides a numerical summary of the dataset
        st.write(df.drop(columns=['ActivityDate']).describe())
        printUniqueUsers(df)
        workoutPerDay(df)
        plot_user_class_distribution(conn)
elif tabs =="Individual Overview":
    user_data = df[df['Id'] == user_id].copy()

    if not user_data.empty:
        st.subheader(f'Statistics for User {user_id}')
        st.write(user_data.drop(columns=['ActivityDate']).describe())
        plot_calories_burnt(df, selected_user, start_date, end_date)
        linear_regression_for_user(df, selected_user)
        plot_regression(df, selected_user)
        heart_rate_vs_intensity(selected_user,conn)
    else:
        st.error("No data available for the selected user.")
elif tabs == "Time-based Analysis":
    st.header("Activity by Time Block")
    activity_by_time_blocks(conn)
    steps_by_time_blocks(conn)
    

elif tabs == "Sleep Analysis":
    st.header("Sleep and Activity Relationships")
    sleep_duration(conn)
    sleep_vs_activity(conn)
    sedentary_vs_sleep(conn)
elif tabs == "Weather":
    st.header("Weather vs Activity")
    weather_vs_activity(weather_df, df)


