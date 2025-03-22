import streamlit as st
#from Fitbit import load_csv, totalDistance, plot_calories_burnt, workoutPerDay, plot_regression, sleep_vs_activity
from Fitbit import *
from datetime import datetime

# Decorate the load_csv function with caching to avoid reloads on every interaction
@st.cache_data
def load_data():
    return load_csv('daily_activity.csv')

df = load_data()

if df is None or df.empty:
    st.error("Failed to load data. Please check the data file and path.")
else:
    st.title('Fitbit Dashboard')

# Sidebar for user ID selection and date filtering
user_id = st.sidebar.selectbox('Select User ID:', df['Id'].unique())
min_date, max_date = df['ActivityDate'].min(), df['ActivityDate'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

tabs = st.sidebar.radio("Navigation", [
    "General Statistics",
    "Individual Overview",
    "Time-based Analysis",
    "Sleep Analysis"
])

if tabs == "General Statistics":
    st.header("General Insights")
    if not df.empty:
        st.write("Graphical Summary:")
        totalDistance(df)  # Assuming this is a graphical summary of total distance per user
        st.write("Numerical Summary:")
        st.write(df.describe())  # Provides a numerical summary of the dataset
        printUniqueUsers(df)
        workoutPerDay(df)
elif tabs =="Individual Overview":
    user_data = df[df['Id'] == user_id]
    if not user_data.empty:
        st.subheader(f'Statistics for User {user_id}')
        st.write(user_data.describe())
    else:
        st.error("No data available for the selected user.")







# # User-specific statistics
# st.sidebar.header("User-specific Statistics")
# if st.sidebar.button('Show User Statistics'):
#     user_data = df[df['Id'] == user_id]
#     if not user_data.empty:
#         st.subheader(f'Statistics for User {user_id}')
#         st.write(user_data.describe())
#     else:
#         st.error("No data available for the selected user.")

# # Interactive plots and analysis
# st.header('Calories Burned Over Time')
# if start_date and end_date:
#     plot_calories_burnt(df, user_id, start_date, end_date)

# Analysis of sleep durations, could not get this one working for now
# st.header('Analysis of Sleep Durations')
# sleep_vs_activity(df)  

# Add a button to clear the cache in the sidebar or main page
if st.button('Clear Cache'):
    st.caching.clear_cache()