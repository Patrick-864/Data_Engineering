# Data_Engineering

# Fitbit Data Analysis

## Overview
This project analyzes Fitbit activity data using Python. The script `Fitbit.py` provides multiple functions to explore and visualize user activity, including calorie expenditure, step count, and workout frequency. It also applies statistical modeling to understand relationships within the dataset.
The end goal is to create a dashboard for statistical analysis and data visualization.

## Table of Contents
- [Introduction](#introduction)
- [Requirements and Dependecies](#requirements-and-dependecies) 
- [Features](#features)
- [Data exploration and visualization](#data-exploration)
- [Database interaction and querying](#databse-interaction)
- [Data wrangling](#data-wrangling)
- [Dashborad](#dashboard)




## Introduction
This project investigates data collected from Fitbit users to identify trends in activity, calories burned, and other fitness metrics. It consists of four main parts:
- **Data exploration and visualization**
- **Database interaction and querying**
- **Data wrangling and preparation for dashboard visualization**
- **Dashborad**

### Setup
Clone the repository and install dependencies:
```
git clone <repository_url>
```


## Requirements and Dependecies
Any Python 3 version needs to be installed and in order for the code to work correctly, the following Python libraries need to be installed
-pandas
-matplotlib 
-seaborn 
-numpy 
-statsmodels 
-plotly


## Features
- **Data Exploration**
  - Count unique users.
  - Visualize daily activity trends.
  - Perform regression analysis between steps and calories.

- **Database Interaction**
  - Query Fitbit activity and sleep data using SQLite.
  - Investigate relationships between activity levels and sleep duration.

- **Data Wrangling**
  - Handle missing values.
  - Aggregate data for dashboard visualization.

- **Dashboard**
  -visualize the data in an comprehensible way.


## Data exploration and visualization

### 1. Run the script
The pyhton script can be executed directly by typing the followng into the terminal:
`python Fitbit.py`


### 2. Function Usage
Modify the script to call different functions as needed. Example:

- To print the number of unique users:
  ```python
  printUniqueUsers()
  ```
- To visualize total distance per user:
  ```python
  totalDistance()
  ```
- To plot calories burnt for a specific user (e.g., `user_id=12345`) over a specfic period of time:
  ```python
  plot_calories_burnt(12345, '2022-01-01', '2022-01-31')
  ```
- To analyze linear regression between calories burnt and steps:
  ```python
  linearRegression()
  ```
- To visualize workout frequency:
  ```python
  workoutPerDay()
  ```

## Dataset
The script reads Fitbit activity data from:
```
daily_activity.csv
```
Ensure the CSV file is correctly placed before running the script.
If the CSV file is located in a different directory copy and paste the path to the CSV file into the variable `FILENAME`. It is important to surond the pathname by `""`

## Future Improvements
- Improve regression analysis visualization.
- Add more user-friendly input handling.
- Optimize performance for larger datasets.



## Database interaction and querying
The code is now modified to be able to read and evaluate data from a database that contains fitness data. 
It includes multiple functions that classify users based on activity levels, examine relationships between sleep and activity, analyze sedentary behavior, and break down calorie expenditure by time blocks.

## Functions

The code is able to perform the following analysis:

1. **Classifying Users Based on Activity Levels:**
   - Categorizes users as Light, Moderate, or Heavy based on their activity frequency.

2. **Analyzing Sleep Duration vs. Active Minutes:**
   - Investigates the correlation between total sleep duration and active minutes.
   - Uses linear regression for statistical analysis.
   - Generates a regression plot.

3. **Analyzing Sedentary Activity vs. Sleep Duration:**
   - Examines the relationship between sedentary minutes and total sleep.
   - Performs linear regression and visualizes the data.

4. **Breaking Down Daily Calories Burned by Time Blocks:**
   - Divides a day into 4-hour segments and calculates total calories burned in each time block.
   - Displays the results in a bar chart.

## Dependencies
Apart from the previous libraries no new library is required to be downloaded. 

## Database Structure

The script expects an SQLite database (`fitbit_database.db`) with the following tables:

- **`daily_activity`**: Contains daily activity summary per user.
- **`minute_sleep`**: Records sleep duration in minutes.
- **`hourly_calories`**: Tracks calories burned per hour.





### Expected Output

- A classification table displaying user activity levels.
- Statistical summaries from regression models.
- Plots visualizing relationships between sleep, activity, and calorie expenditure.





## Authors
Bendel Mees, Karssen Quinten, Schneider Patrick, Tensen Taeke





