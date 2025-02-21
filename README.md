# Data_Engineering

# Fitbit Data Analysis

## Overview
This project analyzes Fitbit activity data using Python. The script `Fitbit.py` provides multiple functions to explore and visualize user activity, including calorie expenditure, step count, and workout frequency. It also applies statistical modeling to understand relationships within the dataset.

## Features
- Count the number of unique Fitbit users.
- Calculate total distance traveled by each user.
- Visualize calories burnt over a date range for a specific user.
- Analyze workout frequency across weekdays.
- Perform linear regression on calories burnt vs. total steps.
- Generate a sunburst chart to visualize activity breakdown by weekday and user.

## Dependencies
In order for the code to work correctly, the following Python libraries need to be installed

-pandas
-matplotlib 
-seaborn 
-numpy 
-statsmodels 
-plotly



## Usage

### 1. Run the script
The pyhton script can be executed directly by typing the followng into the terminal:
`python Fitbit.py`

Note: Python has to be installed for this to work

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

## Author
Group 10



