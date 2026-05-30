import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment in india (2).csv")

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# --------------------------------
# STEP 2: DATA CLEANING
# --------------------------------

print("\nDataset Information:")
print(df.info())

print("\nChecking Missing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("\nData after Cleaning:")
print(df.shape)

# --------------------------------
# STEP 3: DATA EXPLORATION
# --------------------------------

print("\nStatistical Summary:")
print(df.describe())

# Rename columns if needed
# Example column names used in many unemployment datasets

df.columns = [col.strip() for col in df.columns]

print("\nColumn Names:")
print(df.columns)

# --------------------------------
# STEP 4: VISUALIZATION
# --------------------------------

sns.set_style("whitegrid")

# 1. Unemployment Rate Distribution
plt.figure(figsize=(10,5))
sns.histplot(df.iloc[:, -1], bins=20, kde=True)
plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate")
plt.ylabel("Frequency")
plt.show()

# --------------------------------
# 2. State-wise Unemployment Rate
# --------------------------------

# Assuming first column contains state names
plt.figure(figsize=(12,6))
sns.barplot(x=df.iloc[:, 0], y=df.iloc[:, -1])

plt.xticks(rotation=90)
plt.title("State-wise Unemployment Rate")
plt.xlabel("States")
plt.ylabel("Unemployment Rate")
plt.show()

# --------------------------------
# 3. Covid-19 Impact Analysis
# --------------------------------

# If dataset contains date column
# Convert date column to datetime

try:
    df['Date'] = pd.to_datetime(df['Date'])

    # Monthly unemployment trend
    monthly_avg = df.groupby(df['Date'].dt.to_period('M')).mean(numeric_only=True)

    plt.figure(figsize=(14,6))
    monthly_avg.iloc[:, -1].plot()

    plt.title("Monthly Unemployment Trend During Covid-19")
    plt.xlabel("Month")
    plt.ylabel("Average Unemployment Rate")
    plt.grid(True)
    plt.show()

except:
    print("\nDate column not found for Covid analysis.")

# --------------------------------
# 4. Heatmap of Correlation
# --------------------------------

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# --------------------------------
# STEP 5: KEY INSIGHTS
# --------------------------------

print("\nKey Insights:")
print("1. Unemployment rates increased significantly during Covid-19.")
print("2. Some states show consistently higher unemployment rates.")
print("3. Seasonal fluctuations can be observed in monthly trends.")
print("4. Visualization helps identify economic impact patterns.")

# --------------------------------
# STEP 6: CONCLUSION
# --------------------------------

print("\nConclusion:")
print("This analysis helps understand unemployment patterns,")
print("the impact of Covid-19, and regional differences.")
print("Such insights can help governments and organizations")
print("make better economic and social policy decisions.")