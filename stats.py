import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the CSV file
file_path = 'Salary Data.csv'
data = pd.read_csv(file_path)

str_yoe = 'Years of Experience'
str_salary = 'Salary'

"""
This part is for first part of the project,
`calculating average, median, variance standard deviation, quartiles for Salary and Years of Experience`
"""
# statistics for Salary
salary_mean = data[str_salary].mean()
salary_median = data[str_salary].median()
salary_variance = data[str_salary].var()
salary_std_dev = data[str_salary].std()
# I will actually be doing deciles not quartiles
salary_deciles = data[str_salary].quantile([i/10 for i in range(1, 10)])

# statistics for Years of Experience
experience_mean = data[str_yoe].mean()
experience_median = data[str_yoe].median()
experience_variance = data[str_yoe].var()
experience_std_dev = data[str_yoe].std()
# deciles not quartiles
experience_deciles = data[str_yoe].quantile([i/10 for i in range(1, 10)])

salary_stats = {
    "Mean": salary_mean,
    "Median": salary_median,
    "Variance": salary_variance,
    "Standard Deviation": salary_std_dev,
    "Quartiles": salary_deciles
}

experience_stats = {
    "Mean": experience_mean,
    "Median": experience_median,
    "Variance": experience_variance,
    "Standard Deviation": experience_std_dev,
    "Quartiles": experience_deciles
}

print()
for stat_name, stat_value in salary_stats.items():
    print(f"{stat_name}: {stat_value.round(2)}")

print()
for experience_name, experience_value in experience_stats.items():
    print(f"{experience_name}: {experience_value.round(2)}")

"""
Linear Regression and Correlation
"""
correlation_salary_experience = data[[str_salary, str_yoe]].corr().loc[str_salary, str_yoe]

str_correlation_info_range ="""
Correlation can range [-1, 1]
1: Perfect positive linear relationship.
-1: Perfect negative linear relationship.
0: No linear relationship.

|r| > 0.7: Strong correlation.
0.7 to 1: Strong positive correlation.
-1 to -0.7: Strong negative correlation.
0.3 < |r| ≤ 0.7: Moderate correlation.
0.3 to 0.7: Moderate positive correlation.
-0.7 to -0.3: Moderate negative correlation.
|r| ≤ 0.3: Weak correlation.
0 to 0.3: Weak positive correlation.
-0.3 to 0: Weak negative correlation.
"""
# print(str_correlation_info_range)

correlation = correlation_salary_experience.round(3)
abs_corr = abs(correlation)
if (abs_corr >= 0.7 and abs_corr <= 1):
    correlation_strenght = "Strong"
elif (abs_corr >= 0.3 and abs_corr < 0.7):
    correlation_strenght = "Moderate"
elif (abs_corr >= 0 and abs_corr < 0.3):
    correlation_strenght = "Weak"

correlation_direction = "None"
if (correlation > 0):
    correlation_direction = "Positive"
elif (correlation < 0):
    correlation_direction = "Negative"

print(f"Correlation\n Value: {correlation}\n Strength: {correlation_strenght}\n Direction: {correlation_direction}")

data_drop_nan = data.dropna()
X = data_drop_nan[[str_yoe]]
y = data_drop_nan[str_salary]
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=str_yoe, y=str_salary, data=data_drop_nan, color='blue', label='Data Points')

plt.plot(data_drop_nan[str_yoe], predictions, color='red', linewidth=2, label='Linear Regression Line')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience with Linear Regression')
plt.legend()

plt.show()