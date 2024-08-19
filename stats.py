import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

from scipy import stats

separator_ = "=" * 100

def separate() -> None:
    print(separator_)

# Load the CSV file
file_path = "Salary Data.csv"
data = pd.read_csv(file_path)

str_yoe = "Years of Experience"
str_salary = "Salary"
str_gender = "Gender"
str_education = "Education Level"
str_age = "Age"
str_title = "Job Title"


"""
Print number of unique values in each column
"""
separate()

def count_unique_values_from_csv(file_path: str, column_name: str, is_int: bool = False) -> str:
    data = pd.read_csv(file_path)
    
    # I could apply filter like ignore NaN values here
    filtered_data = data

    value_counts = filtered_data[column_name].value_counts()
    
    counts_str = "\n".join([f"  {value}: {count} times" for value, count in value_counts.items()])
    
    if (is_int == True):
        sorted_counts = value_counts.sort_values(ascending=False)
        counts_str = "\n".join([f"  {int(value)}: {count} times" for value, count in sorted_counts.items()])
    
    summary = f"Counts of unique values in '{column_name}':\n{counts_str}"
    
    return summary

UniqueValuesCount = True
if (UniqueValuesCount):
    print(count_unique_values_from_csv(file_path, str_education))
    separate()
    print(count_unique_values_from_csv(file_path, str_salary, is_int=True))
    separate()
    print(count_unique_values_from_csv(file_path, str_yoe, is_int=True))
    separate()
    print(count_unique_values_from_csv(file_path, str_age, is_int=True))
    separate()
    print(count_unique_values_from_csv(file_path, str_gender))
    separate()
    print(count_unique_values_from_csv(file_path, str_title))
    separate()


"""
This part is for first part of the project,
`calculating average, median, variance standard deviation, quartiles for Salary and Years of Experience`
"""
separate()

def summarize_column_stats(data: pd.DataFrame, column_name: str) -> str:
    mean_value = data[column_name].mean()
    median_value = data[column_name].median()
    variance_value = data[column_name].var()
    std_dev_value = data[column_name].std()
    deciles = data[column_name].quantile([i/10 for i in range(1, 10)])

    # formatted string results
    summary = (
        f"Mean: {mean_value.round(2)}\n"
        f"Median: {median_value.round(2)}\n"
        f"Variance: {variance_value.round(2)}\n"
        f"Standard Deviation: {std_dev_value.round(2)}\n"
        f"Deciles:\n" +
        "\n".join([f"  {i+1}0th percentile: {deciles.iloc[i].round(2)}" for i in range(len(deciles))])
    )
    
    return summary

# simply calculated statistics for Salary using function
print("Statistics for Salary")
separate()
print(summarize_column_stats(data, str_salary))
separate()

print()

# statistics for Years of Experience
print("Statistics for Years of Experience")
separate()
print(summarize_column_stats(data, str_yoe))
separate()

print()

# by Gender
# print(summarize_column_stats(data[data[str_gender] == "Male"], str_salary))
# print(summarize_column_stats(data[data[str_gender] == "Female"], str_salary))

"""
Linear Regression and Correlation of Salary and Years of Experience
Correlation calculations with information about what it means
"""
separate()

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
print(str_correlation_info_range)

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

separate()
print(f"Correlation of Salary and Years of Experience\
\n Value: {correlation}\n Strength: {correlation_strenght}\n Direction: {correlation_direction}")
separate()


"""
Linear regression model with showing a graph
"""
separate()

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

separate()


"""
T-Test for Salaries by Gender with information about what its results mean
"""
separate()

alpha = 0.05  # Significance level; 0.05 is used something like by "default"

male_salaries = data[data[str_gender] == "Male"][str_salary]
female_salaries = data[data[str_gender] == "Female"][str_salary]

t_stat, p_value = stats.ttest_ind(male_salaries, female_salaries, equal_var=False)

print("\nT-Test for Salaries by Gender")
print(f" T-Statistic: {t_stat.round(2)}")
print(f" P-Value: {p_value.round(2)}")

print()

if p_value < alpha:
    print("The result is statistically significant so we reject the null hypothesis")
    print("There is a significant difference in salaries between males and females.")
else:
    print("The result is not statistically significant so we fail to reject the null hypothesis")
    print("There is no significant difference in salaries between males and females.")
