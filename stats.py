import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

from scipy import stats

from statsmodels.stats.multicomp import pairwise_tukeyhsd


separator_ = "=" * 100

def separate() -> None:
    print(separator_)

# Load the CSV file
file_path = "Salary Data.csv"
data = pd.read_csv(file_path)

# Significance level; 0.05 is used something like by "default"
alpha_value = 0.05

str_yoe = "Years of Experience"
str_salary = "Salary"
str_gender = "Gender"
str_education = "Education Level"
str_age = "Age"
str_title = "Job Title"

# Cleaning data by some heuristics
data = data[data[str_salary] >= 30000]
data = data[data[str_age] >= 18]
# use only data where genders are Male or Female
data = data[data[str_gender].str.lower().isin(["male", "female"])]

# Make dataset only for female/male gender
data_male = data[data[str_gender].str.lower().isin(["male"])]
data_female = data[data[str_gender].str.lower().isin(["female"])]



"""
Print number of unique values in each column
"""
separate()

def count_unique_values_from_csv(file_path: str, column_name: str, is_int: bool = False) -> str:
    data = pd.read_csv(file_path)
    
    # I could apply filter like ignore NaN values here, but I made some heuristics at the begining
    # to clean up the data
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

print("Statistics for Salary: Male")
separate()
print(summarize_column_stats(data_male, str_salary))
separate()

print("Statistics for Salary: Female")
separate()
print(summarize_column_stats(data_female, str_salary))
separate()

print()

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


correlation_salary_experience = data[[str_salary, str_yoe]].corr().loc[str_salary, str_yoe]

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


"""
Linear regression model with showing a graph
"""
data_drop_nan = data.dropna()
X = data_drop_nan[[str_yoe]]
y = data_drop_nan[str_salary]
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=str_yoe, y=str_salary, data=data_drop_nan, color='blue', label='Data Points')

# PIP upgraded some module and this no longer worked
# plt.plot(data_drop_nan[str_yoe], predictions, color='red', linewidth=2, label='Linear Regression Line')

x_values = np.array(data_drop_nan[str_yoe])
y_values = np.array(predictions)

plt.plot(x_values, y_values, color='red', linewidth=2, label='Linear Regression Line')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience with Linear Regression')
plt.legend()

plt.show()


"""
T-Test for Salaries by Gender with information about what its results mean
"""
separate()


male_salaries = data[data[str_gender] == "Male"][str_salary]
female_salaries = data[data[str_gender] == "Female"][str_salary]

t_stat, p_value = stats.ttest_ind(male_salaries, female_salaries, equal_var=False)

print("\nT-Test for Salaries by Gender, whether or not gender plays a role")
print(f" T-Statistic: {t_stat.round(2)}")
print(f" P-Value: {p_value.round(2)}")

print()

# Simply print() what do results mean based on alpha value - significance
if p_value < alpha_value:
    print("The result is statistically significant so we reject the null hypothesis")
    print("There is a significant difference in salaries between males and females.")
else:
    print("The result is not statistically significant so we fail to reject the null hypothesis")
    print("There is no significant difference in salaries between males and females.")

separate()


"""
Test for Salary vs Education, Education level is independent

Shapiro-Wilk Test Results: Indicates whether the salary distribution is normal for each education level.
ANOVA Results: Shows whether there's a significant difference in mean salaries across education levels.
Tukey's HSD: Identifies specific pairs of education levels that differ in salary.
Kruskal-Wallis Test: Non-parametric test results if ANOVA assumptions are not met.
Boxplot: Visual representation of salary distribution by education level.
"""
# Check if the data is normally distributed for each education level group
# but first clean them because then plotting wont work work
# and data I cleaned before are cleaned using different method()
data_cleaned_salary_education = data.dropna(subset=[str_salary, str_education])

# Prepare data for tests Salary/Education
education_groups = data_cleaned_salary_education.groupby(str_education)[str_salary]
education_levels = data_cleaned_salary_education[str_education]
salary_levels = data_cleaned_salary_education[str_salary]

# Shapiro-Wilk test for normality
# https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
for level, group_data in education_groups:
    stat, p_value = stats.shapiro(group_data)
    print(f"Shapiro-Wilk Test for {level}:")
    print(f"  Statistic: {stat:.3f}, P-Value: {p_value:.3f}")
    print("  Normally Distributed" if p_value > 0.05 else "  Not Normally Distributed")
    separate()


# Perform Analysis of Variance (ANOVA) if the data is normally distributed across groups
# https://www.spotfire.com/glossary/what-is-analysis-of-variance-anova#:~:text=Analysis%20of%20Variance%20(ANOVA)%20is,the%20means%20of%20different%20groups.
# https://en.wikipedia.org/wiki/Analysis_of_variance
anova_stat, anova_p_value = stats.f_oneway(*[group for _, group in education_groups])
print(f"ANOVA Result: F-statistic = {anova_stat:.3f}, P-Value = {anova_p_value:.3f}")

separate()

# If the ANOVA P-Value is significant, perform "Tukey's" post-hoc test
# https://en.wikipedia.org/wiki/Tukey%27s_range_test
if anova_p_value < alpha_value:
    tukey_result = pairwise_tukeyhsd(endog=salary_levels, groups=education_levels, alpha=alpha_value)
    print(tukey_result)

separate()

# If data is not normally distributed or ANOVA assumptions are violated, perform Kruskal-Wallis Test
# https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_test
kruskal_stat, kruskal_p_value = stats.kruskal(*[group for _, group in education_groups])
print(f"Kruskal-Wallis Test: H-statistic = {kruskal_stat:.3f}, P-Value = {kruskal_p_value:.3f}")

separate()

print("Simple statistical visualization, dots (or circles) are outliers")
# Simple visualization of the distribution of salary by education level
plt.figure(figsize=(12, 6))
sns.boxplot(x="Education Level", y="Salary", data=data_cleaned_salary_education)
plt.title("Salary Distribution by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Salary")
plt.show()

separate()