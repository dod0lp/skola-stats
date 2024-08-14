import pandas as pd

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
salary_deciles = data[str_salary].quantile([i/10 for i in range(1, 10)])

# statistics for Years of Experience
experience_mean = data[str_yoe].mean()
experience_median = data[str_yoe].median()
experience_variance = data[str_yoe].var()
experience_std_dev = data[str_yoe].std()
# I will actually be doing decils not quartiles
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

for stat_name, stat_value in salary_stats.items():
    print(f"{stat_name}: {stat_value}")

for experience_name, experience_value in experience_stats.items():
    print(f"{experience_name}: {experience_value}")
