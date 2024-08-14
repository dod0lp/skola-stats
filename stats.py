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
salary_mean = data['Salary'].mean()
salary_median = data['Salary'].median()
salary_variance = data['Salary'].var()
salary_std_dev = data['Salary'].std()
salary_quartiles = data['Salary'].quantile([0.25, 0.5, 0.75])

# statistics for Years of Experience
experience_mean = data['Years of Experience'].mean()
experience_median = data['Years of Experience'].median()
experience_variance = data['Years of Experience'].var()
experience_std_dev = data['Years of Experience'].std()
# I will actually be doing decils not quartiles
experience_deciles = data['Years of Experience'].quantile([i/10 for i in range(1, 10)])

# simply print the results
print("Salary Statistics:")
print(f"Mean: {salary_mean}")
print(f"Median: {salary_median}")
print(f"Variance: {salary_variance}")
print(f"Standard Deviation: {salary_std_dev}")
print(f"Quartiles:\n{salary_quartiles}\n")

print("Years of Experience Statistics:")
print(f"Mean: {experience_mean}")
print(f"Median: {experience_median}")
print(f"Variance: {experience_variance}")
print(f"Standard Deviation: {experience_std_dev}")
print(f"Quartiles:\n{experience_deciles}")
