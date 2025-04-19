# %%
from ucimlrepo import fetch_ucirepo

# fetch dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = (
    fetch_ucirepo(id=544)
)

if estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition is None:
    print("Dataset not found")
    exit()

# %%
# data (as pandas dataframes)
X = (
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
)
y = (
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets
)

from rich import print
import pandas as pd

# metadata
print(
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.metadata
)


# %%
# variable information
print(
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.variables
)

# %%
# Transform target into a pandas DataFrame

if isinstance(y, pd.DataFrame):
    print("Target is already a pandas DataFrame")
else:
    y = pd.DataFrame(y)
    print("Target transformed into a pandas DataFrame")

# Display the first few rows of the target DataFrame
print("\nTarget DataFrame preview:")
print(y.head())
# %%
# Get value counts for the target variable
value_counts = y["NObeyesdad"].value_counts()

# Calculate percentages
percentage_distribution = value_counts / len(y) * 100

# Display both counts and percentages
print("\nTarget class distribution:")
print(value_counts)
print("\nPercentage distribution:")
print(percentage_distribution.round(2))

# Optional: Visual representation
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
percentage_distribution.plot(kind="bar")
plt.title("Class Distribution in Percentage")
plt.ylabel("Percentage (%)")
plt.xlabel("Obesity Level")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
