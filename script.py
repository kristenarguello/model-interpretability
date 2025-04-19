# %%
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# %%

# Load dataset
dataset = fetch_ucirepo(id=544)
if dataset is None:
    print("Dataset not found")
    exit()

# Split features and target
X = dataset.data.features
y = dataset.data.targets

# %%
# Rename columns to more descriptive and Pythonic names
feature_mapping = {
    "Gender": "gender",
    "Age": "age",
    "Height": "height",
    "Weight": "weight",
    "family_history_with_overweight": "family_history_overweight",
    "FAVC": "high_caloric_food",
    "FCVC": "freqof_vegetables",
    "NCP": "number_of_main_meals",
    "CAEC": "food_between_meals",
    "SMOKE": "smoking",
    "CH2O": "water_intake",
    "SCC": "calories_monitoring",
    "FAF": "freq_physical_activity",
    "TUE": "time_using_technology",
    "CALC": "alcohol_consumption",
    "MTRANS": "transportation_type",
}
target_mapping = {
    "NObeyesdad": "obesity_level",
}

X = X.rename(columns=feature_mapping)
y = y.rename(columns=target_mapping)

print("Renamed feature and target columns.")

# %%
# Check for missing values
print("Checking for NaN/null values:")
print("Features (X):")
print(X.isna().sum())
print(f"Total missing in X: {X.isna().sum().sum()}")

print("\nTarget (y):")
print(y.isna().sum())
print(f"Total missing in y: {y.isna().sum().sum()}")

# Basic dataset information
print("\nDataset shape:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")
print("\nFeature columns:", X.columns.tolist())
print("Target column:", y.columns.tolist())

# %%
# Target class distribution
print("\nTarget class distribution:")
print(
    y["obesity_level"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%"
)

# %%
# Check distributions of categorical features
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
print("\nCategorical features in X:")
for col in categorical_columns:
    print(f"\n{col} distribution:")
    print(X[col].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# Check distributions of numerical features
numerical_columns = X.select_dtypes(include=["int", "float"]).columns
numerical_cat_columns = []

print("\nNumerical features in X:")
for col in numerical_columns:
    unique_values = X[col].nunique()
    if col in ["age", "height", "weight", "number_of_main_meals"]:
        # These are clearly continuous variables, so no need to treat as categorical
        print(f"{col} is a numerical column.")
    else:
        # Variables with few unique numeric values might represent categorical choices
        numerical_cat_columns.append(col)
        print(f"\n{col} distribution ({unique_values} unique values):")
        print(X[col].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# %%
# Preprocessing step: round age to nearest integer and convert to float
# Helps model generalize better by focusing on completed years rather than decimal values
X["age"] = X["age"].round().astype(float)
print("Rounded 'age' to 0 decimal places and converted to float.")

# Round continuous variables (height, weight) to 2 decimal points
# Reduces noise and creates more consistent buckets
for col in ["height", "weight"]:
    X[col] = X[col].round(2).astype(float)
    print(f"Rounded '{col}' to 2 decimal points and converted to float.")

# %%

# Round number_of_main_meals to nearest integer
# This is a discrete variable, so rounding to nearest integer makes sense
# Helps model generalize better by focusing on completed meals rather than decimal values
X["number_of_main_meals"] = X["number_of_main_meals"].round().astype(float)

# %%
# Numerical features that are actually categorical (like Likert scales)
# First round to nearest class, then convert to category dtype
for col in numerical_cat_columns:
    X[col] = pd.Categorical(X[col].round().astype(int))
    print(f"Converted '{col}' to category dtype (rounded integer values).")

# %%
# Convert all object-typed columns to category dtype
# Helps reduce memory usage and ensures consistency for categorical encoding
for col in categorical_columns:
    X[col] = X[col].astype("category")
    print(f"Converted '{col}' to category dtype.")

# %%
# Categorizing features for modeling

# Ordinal categories (order matters)
ordinal_cat = [
    "freqof_vegetables",  # frequency of vegetable consumption
    "food_between_meals",  # snacking habits
    "water_intake",  # daily water consumption
    "freq_physical_activity",  # frequency of physical activity
    "time_using_technology",  # sedentary time
    "alcohol_consumption",  # frequency of alcohol use
]

# Nominal categories (no inherent order)
nominal_cat = [
    "gender",
    "family_history_overweight",
    "high_caloric_food",
    "smoking",
    "calories_monitoring",
    "transportation_type",
]

# Continuous numerical features
ratio_cat = ["number_of_main_meals", "height", "weight"]

# Discrete numeric features
discrete_cat = ["age"]

# %%
# Target label is ordinal — represents increasing severity of obesity
# Mapping used for ordinal encoding
obesity_level_mapping = {
    "Insufficient_Weight": -1,  # Less than normal weight
    "Normal_Weight": 0,
    "Overweight_Level_I": 1,
    "Overweight_Level_II": 2,
    "Obesity_Type_I": 3,
    "Obesity_Type_II": 4,
    "Obesity_Type_III": 5,
}
# encoded by hand
y["obesity_level"] = y["obesity_level"].map(obesity_level_mapping)
print("Mapped target labels to ordinal values.")
# %%
# get all possible values for each ordinal category
categories_values = []
for col in ordinal_cat:
    # TODO: need to reorder these values to respect the desired ordinal order
    categories_values.append(X[col].cat.categories.tolist())

print("Gathered all possible values for ordinal categories")

# %%

# setup encoders for categoricals
# use one hot encoder for nominal
nominal_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# ordinal encoder for ordinal
ordinal_encoder = OrdinalEncoder(
    categories=categories_values, handle_unknown="use_encoded_value", unknown_value=-1
)

# %%
print("\nPreprocessing completed. Dataset is ready for modeling or further steps.")

# %%

# TODO
# set up pipeline with encoders and scalers for numericals
# add gridsearch for parameters?
# train with each model
# add expaliner
