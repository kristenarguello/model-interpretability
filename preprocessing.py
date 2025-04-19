# %%
from ucimlrepo import fetch_ucirepo

# %%

estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = (
    fetch_ucirepo(id=544)
)

if estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition is None:
    print("Dataset not found")
    exit()

# %%

X = (
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
)
y = (
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets
)
# 2111 instancias
# 16 colunas

# %%
# column names mapping
mapping = {
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

# Rename feature columns according to mapping
X = X.rename(columns=mapping)
print("Renamed feature columns using mapping")

# Rename target column according to target_mapping
y = y.rename(columns=target_mapping)
print(f"Renamed target column to {list(target_mapping.values())[0]}")


# %%

# analyzing dataset
print("Checking for NaN/null values in features (X):")
print(X.isna().sum())
print(f"Total NaN values in X: {X.isna().sum().sum()}")

print("\nChecking for NaN/null values in target (y):")
print(y.isna().sum())
print(f"Total NaN values in y: {y.isna().sum().sum()}")

# Additional info about dataset
print("\nBasic dataset info:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("\nFeature columns:")
print(X.columns.tolist())
print("\nTarget columns:")
print(y.columns.tolist())

# %%

# show distributions of values and check the unique values to see if there are any typos
# or any values that are none or something that is non-existant
print("Target class distribution:")
print(
    y["obesity_level"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%"
)

# Analyze categorical features in X
categorical_columns = X.select_dtypes(include=["object", "category"]).columns

print("\nCategorical features in X:")
# Plot distribution for each categorical feature
for i, column in enumerate(categorical_columns):
    # Print percentage distribution
    print(f"\n{column} distribution:")
    print(X[column].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
,    
print("\nNumerical features in X:")
numerical_cat_columns = []
# For numerical columns, check if they have few unique values that might be categorical
numerical_columns = X.select_dtypes(include=["int", "float"]).columns
for column in numerical_columns:
    unique_values = X[column].nunique()
    # if unique_values < 10:  # If few unique values, might be treated as categorical
    if column in ["age", "height", "weight"]:
        print("" + column + " is a continuous column.")
        continue
    numerical_cat_columns.append(column)
    print(f"\n{column} distribution (has {unique_values} unique values):")
    print(X[column].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# %%
## ACTUAL PREPROCESSING NOW
# rounding and transforming the age number to integers
# will help the model to understand the data better -- categorize the people according to the completed years instead of the years and months
# converted to float again to make sure eveyrthing is float
X.loc[:, "age"] = X["age"].round().astype(float)
print("Rounded age to 0 decimal points and converted to float")

# %%
# height
# Round Height values to 2 decimal points (less 'buckets' and less 'noise')
# converting to float again to make sure eveyrthing is float
for c in ["height", "weight"]:
    X.loc[:, c] = X[c].round(2).astype(float)
    print(f"Rounded {c} to 2 decimal points and converted to float")
# %%
# convert all of the numerical columns that are actually categorical into int
# and then to categorical
# but first rounded it to the closest class in regarding the number
for c in numerical_cat_columns:
    # Round values to 2 decimal points
    X.loc[:, c] = X[c].round().astype(int)
    # Convert to categorical
    X.loc[:, c] = X[c].astype("int")
    X.loc[:, c] = X[c].astype("category")
    print(f"Converted {c} to category dtype")

# %%
# Convert all object dtypes to categorical
# checked by showing nunique valus of each column, and they were all ok
# no need of preprocessing the values
object_columns = X.select_dtypes(include=["object"]).columns
for col in object_columns:
    X[col] = X[col].astype("category")
    print(f"Converted {col} to category dtype")

# %%
