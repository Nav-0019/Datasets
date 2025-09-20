# ==============================================================================
# Phase 1: Exploratory Data Analysis (EDA) and Data Visualization
# ==============================================================================

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

os.chdir("B:/Projects 2024-2027/titanic")

# Load the training dataset and display the first few rows
df_titanic = pd.read_csv("train.csv")
df_titanic.head()

# Print information about the dataset, including data types and non-null counts
df_titanic.info()

# Calculate and display the number of missing values for each column
df_titanic.isna().sum()

# Get the count of survivors in the dataset
df_titanic["Survived"].sum()

# Display unique values for the 'Sex' column
df_titanic["Sex"].unique()

# Loop through all columns to print unique values for numerical features
for i in df_titanic.columns:
    if df_titanic[i].dtype == object:
        continue
    print("\n\n\n"+"-"*10+ " "+i+" "+"-"*10)
    print(df_titanic[i].unique())
    print("-"*40)

# Create histograms for key numerical features
plt.figure()
plt.hist(df_titanic["Age"].dropna())
plt.xlabel("Age of Passengers")
plt.ylabel("Frequency")
plt.title("Histogram of Age")

plt.figure()
plt.hist(df_titanic["Parch"])
plt.title("Histogram of Parch")

plt.figure()
plt.hist(df_titanic["SibSp"])
plt.title("Histogram of SibSp")

plt.figure()
plt.hist(df_titanic["Fare"])
plt.title("Histogram of Fare")

plt.show()

# Visualize the counts of categorical features
# Pclass counts
plt.figure(figsize=(6,4))
df_titanic["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Count by Class")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

# Sex counts
plt.figure(figsize=(6,4))
df_titanic["Sex"].value_counts().plot(kind="bar")
plt.title("Passenger Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# Embarked counts
plt.figure(figsize=(6,4))
df_titanic["Embarked"].value_counts().plot(kind="bar")
plt.title("Passenger Count by Embarked Port")
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.show()


# Analyze survival rates in relation to different features
# Survival rate by Sex
survival_by_sex = df_titanic.groupby("Sex")["Survived"].mean()
print(survival_by_sex)

survival_by_sex.plot(kind="bar", color=["blue", "pink"])
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()

# Survival rate by Pclass
survival_by_class = df_titanic.groupby("Pclass")["Survived"].mean()
print(survival_by_class)

survival_by_class.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate")
plt.show()

# Create age groups and analyze survival rate by age
bins = [0, 12, 18, 40, 60, 80]
labels = ["Child", "Teen", "Adult", "Middle Age", "Senior"]
df_titanic["AgeGroup"] = pd.cut(df_titanic["Age"], bins, labels=labels)

survival_by_age = df_titanic.groupby("AgeGroup")["Survived"].mean()
print(survival_by_age)

survival_by_age.plot(kind="bar")
plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")
plt.show()

# Analyze survival rate by both Sex and Pclass
survival_sex_class = df_titanic.groupby(["Sex", "Pclass"])["Survived"].mean().unstack()
print(survival_sex_class)

survival_sex_class.plot(kind="bar")
plt.title("Survival Rate by Sex and Class")
plt.ylabel("Survival Rate")
plt.show()


# ==============================================================================
# Phase 2: Data Preprocessing and Feature Engineering
# ==============================================================================

# Impute missing 'Age' values based on the median of 'Sex' and 'Pclass' groups
df_titanic["Age"] = df_titanic.groupby(["Sex", "Pclass"])["Age"].transform(lambda x : x.fillna(x.median()))

# Drop the 'Cabin' column due to a high number of missing values
df_titanic.drop(columns = ["Cabin"], inplace = True)

# Impute missing 'Embarked' values with the most frequent value (mode)
df_titanic["Embarked"] = df_titanic["Embarked"].fillna(df_titanic["Embarked"].mode()[0])

# Convert categorical data into numerical format for the model
# Use LabelEncoder for the 'Sex' column
le = LabelEncoder()
df_titanic["Sex"] = le.fit_transform(df_titanic["Sex"])

# Use one-hot encoding for the 'Embarked' column
Embarked_encoder = pd.get_dummies( df_titanic["Embarked"], prefix="Embarked", drop_first=True)
df_titanic = pd.concat( [df_titanic, Embarked_encoder], axis=1)
df_titanic.drop(columns = ["Embarked"], inplace = True)

# Create a new feature 'FamilySize' from 'SibSp' and 'Parch'
df_titanic["FamilySize"] = df_titanic["SibSp"] + df_titanic["Parch"] + 1

# Create a new binary feature 'IsAlone'
df_titanic["IsAlone"] = 0
df_titanic.loc[df_titanic["FamilySize"] == 1, "IsAlone"] = 1

# Extract titles from the 'Name' column
df_titanic["Title"] = df_titanic["Name"].str.extract( " ([A-Za-z]+)\.", expand = False)

# Group less common titles into a 'Rare' category
df_titanic["Title"] = df_titanic["Title"].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer'], "Rare")
df_titanic["Title"] = df_titanic["Title"].replace({'Mme': 'Mrs', 'Mlle' : 'Miss', 'Ms' : 'Miss'})

# Use one-hot encoding for the new 'Title' feature
Title_encoder = pd.get_dummies( df_titanic["Title"], prefix="Title", drop_first=True)
df_titanic = pd.concat( [df_titanic, Title_encoder], axis=1)
df_titanic.drop(columns= ["Title"], inplace=True)

# Create a new feature 'FarePerPerson'
df_titanic["FarePerPerson"] = df_titanic["Fare"] / df_titanic["FamilySize"]

# Drop columns that are no longer needed for the model
df_titanic.drop(columns = ["Name", "Ticket", "PassengerId", "AgeGroup"], inplace = True)


# ==============================================================================
# Phase 3: Model Selection, Training, and Evaluation
# ==============================================================================

# Separate features (x) and the target variable (y)
x = df_titanic.drop(columns=["Survived"])
y = df_titanic["Survived"]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42, stratify=y)

# Initialize and train the Logistic Regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit( x_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(x_test)

# Evaluate the model's performance using various metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix\n", cm)

cr = classification_report(y_test, y_pred)
print("\nClassification Report\n", cr)

y_pred_prob = log_reg.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, y_pred_prob)
print("\nAUC Score : ", auc)

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr, label= "LogReg = (AUC = %.3f)" % auc)
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ==============================================================================
# Phase 4: Preprocessing and Prediction on Unseen Test Data
# ==============================================================================

# Load the separate test dataset for final predictions
df_test = pd.read_csv("B:/Projects 2024-2027/titanic/test.csv")
df_test_submission = df_test # A reference to the original test dataframe for the PassengerId

# Apply the same preprocessing and feature engineering steps to the test data
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
df_test.drop( columns=["Cabin"], inplace=True)
df_test["Embarked"] = df_test["Embarked"].fillna(df_test["Embarked"].mode()[0])

# Apply the pre-fitted LabelEncoder to the 'Sex' column
df_test["Sex"] = le.transform(df_test["Sex"])

# Apply the same one-hot encoding to 'Embarked'
Embarked_encoder = pd.get_dummies( df_test["Embarked"], prefix="Embarked", drop_first=True)
df_test = pd.concat([ df_test, Embarked_encoder], axis=1)
df_test.drop(columns=["Embarked"], inplace=True)

# Create the same engineered features
df_test["FamilySize"] = df_test["Parch"] + df_test["SibSp"] + 1
df_test["IsAlone"] = 0
df_test.loc[df_test["FamilySize"] == 1, "IsAlone"] = 1
df_test["Title"] = df_test["Name"].str.extract("([A-Za-z]+)\.", expand=False)
df_test["Title"] = df_test["Title"].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer'], "Rare")
df_test["Title"] = df_test["Title"].replace({'Mme': 'Mrs', 'Mlle' : 'Miss', 'Ms' : 'Miss'})

# Apply the same one-hot encoding to 'Title'
Title_encoder = pd.get_dummies(df_test["Title"], prefix="Title", drop_first=True)
df_test = pd.concat([df_test, Title_encoder], axis=1)
df_test.drop(columns=["Title"], inplace = True)

# Impute the single missing 'Fare' value and create 'FarePerPerson'
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())
df_test["FarePerPerson"] = df_test["Fare"] / df_test["FamilySize"]

# Drop unnecessary columns to match the training data's structure
df_test.drop(columns = ["Name", "Ticket", "PassengerId"], inplace = True)

# Re-split the training data to ensure the model is trained on a fresh split
x = df_titanic.drop(columns=["Survived"])
y = df_titanic["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x,y)

# Align the columns of the test data with the training data
df_test = df_test.reindex(columns=x_train.columns, fill_value=0)

# Re-train the model on the full training data (or a new split) for final prediction
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit( x_train, y_train)

# Make the final predictions on the preprocessed test data
y_pred = log_reg.predict(df_test)

# Create the submission DataFrame using the original PassengerId and the predictions
submission_df = pd.DataFrame({"PassengerId": df_test_submission["PassengerId"], "Survived": y_pred})

# Save the DataFrame to a CSV file for submission
submission_df.to_csv("Submission.csv", index=False)

# Print a success message
print("Successfully done!")