# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

def preprocess_and_engineer_features(df_titanic):
    # Impute missing 'Age' values based on the median of 'Sex' and 'Pclass' groups
    df_titanic["Age"] = df_titanic.groupby(["Sex", "Pclass"])["Age"].transform(lambda x : x.fillna(x.median()))

    # Drop the 'Cabin' column due to a high number of missing values
    df_titanic.drop(columns = ["Cabin"], inplace = True, errors='ignore')

    # Impute missing 'Embarked' values with the most frequent value (mode)
    df_titanic["Embarked"] = df_titanic["Embarked"].fillna(df_titanic["Embarked"].mode()[0])

    # Convert categorical data into numerical format for the model
    # Use LabelEncoder for the 'Sex' column

    df_titanic["Sex"] = df_titanic['Sex'].map( {'male': 0, 'female': 1} ).fillna(-1).astype(int)

    # Use one-hot encoding for the 'Embarked' column
    Embarked_encoder = pd.get_dummies( df_titanic["Embarked"], prefix="Embarked", drop_first=True)
    df_titanic = pd.concat( [df_titanic, Embarked_encoder], axis=1)
    df_titanic.drop(columns = ["Embarked"], inplace = True, errors='ignore')

    # Create a new feature 'FamilySize' from 'SibSp' and 'Parch'
    df_titanic["FamilySize"] = df_titanic["SibSp"] + df_titanic["Parch"] + 1

    # Create a new binary feature 'IsAlone'
    df_titanic["IsAlone"] = 0
    df_titanic.loc[df_titanic["FamilySize"] == 1, "IsAlone"] = 1

    # Extract titles from the 'Name' column
    df_titanic["Title"] = df_titanic["Name"].str.extract(r" ([A-Za-z]+)\.", expand = False)

    # Group less common titles into a 'Rare' category
    df_titanic["Title"] = df_titanic["Title"].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer'], "Rare")
    df_titanic["Title"] = df_titanic["Title"].replace({'Mme': 'Mrs', 'Mlle' : 'Miss', 'Ms' : 'Miss'})

    # Use one-hot encoding for the new 'Title' feature
    Title_encoder = pd.get_dummies( df_titanic["Title"], prefix="Title", drop_first=True)
    df_titanic = pd.concat( [df_titanic, Title_encoder], axis=1)
    df_titanic.drop(columns= ["Title"], inplace=True, errors='ignore')

    # Create a new feature 'FarePerPerson'
    df_titanic["FarePerPerson"] = df_titanic["Fare"] / df_titanic["FamilySize"]

    # Drop columns that are no longer needed for the model
    df_titanic.drop(columns = ["Name", "Ticket", "PassengerId"], inplace = True,  errors='ignore')

    return df_titanic

if __name__ == "__main__":

    @st.cache_data
    def load_and_preprocess_data():
        df_titanic = pd.read_csv("Titanic_Dataset/train.csv")
        df_titanic = preprocess_and_engineer_features(df_titanic)
        return df_titanic
        
    @st.cache_resource
    def train_model(x_train, y_train):
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        return model
    
    df_titanic = load_and_preprocess_data()

    # Separate features (x) and the target variable (y)
    x = df_titanic.drop(columns=["Survived"])
    y = df_titanic["Survived"]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42, stratify=y)

    # Initialize and train the Logistic Regression model
    log_reg = train_model(x_train, y_train)

    # Streamlit app for user input and prediction

    st.title("Titanic Survival Prediction")
    st.write("This app predicts whether a passenger survived the Titanic disaster based on their features.")
    st.write("Write Passenger's details below and let me predict for him:")

    Name = st.text_input("Full Name (optional, for Title extraction)", "Mr. Test")
    passenger_id = st.number_input("Passenger ID (optional)", min_value=0, value=0)
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
    cabin = st.text_input("Cabin (optional)", "")

    if st.button("Predict Survival"):

        if age <= 0 or fare < 0:
            st.warning("Please enter valid Age and Fare values.")
        else:
            # Create a DataFrame for the input
            df_input = pd.DataFrame({
                "Pclass": [pclass],
                "Sex": [0 if sex == "male" else 1],
                "Age": [age],
                "SibSp": [sibsp],
                "Parch": [parch],
                "Fare": [fare],
                "Embarked": [embarked],
                "Name": [Name],  
                "Ticket": ["TestTicket"],    
                "Cabin": [cabin if cabin else None],
                "PassengerId": [passenger_id if passenger_id != 0 else 9999]
            })

            df_input = preprocess_and_engineer_features(df_input)

            # Align input columns with training data
            df_input = df_input.reindex(columns=x_train.columns, fill_value=0)

            prediction = log_reg.predict(df_input)
            prediction_proba = log_reg.predict_proba(df_input)[0][1]
                
            result = "Survived" if prediction[0] == 1 else "Did Not Survive"
            st.write(f"The passenger would have: **{result}**")
            st.write(f"Survival Probability: **{prediction_proba*100:.2f}%**")
                
            test_accuracy = log_reg.score(x_test, y_test)
            st.write(f"Model Test Accuracy: **{test_accuracy*100:.2f}%**")
