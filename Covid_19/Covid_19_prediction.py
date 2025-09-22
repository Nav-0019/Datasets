import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load COVID-19 data from a CSV file."""
    data = pd.read_csv(file_path, na_values="-")
    return data

def preprocess_data(data):
    """Preprocess the data by handling missing values and normalizing."""
    
    data['DateTime'] = pd.to_datetime(data['Date'] + " " + data['Time'], format="%Y-%m-%d %I:%M %p")
    
    data.drop(columns=['Date', 'Time', 'Sno'], inplace=True, errors='ignore')
    
    State = {
    "Telengana": "Telangana",
    "Himanchal Pradesh": "Himachal Pradesh",
    "Karanataka": "Karnataka",
    "Bihar****": "Bihar",
    "Madhya Pradesh***": "Madhya Pradesh",
    "Maharashtra***": "Maharashtra",
    "Cases being reassigned to states": "Unassigned",
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu"
    }

    data["State/UnionTerritory"] = data["State/UnionTerritory"].replace(State)
    data = data[data["State/UnionTerritory"] != "Unassigned"]
    
    

if __name__ == "__main__":
    
    # Load and preprocess data
    data = load_data("covid_19_india.csv")
    
    data = preprocess_data(data)