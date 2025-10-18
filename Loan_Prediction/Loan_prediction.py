import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# --- Constants ---
RISK_BINS = [0, 0.10, 0.15, 1.0]
RISK_LABELS = ['stable zone', 'risk zone', 'high risk zone']

PROFESSION_GROUPS = {
    'Physician': 'High_Income_Specialized', 'Surgeon': 'High_Income_Specialized', 
    'Dentist': 'High_Income_Specialized', 'Chartered_Accountant': 'High_Income_Specialized', 
    'Lawyer': 'High_Income_Specialized', 'Magistrate': 'High_Income_Specialized', 
    'Architect': 'High_Income_Specialized', 'Financial_Analyst': 'High_Income_Specialized', 
    'Software_Developer': 'STEM_Technical', 'Mechanical_engineer': 'STEM_Technical', 
    'Chemical_engineer': 'STEM_Technical', 'Civil_engineer': 'STEM_Technical', 
    'Industrial_Engineer': 'STEM_Technical', 'Design_Engineer': 'STEM_Technical', 
    'Computer_hardware_engineer': 'STEM_Technical', 'Biomedical_Engineer': 'STEM_Technical', 
    'Petroleum_Engineer': 'STEM_Technical', 'Engineer': 'STEM_Technical', 
    'Scientist': 'STEM_Technical', 'Technology_specialist': 'STEM_Technical', 
    'Economist': 'STEM_Technical', 'Civil_servant': 'Government_Civil', 
    'Police_officer': 'Government_Civil', 'Firefighter': 'Government_Civil', 
    'Army_officer': 'Government_Civil', 'Official': 'Government_Civil', 
    'Librarian': 'Government_Civil', 'Technician': 'Service_Mid_Range', 
    'Secretary': 'Service_Mid_Range', 'Drafter': 'Service_Mid_Range', 
    'Computer_operator': 'Service_Mid_Range', 'Hotel_Manager': 'Service_Mid_Range', 
    'Surveyor': 'Service_Mid_Range', 'Geologist': 'Service_Mid_Range', 
    'Microbiologist': 'Service_Mid_Range', 'Analyst': 'Analyst_Consulting', 
    'Statistician': 'Analyst_Consulting', 'Consultant': 'Analyst_Consulting', 
    'Psychologist': 'Analyst_Consulting', 'Artist': 'Creative_Hospitality', 
    'Designer': 'Creative_Hospitality', 'Graphic_Designer': 'Creative_Hospitality', 
    'Web_designer': 'Creative_Hospitality', 'Fashion_Designer': 'Creative_Hospitality', 
    'Chef': 'Creative_Hospitality', 'Comedian': 'Creative_Hospitality', 
    'Technical_writer': 'Creative_Hospitality', 'Flight_attendant': 'Aviation_Operations', 
    'Air_traffic_controller': 'Aviation_Operations', 'Aviator': 'Aviation_Operations', 
    'Politician': 'Aviation_Operations'
}

ALL_PROFESSION_GROUPS = list(set(PROFESSION_GROUPS.values()))
TRAINING_COLUMNS = None  # will be set after transformation

# --- Helper Functions ---
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def calculate_risk_maps(df_train):
    state_risk_map = df_train.groupby('STATE')['Risk_Flag'].mean()
    city_risk_map = df_train.groupby('CITY')['Risk_Flag'].mean()
    return state_risk_map, city_risk_map

def apply_transformations(df, state_map=None, city_map=None, is_training=True):
    df = pd.DataFrame(df).copy()
    
    # Binary encodings
    df["Married/Single"] = df["Married/Single"].map({'single':0, 'married':1}).fillna(0).astype(int)
    df["Car_Ownership"] = df["Car_Ownership"].map({'no':0, 'yes':1}).fillna(0).astype(int)

    # House OHE
    House_map = {'rented':1, 'owned':2, 'norent_noown':0}
    df['House_Ownership'] = df['House_Ownership'].map(House_map).fillna(0)
    df['House_Ownership_owned'] = (df['House_Ownership'] == 2).astype(int)
    df['House_Ownership_rented'] = (df['House_Ownership'] == 1).astype(int)

    # Profession group + OHE
    df['Profession_Group'] = df['Profession'].map(PROFESSION_GROUPS).fillna(ALL_PROFESSION_GROUPS[0])
    Prof_Encoder = pd.get_dummies(df['Profession_Group'], prefix='Profession')
    df = pd.concat([df, Prof_Encoder], axis=1)

    # State/City risk scores
    if is_training:
        df['State_Risk_Score'] = df['STATE'].map(state_map).fillna(df['STATE'].map(state_map).mean())
        df['City_Risk_Score'] = df['CITY'].map(city_map).fillna(df['CITY'].map(city_map).mean())
    else:
        mean_state_risk = state_map.mean() if not state_map.empty else 0.125
        mean_city_risk = city_map.mean() if not city_map.empty else 0.125
        df['State_Risk_Score'] = df['STATE'].map(state_map).fillna(mean_state_risk)
        df['City_Risk_Score'] = df['CITY'].map(city_map).fillna(mean_city_risk)

    # Risk zones
    df['State_Risk_Zone'] = pd.cut(df['State_Risk_Score'], bins=RISK_BINS, labels=RISK_LABELS, right=True, include_lowest=True)
    df['City_Risk_Zone'] = pd.cut(df['City_Risk_Score'], bins=RISK_BINS, labels=RISK_LABELS, right=True, include_lowest=True)
    Encoder_State = pd.get_dummies(df['State_Risk_Zone'], prefix='Zone_Risk_State', drop_first=True)
    Encoder_City = pd.get_dummies(df['City_Risk_Zone'], prefix='Zone_Risk_City', drop_first=True)
    df = pd.concat([df, Encoder_State, Encoder_City], axis=1)

    # Drop unnecessary columns
    cols_to_drop = ['CITY', 'STATE', 'Profession', 'Profession_Group', 'Profession_Analyst_Consulting',
                    'State_Risk_Score', 'City_Risk_Score', 'State_Risk_Zone', 'City_Risk_Zone',
                    'House_Ownership', 'Id']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

@st.cache_resource
def train_model(x_train, y_train):
    X_train = x_train.astype(float)
    model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Main App ---
if __name__ == "__main__":
    st.set_page_config(page_title="Loan Risk Predictor", layout="centered")
    DATA_FILE_PATH = 'Loan_Prediction/Training Data.csv'
    
    try:
        data = load_data(DATA_FILE_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data.drop(columns=['Risk_Flag']),
        data['Risk_Flag'],
        test_size=0.2,
        random_state=42,
        stratify=data['Risk_Flag']
    )

    # Risk maps
    state_map_train, city_map_train = calculate_risk_maps(pd.concat([X_train_raw, y_train], axis=1))

    # Transform training and test sets
    X_train = apply_transformations(X_train_raw, state_map_train, city_map_train, is_training=True)
    X_test = apply_transformations(X_test_raw, state_map_train, city_map_train, is_training=False)

    # Ensure same columns
    TRAINING_COLUMNS = X_train.columns.tolist()
    X_train = X_train.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    X_test = X_test.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    # Train model
    model = train_model(X_train, y_train)

    # --- Streamlit UI ---
    st.title("🏦 Loan Risk Prediction Model")
    st.markdown("---")
    st.markdown("### Applicant Details")

    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income", min_value=100000, value=3000000, step=100000)
        age = st.number_input("Age", min_value=18, max_value=85, value=35)
        experience_years = st.number_input("Total Work Experience (Years)", min_value=0, max_value=60, value=10)
        current_job_years = st.number_input("Years in Current Job", min_value=0, max_value=15, value=5)
        current_house_years = st.number_input("Years in Current House", min_value=0, max_value=20, value=12)

    with col2:
        married_single = st.selectbox("Marital Status", ["single", "married"])
        house_ownership_input = st.selectbox("House Ownership", ["rented", "owned", "norent_noown"])
        car_ownership = st.selectbox("Car Ownership", ["no", "yes"])
        profession_input = st.selectbox("Profession", list(PROFESSION_GROUPS.keys()))
        state_input = st.selectbox("State of Residence", sorted(state_map_train.index.tolist()))
        city_input = st.selectbox("City of Residence", sorted(city_map_train.index.tolist()))

    if st.button("Predict Loan Risk", type="primary"):
        input_data_raw = pd.DataFrame({
            'Income':[income],
            'Age':[age],
            'Experience':[experience_years],
            'Married/Single':[married_single],
            'House_Ownership':[house_ownership_input],
            'Car_Ownership':[car_ownership],
            'Profession':[profession_input],
            'CITY':[city_input],
            'STATE':[state_input],
            'CURRENT_JOB_YRS':[current_job_years],
            'CURRENT_HOUSE_YRS':[current_house_years],
        })
        # Apply transformations
        input_data_processed = apply_transformations(input_data_raw, state_map_train, city_map_train, is_training=False)
        input_data_final = input_data_processed.reindex(columns=TRAINING_COLUMNS, fill_value=0)

        # Predictions
        prediction_proba = model.predict_proba(input_data_final)[0]
        risk_prediction = model.predict(input_data_final)[0]

        risk_label = "HIGH RISK (Predicted Default)" if risk_prediction==1 else "LOW RISK (Predicted Safe)"
        risk_color = "red" if risk_prediction==1 else "green"

        st.markdown("### Prediction Result")
        st.markdown(f"The predicted risk level is: **<span style='color:{risk_color}; font-size: 24px;'>{risk_label}</span>**", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
            <p>Probability of Default (Risk=1): <strong>{prediction_proba[1]:.2%}</strong></p>
            <p>Probability of Safety (Risk=0): <strong>{prediction_proba[0]:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"Model Training Accuracy: **{accuracy_score(y_test, model.predict(X_test)): .2%}**")
