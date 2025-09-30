import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path):
    """Load COVID-19 data from a CSV file."""
    data = pd.read_csv(file_path, na_values="-")
    return data

@st.cache_data
def preprocess_data(data):
    """Preprocess the data by handling missing values and converting date columns."""
    if 'Date' in data.columns and 'Time' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data.drop(columns=['Time', 'Sno'], inplace=True, errors='ignore')    
    elif 'Updated On' in data.columns:
        data.rename(columns={'Updated On': 'Date'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data.dropna(subset=['Date'], inplace=True)
    elif 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data.dropna(subset=['Date'], inplace=True)
    
    if 'State/UnionTerritory' in data.columns:
        data.rename(columns={'State/UnionTerritory': 'State'}, inplace=True)
    
    if 'State' in data.columns:
        State = {
            "Telengana": "Telangana",
            "Himanchal Pradesh": "Himachal Pradesh",
            "Karanataka": "Karnataka",
            "Bihar****": "Bihar",
            "Madhya Pradesh***": "Madhya Pradesh",
            "Maharashtra***": "Maharashtra",
            "Cases being reassigned to states": "Unassigned",
            "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
            "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
            "A & N Islands": "Andaman and Nicobar Islands",
            "India": "All States/UTs"
        }

        data['State'] = data['State'].replace(State)
        data = data[data['State'] != 'Unassigned']

    for col in data.columns:
        if data[col].dtype == 'object' and col != 'State':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
        elif data[col].dtype in ['float64', 'Int64']:
            data[col] = data[col].fillna(0).astype('Int64')
    
    return data

def run_app():
    st.set_page_config(page_title="COVID-19 Prediction", page_icon="🦠",layout="wide")
    st.title("COVID-19 Prediction App")
    st.write("This app predicts COVID-19 cases using historical data.")

    try:
        df_cases = load_data("Covid_19/covid_19_india.csv")
        df_testing = load_data("Covid_19/StatewiseTestingDetails.csv")
        df_vaccination = load_data("Covid_19/covid_vaccine_statewise.csv")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    df_cases = preprocess_data(df_cases)
    df_testing = preprocess_data(df_testing)
    df_vaccination = preprocess_data(df_vaccination)

    st.subheader("COVID-19 Cases Data")

    latest_cases = df_cases[df_cases['State'] == 'All States/UTs'].sort_values(by='Date', ascending=False).iloc[0]
    latest_vaccine_india = df_vaccination[df_vaccination['State'] == 'All States/UTs'].sort_values(by='Date', ascending=False).iloc[0] if 'All States/UTs' in df_vaccination['State'].unique() else None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Confirmed Cases", f'{latest_cases["Confirmed"]:, .0f}')
    with col2:
        st.metric("Total Deaths", f'{latest_cases["Deaths"]:, .0f}')

    if latest_vaccine_india is not None:
        with col3:
            st.metric("Total Doses Administered", f'{latest_vaccine_india["Total Doses Administered"]:, .0f}')
    else:
        with col3:
            st.metric("Total Doses Administered", "Data not available")

    st.subheader("---")

    st.subheader("State-wise COVID-19 Cases Over Time")

    state_list = sorted([s for s in df_cases['State'].unique() if s != 'All States/UTs'])

    selected_state = st.selectbox(
        "Select a State/UT",
        state_list,
        index=state_list.index("Kerala") if "Kerala" in state_list else 0
    )

    df_state = df_cases[df_cases['State'] == selected_state].sort_values(by='Date')

    df_state['Active'] = df_state['Confirmed'] - df_state['Cured'] - df_state['Deaths']


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_state['Date'], df_state['Confirmed'], label='Confirmed Cases', color='blue')
    ax.plot(df_state['Date'], df_state['Cured'], label='Cured Cases', color='green')
    ax.plot(df_state['Date'], df_state['Active'], label='Active Cases', color='orange', linestyle=':')

    ax.set_title(f'COVID-19 Cases in {selected_state} Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Number of Cases', fontsize=14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("---")

    st.subheader("Latest Sate-wise Testing and Vaccination Summary")

    latest_testing = df_testing.sort_values(by='Date', ascending=False).groupby('State').first().reset_index()

    latest_vaccination = df_vaccination[df_vaccination['State'] != 'All States/UTs'].sort_values(by='Date', ascending=False).groupby('State').first().reset_index()

    display_df = latest_testing[['State', 'TotalSamples']].merge(
        latest_vaccination[['State', 'Total Doses Administered', 'Total Individuals Vaccinated']],
        on='State',
        how='outer'
    ).rename(columns={
        'TotalSamples': 'Latest Total Samples Tested',
        'Total Doses Administered': 'Latest Total Doses Administered',
        'Total Individuals Vaccinated': 'Latest Total Individuals Vaccinated'
    })

    st.dataframe(
        display_df.set_index('State')
            .fillna(0)
            .astype({'Latest Total Samples Tested': 'Int64', 'Latest Total Doses Administered': 'Int64', 'Latest Total Individuals Vaccinated': 'Int64'})
            .sort_values(by='Latest Total Samples Tested', ascending=False),
        use_container_width=True
    )


if __name__ == "__main__":
    
    run_app()
