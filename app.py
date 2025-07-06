
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import joblib

st.set_page_config(layout="wide", page_title="GNPOC Wells Advanced Analytics")

@st.cache_data
def load_data():
    df = pd.read_csv('GNPOC_Wells.csv', encoding='latin1')
    df = df.dropna(how='all').dropna(axis=1, how='all')
    cols = ['WELL NAME', 'WELL TYPE', 'LATITUDE', 'LONGITUDE', 
            'NORTHING', 'EASTING', 'OPERATOR', 'DATE', 'SURVEYOR', 'BLOCK #']
    df = df[cols].copy()

    def convert_coord(coord):
        try:
            if 'N' in str(coord):
                return float(str(coord).split('N')[0])
            elif 'E' in str(coord):
                return float(str(coord).split('E')[0])
            else:
                return float(coord)
        except:
            return np.nan

    df['LATITUDE'] = df['LATITUDE'].apply(convert_coord)
    df['LONGITUDE'] = df['LONGITUDE'].apply(convert_coord)
    df['WELL TYPE'] = df['WELL TYPE'].fillna('UNKNOWN').str.strip()
    df['OPERATOR'] = df['OPERATOR'].fillna('UNKNOWN').str.strip()
    df['BLOCK #'] = df['BLOCK #'].fillna('UNKNOWN').str.strip()
    df['DATE_DT'] = pd.to_datetime(df['DATE'], errors='coerce', dayfirst=True)
    df['YEAR'] = df['DATE_DT'].dt.year
    df['MONTH'] = df['DATE_DT'].dt.month
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE_DT']).reset_index(drop=True)
    return df

df = load_data()

@st.cache_resource
def train_models(df):
    type_df = df[df['WELL TYPE'].isin(['SUSP', 'ABND', 'D&A', 'Development', 'UNKNOWN'])]
    X_type = type_df[['LATITUDE', 'LONGITUDE', 'NORTHING', 'EASTING', 'YEAR']]
    y_type = type_df['WELL TYPE']
    le_type = LabelEncoder()
    y_type_encoded = le_type.fit_transform(y_type)
    X_train, X_test, y_train, y_test = train_test_split(X_type, y_type_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    history = {'accuracy': [], 'val_accuracy': []}

    coords = df[['LATITUDE', 'LONGITUDE']].values
    n_clusters = min(5, len(coords))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(coords)

    anomaly_features = df[['LATITUDE', 'LONGITUDE', 'YEAR']].dropna()
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(anomaly_features)

    ts_data = df.groupby('YEAR').size().reset_index(name='count')

    return {
        'nn_model': rf_model,
        'le_type': le_type,
        'scaler': scaler,
        'type_test': (X_test_scaled, y_test),
        'history': history,
        'kmeans': kmeans,
        'clusters': clusters,
        'iso_forest': iso_forest,
        'anomalies': anomalies,
        'ts_data': ts_data
    }

models = train_models(df)

st.title('GNPOC Wells Advanced Analytics Dashboard')
st.sidebar.header("Controls")
view_data = st.sidebar.checkbox("View Raw Data", True)
show_advanced = st.sidebar.checkbox("Show Advanced Analytics", False)

if view_data:
    st.header('Dataset Overview')
    st.write(f"Total wells: {len(df)}")
    st.dataframe(df.head())

tab1, tab2, tab3, tab4 = st.tabs(["Basic Analytics", "Predictive Models", "Anomaly Detection", "Time Series"])

with tab1:
    st.header('Basic Analytics')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Well Type Distribution')
        type_counts = df['WELL TYPE'].value_counts()
        fig = px.pie(type_counts, values=type_counts.values, names=type_counts.index)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Drilling Activity Over Time')
        year_counts = df['YEAR'].value_counts().sort_index()
        fig = px.line(year_counts, markers=True)
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Wells Drilled')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader('Operator Distribution')
        operator_counts = df['OPERATOR'].value_counts()
        fig = px.bar(operator_counts)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Block Distribution')
        block_counts = df['BLOCK #'].value_counts().head(10)
        fig = px.bar(block_counts)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader('Geographic Distribution')
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)
    for idx, row in df.iterrows():
        folium.Marker(
            [row['LATITUDE'], row['LONGITUDE']],
            popup=f"{row['WELL NAME']} ({row['WELL TYPE']}, {row['YEAR']})",
            icon=folium.Icon(color=['red', 'blue', 'green', 'purple', 'orange'][models['clusters'][idx]])
        ).add_to(m)
    folium_static(m, width=1200)

with tab2:
    st.header('Predictive Models')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Well Type Prediction (Random Forest)')
        lat = st.number_input('Latitude', value=float(df['LATITUDE'].mean()))
        lon = st.number_input('Longitude', value=float(df['LONGITUDE'].mean()))
        north = st.number_input('Northing', value=float(df['NORTHING'].mean()))
        east = st.number_input('Easting', value=float(df['EASTING'].mean()))
        year = st.number_input('Year', min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), value=int(df['YEAR'].mean()))
        if st.button('Predict Well Type'):
            input_data = [[lat, lon, north, east, year]]
            input_scaled = models['scaler'].transform(input_data)
            prediction = models['nn_model'].predict(input_scaled)
            well_type = models['le_type'].classes_[prediction[0]]
            st.success(f'Predicted Well Type: {well_type}')
            st.subheader('Model Performance')
            X_test, y_test = models['type_test']
            y_pred = models['nn_model'].predict(X_test)
            st.text(classification_report(y_test, y_pred, target_names=models['le_type'].classes_))
    with col2:
        st.subheader('Well Clustering Analysis')
        st.write('Wells clustered into 5 geographic groups:')
        fig = px.scatter(df, x='LONGITUDE', y='LATITUDE', color=models['clusters'], hover_name='WELL NAME',
                         hover_data=['WELL TYPE', 'YEAR'])
        st.plotly_chart(fig, use_container_width=True)
        if show_advanced:
            st.subheader('Cluster Characteristics')
            df['Cluster'] = models['clusters']
            cluster_stats = df.groupby('Cluster').agg({'YEAR': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'})
            st.dataframe(cluster_stats)
