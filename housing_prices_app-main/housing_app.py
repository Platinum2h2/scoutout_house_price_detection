import streamlit as st
st.set_page_config(page_title="ScoutOut AI: Housing Prices", page_icon=":house:")

import pandas as pd
import numpy as np
import pickle
import folium
import os
from geopy.geocoders import Nominatim
import geopy.distance
from streamlit_folium import st_folium
from utils.combiner import CombinedAttributesAdder

# Max width setup
def _max_width_(prcnt_width: int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
        <style> 
        .appview-container .main .block-container{{{max_width_str}}}
        </style>    
    """, unsafe_allow_html=True)

# Default random addresses
rand_addresses = [
    '1219 Carleton Street, Berkeley CA 94702',
    '24147 Clinton Court, Hayward CA 94545',
    '560 Penstock Drive, Grass Valley CA 95945',
    '1238 Roanwood Way, Concord CA 94521',
    '2807 Huxley Place, Fremont CA 94555',
    '441 Merritt Avenue, Oakland CA 94610',
    '3377 Sandstone Court, Pleasanton CA 94588',
    '2443 Sierra Nevada Road, Mammoth Lakes CA 93546'
]

def get_rand_addr(addresses):
    return np.random.choice(addresses, replace=False)

def initialize_session_states():
    if 'markers' not in st.session_state:
        st.session_state['markers'] = []
    if 'lines' not in st.session_state:
        st.session_state['lines'] = []
    if 'fg' not in st.session_state:
        st.session_state['fg'] = None
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'address' not in st.session_state:
        st.session_state['address'] = None
    if 'address_output' not in st.session_state:
        st.session_state['address_output'] = ""
    if 'location' not in st.session_state:
        st.session_state['location'] = None
    if 'random_values' not in st.session_state:
        random_rooms = np.random.randint(40, 600)
        random_bedrooms = np.random.randint(30, random_rooms)
        random_households = np.random.randint(0, 60) + random_bedrooms - np.random.randint(0, 20)
        random_addr = get_rand_addr(rand_addresses)
        st.session_state['random_values'] = dict(
            random_rooms=random_rooms,
            random_bedrooms=random_bedrooms,
            random_households=random_households,
            random_address=random_addr
        )

@st.cache_resource
def initialize_nominatim(user_agent=f'scoutout_ai_{np.random.randint(0,200)}'):
    with st.spinner('Initializing geolocator...'):
        return Nominatim(user_agent=user_agent)

@st.cache_resource
def load_model():
    with st.spinner('Loading model...'):
        model_path = os.path.join("model", "linear_reg_model.pkl")
        return pickle.load(open(model_path, 'rb'))

@st.cache_resource
def load_combiner():
    with st.spinner('Loading components...'):
        return CombinedAttributesAdder()

geolocator = initialize_nominatim()
loaded_model = load_model()
combiner = load_combiner()

def get_location(address: str):
    return geolocator.geocode(address, addressdetails=True)

def transform_data(data: pd.DataFrame):
    return combiner.add_nearest_cities(data)

def get_nearest_city(location):
    lon, lat = location.longitude, location.latitude
    data = pd.DataFrame(dict(lon=lon, lat=lat), index=[0])
    transformed = transform_data(data)
    return transformed['nearest_city'].values.squeeze()

def create_marker(m: folium.Map, location, icon_color='red', **kwargs):
    return folium.Marker(
        location=[location.latitude, location.longitude],
        icon=folium.Icon(color=icon_color),
        **kwargs
    )

def get_markers_addresses():
    return [m['address'] for m in st.session_state['markers']]

def link_two_markers(marker1, marker2, **kwargs):
    return folium.PolyLine(locations=(marker1.location, marker2.location), **kwargs)

def clear_markers():
    st.session_state['markers'] = []
    st.session_state['lines'] = []
    return folium.FeatureGroup('objects')

def create_map():
    map_ca = folium.Map(
        location=[37.7749, -122.4194], 
        zoom_start=6,
        min_lat=32.5295, min_lon=-124.4820, 
        max_lat=42.0095, max_lon=-114.1315, 
        no_wrap=True, max_bounds=True
    )
    return map_ca

# -----------------------------------------------------------------------------------
# UI Setup
_max_width_(70)
st.title("🏡 ScoutOut AI — California Housing Price Prediction")
st.markdown("""
##### Predict the median house value using AI.

This app uses machine learning to predict housing prices in California. Simply enter the property attributes and ScoutOut AI will do the rest!
""")

data = pd.read_csv('data/housing.csv')
max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

map_ca = create_map()
initialize_session_states()
st.session_state['fg'] = folium.FeatureGroup(name="objects", control=True)

rand_vals = st.session_state['random_values']
col1, col2 = st.columns([1, 2], gap='large')

with col1:
    st.header("📋 Input Housing Attributes")
    subcol1, subcol2 = st.columns(2)

    with subcol1:
        housing_median_age = np.nan
        total_rooms = st.number_input(
            "Total Rooms", 
            value=rand_vals['random_rooms'],
            min_value=int(min_values['total_rooms']),
            max_value=int(max_values['total_rooms']), step=5)
        total_bedrooms = st.number_input(
            "Total Bedrooms", 
            value=rand_vals['random_bedrooms'],
            min_value=int(min_values['total_bedrooms']), 
            max_value=int(max_values['total_bedrooms']), step=5)
        ocean_proximity = st.selectbox(
            'Ocean Proximity:',
            ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))
        population = np.nan

    with subcol2:
        households = st.number_input(
            "Households", 
            value=rand_vals['random_households'],
            min_value=int(min_values['households']), 
            max_value=int(max_values['households']), step=5)
        median_income = st.slider(
            "Median Income (in thousands USD)",
            value=2.0,
            min_value=float(min_values['median_income']), 
            max_value=float(max_values['median_income']), step=0.5)
        if st.button("🎲 Random Address"):
            rand_vals['random_address'] = get_rand_addr(rand_addresses)
            st.session_state['address_output'] = ""

    address = st.text_input("📍 Address", value=rand_vals['random_address'])
    if st.button("📌 Locate"):
        if address not in get_markers_addresses():
            location = get_location(address)
            st.session_state['address'] = address
            st.session_state['location'] = location
            if location:
                if location.raw['address']['state'] == 'California':
                    housing_coords = (location.latitude, location.longitude)
                    housing_marker = create_marker(map_ca, location, popup=location)
                    nearest_city = get_nearest_city(location)
                    nearest_city_loc = get_location(nearest_city + ", CA")
                    nearest_coords = (nearest_city_loc.latitude, nearest_city_loc.longitude)
                    dist_km = geopy.distance.distance(nearest_coords, housing_coords).km
                    st.session_state['address_output'] = f'Nearest City: {nearest_city} | Distance: {dist_km:.2f} km'
                    nearest_marker = create_marker(map_ca, nearest_city_loc, icon_color='green', popup=nearest_city)
                    line = link_two_markers(housing_marker, nearest_marker, tooltip=f'{dist_km:.2f} km')
                    st.session_state['markers'].append({'marker': housing_marker, 'address': address})
                    st.session_state['markers'].append({'marker': nearest_marker, 'address': "n_city_" + address})
                    st.session_state['lines'].append(line)
                else:
                    st.warning("Please enter an address in California.")
            else:
                st.error("Location not found.")
    st.write(st.session_state['address_output'])

    if st.button("🔮 Predict", use_container_width=True):
        if address not in get_markers_addresses():
            st.error("📍 Please locate the address first.")
        elif total_bedrooms > total_rooms:
            st.error("🛑 Total bedrooms cannot exceed total rooms.")
        else:
            location = st.session_state['location']
            input_df = pd.DataFrame([{
                "lon": location.longitude,
                "lat": location.latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }])
            prediction = loaded_model.predict(input_df).squeeze()
            st.session_state['prediction'] = prediction
            st.success("Prediction complete! 🎉")

    if st.session_state['prediction']:
        pred = st.session_state['prediction']
        st.markdown("""<style>[data-testid="stMetricValue"] { font-size: 34px; color: green; }</style>""", unsafe_allow_html=True)
        st.metric(label='🏠 Predicted Median House Value', value=f"$ {pred:.2f}")

with col2:
    for item in st.session_state["markers"]:
        st.session_state['fg'].add_child(item['marker'])
    for line in st.session_state["lines"]:
        st.session_state['fg'].add_child(line)
    if st.button("🧹 Clear Markers"):
        st.session_state['fg'] = clear_markers()
    st_folium(map_ca, width=1200, height=800, feature_group_to_add=st.session_state['fg'])
