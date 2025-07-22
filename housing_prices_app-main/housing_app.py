import streamlit as st
st.set_page_config(page_title="ScoutOut AI - Housing Prices", page_icon=":house:")

import os
import pandas as pd
import numpy as np
import pickle
import folium
from geopy.geocoders import Nominatim
import geopy.distance
from streamlit_folium import st_folium
from utils.combiner import CombinedAttributesAdder

# ----------------------------- Functions ----------------------------- #

def _max_width_(prcnt_width: int = 70):
    st.markdown(
        f"""<style> .appview-container .main .block-container{{max-width: {prcnt_width}%;}}</style>""",
        unsafe_allow_html=True,
    )

rand_addresses = [
    '1219 Carleton Street, Berkeley CA 94702', '24147 Clinton Court, Hayward CA 94545',
    '560 Penstock Drive, Grass Valley CA 95945', '1238 Roanwood Way, Concord CA 94521',
    '2807 Huxley Place, Fremont CA 94555', '441 Merritt Avenue, Oakland CA 94610',
    '3377 Sandstone Court, Pleasanton CA 94588', '2443 Sierra Nevada Road, Mammoth Lakes CA 93546'
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
def initialize_nominatim(user_agent=f'scoutout_housing_app_{np.random.randint(0,200)}'):
    with st.spinner('Initializing geolocator...'):
        return Nominatim(user_agent=user_agent)

@st.cache_resource
def load_model():
    with st.spinner('Loading model...'):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'model', 'linear_reg_model.pkl')
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at: {model_path}")
            st.stop()
        return pickle.load(open(model_path, 'rb'))

@st.cache_resource
def load_combiner():
    with st.spinner('Loading combiner...'):
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
    df = pd.DataFrame(dict(lon=lon, lat=lat), index=[0])
    transformed = transform_data(df)
    return transformed['nearest_city'].values.squeeze()

def create_marker(m: folium.Map, location, icon_color='red', **kwargs):
    coords = [location.latitude, location.longitude]
    return folium.Marker(location=coords, icon=folium.Icon(color=icon_color), **kwargs)

def get_markers_addresses():
    return [marker['address'] for marker in st.session_state['markers']]

def link_two_markers(marker1, marker2, **kwargs):
    return folium.PolyLine(locations=(marker1.location, marker2.location), **kwargs)

def clear_markers():
    st.session_state['markers'] = []
    st.session_state['lines'] = []
    return folium.FeatureGroup('objects')

def create_map():
    map_ca = folium.Map(location=[37.7749, -122.4194], zoom_start=6, max_bounds=True)
    return map_ca

# ----------------------------- Page ----------------------------- #

_max_width_(70)
st.title("🏡 ScoutOut AI — Housing Price Predictor")
st.markdown("""
A California housing market prediction tool powered by machine learning and location-aware intelligence.
""")

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'housing.csv'))
max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

map_ca = create_map()
initialize_session_states()
st.session_state['fg'] = folium.FeatureGroup(name="objects", control=True)

rand_vals = st.session_state['random_values']

# --- UI ---
col1, col2 = st.columns([1, 2], gap='large')
with col1:
    st.header("🏠 Property Attributes")
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        housing_median_age = np.nan
        total_rooms = st.number_input("Total Rooms", value=rand_vals['random_rooms'],
                                      min_value=int(min_values['total_rooms']), max_value=int(max_values['total_rooms']), step=5)
        total_bedrooms = st.number_input("Total Bedrooms", value=rand_vals['random_bedrooms'],
                                         min_value=int(min_values['total_bedrooms']), max_value=int(max_values['total_bedrooms']), step=5)
        ocean_proximity = st.selectbox("Ocean Proximity", ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))
        population = np.nan

    with subcol2:
        households = st.number_input("Households", value=rand_vals['random_households'],
                                     min_value=int(min_values['households']), max_value=int(max_values['households']), step=5)
        median_income = st.slider("Median Income ($K)", value=2.0,
                                  min_value=float(min_values['median_income']),
                                  max_value=float(max_values['median_income']), step=0.5)
        if st.button("🔄 Generate Random Address"):
            rand_vals['random_address'] = get_rand_addr(rand_addresses)
            st.session_state['address_output'] = ""

    address = st.text_input("📍 Property Address", value=rand_vals['random_address'])
    if st.button("📌 Locate Address"):
        if address and address not in get_markers_addresses():
            location = get_location(address)
            st.session_state['location'] = location
            if location:
                if location.raw['address'].get('state') == 'California':
                    housing_coords = (location.latitude, location.longitude)
                    housing_marker = create_marker(map_ca, location, popup=location)
                    nearest_city = get_nearest_city(location)
                    nearest_city_loc = get_location(nearest_city + ", CA")
                    nearest_city_coords = (nearest_city_loc.latitude, nearest_city_loc.longitude)
                    distance_km = geopy.distance.distance(nearest_city_coords, housing_coords).km
                    st.session_state['address_output'] = f'Nearest City: {nearest_city} | Distance: {distance_km:.2f} km'
                    nearest_marker = create_marker(map_ca, nearest_city_loc, icon_color='green', popup=nearest_city)
                    line = link_two_markers(housing_marker, nearest_marker, tooltip=f"{distance_km:.2f} km")
                    st.session_state['markers'].extend([
                        {'marker': housing_marker, 'address': address},
                        {'marker': nearest_marker, 'address': f"n_city_{address}"}
                    ])
                    st.session_state['lines'].append(line)
                else:
                    st.warning("⚠️ Please enter an address within California.")
            else:
                st.error("❌ Address not found.")

    st.write(st.session_state['address_output'])

    if st.button("📈 Predict Price", use_container_width=True):
        if address not in get_markers_addresses():
            st.error("📌 Please locate the address before predicting.")
        elif total_bedrooms > total_rooms:
            st.error("❌ Total bedrooms cannot exceed total rooms.")
        else:
            loc = st.session_state['location']
            input_data = pd.DataFrame([{
                "lon": loc.longitude, "lat": loc.latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }])
            st.session_state['prediction'] = loaded_model.predict(input_data).squeeze()
            st.success("Prediction Complete!")

    if st.session_state['prediction']:
        st.markdown("<style>[data-testid='stMetricValue'] { font-size: 34px; color: green; }</style>", unsafe_allow_html=True)
        st.metric("Estimated House Value", f"${st.session_state['prediction']:.2f}")

with col2:
    for marker in st.session_state['markers']:
        st.session_state['fg'].add_child(marker['marker'])
    for line in st.session_state['lines']:
        st.session_state['fg'].add_child(line)
    if st.button("🧹 Clear Map"):
        st.session_state['fg'] = clear_markers()
    st_data = st_folium(map_ca, width=1200, height=800, feature_group_to_add=st.session_state['fg'])
