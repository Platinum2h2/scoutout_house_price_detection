import streamlit as st
st.set_page_config(page_title="Housing Prices Prediction", page_icon=":house:")
import pandas as pd
import numpy as np
import pickle
import folium
from geopy.geocoders import Nominatim
import geopy.distance
from streamlit_folium import st_folium

## -----------------------------------------------------------------------------------------##
## Functions

def _max_width_(prcnt_width: int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
        <style> 
        .appview-container .main .block-container{{{max_width_str}}}
        </style>    
        """, unsafe_allow_html=True)

rand_addresses = [
    '1219 Carleton Street, Berkeley CA 94702','24147 Clinton Court, Hayward CA 94545',
    '560 Penstock Drive, Grass Valley CA 95945','1238 Roanwood Way, Concord CA 94521',
    '2807 Huxley Place, Fremont CA 94555','441 Merritt Avenue, Oakland CA 94610',
    '3377 Sandstone Court, Pleasanton CA 94588', '2443 Sierra Nevada Road, Mammoth Lakes CA 93546']

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
        random_households = np.random.randint(0, 60) + random_bedrooms -  np.random.randint(0, 20)
        random_addr = get_rand_addr(rand_addresses)
        st.session_state['random_values'] = dict(
            random_rooms=random_rooms,
            random_bedrooms=random_bedrooms,
            random_households=random_households,
            random_address=random_addr
        )

@st.cache_resource
def initialize_nominatim():
    return Nominatim(user_agent="housing_price_app")

@st.cache_resource
def load_model():
    return pickle.load(open(r'C:\Users\User\Desktop\Final_Internship_Project\model\linear_reg_model.pkl', 'rb'))

@st.cache_resource
def load_city_data():
    df = pd.read_csv(r'C:\Users\User\Desktop\Final_Internship_Project\utils\cal_cities_lat_long.csv')
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        'city': 'city',
        'latitude': 'latitude',
        'lat': 'latitude',
        'longitude': 'longitude',
        'lon': 'longitude'
    }
    df = df.rename(columns=rename_map)
    if not all(col in df.columns for col in ['city', 'latitude', 'longitude']):
        raise ValueError(f"Expected columns 'city', 'latitude', 'longitude' in city CSV, got {df.columns.tolist()}")
    return df

# Distance to nearest city

def get_nearest_city(lat, lon, city_df):
    city_df['distance'] = city_df.apply(lambda row: geopy.distance.distance((lat, lon), (row['latitude'], row['longitude'])).km, axis=1)
    nearest = city_df.loc[city_df['distance'].idxmin()]
    return nearest['city'], nearest['latitude'], nearest['longitude'], nearest['distance']

# Map markers

def create_marker(map_obj, lat, lon, text, color='red'):
    return folium.Marker(location=[lat, lon], popup=text, icon=folium.Icon(color=color))

def link_two_markers(latlon1, latlon2, **kwargs):
    return folium.PolyLine(locations=(latlon1, latlon2), **kwargs)

def clear_markers():
    st.session_state['markers'] = []
    st.session_state['lines'] = []
    return folium.FeatureGroup('objects')

# Create map

def create_map():
    return folium.Map(location=[37.7749, -122.4194], zoom_start=6)

## -------------------------------------------------------------------------------------------------------##
## Webpage

_max_width_(70)
st.title("California Housing Prices Prediction")
st.markdown("""
##### A web application for predicting California Housing Prices.

Uses a trained linear regression model based on housing data and nearest city proximity.
""")

model = load_model()
city_data = load_city_data()
data = pd.read_csv(r'C:\Users\User\Desktop\Final_Internship_Project\data\housing.csv')
max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

initialize_session_states()
geolocator = initialize_nominatim()
map_ca = create_map()
st.session_state['fg'] = folium.FeatureGroup(name="objects", control=True)

rand_vals = st.session_state['random_values']

# layout and input data
col1, col2 = st.columns([1, 2], gap='large')
with col1:
    st.header("Enter the attributes of the housing.")
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        total_rooms = st.number_input("Total Rooms", value=rand_vals['random_rooms'], min_value=int(min_values['total_rooms']), max_value=int(max_values['total_rooms']), step=5)
        total_bedrooms = st.number_input("Total Bedrooms", value=rand_vals['random_bedrooms'], min_value=int(min_values['total_bedrooms']), max_value=int(max_values['total_bedrooms']), step=5)
        ocean_proximity = st.selectbox("Ocean Proximity", ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))
    with subcol2:
        households = st.number_input("Households", value=rand_vals['random_households'], min_value=int(min_values['households']), max_value=int(max_values['households']), step=5)
        median_income = st.slider("Median income (in thousands)", value=2.0, min_value=float(min_values['median_income']), max_value=float(max_values['median_income']), step=0.5)
        if st.button("Random address"):
            rand_vals['random_address'] = get_rand_addr(rand_addresses)
            st.session_state['address_output'] = ""

    address = st.text_input("Address", value=rand_vals['random_address'])
    if st.button("Locate"):
        location = geolocator.geocode(address)
        if location and "California" in location.address:
            st.session_state['location'] = location
            lat, lon = location.latitude, location.longitude
            nearest_city, city_lat, city_lon, dist_km = get_nearest_city(lat, lon, city_data)
            st.session_state['address_output'] = f"Nearest City: {nearest_city} | Distance: {dist_km:.2f} km"
            st.session_state['markers'].append(create_marker(map_ca, lat, lon, "Property", color='red'))
            st.session_state['markers'].append(create_marker(map_ca, city_lat, city_lon, f"Nearest: {nearest_city}", color='green'))
            st.session_state['lines'].append(link_two_markers((lat, lon), (city_lat, city_lon), tooltip=f"{dist_km:.2f} km"))
        else:
            st.error("Address must be in California.")

    st.write(st.session_state['address_output'])
    if st.button("Predict"):
        if total_bedrooms > total_rooms:
            st.error("Bedrooms cannot exceed total rooms.")
        elif not st.session_state.get('location'):
            st.error("Please locate the property first.")
        else:
            loc = st.session_state['location']
            input_df = pd.DataFrame([{
                "longitude": loc.longitude,
                "latitude": loc.latitude,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }])
            prediction = model.predict(input_df)[0]
            st.metric(label='Predicted Median House Value', value=f"$ {prediction:,.2f}")

with col2:
    for m in st.session_state['markers']:
        st.session_state['fg'].add_child(m)
    for l in st.session_state['lines']:
        st.session_state['fg'].add_child(l)
    if st.button("Clear markers"):
        st.session_state['fg'] = clear_markers()
    st_folium(map_ca, width=1100, height=700, feature_group_to_add=st.session_state['fg'])