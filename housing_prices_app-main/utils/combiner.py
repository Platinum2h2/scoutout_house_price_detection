from sklearn.base import BaseEstimator, TransformerMixin
import geopandas as gpd
import pandas as pd
import os

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    DISTANCE_CRS = 3310
    CRS = 4326

    def __init__(self, add_bedrooms_per_room=True):
        print("✅ USING NEW CombinedAttributesAdder (CSV version)")
        self.add_bedrooms_per_room = add_bedrooms_per_room

        # Load city coordinates from CSV
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.city_path = os.path.join(base_path, '../data/cal_cities_lat_long.csv')

        california_cities = pd.read_csv(self.city_path)
        geometry = gpd.points_from_xy(california_cities['Longitude'], california_cities['Latitude'])
        self.points_cal = gpd.GeoDataFrame(california_cities, geometry=geometry)
        self.points_cal.set_crs(self.CRS, inplace=True)

        self._feature_names_out = None

    @staticmethod
    def dataframe_to_geo(data: pd.DataFrame):
        geometry = gpd.points_from_xy(data['lon'], data['lat'])
        data = gpd.GeoDataFrame(data).set_geometry(geometry).set_crs(CombinedAttributesAdder.CRS)
        return data

    def add_nearest_cities(self, data):
        if not isinstance(data, gpd.GeoDataFrame):
            data = self.dataframe_to_geo(data)

        nearest_cities = gpd.sjoin_nearest(
            data.to_crs(self.DISTANCE_CRS),
            self.points_cal.to_crs(self.DISTANCE_CRS),
            how='left',
            distance_col='distance_nearest_city'
        ).drop(columns=['index_right'])

        # Rename to generic label
        nearest_cities = nearest_cities.rename(columns={'Name': 'nearest_city'})  # Optional if 'Name' exists
        nearest_cities = nearest_cities[~nearest_cities.index.duplicated(keep='first')]

        return nearest_cities

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = self.dataframe_to_geo(X)

        joined_data = self.add_nearest_cities(data).to_crs(self.DISTANCE_CRS)

        X = joined_data.drop(columns=['geometry', 'lat', 'lon', 'Latitude', 'Longitude'], errors='ignore')
        X = X.assign(
            rooms_per_household=X.total_rooms / X.households,
            bedrooms_per_room=X.total_bedrooms / X.total_rooms if self.add_bedrooms_per_room else None
        )

        self._feature_names_out = X.columns.to_list()
        return X

    def get_feature_names_out(self, input_feature=None):
        return self._feature_names_out
