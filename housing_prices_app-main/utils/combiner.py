# combiner.py (utils/combiner.py)

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.city_path = os.path.join(base_path, '../data/cities.geojson')
        self.city_gdf = gpd.read_file(self.city_path)

    def add_nearest_cities(self, df):
        if 'lon' not in df.columns or 'lat' not in df.columns:
            raise ValueError("DataFrame must contain 'lon' and 'lat' columns")

        geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        geo_df.set_crs(epsg=4326, inplace=True)

        distances = geo_df.geometry.apply(lambda point: self.city_gdf.distance(point))
        nearest_city_indices = distances.idxmin(axis=1)
        geo_df['nearest_city'] = self.city_gdf.loc[nearest_city_indices, 'city'].values

        return geo_df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.add_nearest_cities(X)
