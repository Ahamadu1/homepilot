import pandas as pd
from utils.helpers import haversine_km

# Example enrichment: distance to a reference point (e.g., city center)
NYC = (40.7128, -74.0060)

def enrich_listings(df: pd.DataFrame) -> pd.DataFrame:
    def _dist(row):
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")): return None
        return haversine_km((row["lat"], row["lon"]), NYC)
    df["distance_city_km"] = df.apply(_dist, axis=1)
    return df
