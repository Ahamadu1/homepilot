import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class HouseScorer:
    """
    Score and rank houses based on user preferences
    Uses weighted scoring across multiple criteria
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def score_listings(self, listings, user_preferences):
        """
        Score listings based on user preferences

        Args:
            listings: List of listing dicts
            user_preferences: Dict with max_price, min_beds, min_baths, priorities

        Example user_preferences:
        {
            "max_price": 500000,
            "min_beds": 3,
            "min_baths": 2,
            "priorities": {
                "price": 0.3,        # 30% weight
                "location": 0.25,    # 25% weight
                "size": 0.2,         # 20% weight
                "bedrooms": 0.15,    # 15% weight
                "age": 0.1           # 10% weight
            }
        }

        Returns:
            List of scored listings sorted by score (highest first)
        """
        if not listings:
            return []

        df = pd.DataFrame(listings)

        # Filter hard requirements
        df = df[
            (df["price"] <= user_preferences.get("max_price", float('inf'))) &
            (df["beds"] >= user_preferences.get("min_beds", 0)) &
            (df["baths"] >= user_preferences.get("min_baths", 0))
            ]

        if df.empty:
            return []

        # Calculate component scores (0-100 scale)
        scores = pd.DataFrame(index=df.index)
        priorities = user_preferences.get("priorities", {})

        # 1. Price Score (lower is better)
        if "price" in df.columns and priorities.get("price", 0) > 0:
            scores["price_score"] = 100 - self._normalize(df["price"]) * 100

        # 2. Location Score (closer to preferred location is better)
        if "distance_to_downtown" in df.columns and priorities.get("location", 0) > 0:
            scores["location_score"] = 100 - self._normalize(df["distance_to_downtown"]) * 100

        # 3. Size Score (bigger is better)
        if "sqft" in df.columns and priorities.get("size", 0) > 0:
            scores["size_score"] = self._normalize(df["sqft"]) * 100

        # 4. Bedroom Score (closer to target is better)
        if "beds" in df.columns and priorities.get("bedrooms", 0) > 0:
            target_beds = user_preferences.get("min_beds", 3)
            bed_diff = np.abs(df["beds"] - target_beds)
            scores["bedroom_score"] = 100 - self._normalize(bed_diff) * 100

        # 5. Age Score (newer is better)
        if "year_built" in df.columns and priorities.get("age", 0) > 0:
            current_year = 2024
            age = current_year - df["year_built"].fillna(1900)
            scores["age_score"] = 100 - self._normalize(age) * 100

        # Calculate weighted final score
        final_scores = np.zeros(len(df))
        for component in scores.columns:
            weight_key = component.replace("_score", "")
            weight = priorities.get(weight_key, 0)
            final_scores += scores[component].fillna(50) * weight

        df["score"] = final_scores.round(2)
        df["rank"] = df["score"].rank(ascending=False, method="dense").astype(int)

        # Sort by score (highest first)
        df = df.sort_values("score", ascending=False)

        return df.to_dict("records")

    def _normalize(self, series):
        """Normalize series to 0-1 range"""
        if series.std() == 0 or len(series) == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.min()) / (series.max() - series.min())


# ====== TEST ======
if __name__ == "__main__":
    # Test data
    sample_listings = [
        {
            "id": "1",
            "address": "123 Main St",
            "city": "Austin",
            "state": "TX",
            "zip": "78701",
            "price": 400000,
            "beds": 3,
            "baths": 2,
            "sqft": 2000,
            "year_built": 2015,
            "distance_to_downtown": 5,
            "lat": 30.27,
            "lon": -97.74
        },
        {
            "id": "2",
            "address": "456 Oak Ave",
            "city": "Austin",
            "state": "TX",
            "zip": "78702",
            "price": 350000,
            "beds": 4,
            "baths": 2.5,
            "sqft": 2200,
            "year_built": 2018,
            "distance_to_downtown": 3,
            "lat": 30.26,
            "lon": -97.72
        },
        {
            "id": "3",
            "address": "789 Elm St",
            "city": "Austin",
            "state": "TX",
            "zip": "78703",
            "price": 450000,
            "beds": 3,
            "baths": 2,
            "sqft": 1800,
            "year_built": 2010,
            "distance_to_downtown": 8,
            "lat": 30.28,
            "lon": -97.76
        }
    ]

    # Test preferences
    preferences = {
        "max_price": 500000,
        "min_beds": 3,
        "min_baths": 2,
        "priorities": {
            "price": 0.3,
            "location": 0.25,
            "size": 0.2,
            "bedrooms": 0.15,
            "age": 0.1
        }
    }

    # Score
    scorer = HouseScorer()
    scored = scorer.score_listings(sample_listings, preferences)

    print("🏆 SCORED LISTINGS:")
    print("=" * 50)
    for house in scored:
        print(f"\nRank #{house['rank']}: {house['address']}")
        print(f"  Score: {house['score']}")
        print(f"  Price: ${house['price']:,}")
        print(f"  Beds/Baths: {house['beds']}/{house['baths']}")
        print(f"  Distance to downtown: {house['distance_to_downtown']} mi")