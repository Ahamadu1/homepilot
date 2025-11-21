import requests, pandas as pd
from config.settings import Settings

# Uses a typical Realtor endpoint on RapidAPI (you can swap to your chosen one)
BASE_URL = "https://realtor16.p.rapidapi.com/properties/list-for-sale"  # Example; adjust to your provider

def search_listings(city: str, state_code: str, min_price: int, max_price: int, beds: int) -> pd.DataFrame:
    s = Settings()
    headers = {
        "x-rapidapi-key": s.rapidapi_key,
        "x-rapidapi-host": BASE_URL.split("//")[1].split("/")[0]
    }
    params = {
        "city": city,
        "state_code": state_code,
        "limit": 30,
        "offset": 0,
        "price_min": min_price,
        "price_max": max_price,
        "beds_min": beds
    }
    try:
        r = requests.get(BASE_URL, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("RapidAPI error:", e)
        return pd.DataFrame()

    # Normalize response shape (adjust mapping to your API)
    items = []
    for it in (data.get("data") or data.get("properties") or []):
        items.append({
            "listing_id": it.get("property_id") or it.get("listing_id") or it.get("id"),
            "address": it.get("address") or it.get("location",{}).get("address"),
            "price": it.get("price") or it.get("list_price"),
            "beds": it.get("beds") or it.get("bedrooms"),
            "baths": it.get("baths") or it.get("bathrooms"),
            "desc": it.get("description") or "",
            "lat": (it.get("lat") or it.get("location",{}).get("lat")),
            "lon": (it.get("lon") or it.get("location",{}).get("lon")),
            "photo": (it.get("photo") or (it.get("photos") or [None])[0]),
        })
    df = pd.DataFrame(items).dropna(subset=["listing_id"]).drop_duplicates("listing_id")
    return df
