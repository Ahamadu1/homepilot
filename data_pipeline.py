import os
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time

load_dotenv()

# ====== CONFIG ======
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize geocoder
geolocator = Nominatim(user_agent="house_assistant_v1")


# ====== REALTY IN US API (via RapidAPI) ======
def fetch_redfin_listings(city, state, limit=20, min_price=None, max_price=None, beds=None):
    """
    Fetch listings from Realty in US API via RapidAPI
    """
    url = "https://realty-in-us.p.rapidapi.com/properties/v3/list"

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    # Use ZIP code format for better results - you can look up ZIP for cities
    # or use city, state format
    payload = {
        "limit": limit,
        "offset": 0,
        "postal_code": "78701",  # Austin, TX downtown ZIP (we'll make this dynamic later)
        "status": ["for_sale", "ready_to_build"],
        "sort": {
            "direction": "desc",
            "field": "list_date"
        }
    }

    if min_price:
        payload["price_min"] = min_price
    if max_price:
        payload["price_max"] = max_price
    if beds:
        payload["beds_min"] = beds

    try:
        print(f"🔍 Fetching listings for {city}, {state}...")
        print(f"📡 Using endpoint: {url}")

        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()

        data = response.json()

        # Print response structure for debugging
        print(f"📦 Response keys: {list(data.keys())}")

        # Realty in US API structure (based on Realtor.com)
        listings = data.get("data", {}).get("home_search", {}).get("results", [])

        if not listings:
            print(f"⚠️  No results in expected location. Full response structure:")
            print(f"   Top-level keys: {list(data.keys())}")
            if "data" in data:
                print(f"   Data keys: {list(data['data'].keys())}")

        print(f"✅ Found {len(listings)} listings")

        # Debug: print first listing structure if available
        if listings and len(listings) > 0:
            print(f"📋 Sample data keys: {list(listings[0].keys())[:10]}")

        return listings

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching listings: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text[:500]}")
        return []


# ====== CLEAN & STRUCTURE DATA ======
def clean_listing(listing):
    """
    Extract fields from Realty in US API response
    This API follows Realtor.com structure
    """
    # Get location data
    location = listing.get("location", {})
    address_data = location.get("address", {})
    coordinate = address_data.get("coordinate", {})

    # Get description data
    description = listing.get("description", {})

    # Get photos
    photos_raw = listing.get("photos", [])
    photos = []
    if isinstance(photos_raw, list):
        for photo in photos_raw[:5]:
            if isinstance(photo, dict):
                photos.append(photo.get("href", ""))
            else:
                photos.append(str(photo))

    return {
        "id": str(listing.get("property_id", "")),
        "address": address_data.get("line", ""),
        "city": address_data.get("city", ""),
        "state": address_data.get("state_code", ""),
        "zip": address_data.get("postal_code", ""),
        "lat": coordinate.get("lat"),
        "lon": coordinate.get("lon"),
        "price": listing.get("list_price"),
        "beds": description.get("beds"),
        "baths": description.get("baths"),
        "sqft": description.get("sqft"),
        "lot_sqft": description.get("lot_sqft"),
        "year_built": description.get("year_built"),
        "property_type": description.get("type"),
        "description": description.get("text", "")[:1000],  # Limit length
        "photos": photos,
        "url": listing.get("href", ""),
        "raw_data": listing
    }


# ====== CALCULATE DISTANCES ======
def calculate_distance(listing_coords, target_coords):
    """
    Calculate distance in miles between two lat/lon points
    """
    if not listing_coords[0] or not listing_coords[1]:
        return None
    return round(geodesic(listing_coords, target_coords).miles, 2)


def get_coordinates(address):
    """
    Geocode an address to get lat/lon
    """
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None


def enrich_with_distances(listings, poi_dict=None):
    """
    Add distance calculations to listings

    poi_dict example:
    {
        "downtown": (30.2672, -97.7431),
        "airport": (30.1975, -97.6664)
    }
    """
    if not poi_dict:
        return listings

    for listing in listings:
        if listing["lat"] and listing["lon"]:
            listing_coords = (listing["lat"], listing["lon"])

            for poi_name, poi_coords in poi_dict.items():
                distance = calculate_distance(listing_coords, poi_coords)
                listing[f"distance_to_{poi_name}"] = distance

    return listings


# ====== SAVE TO SUPABASE ======
def save_to_supabase(listings):
    """
    Insert listings into Supabase
    """
    if not listings:
        print("⚠️  No listings to save")
        return

    try:
        # Prepare data for insertion
        records = []
        for listing in listings:
            # Skip listings without required fields
            if not listing.get("id"):
                print(f"⚠️  Skipping listing without ID")
                continue

            # Remove fields that aren't in the table schema
            record = {
                "id": listing["id"],
                "address": listing["address"],
                "city": listing["city"],
                "state": listing["state"],
                "zip": listing["zip"],
                "lat": listing["lat"],
                "lon": listing["lon"],
                "price": listing["price"],
                "beds": listing["beds"],
                "baths": listing["baths"],
                "sqft": listing["sqft"],
                "lot_sqft": listing["lot_sqft"],
                "year_built": listing["year_built"],
                "property_type": listing["property_type"],
                "description": listing["description"],
                "photos": listing["photos"],
                "url": listing["url"],
                "raw_data": listing["raw_data"]
            }
            records.append(record)

        if not records:
            print("⚠️  No valid records to save")
            return

        # Insert (upsert to avoid duplicates)
        response = supabase.table("listings").upsert(records).execute()

        print(f"✅ Saved {len(records)} listings to Supabase")
        return response

    except Exception as e:
        print(f"❌ Error saving to Supabase: {e}")
        print(f"Error details: {str(e)}")
        return None


# ====== MAIN PIPELINE ======
def run_pipeline(city, state, limit=20, poi_dict=None):
    """
    Full data pipeline: Fetch → Clean → Enrich → Save
    """
    print("=" * 50)
    print(f"🏠 HOUSE DATA PIPELINE STARTED")
    print("=" * 50)

    # Step 1: Fetch listings
    raw_listings = fetch_redfin_listings(city, state, limit=limit)

    if not raw_listings:
        print("❌ No listings fetched. Check your API key and params.")
        return

    # Step 2: Clean data
    print(f"\n🧹 Cleaning {len(raw_listings)} listings...")
    cleaned = [clean_listing(l) for l in raw_listings]

    # Step 3: Enrich with distances
    if poi_dict:
        print(f"\n📍 Calculating distances to {len(poi_dict)} points of interest...")
        cleaned = enrich_with_distances(cleaned, poi_dict)

    # Step 4: Save to Supabase
    print(f"\n💾 Saving to database...")
    save_to_supabase(cleaned)

    print("\n" + "=" * 50)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 50)

    return cleaned


# ====== NATIONWIDE FETCH ======
def fetch_nationwide_listings(limit_per_city=20):
    """
    Fetch listings from multiple major US cities for nationwide coverage
    """
    # Top 30 US cities by population
    cities = [
        ("New York", "NY", (40.7128, -74.0060)),
        ("Los Angeles", "CA", (34.0522, -118.2437)),
        ("Chicago", "IL", (41.8781, -87.6298)),
        ("Houston", "TX", (29.7604, -95.3698)),
        ("Phoenix", "AZ", (33.4484, -112.0740)),
        ("Philadelphia", "PA", (39.9526, -75.1652)),
        ("San Antonio", "TX", (29.4241, -98.4936)),
        ("San Diego", "CA", (32.7157, -117.1611)),
        ("Dallas", "TX", (32.7767, -96.7970)),
        ("San Jose", "CA", (37.3382, -121.8863)),
        ("Austin", "TX", (30.2672, -97.7431)),
        ("Jacksonville", "FL", (30.3322, -81.6557)),
        ("Fort Worth", "TX", (32.7555, -97.3308)),
        ("Columbus", "OH", (39.9612, -82.9988)),
        ("Charlotte", "NC", (35.2271, -80.8431)),
        ("San Francisco", "CA", (37.7749, -122.4194)),
        ("Indianapolis", "IN", (39.7684, -86.1581)),
        ("Seattle", "WA", (47.6062, -122.3321)),
        ("Denver", "CO", (39.7392, -104.9903)),
        ("Washington", "DC", (38.9072, -77.0369)),
        ("Boston", "MA", (42.3601, -71.0589)),
        ("Nashville", "TN", (36.1627, -86.7816)),
        ("Detroit", "MI", (42.3314, -83.0458)),
        ("Portland", "OR", (45.5152, -122.6784)),
        ("Las Vegas", "NV", (36.1699, -115.1398)),
        ("Memphis", "TN", (35.1495, -90.0490)),
        ("Louisville", "KY", (38.2527, -85.7585)),
        ("Baltimore", "MD", (39.2904, -76.6122)),
        ("Milwaukee", "WI", (43.0389, -87.9065)),
        ("Albuquerque", "NM", (35.0844, -106.6504)),
    ]

    all_listings = []
    successful_cities = 0
    failed_cities = []

    print("\n" + "=" * 60)
    print("🌎 FETCHING NATIONWIDE LISTINGS")
    print("=" * 60)

    for i, (city, state, coords) in enumerate(cities, 1):
        print(f"\n[{i}/{len(cities)}] Fetching {city}, {state}...")

        try:
            # Fetch listings for this city
            raw_listings = fetch_redfin_listings(city, state, limit=limit_per_city)

            if raw_listings:
                # Clean listings
                cleaned = [clean_listing(l) for l in raw_listings]

                # Add POI (use city center as downtown)
                poi_dict = {"downtown": coords}
                enriched = enrich_with_distances(cleaned, poi_dict)

                all_listings.extend(enriched)
                successful_cities += 1
                print(f"   ✅ Got {len(enriched)} listings")
            else:
                print(f"   ⚠️  No listings found")
                failed_cities.append(f"{city}, {state}")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            failed_cities.append(f"{city}, {state}")
            continue

    print("\n" + "=" * 60)
    print(f"✅ COMPLETE!")
    print(f"   Total listings: {len(all_listings)}")
    print(f"   Successful cities: {successful_cities}/{len(cities)}")
    if failed_cities:
        print(f"   Failed cities: {', '.join(failed_cities[:5])}")
    print("=" * 60)

    return all_listings


# ====== TEST IT ======
if __name__ == "__main__":
    import sys

    # Check if nationwide flag is passed
    if len(sys.argv) > 1 and sys.argv[1] == "nationwide":
        print("🌎 Running NATIONWIDE fetch...")

        # Fetch from all cities
        all_listings = fetch_nationwide_listings(limit_per_city=20)

        if all_listings:
            # Save to Supabase
            print(f"\n💾 Saving {len(all_listings)} listings to database...")
            save_to_supabase(all_listings)

            print("\n✅ Nationwide data fetch complete!")
            print(f"   Total properties: {len(all_listings)}")
            print(f"   Ready to create embeddings with: python test_ai_system.py")
        else:
            print("❌ No listings fetched.")

    else:
        # Single city mode (original behavior)
        city = sys.argv[1] if len(sys.argv) > 1 else "Austin"
        state = sys.argv[2] if len(sys.argv) > 2 else "TX"

        poi_dict = {
            "downtown": (30.2672, -97.7431),
            "ut_austin": (30.2849, -97.7341),
            "airport": (30.1975, -97.6664)
        }

        listings = run_pipeline(
            city=city,
            state=state,
            limit=20,
            poi_dict=poi_dict
        )

        if listings:
            print(f"\n📊 Sample listing:")
            sample = listings[0]
            print(f"Address: {sample['address']}")
            print(f"Price: ${sample['price']:,}" if sample['price'] else "Price: N/A")
            print(f"Beds/Baths: {sample['beds']}/{sample['baths']}")
            if "distance_to_downtown" in sample:
                print(f"Distance to downtown: {sample['distance_to_downtown']} miles")