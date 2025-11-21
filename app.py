import streamlit as st
import pandas as pd
from ml.embeddings import ListingEmbedder
from ml.qa_chain import HouseQA, ConversationalHouseQA
from ml.scorer import HouseScorer
from supabase import create_client
import os
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium

load_dotenv()

# Page config
st.set_page_config(
    page_title="🏠 AI House Shopping Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .property-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .price-tag {
        font-size: 1.5rem;
        color: #2E7D32;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'conv_qa' not in st.session_state:
    st.session_state.conv_qa = None
if 'listings' not in st.session_state:
    st.session_state.listings = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Load embeddings on startup
@st.cache_resource
def load_ai_system():
    """Load the AI system (embeddings + Q&A)"""
    try:
        embedder = ListingEmbedder()
        embedder.load_vectorstore()
        qa = HouseQA(embedder.vectorstore)
        conv_qa = ConversationalHouseQA(embedder.vectorstore)
        return embedder, qa, conv_qa
    except Exception as e:
        st.error(f"Error loading AI system: {e}")
        st.info("Run `python test_ai_system.py` first to create embeddings!")
        return None, None, None


# Load listings from Supabase
@st.cache_data
def load_listings():
    """Load all listings from Supabase"""
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        response = supabase.table("listings").select("*").execute()
        return response.data
    except Exception as e:
        st.error(f"Error loading listings: {e}")
        return []


# Initialize
if st.session_state.embedder is None:
    with st.spinner("🔄 Loading AI system..."):
        embedder, qa, conv_qa = load_ai_system()
        st.session_state.embedder = embedder
        st.session_state.qa = qa
        st.session_state.conv_qa = conv_qa

if not st.session_state.listings:
    with st.spinner("📊 Loading listings..."):
        st.session_state.listings = load_listings()

# Header
st.markdown('<h1 class="main-header">🏠 AI House Shopping Assistant</h1>', unsafe_allow_html=True)
st.markdown("### Find your perfect home with AI-powered search and recommendations")

# Sidebar - User Preferences
with st.sidebar:
    st.header("🎯 Your Preferences")

    # ====== LOCATION SELECTION & FETCH ======
    st.subheader("🌎 Search Location")

    fetch_mode = st.radio(
        "Fetch Mode",
        ["Use Existing Data", "Fetch New Listings"],
        horizontal=True
    )

    if fetch_mode == "Fetch New Listings":
        st.markdown("---")

        search_type = st.selectbox(
            "Search Type",
            ["Single City", "Multiple Cities", "Nationwide (30 Cities)"]
        )

        if search_type == "Single City":
            col1, col2 = st.columns(2)
            with col1:
                city = st.text_input("City", value="Austin")
            with col2:
                state = st.selectbox(
                    "State",
                    ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                     "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                     "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                     "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                     "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"],
                    index=43  # TX
                )

            if st.button("🔍 Fetch Listings", type="primary"):
                with st.spinner(f"Fetching listings for {city}, {state}..."):
                    from data_pipeline import fetch_redfin_listings, clean_listing, enrich_with_distances, \
                        save_to_supabase

                    # Fetch
                    raw = fetch_redfin_listings(city, state, limit=50)

                    if raw:
                        # Clean and enrich
                        cleaned = [clean_listing(l) for l in raw]

                        # Get city coordinates for downtown
                        from geopy.geocoders import Nominatim

                        geolocator = Nominatim(user_agent="house_assistant")
                        try:
                            location = geolocator.geocode(f"{city}, {state}")
                            if location:
                                poi_dict = {"downtown": (location.latitude, location.longitude)}
                                cleaned = enrich_with_distances(cleaned, poi_dict)
                        except:
                            pass

                        # Save
                        save_to_supabase(cleaned)

                        # Update session state
                        st.session_state.listings = cleaned

                        # Re-create embeddings
                        from ml.embeddings import ListingEmbedder

                        embedder = ListingEmbedder()
                        st.session_state.embedder = embedder
                        vectorstore = embedder.create_vectorstore(cleaned)

                        from ml.qa_chain import HouseQA, ConversationalHouseQA

                        st.session_state.qa = HouseQA(vectorstore)
                        st.session_state.conv_qa = ConversationalHouseQA(vectorstore)

                        st.success(f"✅ Loaded {len(cleaned)} properties from {city}, {state}!")
                        st.rerun()
                    else:
                        st.error(f"❌ No listings found for {city}, {state}")

        elif search_type == "Multiple Cities":
            cities_input = st.text_area(
                "Enter cities (one per line, format: City, STATE)",
                value="Austin, TX\nDallas, TX\nHouston, TX",
                height=150
            )

            if st.button("🔍 Fetch Multiple Cities", type="primary"):
                cities_list = [line.strip().split(",") for line in cities_input.split("\n") if line.strip()]

                with st.spinner(f"Fetching from {len(cities_list)} cities..."):
                    from data_pipeline import fetch_redfin_listings, clean_listing, enrich_with_distances, \
                        save_to_supabase
                    from geopy.geocoders import Nominatim
                    import time

                    all_listings = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    geolocator = Nominatim(user_agent="house_assistant")

                    for i, city_state in enumerate(cities_list):
                        if len(city_state) != 2:
                            continue

                        city, state = city_state[0].strip(), city_state[1].strip()
                        status_text.text(f"Fetching {city}, {state}... ({i + 1}/{len(cities_list)})")

                        try:
                            raw = fetch_redfin_listings(city, state, limit=30)
                            if raw:
                                cleaned = [clean_listing(l) for l in raw]

                                # Get coordinates
                                try:
                                    location = geolocator.geocode(f"{city}, {state}")
                                    if location:
                                        poi_dict = {"downtown": (location.latitude, location.longitude)}
                                        cleaned = enrich_with_distances(cleaned, poi_dict)
                                except:
                                    pass

                                all_listings.extend(cleaned)

                            time.sleep(0.5)  # Rate limiting
                        except:
                            pass

                        progress_bar.progress((i + 1) / len(cities_list))

                    if all_listings:
                        save_to_supabase(all_listings)
                        st.session_state.listings = all_listings

                        # Re-create embeddings
                        status_text.text("Creating AI embeddings...")
                        from ml.embeddings import ListingEmbedder
                        from ml.qa_chain import HouseQA, ConversationalHouseQA

                        embedder = ListingEmbedder()
                        vectorstore = embedder.create_vectorstore(all_listings)
                        st.session_state.embedder = embedder
                        st.session_state.qa = HouseQA(vectorstore)
                        st.session_state.conv_qa = ConversationalHouseQA(vectorstore)

                        st.success(f"✅ Loaded {len(all_listings)} properties from {len(cities_list)} cities!")
                        st.rerun()
                    else:
                        st.error("❌ No listings found")

        else:  # Nationwide
            st.info("This will fetch from 30 major US cities (~600 properties). Takes 2-3 minutes.")

            if st.button("🌎 Fetch Nationwide", type="primary"):
                with st.spinner("Fetching nationwide listings... this may take 2-3 minutes"):
                    import subprocess
                    import sys

                    # Run the nationwide script
                    result = subprocess.run(
                        [sys.executable, "data_pipeline.py", "nationwide"],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        # Reload data
                        supabase = create_client(
                            os.getenv("SUPABASE_URL"),
                            os.getenv("SUPABASE_KEY")
                        )
                        response = supabase.table("listings").select("*").execute()
                        st.session_state.listings = response.data

                        # Re-create embeddings
                        st.info("Creating AI embeddings for all properties...")
                        from ml.embeddings import ListingEmbedder
                        from ml.qa_chain import HouseQA, ConversationalHouseQA

                        embedder = ListingEmbedder()
                        vectorstore = embedder.create_vectorstore(st.session_state.listings)
                        st.session_state.embedder = embedder
                        st.session_state.qa = HouseQA(vectorstore)
                        st.session_state.conv_qa = ConversationalHouseQA(vectorstore)

                        st.success(f"✅ Loaded {len(st.session_state.listings)} nationwide properties!")
                        st.rerun()
                    else:
                        st.error("❌ Nationwide fetch failed. Check terminal for errors.")

        st.markdown("---")

    # City Filter for existing data
    if st.session_state.listings:
        st.subheader("📍 Filter by Location")
        all_cities = sorted(list(set([
            f"{l['city']}, {l['state']}"
            for l in st.session_state.listings
            if l.get('city') and l.get('state')
        ])))

        st.caption(f"{len(all_cities)} cities available")

        selected_cities = st.multiselect(
            "Show properties from:",
            options=["All"] + all_cities,
            default=["All"]
        )
    else:
        selected_cities = ["All"]

    st.markdown("---")

    max_price = st.slider(
        "Maximum Price",
        min_value=100000,
        max_value=5000000,
        value=1000000,
        step=50000,
        format="$%d"
    )

    min_beds = st.selectbox("Minimum Bedrooms", [1, 2, 3, 4, 5], index=1)
    min_baths = st.selectbox("Minimum Bathrooms", [1, 1.5, 2, 2.5, 3], index=1)

    st.markdown("---")
    st.subheader("📍 Priority Weights")
    st.caption("How important is each factor? (Total should = 100%)")

    price_weight = st.slider("Price", 0, 100, 30, 5)
    location_weight = st.slider("Location", 0, 100, 25, 5)
    size_weight = st.slider("Size", 0, 100, 20, 5)
    bedroom_weight = st.slider("Bedrooms", 0, 100, 15, 5)
    age_weight = st.slider("Age/Condition", 0, 100, 10, 5)

    total_weight = price_weight + location_weight + size_weight + bedroom_weight + age_weight
    if total_weight != 100:
        st.warning(f"⚠️ Weights sum to {total_weight}% (should be 100%)")

    preferences = {
        "max_price": max_price,
        "min_beds": min_beds,
        "min_baths": min_baths,
        "priorities": {
            "price": price_weight / 100,
            "location": location_weight / 100,
            "size": size_weight / 100,
            "bedrooms": bedroom_weight / 100,
            "age": age_weight / 100
        }
    }

    # Apply city filter to listings
    if "All" not in selected_cities and selected_cities:
        filtered_listings = [
            l for l in st.session_state.listings
            if f"{l.get('city', '')}, {l.get('state', '')}" in selected_cities
        ]
    else:
        filtered_listings = st.session_state.listings

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Matches", "💬 Ask AI", "🗺️ Map View", "📊 All Listings"])

# TAB 1: Top Matches (Scored & Ranked)
with tab1:
    st.header("🏆 Your Top Matches")
    st.caption("Houses ranked by your preferences using AI scoring")

    # Show filtered count
    if "All" not in selected_cities and selected_cities:
        st.info(f"📍 Showing properties from: {', '.join(selected_cities)}")
    else:
        st.info(f"📍 Showing all {len(st.session_state.listings)} properties nationwide")

    if st.button("🔄 Calculate Matches", type="primary"):
        with st.spinner("🤖 Scoring houses based on your preferences..."):
            # Filter and score listings
            scorer = HouseScorer()
            scored_listings = scorer.score_listings(
                filtered_listings,
                preferences
            )

            if scored_listings:
                st.success(f"✅ Found {len(scored_listings)} matching properties!")

                # Display top 10
                for listing in scored_listings[:10]:
                    with st.container():
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"### 🏠 Rank #{listing['rank']} - {listing['address']}")
                            st.markdown(f"<span class='price-tag'>${listing['price']:,}</span>", unsafe_allow_html=True)
                            st.write(f"📍 {listing['city']}, {listing['state']} {listing['zip']}")
                            st.write(
                                f"🛏️ {listing['beds']} beds | 🛁 {listing['baths']} baths | 📐 {listing['sqft']} sq ft")
                            if listing.get('year_built'):
                                st.write(f"🏗️ Built in {listing['year_built']}")

                        with col2:
                            st.metric("Match Score", f"{listing['score']:.0f}%")
                            if listing.get('url'):
                                st.link_button("View Details", listing['url'])

                        # Distances
                        if any(k.startswith('distance_to_') for k in listing.keys()):
                            dist_cols = st.columns(3)
                            for i, (key, val) in enumerate(
                                    [k for k in listing.items() if k[0].startswith('distance_to_')]):
                                if val:
                                    place = key.replace('distance_to_', '').replace('_', ' ').title()
                                    dist_cols[i % 3].metric(f"📍 {place}", f"{val} mi")

                        st.markdown("---")
            else:
                st.warning("No properties match your criteria. Try adjusting filters!")

# TAB 2: Ask AI
with tab2:
    st.header("💬 Chat with AI Assistant")
    st.caption("Ask natural language questions about the properties")

    # Chat mode selector
    chat_mode = st.radio(
        "Chat Mode",
        ["Single Questions", "Conversation (with memory)"],
        horizontal=True
    )

    # Chat interface
    question = st.text_input(
        "Ask anything about the houses:",
        placeholder="e.g., 'Show me affordable houses near downtown with parking'"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary")
    with col2:
        if chat_mode == "Conversation (with memory)":
            if st.button("Clear History"):
                st.session_state.conv_qa.clear_history()
                st.session_state.chat_history = []
                st.success("Chat history cleared!")

    if ask_button and question:
        with st.spinner("🤔 Thinking..."):
            if chat_mode == "Single Questions":
                result = st.session_state.qa.ask(question)
            else:
                result = st.session_state.conv_qa.ask(question)
                st.session_state.chat_history.append({"q": question, "a": result['answer']})

            # Display answer
            st.markdown("### 🤖 AI Answer:")
            st.info(result['answer'])

            # Show sources
            st.markdown("### 📚 Based on these properties:")
            for i, doc in enumerate(result['sources'][:3], 1):
                with st.expander(f"{i}. {doc.metadata['address']} - ${doc.metadata['price']:,}"):
                    st.write(doc.page_content)

    # Show chat history for conversation mode
    if chat_mode == "Conversation (with memory)" and st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            st.markdown(f"**You:** {chat['q']}")
            st.markdown(f"**AI:** {chat['a']}")
            st.markdown("---")

# TAB 3: Map View
with tab3:
    st.header("🗺️ Property Map")

    # Show count based on filter
    if "All" not in selected_cities and selected_cities:
        st.caption(f"Interactive map of properties in: {', '.join(selected_cities)}")
        map_listings = [l for l in filtered_listings if l.get('lat') and l.get('lon')]
    else:
        st.caption(f"Interactive map of all {len(st.session_state.listings)} properties nationwide")
        map_listings = [l for l in st.session_state.listings if l.get('lat') and l.get('lon')]

    if map_listings:
        # Calculate map center
        avg_lat = sum(l['lat'] for l in map_listings) / len(map_listings)
        avg_lon = sum(l['lon'] for l in map_listings) / len(map_listings)

        # Create map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

        # Add markers
        for listing in map_listings:
            popup_html = f"""
            <b>{listing['address']}</b><br>
            <b>${listing['price']:,}</b><br>
            {listing['beds']} bd | {listing['baths']} ba<br>
            {listing['sqft']} sq ft
            """

            folium.Marker(
                location=[listing['lat'], listing['lon']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"${listing['price']:,}",
                icon=folium.Icon(color='green', icon='home', prefix='fa')
            ).add_to(m)

        # Display map
        st_folium(m, width=1200, height=600)
    else:
        st.warning("No properties with valid coordinates to display.")

# TAB 4: All Listings
with tab4:
    st.header("📊 All Listings")

    # Show filtered count
    display_listings = filtered_listings if "All" not in selected_cities and selected_cities else st.session_state.listings
    st.caption(f"Showing {len(display_listings)} properties")

    # Convert to DataFrame
    if display_listings:
        df = pd.DataFrame(display_listings)

        # Select columns to display
        display_cols = ['address', 'city', 'price', 'beds', 'baths', 'sqft', 'year_built', 'property_type']
        display_df = df[[col for col in display_cols if col in df.columns]]

        # Format price
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:,}" if x else "N/A")

        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="house_listings.csv",
            mime="text/csv"
        )
    else:
        st.info("No listings available. Run the data pipeline first!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🏠 AI House Shopping Assistant | Built with LangChain, OpenAI & Streamlit"
    "</div>",
    unsafe_allow_html=True
)