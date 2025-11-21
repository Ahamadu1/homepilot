import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = "./chroma_db"


class ListingEmbedder:
    """
    Handles embedding listings and storing them in Chroma vector database
    for semantic search
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"  # Latest, cheapest model
        )
        self.vectorstore = None

    def create_rich_text(self, listing):
        """
        Convert listing dict into rich text representation for better embeddings
        """
        # Format price nicely
        price_str = f"${listing['price']:,}" if listing['price'] else "Price not listed"

        # Build description
        parts = [
            f"Property at {listing['address']}, {listing['city']}, {listing['state']} {listing['zip']}",
            f"Listed at {price_str}",
            f"{listing['beds']} bedrooms, {listing['baths']} bathrooms",
            f"{listing['sqft']} square feet" if listing['sqft'] else "",
            f"Built in {listing['year_built']}" if listing['year_built'] else "",
            f"Property type: {listing['property_type']}" if listing['property_type'] else "",
            f"Lot size: {listing['lot_sqft']} sq ft" if listing['lot_sqft'] else "",
        ]

        # Add distances if available
        distance_parts = []
        for key, value in listing.items():
            if key.startswith("distance_to_") and value:
                place = key.replace("distance_to_", "").replace("_", " ").title()
                distance_parts.append(f"{value} miles to {place}")

        if distance_parts:
            parts.append("Distances: " + ", ".join(distance_parts))

        # Add property description
        if listing.get('description'):
            parts.append(f"\nDescription: {listing['description'][:500]}")  # Limit length

        # Join all parts
        text = "\n".join(filter(None, parts))
        return text.strip()

    def create_vectorstore(self, listings):
        """
        Convert listings to embeddings and store in Chroma

        Args:
            listings: List of listing dicts from data pipeline

        Returns:
            Chroma vectorstore instance
        """
        print(f"\n🧠 Creating embeddings for {len(listings)} listings...")

        documents = []
        for listing in listings:
            # Create rich text representation
            text = self.create_rich_text(listing)

            # Create LangChain Document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "id": listing["id"],
                    "address": listing["address"],
                    "city": listing["city"],
                    "state": listing["state"],
                    "price": listing["price"] or 0,
                    "beds": listing["beds"] or 0,
                    "baths": listing["baths"] or 0,
                    "sqft": listing["sqft"] or 0,
                    "url": listing.get("url", "")
                }
            )
            documents.append(doc)

        # Create Chroma vectorstore
        print(f"💾 Storing embeddings in Chroma...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )

        print(f"✅ Embeddings created and stored!")
        return self.vectorstore

    def load_vectorstore(self):
        """
        Load existing vectorstore from disk
        """
        print(f"📂 Loading existing vectorstore...")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )
        return self.vectorstore

    def search(self, query, k=5, filter_dict=None):
        """
        Search for relevant listings using natural language

        Args:
            query: Natural language search query
            k: Number of results to return
            filter_dict: Optional filters like {"price": {"$lt": 500000}}

        Returns:
            List of Document objects with scores
        """
        if not self.vectorstore:
            try:
                self.load_vectorstore()
            except:
                print("❌ No vectorstore found. Run create_vectorstore first!")
                return []

        print(f"🔍 Searching for: '{query}'")

        # Perform similarity search
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        print(f"✅ Found {len(results)} results")
        return results

    def search_with_scores(self, query, k=5):
        """
        Search with relevance scores
        """
        if not self.vectorstore:
            self.load_vectorstore()

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results


# ====== TEST FUNCTIONS ======
def test_embeddings(listings):
    """
    Test the embedding system with sample queries
    """
    embedder = ListingEmbedder()

    # Create vectorstore
    embedder.create_vectorstore(listings)

    # Test queries
    test_queries = [
        "affordable 3 bedroom house",
        "modern home near downtown",
        "house with large backyard",
        "property close to airport"
    ]

    print("\n" + "=" * 50)
    print("🧪 TESTING SEARCH QUERIES")
    print("=" * 50)

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = embedder.search(query, k=3)

        for i, doc in enumerate(results, 1):
            print(f"\n  {i}. {doc.metadata['address']}")
            print(f"     Price: ${doc.metadata['price']:,}")
            print(f"     {doc.metadata['beds']} beds, {doc.metadata['baths']} baths")


if __name__ == "__main__":
    # This would normally import from data_pipeline
    # For now, you'd run this after fetching listings
    print("Run this module after data_pipeline.py to embed listings!")
    print("\nExample usage:")
    print("  from ml.embeddings import ListingEmbedder")
    print("  embedder = ListingEmbedder()")
    print("  embedder.create_vectorstore(listings)")
    print("  results = embedder.search('affordable house near schools')")