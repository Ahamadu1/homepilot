import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class HouseQA:
    """
    Question-answering system for house listings using LangChain
    Updated for LangChain 1.0+
    """

    def __init__(self, vectorstore):
        """
        Initialize Q&A chain with vectorstore

        Args:
            vectorstore: Chroma vectorstore from ListingEmbedder
        """
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        self.qa_chain = self._create_qa_chain()

    def _format_docs(self, docs):
        """Format documents for context"""
        return "\n\n".join([doc.page_content for doc in docs])

    def _create_qa_chain(self):
        """
        Create the Q&A chain with custom prompt (LangChain 1.0+ style)
        """
        template = """You are a helpful real estate assistant. Use the following property listings to answer the user's question.
If you don't know the answer, just say you don't have enough information. Don't make up information.

Context (Property Listings):
{context}

Question: {question}

Provide a helpful answer with specific property details (address, price, beds/baths) when relevant.

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create chain using LCEL (LangChain Expression Language)
        qa_chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return qa_chain

    def ask(self, question):
        """
        Ask a question about the house listings

        Args:
            question: Natural language question

        Returns:
            dict with 'answer' and 'sources' (the listings used)
        """
        print(f"\n❓ Question: {question}")

        # Get answer
        answer = self.qa_chain.invoke(question)

        # Get source documents separately
        sources = self.retriever.invoke(question)

        print(f"\n💬 Answer: {answer}")
        print(f"\n📚 Based on {len(sources)} properties:")
        for i, doc in enumerate(sources[:3], 1):
            print(f"   {i}. {doc.metadata['address']} - ${doc.metadata['price']:,}")

        return {
            "answer": answer,
            "sources": sources
        }

    def ask_with_filters(self, question, max_price=None, min_beds=None):
        """
        Ask a question with filters applied

        Args:
            question: Natural language question
            max_price: Maximum price filter
            min_beds: Minimum bedrooms filter
        """
        # Build filter dict
        filters = {}
        if max_price:
            filters["price"] = {"$lte": max_price}
        if min_beds:
            filters["beds"] = {"$gte": min_beds}

        # Create filtered retriever
        filtered_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": filters}
        )

        # Get sources and answer
        sources = filtered_retriever.invoke(question)
        context = self._format_docs(sources)

        template = f"""You are a helpful real estate assistant. Use the following property listings to answer the user's question.

Context (Property Listings):
{context}

Question: {question}

Answer:"""

        answer = self.llm.invoke(template).content

        return {
            "answer": answer,
            "sources": sources
        }


# ====== CONVERSATIONAL MEMORY (Bonus) ======
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class ConversationalHouseQA:
    """
    Q&A system with conversation memory (remembers previous questions)
    Updated for LangChain 1.0+
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        self.chat_history = InMemoryChatMessageHistory()
        self.qa_chain = self._create_conversational_chain()

    def _format_docs(self, docs):
        """Format documents for context"""
        return "\n\n".join([doc.page_content for doc in docs])

    def _create_conversational_chain(self):
        """Create conversational chain with memory"""
        template = """You are a helpful real estate assistant. Use the following property listings and conversation history to answer the user's question.

Context (Property Listings):
{context}

Conversation History:
{chat_history}

Current Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )

        def get_chat_history():
            messages = self.chat_history.messages
            if not messages:
                return "No previous conversation."
            return "\n".join([f"{m.type}: {m.content}" for m in messages])

        chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "chat_history": lambda x: get_chat_history(),
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return chain

    def ask(self, question):
        """
        Ask with conversation history
        """
        # Get answer
        answer = self.qa_chain.invoke(question)

        # Get sources
        sources = self.retriever.invoke(question)

        # Store in history
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(answer)

        return {
            "answer": answer,
            "sources": sources
        }

    def clear_history(self):
        """Clear conversation memory"""
        self.chat_history.clear()


# ====== TEST FUNCTIONS ======
def test_qa_system(vectorstore):
    """
    Test the Q&A system with sample questions
    """
    qa = HouseQA(vectorstore)

    print("\n" + "=" * 50)
    print("🧪 TESTING Q&A SYSTEM")
    print("=" * 50)

    test_questions = [
        "What's the cheapest house available?",
        "Show me houses with 3 or more bedrooms",
        "Which properties are closest to downtown?",
        "Are there any houses with large lot sizes?",
        "What's the average price of the listings?"
    ]

    for question in test_questions:
        qa.ask(question)
        print("\n" + "-" * 50)


def test_conversational_qa(vectorstore):
    """
    Test conversational Q&A with follow-up questions
    """
    qa = ConversationalHouseQA(vectorstore)

    print("\n" + "=" * 50)
    print("🧪 TESTING CONVERSATIONAL Q&A")
    print("=" * 50)

    # Conversation flow
    questions = [
        "Show me 3 bedroom houses",
        "Which one is closest to the airport?",
        "What's the price of that one?",
        "Does it have a big yard?"
    ]

    for q in questions:
        result = qa.ask(q)
        print(f"\n❓ {q}")
        print(f"💬 {result['answer']}\n")


if __name__ == "__main__":
    print("Run this after creating embeddings!")
    print("\nExample usage:")
    print("  from ml.embeddings import ListingEmbedder")
    print("  from ml.qa_chain import HouseQA")
    print("  ")
    print("  embedder = ListingEmbedder()")
    print("  embedder.load_vectorstore()")
    print("  ")
    print("  qa = HouseQA(embedder.vectorstore)")
    print("  qa.ask('Show me affordable houses near downtown')")