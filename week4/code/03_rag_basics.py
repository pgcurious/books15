"""
Module 4.2: RAG Basics
======================
Demonstrates the fundamentals of Retrieval-Augmented Generation:
- Document loading
- Text chunking
- Creating embeddings
- Semantic similarity
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Some demos will be skipped.")
    HAS_API_KEY = False
else:
    HAS_API_KEY = True


# =============================================================================
# DEMO 1: Document Loading
# =============================================================================

def demo_document_loading():
    """Demonstrate loading documents from different sources."""
    print("=" * 60)
    print("DEMO 1: Document Loading")
    print("=" * 60)

    from langchain_core.documents import Document

    # Create sample documents manually (simulating loaded documents)
    documents = [
        Document(
            page_content="""
            Getting Started with Our Product

            Welcome to our product documentation. This guide will help you
            get started quickly with the main features.

            Installation:
            1. Download the installer from our website
            2. Run the installer and follow the prompts
            3. Launch the application from your applications folder
            """,
            metadata={"source": "getting_started.md", "section": "intro"}
        ),
        Document(
            page_content="""
            Frequently Asked Questions

            Q: How do I reset my password?
            A: Click on "Forgot Password" on the login screen and follow
            the instructions sent to your email.

            Q: How do I contact support?
            A: You can reach our support team at support@example.com or
            through the in-app chat feature.

            Q: What are the system requirements?
            A: You need Windows 10+, macOS 10.15+, or Ubuntu 20.04+.
            Minimum 4GB RAM and 500MB disk space.
            """,
            metadata={"source": "faq.md", "section": "support"}
        ),
        Document(
            page_content="""
            Billing and Subscriptions

            We offer three subscription tiers:
            - Basic: $9.99/month - Core features
            - Pro: $24.99/month - Advanced features + priority support
            - Enterprise: Custom pricing - Full features + dedicated support

            All plans include a 14-day free trial. Cancel anytime.
            Refunds are available within 30 days of purchase.
            """,
            metadata={"source": "billing.md", "section": "pricing"}
        )
    ]

    print(f"Loaded {len(documents)} documents\n")
    for doc in documents:
        print(f"Source: {doc.metadata['source']}")
        print(f"Section: {doc.metadata['section']}")
        print(f"Content preview: {doc.page_content[:100].strip()}...")
        print()

    return documents


# =============================================================================
# DEMO 2: Text Chunking
# =============================================================================

def demo_chunking():
    """Demonstrate different chunking strategies."""
    print("=" * 60)
    print("DEMO 2: Text Chunking Strategies")
    print("=" * 60)

    from langchain.text_splitter import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter
    )

    # Sample long text
    long_text = """
    Chapter 1: Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables
    systems to learn and improve from experience without being explicitly
    programmed. The field has grown significantly in recent years.

    There are three main types of machine learning:

    1. Supervised Learning: The algorithm learns from labeled training data.
    Examples include classification and regression tasks.

    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
    Examples include clustering and dimensionality reduction.

    3. Reinforcement Learning: The algorithm learns through trial and error,
    receiving rewards or penalties for its actions.

    Each type has its own use cases and is suited for different problems.
    The choice of approach depends on the available data and the goal.
    """

    # Strategy 1: Simple character splitting
    print("\n--- Character Splitter (chunk_size=200) ---")
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=50
    )
    char_chunks = char_splitter.split_text(long_text)
    for i, chunk in enumerate(char_chunks[:3]):
        print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:80]}...")
    print(f"Total chunks: {len(char_chunks)}")

    # Strategy 2: Recursive splitting (smarter)
    print("\n--- Recursive Splitter (chunk_size=200) ---")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_text(long_text)
    for i, chunk in enumerate(recursive_chunks[:3]):
        print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:80]}...")
    print(f"Total chunks: {len(recursive_chunks)}")

    return recursive_chunks


# =============================================================================
# DEMO 3: Embeddings
# =============================================================================

def demo_embeddings():
    """Demonstrate how embeddings capture semantic meaning."""
    if not HAS_API_KEY:
        print("\n--- Skipping embeddings demo (no API key) ---")
        return

    print("\n" + "=" * 60)
    print("DEMO 3: Embeddings and Semantic Similarity")
    print("=" * 60)

    from langchain_openai import OpenAIEmbeddings
    import numpy as np

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Test sentences
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "Dogs are great pets",
        "The stock market crashed today",
        "How do I reset my password?"
    ]

    print("\nEmbedding sentences...")
    vectors = embeddings.embed_documents(sentences)

    print(f"Embedding dimension: {len(vectors[0])}")

    # Calculate similarity matrix
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nSimilarity Matrix:")
    print("-" * 80)

    # Header
    print(f"{'':30}", end="")
    for i in range(len(sentences)):
        print(f"S{i+1:2}", end="   ")
    print()

    for i, s1 in enumerate(sentences):
        print(f"{s1[:28]:30}", end="")
        for j, s2 in enumerate(sentences):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"{sim:.2f}", end="  ")
        print()

    print("\nKey observations:")
    print("- S1 (cat/mat) and S2 (feline/rug) have HIGH similarity (similar meaning)")
    print("- S1 (cat) and S4 (stock market) have LOW similarity (different topics)")


# =============================================================================
# DEMO 4: Simple Vector Store
# =============================================================================

def demo_vector_store():
    """Demonstrate creating and querying a vector store."""
    if not HAS_API_KEY:
        print("\n--- Skipping vector store demo (no API key) ---")
        return

    print("\n" + "=" * 60)
    print("DEMO 4: Vector Store Operations")
    print("=" * 60)

    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    # Create sample documents
    documents = [
        Document(
            page_content="To reset your password, go to Settings > Security > Change Password",
            metadata={"source": "help.md", "topic": "password"}
        ),
        Document(
            page_content="Our refund policy allows returns within 30 days of purchase",
            metadata={"source": "policy.md", "topic": "refunds"}
        ),
        Document(
            page_content="Contact support at support@example.com or call 1-800-SUPPORT",
            metadata={"source": "contact.md", "topic": "support"}
        ),
        Document(
            page_content="Premium plans include priority support and advanced features",
            metadata={"source": "pricing.md", "topic": "plans"}
        ),
        Document(
            page_content="System requirements: Windows 10+, 4GB RAM, 500MB disk space",
            metadata={"source": "requirements.md", "topic": "technical"}
        )
    ]

    print("Creating vector store from documents...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"Vector store created with {len(documents)} documents")

    # Query the vector store
    queries = [
        "How do I change my password?",
        "Can I get my money back?",
        "What computer do I need?"
    ]

    print("\n--- Similarity Search Results ---")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vector_store.similarity_search_with_score(query, k=2)
        for doc, score in results:
            print(f"  [{score:.3f}] {doc.page_content[:60]}... (from {doc.metadata['source']})")

    return vector_store


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    documents = demo_document_loading()
    chunks = demo_chunking()
    demo_embeddings()
    demo_vector_store()

    print("\n" + "=" * 60)
    print("RAG Basics Demo Complete!")
    print("=" * 60)
