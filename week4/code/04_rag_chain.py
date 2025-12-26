"""
Module 4.2: Complete RAG Chain
==============================
Demonstrates building a complete RAG pipeline:
- Document ingestion
- Vector store creation
- Retrieval chain
- Question answering
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =============================================================================
# Sample Documentation
# =============================================================================

SAMPLE_DOCS = [
    {
        "content": """
# Getting Started Guide

Welcome to Acme Software! This guide will help you get up and running quickly.

## Installation

1. Download the installer from https://acme.example.com/download
2. Run the installer (acme-setup.exe on Windows, acme-setup.dmg on Mac)
3. Follow the installation wizard prompts
4. Launch Acme from your Applications folder

## First Steps

After installation, you'll need to:
1. Create an account or sign in
2. Complete the onboarding tutorial (about 5 minutes)
3. Connect your first data source

## System Requirements

- Windows 10/11 or macOS 10.15+
- 8GB RAM minimum (16GB recommended)
- 2GB disk space
- Internet connection required
        """,
        "source": "getting_started.md",
        "category": "onboarding"
    },
    {
        "content": """
# Account Management

## Password Reset

To reset your password:
1. Go to the login page
2. Click "Forgot Password"
3. Enter your email address
4. Check your email for a reset link (valid for 24 hours)
5. Click the link and create a new password

Password requirements:
- Minimum 12 characters
- At least one uppercase letter
- At least one number
- At least one special character

## Two-Factor Authentication (2FA)

We strongly recommend enabling 2FA:
1. Go to Settings > Security
2. Click "Enable Two-Factor Authentication"
3. Scan the QR code with your authenticator app
4. Enter the verification code to confirm

Supported authenticator apps:
- Google Authenticator
- Authy
- Microsoft Authenticator
        """,
        "source": "account.md",
        "category": "security"
    },
    {
        "content": """
# Billing and Subscriptions

## Pricing Plans

### Free Tier
- Up to 100 queries/month
- Basic features
- Community support

### Professional ($29/month)
- Unlimited queries
- Advanced features
- Email support
- API access

### Enterprise (Custom pricing)
- Everything in Professional
- Dedicated support
- Custom integrations
- SLA guarantee

## Refund Policy

We offer a 30-day money-back guarantee for all paid plans.

To request a refund:
1. Go to Settings > Billing
2. Click "Request Refund"
3. Select a reason (optional)
4. Confirm your request

Refunds are processed within 5-7 business days.
        """,
        "source": "billing.md",
        "category": "billing"
    },
    {
        "content": """
# API Documentation

## Authentication

All API requests require an API key. Get your key from Settings > API.

Include the key in the Authorization header:
```
Authorization: Bearer your_api_key_here
```

## Rate Limits

- Free: 100 requests/day
- Professional: 10,000 requests/day
- Enterprise: Unlimited

## Endpoints

### GET /api/v1/status
Returns the service status.

### POST /api/v1/query
Submit a query for processing.

Request body:
```json
{
  "query": "your query here",
  "options": {}
}
```

### GET /api/v1/results/{id}
Get results for a submitted query.
        """,
        "source": "api.md",
        "category": "technical"
    },
    {
        "content": """
# Troubleshooting Guide

## Common Issues

### "Connection Failed" Error
This usually means:
- Check your internet connection
- Verify our service status at status.acme.example.com
- Try disabling your VPN temporarily
- Check if a firewall is blocking the connection

### Slow Performance
If the application is running slowly:
- Close other applications to free up memory
- Check if background sync is running
- Try clearing the cache (Settings > Advanced > Clear Cache)
- Ensure you meet the minimum system requirements

### Login Issues
Can't log in?
- Verify you're using the correct email
- Reset your password using "Forgot Password"
- Check if 2FA is enabled and use the correct code
- Try a different browser or clear cookies

## Contact Support

If issues persist:
- Email: support@acme.example.com
- Live chat: Available 9am-5pm EST
- Phone: 1-800-ACME-HELP (Enterprise only)
        """,
        "source": "troubleshooting.md",
        "category": "support"
    }
]


# =============================================================================
# RAG Pipeline Components
# =============================================================================

def create_documents() -> List[Document]:
    """Create Document objects from sample data."""
    documents = []
    for doc in SAMPLE_DOCS:
        documents.append(Document(
            page_content=doc["content"],
            metadata={
                "source": doc["source"],
                "category": doc["category"]
            }
        ))
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 500) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks: List[Document]) -> FAISS:
    """Create a FAISS vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(vector_store: FAISS):
    """Create the complete RAG chain."""

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # RAG prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant for Acme Software.
Answer the user's question based ONLY on the following context.
If the context doesn't contain enough information to answer, say so honestly.
Always cite your sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer:
""")

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# =============================================================================
# Demo
# =============================================================================

def main():
    """Run the complete RAG demo."""
    print("=" * 60)
    print("Complete RAG Chain Demo")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[Step 1] Loading documents...")
    documents = create_documents()
    print(f"  Loaded {len(documents)} documents")

    # Step 2: Chunk documents
    print("\n[Step 2] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # Step 3: Create vector store
    print("\n[Step 3] Creating vector store...")
    vector_store = create_vector_store(chunks)
    print("  Vector store ready")

    # Step 4: Create RAG chain
    print("\n[Step 4] Building RAG chain...")
    rag_chain, retriever = create_rag_chain(vector_store)
    print("  RAG chain ready")

    # Step 5: Test queries
    print("\n" + "=" * 60)
    print("Testing RAG Chain")
    print("=" * 60)

    questions = [
        "How do I reset my password?",
        "What are the pricing plans?",
        "How do I get an API key?",
        "The app is running slowly, what should I do?",
        "Can I get a refund?",
        "What is the meaning of life?"  # Out of scope question
    ]

    for question in questions:
        print(f"\n{'─' * 60}")
        print(f"Q: {question}")
        print("─" * 60)

        # Get answer
        answer = rag_chain.invoke(question)
        print(f"A: {answer}")

        # Show retrieved sources
        docs = retriever.invoke(question)
        sources = [doc.metadata.get('source') for doc in docs]
        print(f"\nSources consulted: {', '.join(sources)}")

    # Step 6: Show vector store info
    print("\n" + "=" * 60)
    print("Vector Store Statistics")
    print("=" * 60)
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Categories: {set(d.metadata.get('category') for d in chunks)}")

    # Save vector store for later use
    print("\n[Optional] Saving vector store to disk...")
    vector_store.save_local("./acme_docs_store")
    print("  Saved to ./acme_docs_store")


if __name__ == "__main__":
    main()
