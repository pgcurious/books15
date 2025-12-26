# Module 4.2: RAG - Retrieval-Augmented Generation

## What You'll Learn
- Understand why RAG solves the fundamental knowledge problem in LLMs
- Master the RAG pipeline: load, chunk, embed, store, retrieve, generate
- Build semantic search using vector databases
- Create a complete document Q&A system from scratch
- Apply quality patterns for accurate, grounded responses

---

## First Principles: The Knowledge Problem

### Why LLMs Need External Knowledge

Let's reason carefully about LLM limitations:

**First Principle #1:** LLMs have a knowledge cutoff.

```
Training Data Timeline:
├── Historical events: ✓ Knows
├── Recent events: ✗ Doesn't know
├── Your company docs: ✗ Never saw them
├── Private information: ✗ Not in training
└── Updated information: ✗ Frozen in time
```

**First Principle #2:** LLMs are pattern matchers, not knowledge bases.

An LLM doesn't "know" facts the way a database stores them. It has learned patterns that often produce correct-sounding answers. When knowledge is uncertain, it may:
- Hallucinate confidently
- Mix up similar concepts
- Fabricate plausible-sounding details

**First Principle #3:** The solution is retrieval, not training.

Fine-tuning an LLM on your data is expensive and inflexible. Instead:

```
Traditional Approach:          RAG Approach:
┌─────────────────────┐        ┌─────────────────────┐
│  Fine-tune LLM      │        │  LLM (unchanged)    │
│  on your data       │        │       +             │
│                     │        │  External knowledge │
│  Cost: $$$$$        │        │  Cost: $            │
│  Time: Days/weeks   │        │  Time: Minutes      │
│  Update: Retrain    │        │  Update: Re-index   │
└─────────────────────┘        └─────────────────────┘
```

---

## The RAG Mental Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INDEXING PHASE (Offline, done once)                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ Documents│───►│ Chunker  │───►│ Embedder │───►│  Vector Store    │   │
│  │ (PDF,    │    │ (Split   │    │ (Text →  │    │  (FAISS, Chroma, │   │
│  │  TXT,    │    │  into    │    │  Vector) │    │   Pinecone...)   │   │
│  │  MD...)  │    │  pieces) │    │          │    │                  │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘   │
│                                                                          │
│  QUERY PHASE (Online, per question)                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ Question │───►│ Embedder │───►│ Retriever│───►│  Relevant Chunks │   │
│  └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘   │
│                                                           │              │
│  ┌──────────────────────────────────────────────────────┐ │              │
│  │                         LLM                          │◄┘              │
│  │  "Based on the following context, answer the        │                │
│  │   question: {context} Question: {question}"         │                │
│  └──────────────────────────────────────────────────────┘                │
│                              │                                           │
│                              ▼                                           │
│                      ┌──────────────┐                                   │
│                      │   Grounded   │                                   │
│                      │   Answer     │                                   │
│                      └──────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Analogical Thinking: The Librarian

RAG works exactly like a skilled research librarian:

| Librarian | RAG Component |
|-----------|---------------|
| Library books | Document collection |
| Card catalog | Vector store index |
| Book chapters | Document chunks |
| "Books about..." | Semantic similarity search |
| Reading relevant passages | Context retrieval |
| Synthesizing an answer | LLM generation |
| Citing sources | Including references |
| "I found it in Volume 3, page 42" | Source attribution |

**The key insight:** A librarian doesn't memorize every book. They know how to find relevant information quickly. RAG gives your agent the same capability.

---

## Part 1: Document Loading

### Loading Different Document Types

```python
# See code/06_rag_basics.py for executable version

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)

# Load a single text file
def load_text_file(file_path: str):
    """Load a plain text file."""
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# Load a PDF
def load_pdf_file(file_path: str):
    """Load a PDF file (one document per page)."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()  # Returns list of documents, one per page
    return documents

# Load all files from a directory
def load_directory(dir_path: str, glob_pattern: str = "**/*.txt"):
    """Load all matching files from a directory."""
    loader = DirectoryLoader(
        dir_path,
        glob=glob_pattern,
        show_progress=True
    )
    documents = loader.load()
    return documents

# Load markdown files
def load_markdown(file_path: str):
    """Load a markdown file preserving structure."""
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    return documents
```

### Understanding Document Objects

```python
from langchain_core.documents import Document

# A document has two parts:
doc = Document(
    page_content="This is the actual text content...",
    metadata={
        "source": "report.pdf",
        "page": 1,
        "author": "Jane Smith",
        "date": "2024-01-15"
    }
)

# Metadata enables:
# - Source attribution
# - Filtering by date/author
# - Page number references
# - Custom categorization
```

---

## Part 2: Chunking Strategies

### Why Chunking Matters

**First Principle:** Embeddings work best on focused, coherent text.

A 100-page document as a single vector loses nuance. Chunking creates semantic units.

```
Document without chunking:        Document with chunking:
┌────────────────────────────┐    ┌───────────┐
│ Chapter 1: Introduction    │    │ Chunk 1   │ ← About topic A
│ Chapter 2: Methods         │    └───────────┘
│ Chapter 3: Results         │    ┌───────────┐
│ Chapter 4: Discussion      │    │ Chunk 2   │ ← About topic B
│ ...                        │    └───────────┘
│ References                 │    ┌───────────┐
└────────────────────────────┘    │ Chunk 3   │ ← About topic C
                                  └───────────┘
  Single vector for everything    Specific vector for each topic
  ↓                               ↓
  Poor retrieval precision        High retrieval precision
```

### Chunking Strategies

```python
# See code/06_rag_basics.py for executable version

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

# Strategy 1: Fixed size chunks (simple but naive)
def chunk_by_characters(documents, chunk_size=1000, overlap=200):
    """Split by character count with overlap."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)

# Strategy 2: Recursive splitting (respects structure)
def chunk_recursively(documents, chunk_size=1000, overlap=200):
    """Split recursively, preserving structure.

    Tries to split on:
    1. Paragraphs (\n\n)
    2. Lines (\n)
    3. Sentences (. )
    4. Words ( )
    5. Characters
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

# Strategy 3: Token-based (matches LLM processing)
def chunk_by_tokens(documents, chunk_size=500, overlap=50):
    """Split by token count (more accurate for LLMs)."""
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)
```

### Choosing the Right Chunk Size

| Chunk Size | Pros | Cons |
|------------|------|------|
| Small (200-500 tokens) | Precise retrieval | May lose context |
| Medium (500-1000 tokens) | Good balance | General purpose |
| Large (1000-2000 tokens) | More context | Less precise |

**Rule of thumb:** Start with 500-1000 tokens, adjust based on retrieval quality.

---

## Part 3: Embeddings

### What Are Embeddings?

**First Principle:** Computers can't understand meaning directly. Embeddings convert meaning to numbers.

```
Text:                           Embedding:
"The cat sat on the mat"   →    [0.12, -0.45, 0.78, 0.23, ...]

"A feline rested on the rug" → [0.11, -0.44, 0.77, 0.25, ...]
                                  ↑
                                Similar vectors! (Similar meaning)

"Stock prices rose today"   →   [-0.89, 0.34, -0.12, 0.67, ...]
                                  ↑
                                Different vector (different meaning)
```

### Creating Embeddings

```python
# See code/06_rag_basics.py for executable version

from langchain_openai import OpenAIEmbeddings
import numpy as np

def create_embeddings():
    """Create an embeddings model."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Fast and cost-effective
        # model="text-embedding-3-large"  # More accurate, higher cost
    )
    return embeddings

def demonstrate_embeddings():
    """Show how embeddings capture meaning."""
    embeddings = create_embeddings()

    # Embed some sentences
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "Stock prices rose sharply",
        "The dog played in the yard"
    ]

    vectors = embeddings.embed_documents(sentences)

    # Calculate similarity between all pairs
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("Similarity Matrix:")
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  {s1[:20]:20} vs {s2[:20]:20}: {sim:.3f}")

# Results show:
# "cat on mat" ↔ "feline on rug" = 0.92 (high similarity)
# "cat on mat" ↔ "stock prices" = 0.23 (low similarity)
```

---

## Part 4: Vector Stores

### What Is a Vector Store?

A vector store is a specialized database for embeddings:
- Stores vectors with their metadata
- Enables fast similarity search
- Scales to millions of vectors

### Using FAISS (Local, Free)

```python
# See code/07_vector_store.py for executable version

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_vector_store(documents: list[Document]) -> FAISS:
    """Create a FAISS vector store from documents."""
    embeddings = OpenAIEmbeddings()

    # Create vector store from documents
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vector_store

def save_vector_store(vector_store: FAISS, path: str):
    """Save vector store to disk."""
    vector_store.save_local(path)

def load_vector_store(path: str) -> FAISS:
    """Load vector store from disk."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
```

### Searching the Vector Store

```python
def search_similar(vector_store: FAISS, query: str, k: int = 4):
    """Find documents similar to a query."""

    # Method 1: Simple similarity search
    docs = vector_store.similarity_search(query, k=k)

    # Method 2: Search with relevance scores
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

    # Method 3: Maximum Marginal Relevance (diverse results)
    diverse_docs = vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=20  # Fetch more, then select diverse subset
    )

    return docs_with_scores

# Example usage
results = search_similar(vector_store, "How do I reset my password?")
for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print()
```

---

## Part 5: Building the Complete RAG Chain

### The Retrieval Chain

```python
# See code/08_rag_chain.py for executable version

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs) -> str:
    """Format retrieved documents for the prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def create_rag_chain(vector_store: FAISS):
    """Create a complete RAG chain."""

    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # RAG prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the following context.
    If the context doesn't contain enough information, say so.
    Always cite your sources using [1], [2], etc.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build the chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Usage
def ask_documents(rag_chain, question: str) -> str:
    """Ask a question and get a grounded answer."""
    return rag_chain.invoke(question)
```

### Complete Example: Document Q&A System

```python
# See code/08_rag_chain.py for executable version

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_qa_system(documents_path: str):
    """Build a complete Q&A system from a directory of documents."""

    print("Step 1: Loading documents...")
    loader = DirectoryLoader(
        documents_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"  Loaded {len(documents)} documents")

    print("Step 2: Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Step 3: Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"  Vector store ready")

    print("Step 4: Creating RAG chain...")
    rag_chain = create_rag_chain(vector_store)
    print("  RAG chain ready!")

    return rag_chain, vector_store

# Example session
if __name__ == "__main__":
    rag_chain, vector_store = build_qa_system("./documents")

    questions = [
        "What is the company's refund policy?",
        "How do I contact customer support?",
        "What are the main features of the product?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        answer = ask_documents(rag_chain, question)
        print(answer)
```

---

## Part 6: Advanced RAG Patterns

### Pattern 1: Query Enhancement

Sometimes the user's query isn't optimal for retrieval:

```python
# See code/08_rag_chain.py for executable version

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def enhance_query(original_query: str) -> str:
    """Rewrite query to improve retrieval."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Rewrite the following query to be more specific and likely to
    retrieve relevant information. Add relevant keywords and context.

    Original query: {query}

    Enhanced query:
    """)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": original_query})

# Example:
# Original: "reset password"
# Enhanced: "How to reset account password, password recovery process,
#           forgot password, change login credentials"
```

### Pattern 2: Hybrid Search

Combine semantic search with keyword matching:

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(documents, vector_store):
    """Create a retriever that combines semantic and keyword search."""

    # Semantic retriever (vector similarity)
    semantic_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    # Keyword retriever (BM25)
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = 4

    # Combine with equal weights
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.5, 0.5]  # Adjust based on use case
    )

    return hybrid_retriever
```

### Pattern 3: Self-Query (Filtering by Metadata)

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

def create_self_query_retriever(vector_store, llm):
    """Create a retriever that can filter by metadata."""

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The document source file",
            type="string"
        ),
        AttributeInfo(
            name="date",
            description="The document date",
            type="string"
        ),
        AttributeInfo(
            name="category",
            description="Document category: policy, technical, faq",
            type="string"
        )
    ]

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents="Company documentation",
        metadata_field_info=metadata_field_info
    )

    return retriever

# Now queries like "Show me policies from 2024" work automatically
```

### Pattern 4: Parent Document Retriever

Retrieve small chunks but return larger context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

def create_parent_document_retriever(documents, embeddings):
    """Retrieve with small chunks, return with full context."""

    # Store for parent documents
    store = InMemoryStore()

    # Child splitter (small, for precise retrieval)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # Parent splitter (larger, for context)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    vector_store = FAISS.from_documents([], embeddings)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    # Add documents
    retriever.add_documents(documents)

    return retriever
```

---

## Emergence Thinking: From Retrieval to Understanding

Watch how intelligence emerges from simple components:

```
Component Layer:
├── Embeddings: Convert text to vectors (mechanical)
├── Vector Store: Find similar vectors (mathematical)
├── Retriever: Return top matches (algorithmic)
└── LLM: Generate response (pattern matching)

Emergent Behavior:
├── Semantic understanding (not in any single component)
├── Context synthesis (combining multiple chunks)
├── Reasoning over sources (comparing/contrasting)
├── Citation and attribution (self-organizing)
└── Knowledge grounding (reducing hallucination)
```

**The profound insight:** No single component "understands" the documents. Understanding emerges from their interaction:

1. Embeddings capture **semantic proximity**
2. Retrieval surfaces **relevant context**
3. LLM synthesizes **coherent response**
4. Together they produce **grounded understanding**

---

## Common RAG Pitfalls and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Irrelevant retrieval | Poor chunking | Adjust chunk size, add overlap |
| Missing context | Chunks too small | Use parent document retriever |
| Hallucination | Weak prompt | Add "only use provided context" |
| Wrong sources cited | Generic prompt | Ask for specific citations |
| Slow retrieval | Large index | Use approximate search, add filters |
| High cost | Too many embeddings | Cache embeddings, batch queries |

---

## Summary

### What We Learned

1. **The Knowledge Problem**
   - LLMs have frozen knowledge
   - Fine-tuning is expensive and inflexible
   - RAG provides dynamic knowledge access

2. **The RAG Pipeline**
   - Load → Chunk → Embed → Store → Retrieve → Generate
   - Each step affects quality
   - Start simple, optimize iteratively

3. **Key Components**
   - Embeddings convert meaning to numbers
   - Vector stores enable fast similarity search
   - Retrievers surface relevant context
   - Prompts guide grounded generation

4. **Advanced Patterns**
   - Query enhancement improves retrieval
   - Hybrid search combines approaches
   - Self-query enables metadata filtering
   - Parent document provides more context

### The RAG Quality Formula

```
RAG Quality =
  Chunk Quality × Embedding Quality × Retrieval Precision × Prompt Quality

Improve any factor, improve the whole.
```

---

## Practice Exercises

### Exercise 1: Build a Documentation Q&A
Create a RAG system that:
- Ingests your project's README and docs
- Answers questions about your own code
- Cites specific files and sections

### Exercise 2: Compare Chunk Sizes
Experiment with:
- 200, 500, 1000, 2000 token chunks
- Measure retrieval quality
- Document trade-offs

### Exercise 3: Add Metadata Filtering
Enhance your RAG with:
- Document dates
- Categories
- Importance levels
- Self-query capability

---

## Next Steps

In [Module 4.3](03_production_data_pipelines.md), we'll build production-ready data pipelines:
- Multi-source integration
- Caching and performance
- Quality gates
- Monitoring and observability

---

*"RAG transforms an agent from someone who can only remember into someone who can research. That's not just a feature—it's a fundamental expansion of capability."*
