# Galactic Gadgets RAG Chatbot

Author: Alok Katiyar\
Course: ARIN 5360 -- AI Systems Engineering\
Project: Galactic Gadgets RAG Chatbot

---
# Pipeline Status
![CI Pipeline](https://github.com/alokkatiyar/AskGalacticGadgets/actions/workflows/ci.yml/badge.svg)\
![Python](https://img.shields.io/badge/python-3.12-blue)\
![FastAPI](https://img.shields.io/badge/FastAPI-RAG-green)\
![License](https://img.shields.io/badge/license-educational-lightgrey)


---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** for the fictional company **Galactic Gadgets**.

The system allows customers to ask questions about product documentation and receive **context-aware answers with citations**.

The chatbot integrates:

- document retrieval
- reranking
- LLM generation
- conversational chat interface
- source transparency

---

## RAG Pipeline

------------------------------------------------------------------------

# Overview

This project implements a **production-style Retrieval-Augmented
Generation (RAG) chatbot** for the fictional company **Galactic
Gadgets**.

Customers can ask natural language questions about company documentation
including:

-   Product specifications
-   Setup guides
-   Troubleshooting instructions
-   Technical documentation

Instead of returning raw search results, the system combines:

1.  Document retrieval
2.  LLM generation
3.  Chat interface

to produce contextual answers with **document citations**.

------------------------------------------------------------------------

# RAG Architecture

The chatbot uses a Retrieval-Augmented Generation pipeline.

User Question\
↓\
Semantic Search (vector embeddings)\
↓\
Reranking (improves relevance)\
↓\
Context Construction\
↓\
Prompt Creation\
↓\
LLM Generation (Ollama)\
↓\
Answer + Source Documents

------------------------------------------------------------------------

# Key Features

## Modern Chat Interface

The UI provides a conversational experience similar to ChatGPT.

Features include:

-   Message bubbles for user and assistant
-   Scrollable conversation history
-   Auto-scroll to latest message
-   Loading indicator ("Assistant is typing...")
-   Timestamps on messages
-   Export chat functionality
-   Clear conversation button
-   Responsive mobile/desktop layout

------------------------------------------------------------------------

## Source Transparency

Each assistant response includes a **collapsible source panel**
displaying:

-   Document filename
-   Similarity score
-   Content preview
-   Expandable full context

This improves **traceability and explainability**.

------------------------------------------------------------------------

## Chat Settings

Users can adjust response behavior:

  Setting              Description
  -------------------- ----------------------------------------
  Temperature          Controls creativity of responses
  Context Docs         Number of retrieved documents
  Theme Toggle         Light / dark mode
  Keyboard Shortcuts   Enter to send, Shift+Enter for newline

Theme preference persists using **localStorage**.

------------------------------------------------------------------------

# Optional Feature Implemented

## Option A -- Conversation Memory

The chatbot supports **multi-turn conversations**.

Recent conversation history is included in prompts sent to the LLM,
enabling follow‑up questions.

Example:

User: What is AstroLamp?\
Assistant: AstroLamp is a compact desk lamp...\
User: How do I change its brightness?

The system correctly understands that *"its" refers to AstroLamp*.

Conversation history:

-   Stored in memory
-   Cleared on page refresh
-   Can be cleared manually via UI

------------------------------------------------------------------------
## Project Structure

```text
AskGalacticGadgets/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
│
├── documents/
│   ├── faq.txt
│   ├── astrolamp.txt
│   └── nebulanoise.txt
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── chat.js
│
├── src/
│   └── retrieval/
│       ├── __init__.py
│       ├── main.py
│       ├── loader.py
│       ├── embeddings.py
│       ├── store.py
│       ├── retriever.py
│       ├── reranker.py
│       ├── llm.py
│       └── rag.py
│
└── tests/
    ├── __init__.py
    ├── test_smoke.py
    ├── test_llm.py
    ├── test_rag.py
    └── test_integration.py
```
------------------------------------------------------------------------

# Setup Instructions

## 1 Install dependencies

uv sync

------------------------------------------------------------------------

## 2 Install Ollama

Download from:

https://ollama.com/download

Verify installation:

ollama list

------------------------------------------------------------------------

## 3 Download the default model

ollama pull qwen2.5:3b

Verify Ollama is running:

curl http://localhost:11434/api/tags

------------------------------------------------------------------------

# Environment Configuration

Copy `.env.example`:

cp .env.example .env

Example configuration:

LLM_BASE_URL=http://localhost:11434\
LLM_MODEL=qwen2.5:3b\
DEFAULT_CONTEXT_DOCS=3\
DEFAULT_TEMPERATURE=0.7\
PORT=8081

------------------------------------------------------------------------

# Start the Server

uv run uvicorn src.retrieval.main:app --reload

Open the interface:

http://localhost:8081

------------------------------------------------------------------------

# Example Questions

Try asking:

What is RAG?

What features do NebulaNoise headphones support?

How do I adjust AstroLamp brightness?

Each response includes **document citations**.

------------------------------------------------------------------------

# Testing

Run tests with coverage:

uv run pytest --cov=src/retrieval tests/

Expected coverage: **≥ 80%**

Tests include:

-   LLM client tests
-   RAG system tests
-   Integration tests
-   Smoke tests

LLM calls are **mocked**, so tests pass even when Ollama is not running.

------------------------------------------------------------------------

# CI/CD Pipeline

GitHub Actions automatically runs on:

-   push
-   pull requests

Pipeline tasks:

✔ Ruff formatting check\
✔ Ruff linting\
✔ MyPy type checking\
✔ Pytest test suite\
✔ Coverage reporting

Workflow file:

.github/workflows/ci.yml

------------------------------------------------------------------------

# Comparison to Previous Work

### Compared to P2

P2 implemented the **retrieval pipeline**:

-   embeddings
-   semantic search
-   reranking

P3 adds:

-   conversational UI
-   LLM generation
-   source citations
-   configuration system

### Compared to Lab 6

Lab 6 introduced **basic RAG with Ollama**.

P3 improves this with:

-   conversation interface
-   settings panel
-   conversation management
-   improved prompts
-   error handling
-   CI/CD integration

------------------------------------------------------------------------

# Future Improvements

Possible enhancements:

-   streaming responses
-   OpenAI model integration
-   document upload support
-   citation highlighting
-   feedback system

------------------------------------------------------------------------

# Delivery Video

The delivery video demonstrates:

-   chat interface
-   RAG responses
-   source citations
-   theme toggle
-   settings
-   conversation memory
-   tests and coverage

Video length: **10--15 minutes**
