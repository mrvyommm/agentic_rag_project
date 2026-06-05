# Agentic RAG Project

## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) system using LangGraph, ChromaDB, Ollama, and local embedding models.

The system can:

* Accept user questions
* Retrieve relevant context from a vector database
* Evaluate whether the retrieved context is sufficient
* Rewrite retrieval queries when additional information is needed
* Generate grounded answers using retrieved context

## Architecture

Question
↓
Planner
↓
Retrieve
↓
Evaluate Context
├── Answer
└── Rewrite Query → Retrieve Again

## Technologies Used

* Python
* LangGraph
* Ollama
* ChromaDB
* Qwen Embeddings
* Gemma

## Features

* Local RAG pipeline
* ChromaDB vector storage
* Query rewriting loop
* Context evaluation node
* Conditional routing using LangGraph
* Fully local inference

## Models

Embedding Model:

* qwen3-embedding:0.6b

Generation Model:

* gemma4:e2b

## Run

```bash
python src/main.py
```

Example:

Please ask your question or enter /bye:
What is AI Engineering?

The system retrieves relevant chunks, evaluates context sufficiency, and generates a final answer.

## Future Improvements

* Tool Calling
* Web Search Integration
* Memory
* Multi-Agent Workflows
* Improved Retrieval Strategies
* Hybrid Search
