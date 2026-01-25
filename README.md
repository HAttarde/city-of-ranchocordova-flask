---
title: City of Rancho Cordova Chatbot
emoji: 🏙️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# City of Rancho Cordova Energy & Customer Service Chatbot

An AI-powered chatbot for Rancho Cordova residents, providing information about:
- ⚡ Energy efficiency and SMUD rates
- 📊 Data visualizations and charts
- 🏛️ City services and customer support

## Features

- **RAG-powered responses** using ChromaDB vector store
- **Auto-generated charts** with Chart.js
- **Web-scraped content** from city/utility websites
- **PDF document processing** for city reports

## Environment Variables

Set the following secret in your Space settings:
- `GROQ_API_KEY` - Your Groq API key (get free at https://console.groq.com)

## Tech Stack

- Flask backend
- Groq API (Llama 3.3 70B)
- Sentence Transformers for embeddings
- ChromaDB for vector storage
