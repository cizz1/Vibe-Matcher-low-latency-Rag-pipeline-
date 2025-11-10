
# Nexora Vibe Matcher — AI-Powered Fashion Vibe Recommender

### Objective
This project implements a "Vibe Matcher" prototype — a semantic recommendation system that connects a user's vibe query (e.g., "energetic urban chic") to the most relevant fashion products using embeddings and cosine similarity.

The task was assigned as part of Nexora’s AI developer assessment to design a mini RAG-style retrieval system that ranks the top 3 most semantically similar fashion items for each query.

---

## How It Works

| Step | Process | Description |
|------|----------|-------------|
| 1 | Data Preparation | Created a dataset of 20 fashion items with descriptions and vibe tags (e.g., "boho", "urban", "chic", "relaxed"). |
| 2 | Embeddings | Used Hugging Face’s `BAAI/bge-large-en-v1.5` model to embed product descriptions (instead of OpenAI’s `text-embedding-ada-002`). |
| 3 | Vector Search | Stored embeddings locally in FAISS (Facebook AI Similarity Search) for efficient cosine similarity search. |
| 4 | Retrieval | Computed cosine similarities between query embeddings and stored vectors to return top-3 ranked matches. |
| 5 | Evaluation | Logged similarity scores, computed “good match” rate (similarity > 0.7), and plotted latency per query. |

---

## Why Hugging Face Instead of OpenAI

OpenAI’s embedding API (`text-embedding-ada-002`) was originally recommended, but OpenAI no longer provides free trial credits in India, so an equivalent open-source alternative was used.

The Hugging Face `BAAI/bge-large-en-v1.5` model was selected because:
- It performs on par or better than Ada on semantic similarity benchmarks.
- It’s freely accessible via the Hugging Face Inference API.
- It captures aesthetic and stylistic vibes (e.g., "romantic summer", "urban chic") better than smaller transformer models.

---

## Tech Stack

| Component | Tool |
|------------|------|
| Embedding Model | `BAAI/bge-large-en-v1.5` (Hugging Face Inference API) |
| Vector Database | FAISS (Local Index) |
| Language | Python |
| Libraries | `pandas`, `numpy`, `requests`, `matplotlib`, `scikit-learn`, `faiss-cpu` |
| Runtime | Jupyter / Google Colab / VS Code |

---

## Metrics & Results

| Metric | Result |
|--------|---------|
| Average Similarity (Best Match) | 0.73 |
| Queries ≥ 0.7 Similarity | 7 / 10 |
| Fallbacks Triggered | 3 / 10 |
| Average Latency (Excl. Warm-up) | ~2.2 seconds |
| Retrieval Speed (FAISS) | < 1 ms |

**Top examples:**
- "urban chic" → Street Hoodie (0.70)
- "luxury evening gown" → Evening Gown (0.77)
- "retro vintage nostalgia" → Vintage Cardigan (0.80)
- "party night sparkle dress" → Sequin Dress (0.79)

---
## Edge Cases Handled

1. **Low Similarity Handling**  
   If no cosine similarity exceeds 0.6, the system gracefully shows a fallback suggestion (top product by default) instead of returning empty results.
2. **Empty Query Validation**  
   The input is validated to ensure non-empty, meaningful text before embedding.
3. **API Resilience**  
   Errors from the Hugging Face API (rate limit, timeout) are caught with retry or fallback logic.
4. **Deterministic Ranking**  
   Products with identical scores are secondarily sorted alphabetically for stable results.
5. **Scalability Consideration**  
   FAISS and caching ensure performance even for repeated queries or degraded API conditions.


## Future Improvements

- Switch to local embedding inference to avoid network overhead.
- Use vector DBs like Pinecone or Weaviate for scalable cloud retrieval.
- Integrate session caching to skip re-embedding repeated queries.
- Deploy embedding and FAISS search clusters geographically close to users (reduce API round-trip time).

---

## Folder Structure
```
nexora-vibe-matcher
┣  faiss/
┃ ┣ bge-large-en-v1.5_index.faiss
┃ ┣ bge-large-en-v1.5_embeddings.pkl
┣ nexora_vibe_matcher.ipynb
┣ README.md
┣ analysis.md

```

---

## Key Insights
- Latency was dominated by the embedding API, not FAISS.
- FAISS retrieval remains microsecond-fast even on CPU.
- For production, local inference or Pinecone (with vector replication near deployment zones) would yield near real-time response times (<300 ms).

---

## Credits
- Model: BAAI/bge-large-en-v1.5 (Hugging Face)
- Vector Engine: FAISS



