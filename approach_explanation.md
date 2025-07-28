# Approach Explanation: Round-1.b

Our solution implements a multi-stage Retrieval-Augmented Generation (RAG) pipeline designed to act as an intelligent document analyst. [cite_start]It extracts and prioritizes the most relevant sections from a collection of documents based on a user's persona and specific job-to-be-done[cite: 109]. [cite_start]The entire process runs offline and is optimized for CPU execution[cite: 151].

### 1. Hybrid Query Generation

The pipeline begins by understanding the user's request. It first attempts to extract key search terms from the "job-to-be-done" using a fast, rule-based approach that filters out common stopwords and generic terms. If this initial extraction yields too few keywords, the system uses the Qwen2-0.5B LLM to generate more specific, actionable search terms related to the persona's task. This hybrid approach ensures both speed and high-quality search queries.

### 2. SBERT Broad Retrieval

With a focused search query, the system uses a highly efficient `all-MiniLM-L6-v2` Sentence Transformer model to perform a broad semantic search across all text chunks parsed from the document collection. This stage rapidly identifies the top 20 most relevant document chunks based on cosine similarity, forming a strong candidate pool for further analysis.

### 3. Rule-Based Pre-Filtering

Before involving the more powerful LLM, we apply a smart pre-filtering step. This function analyzes the user's request for contextual cues. For example, if the request is for a "group of college friends," the filter automatically removes any document chunks containing "family-friendly" or "children-related" themes. This step cleans the candidate pool, improving the efficiency and accuracy of the final stage.

### 4. LLM Re-Ranking and Final Output

Finally, the cleaned list of candidate chunks is passed to the local Qwen2-0.5B LLM. The LLM is prompted to act as the specified persona and re-rank the candidates based on the core job requirements. It selects the absolute best 5 results, ensuring relevance and avoiding topic repetition. This last step provides a highly refined and context-aware final output, which is then formatted into the required JSON structure.