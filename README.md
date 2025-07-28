# Adobe Hackathon: Persona-Driven Document Intelligence

[cite_start]This project is a solution for Round 1B of the Adobe "Connecting the Dots" Hackathon[cite: 2, 27]. [cite_start]It is an intelligent document analyst that processes a collection of PDFs to find the most relevant sections based on a user persona and a specific task[cite: 109].

## Approach Summary

The solution uses a multi-stage RAG pipeline that runs completely offline:
1.  **Query Generation**: A hybrid rule-based and LLM approach creates a focused search query.
2.  **Retrieval**: A Sentence Transformer (`all-MiniLM-L6-v2`) retrieves the top 20 relevant text chunks.
3.  **Pre-Filtering**: A smart filter removes obviously irrelevant content.
4.  **Re-Ranking**: A local LLM (`Qwen2-0.5B`) performs a final re-ranking to select the top 5 results.

## Models & Libraries

* **Models**:
    * `all-MiniLM-L6-v2` (for sentence similarity)
    * `Qwen/Qwen2-0.5B-Instruct-GGUF` (for re-ranking)
* **Key Libraries**:
    * `sentence-transformers`
    * `llama-cpp-python`
    * `PyMuPDF`
    * `torch`

## How to Build and Run

1.  **Download Models**:
    ```sh
    python download_model.py
    ```
2.  **Build the Docker Image**:
    ```sh
    docker build -t adobe-hackathon-solution .
    ```
3.  **Run the Solution**:
    *Place your input files (PDFs and one `.json` config file) in a folder named `my_input`.*
    *Create an empty folder named `my_output`.*

    ```sh
    docker run --rm -v $(pwd)/my_input:/app/input -v $(pwd)/my_output:/app/output adobe-hackathon-solution
    ```
    The results will be saved in the `my_output` folder.