# download_models.py
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os

def download_all_models():
    """Downloads all required models for a fully offline RAG pipeline."""
    os.makedirs('models', exist_ok=True)
    try:
        # 1. Download the Sentence Transformer for fast retrieval
        sbert_model_name = 'all-MiniLM-L6-v2'
        sbert_save_path = 'models/sentence-transformer'
        print(f"Downloading ranking model: {sbert_model_name}...")
        sbert = SentenceTransformer(sbert_model_name)
        sbert.save(sbert_save_path)
        print(f"✅ Ranking model saved to '{sbert_save_path}'.")

        # 2. Download the LLM for the re-ranking stage
        llm_repo_id = "Qwen/Qwen2-0.5B-Instruct-GGUF"
        llm_filename = "qwen2-0_5b-instruct-q4_k_m.gguf"
        llm_save_path = 'models/llm'
        print(f"\nDownloading LLM: {llm_filename} from {llm_repo_id}...")
        hf_hub_download(
            repo_id=llm_repo_id,
            filename=llm_filename,
            local_dir=llm_save_path,
            local_dir_use_symlinks=False
        )
        print(f"✅ LLM downloaded successfully to '{llm_save_path}'.")
        print("\n🎉 All models are ready for offline use.")
    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    download_all_models()