# main.py
import json
import os
import argparse
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from analyzer import run_analysis
import ctypes
import sys

# Fix for "OSError: [WinError 6] The handle is invalid" on Windows
class SuppressStderr:
    def __enter__(self):
        self.original_stderr_fd = sys.stderr.fileno()
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.original_stderr_dup = os.dup(self.original_stderr_fd)
        ctypes.windll.kernel32.SetStdHandle(ctypes.c_ulong(-12), ctypes.c_void_p(self.devnull))
    def __exit__(self, exc_type, exc_val, exc_tb):
        ctypes.windll.kernel32.SetStdHandle(ctypes.c_ulong(-12), ctypes.c_void_p(self.original_stderr_dup))
        os.close(self.devnull)
        os.close(self.original_stderr_dup)

def load_models():
    """Loads both the SBERT and LLM models."""
    print("Loading all models for offline use...")
    sbert_model, llm = None, None
    try:
        sbert_path = 'models/sentence-transformer'
        sbert_model = SentenceTransformer(sbert_path)
        print("✅ SBERT model loaded.")

        llm_path = "models/llm/qwen2-0_5b-instruct-q4_k_m.gguf"
        if not os.path.exists(llm_path):
            raise FileNotFoundError(f"LLM not found at {llm_path}. Please run 'python download_models.py'.")
        
        with SuppressStderr():
            llm = Llama(model_path=llm_path, n_ctx=4096, n_threads=8, verbose=False)
        print("✅ LLM loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None
    return sbert_model, llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adobe Hackathon - RAG Pipeline")
    parser.add_argument('--input-dir', type=str, default='input')
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()

    sbert_model, llm = load_models()

    if sbert_model and llm:
        input_json_path = next((os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(".json")), None)
        if not input_json_path:
            print(f"Error: No input JSON file found in {args.input_dir}.")
        else:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            result = run_analysis(input_data, args.input_dir, sbert_model, llm)

            if result and "error" not in result and result.get("extracted_sections"):
                output_path = os.path.join(args.output_dir, "analysis_output.json")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Analysis complete. Result saved to: {output_path}")
            else:
                print(f"\n❌ Analysis failed or produced no results. Result: {result}")