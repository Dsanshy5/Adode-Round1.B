import os
import re
import json
from datetime import datetime
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from pdf_processor import extract_meaningful_chunks

# ------------------------------------------
# Phase 1a: Extracting Focused Search Terms
# ------------------------------------------

# Common stopwords to filter out
STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "for", "to", "with", "by", "of", "and", "or", "as",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "our", "we", "i", "you", "they", "them", "us",
    "my", "your", "their", "our", "me", "him", "her", "it", "which", "what", "who", "whom",
    "where", "when", "why", "how", "there", "here", "then", "than", "so", "such", "some", "any",
    "all", "both", "each", "few", "more", "most", "other", "many", "much", "little", "own", "same",
    "so", "too", "very", "will", "would", "should", "can", "could", "may", "might", "must", "shall",
    "need", "dare", "ought", "let", "make", "like", "want", "try", "use", "take", "give", "get"
}

# Generic terms to exclude
GENERIC_TERMS = {
    "trip", "plan", "travel", "vacation", "itinerary", "planner", "days", "day", "group", "friends",
    "college", "people", "number", "size", "count", "quantity", "person", "individual", "party"
}

def extract_salient_terms(text: str, max_terms=8) -> str:
    """Extract only the most salient keywords using pure Python"""
    # Convert to lowercase
    text = text.lower()

    # Extract meaningful words (4+ characters)
    words = re.findall(r'\b[a-z]{4,}\b', text)

    # Filter out stopwords and generic terms
    filtered_words = [
        word for word in words
        if word not in STOPWORDS and word not in GENERIC_TERMS
    ]

    # Count term frequency
    term_counter = Counter(filtered_words)

    # Get most common terms
    return " ".join([term for term, _ in term_counter.most_common(max_terms)])

def get_search_terms_from_llm(persona: str, job: str, llm) -> str:
    """
    Generates search terms focused exclusively on actionable entities
    """
    print("Phase 1a: Extracting key search terms...")

    # First try pure extraction
    extracted_terms = extract_salient_terms(job)
    print(f"🔍 Extracted terms: {extracted_terms}")

    # Only use LLM if we have too few terms
    if len(extracted_terms.split()) < 3:
        print("⚠️ Few terms - using LLM enhancement")
        try:
            prompt = f"""
Based on this travel planning request: "{job}",
list ONLY the 5 most specific activity/experience types that would be relevant.

Exclude:
- Generic terms like 'travel', 'plan', or 'days'
- Group descriptors like 'friends' or 'students'
- Numbers or quantities

Example Output: beaches, nightlife, adventure sports, local cuisine, cultural sites

Specific Activities: """
            output = llm(prompt, max_tokens=100, temperature=0.0, stop=["\n"])
            llm_terms = output['choices'][0]['text'].strip()

            # Clean LLM response
            cleaned_terms = re.sub(r'\d+[\.\)]|\b(and|or)\b|[^a-zA-Z,\s]', '', llm_terms, flags=re.IGNORECASE)
            cleaned_terms = cleaned_terms.replace(',', ' ').replace('  ', ' ').strip()

            # Combine with extracted terms
            combined = f"{extracted_terms} {cleaned_terms}"[:100]
            print(f"✅ Combined terms: '{combined}'")
            return combined
        except Exception as e:
            print(f"⚠️ LLM enhancement failed: {e}")

    return extracted_terms

def fallback_keyword_extraction(job: str) -> str:
    """
    Fallback method to extract basic keywords from job description.
    """
    core_terms = []

    duration_match = re.search(r'(\d+)\s*(day|week|month)', job, re.IGNORECASE)
    if duration_match:
        core_terms.append(f"{duration_match.group(1)} {duration_match.group(2)}")

    if "friends" in job:
        core_terms.extend(["friends", "group travel"])
    elif "family" in job:
        core_terms.extend(["family", "kids"])

    if "trip" in job or "vacation" in job:
        core_terms.extend(["itinerary", "activities", "accommodation"])

    nouns = re.findall(r'\b[a-z]{4,}\b', job.lower())
    core_terms.extend(nouns[:5])

    return " ".join(list(set(core_terms))[:8])

# ------------------------------------------
# Phase 2: SBERT Broad Retrieval
# ------------------------------------------

def retrieve_top_k_chunks(query: str, all_chunks: List[Dict], sbert_model: SentenceTransformer, k: int = 20) -> List[Dict]:
    """Enhanced retrieval with detailed debugging"""
    print(f"Phase 2: Running Broad Retrieval with query: '{query}'")

    # Encode query
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)

    # Print debug header
    print("\n🔎 Top 20 chunks by relevance:")
    print(f"{'Rank':<5} | {'Score':<6} | {'Document':<40} | {'Heading'}")
    print("-" * 80)

    # Process each chunk
    for chunk in all_chunks:
        chunk_embedding = sbert_model.encode(chunk['paragraph_text'], convert_to_tensor=True)
        chunk['relevance_score'] = util.cos_sim(query_embedding, chunk_embedding).item()

    # Sort and select top k
    sorted_chunks = sorted(all_chunks, key=lambda x: x['relevance_score'], reverse=True)[:k]

    # Print top candidates
    for i, chunk in enumerate(sorted_chunks):
        doc_name = os.path.basename(chunk['document'])[:35] + ("..." if len(chunk['document']) > 35 else "")
        heading = chunk['heading'][:35] + ("..." if len(chunk['heading']) > 35 else "")
        print(f"{i+1:<5} | {chunk['relevance_score']:.4f} | {doc_name:<40} | {heading}")

    return sorted_chunks

# ------------------------------------------
# Phase 3: LLM Re-ranking of Candidate Chunks
# ------------------------------------------

def llm_re_rank(job_to_be_done: str, persona: str, search_terms: str, candidate_chunks: List[Dict], llm) -> List[Dict]:
    """Simplified re-ranking with search context"""
    print("Phase 3: Simplified LLM Re-Ranking...")
    
    candidate_list_str = "\n".join([
        f"[Item {i+1}] {chunk['heading']}: {chunk['paragraph_text'][:150]}..."
        for i, chunk in enumerate(candidate_chunks)
    ])
    
    prompt = f"""
As a {persona}, just shortlist the most relevant things". 
We've identified these key aspects: {search_terms}

Select the 5 most relevant items from this list and make sure not to repeat items of the same topic for example-  must-visit restuarants and upscale restaurants comes under same topic (respond ONLY with numbers 1-20 in order of importance, comma-separated):

{candidate_list_str}
just avoid nultiple items on the same topic.

Top 5 Items: """
    
    output = llm(prompt, max_tokens=50, temperature=0.0)
    response_text = output['choices'][0]['text'].strip()
    print(f"DEBUG: LLM re-ranking response: '{response_text}'")
    
    # Parse response
    try:
        # Extract all numbers from the response
        numbers = [int(num) for num in re.findall(r'\d+', response_text)]
        valid_indices = [num-1 for num in numbers if 1 <= num <= len(candidate_chunks)]
        
        # Deduplicate while preserving order
        seen = set()
        final_indices = [idx for idx in valid_indices if not (idx in seen or seen.add(idx))]
        
        return [candidate_chunks[i] for i in final_indices[:5]]
    except:
        print("❌ LLM re-ranking failed. Using top 5 by similarity.")
        return candidate_chunks[:5]

def filter_irrelevant_chunks(chunks: List[Dict], job: str) -> List[Dict]:
    """Filter out chunks that are clearly irrelevant based on content"""
    print("Applying content-based pre-filtering...")

    # Define exclusion terms based on job context
    exclusion_terms = []
    if "college friends" in job.lower() or "students" in job.lower():
        exclusion_terms = ["family", "children", "kids", "child", "baby", "toddler", "diaper", "family-friendly"]

    filtered_chunks = []
    for chunk in chunks:
        chunk_text = chunk['paragraph_text'].lower()

        # Skip chunks containing exclusion terms
        if any(term in chunk_text for term in exclusion_terms):
            print(f"  Filtered out: {os.path.basename(chunk['document'])} - {chunk['heading']}")
            continue

        # Skip packing/tips sections unless specifically relevant
        if ("packing" in chunk_text or "tips" in chunk_text) and "group" not in chunk_text:
            print(f"  Filtered out (packing/tips): {os.path.basename(chunk['document'])} - {chunk['heading']}")
            continue

        filtered_chunks.append(chunk)

    print(f"  {len(filtered_chunks)} chunks remain after filtering")
    return filtered_chunks

# ------------------------------------------
# Phase 4: Full Orchestration
# ------------------------------------------

def consolidate_topics(final_chunks: List[Dict], llm) -> List[Dict]:
    """
    Consolidates similar topics and removes duplicates to avoid repetition in final output
    """
    print("Consolidating similar topics...")
    
    # Step 1: Identify topics
    topics = []
    for chunk in final_chunks:
        prompt = f"""
Identify the main topic of the following text. Respond with ONLY one word or short phrase.

Text: {chunk['paragraph_text'][:300]}

Topic: """
        output = llm(prompt, max_tokens=20, temperature=0.0)
        topic = output['choices'][0]['text'].strip()
        topics.append(topic)
    
    # Step 2: Group chunks by topic
    topic_groups = {}
    for i, topic in enumerate(topics):
        topic = topic.lower()
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(final_chunks[i])
    
    # Step 3: Select best chunk per topic
    consolidated = []
    for topic, chunks in topic_groups.items():
        if len(chunks) == 1:
            consolidated.append(chunks[0])
        else:
            # Select the chunk with the most comprehensive content
            best_chunk = max(chunks, key=lambda x: len(x['paragraph_text']))
            consolidated.append(best_chunk)
            print(f"  Consolidated {len(chunks)} chunks about '{topic}'")
    
    # Ensure we have exactly 5 chunks
    return consolidated[:5]
    
def run_analysis(input_data: Dict, input_dir: str, sbert_model, llm):
    """
    Main entry point to process documents using the final hybrid strategy.
    """
    # Step 1: Get Persona and Job
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]

    # Step 2: Generate the Smart Search Query
    search_query = get_search_terms_from_llm(persona, job, llm)

    # Step 3: Parse All Documents
    print("Phase 1: Parsing all documents...")
    all_chunks = [
        chunk
        for doc in input_data["documents"]
        if os.path.exists(os.path.join(input_dir, doc["filename"]))
        for chunk in extract_meaningful_chunks(os.path.join(input_dir, doc["filename"]))
    ]
    if not all_chunks:
        return {"error": "Document parsing yielded no content."}

    # Step 4: Fast SBERT Retrieval
    top_candidates = retrieve_top_k_chunks(search_query, all_chunks, sbert_model, k=20)

    # Step 5: Apply the Rule-Based Pre-Filter
    filtered_candidates = filter_irrelevant_chunks(top_candidates, job)
    search_terms = get_search_terms_from_llm(persona, job, llm)

    # Step 6: Final LLM Re-Ranking on the Cleaned List
    final_chunks = llm_re_rank(job, persona, search_terms, filtered_candidates, llm)

    # Step 7: Construct Final Output
    print("Phase 4: Constructing Final JSON...")

    
    # Generate summaries for final chunks

    return {
        "metadata": {
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": c["document"],
                "section_title": c["heading"],
                "importance_rank": i + 1,
                "page_number": c["page_number"]
            } for i, c in enumerate(final_chunks)
        ],
        "subsection_analysis": [
            {
                "document": c["document"],
                "refined_text": c["paragraph_text"],
                "page_number": c["page_number"]
            } for c in final_chunks
        ]
    }
