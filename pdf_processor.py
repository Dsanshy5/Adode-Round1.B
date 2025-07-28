# pdf_processor.py
import fitz  # PyMuPDF
import os
import re

def clean_text(text):
    """Cleans text by collapsing whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_meaningful_chunks(doc_path: str):
    """Final reliable parser using a ToC-first, page-by-page fallback strategy."""
    try:
        doc = fitz.open(doc_path)
        doc_filename = os.path.basename(doc_path)
    except Exception as e:
        print(f"Error opening {doc_path}: {e}")
        return []

    chunks = []
    toc = doc.get_toc()

    if toc:
        print(f"Found ToC in {doc_filename}. Processing by section...")
        for i, item in enumerate(toc):
            _, title, page_num = item
            start_page = page_num - 1
            end_page = doc.page_count if i + 1 >= len(toc) else toc[i+1][2] - 1
            section_text = "".join(doc[pg].get_text() for pg in range(start_page, end_page) if pg < doc.page_count)
            if section_text.strip():
                chunks.append({
                    "document": doc_filename, "heading": clean_text(title),
                    "page_number": page_num, "paragraph_text": clean_text(section_text)
                })
    else:
        print(f"No ToC in {doc_filename}. Processing page by page with title detection...")
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if not page_text.strip():
                continue

            heading = f"{doc_filename} - Page {page_num + 1}"
            blocks = page.get_text("dict", flags=0)["blocks"]
            text_blocks = [b for b in blocks if b['type'] == 0]
            page_height = page.rect.height
            top_blocks = [b for b in text_blocks if b['bbox'][1] < page_height * 0.3]

            if top_blocks:
                top_heading_block = max(top_blocks, key=lambda b: b['lines'][0]['spans'][0]['size'] if b.get('lines') and b['lines'][0].get('spans') else 0)
                try:
                    title_text = " ".join(s['text'] for l in top_heading_block['lines'] for s in l['spans'])
                    if title_text and len(title_text) < 100:
                        heading = clean_text(title_text)
                except (IndexError, KeyError):
                    pass

            chunks.append({
                "document": doc_filename, "heading": heading,
                "page_number": page_num + 1, "paragraph_text": clean_text(page_text)
            })
    doc.close()
    return chunks