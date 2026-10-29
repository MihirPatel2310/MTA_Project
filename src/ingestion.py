# import os
# import re
# import gc
# from pathlib import Path
# from typing import Tuple

# import torch
# import chromadb
# import pypdfium2 as pdfium
# from transformers import BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor

# # Optimize CUDA allocator
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# KNOWN_NOISE_TOKENS = {
#     "esg", "sustainability", "environmental", "progress", "report", "datasheet",
#     "data", "factsheet", "annual", "csr", "update", "impact", "overview", "pdf"
# }


# def extract_company_from_cover_text(pdf_path: Path) -> str:
#     """Fallback: Extracts candidate company name from the first page text."""
#     try:
#         pdf = pdfium.PdfDocument(pdf_path)
#         page = pdf[0]
#         text_page = page.get_textpage()
#         first_page_text = text_page.get_text_range().strip()
#         pdf.close()

#         # Find first prominent uppercase/title words
#         lines = [line.strip() for line in first_page_text.splitlines() if len(line.strip()) > 2]
#         for line in lines[:5]:
#             clean_line = re.sub(r"[^a-zA-Z0-9\s]", "", line).strip()
#             words = clean_line.split()
#             filtered_words = [w for w in words if w.lower() not in KNOWN_NOISE_TOKENS and not w.isdigit()]
#             if filtered_words:
#                 return " ".join(filtered_words[:3]).title()
#     except Exception:
#         pass
#     return "Unknown Company"


# def parse_metadata_dynamically(pdf_path: Path) -> Tuple[str, int]:
#     """
#     Dynamically identifies company name and fiscal year for ANY PDF.
#     No hardcoded company lists.
#     """
#     filename = pdf_path.name
#     stem = pdf_path.stem.lower()

#     # 1. Detect 4-digit fiscal year (e.g., 2018-2029)
#     year_match = re.search(r"\b(20[1-3][0-9])\b", stem)
#     if not year_match:
#         # Fallback: search within the whole filename string
#         year_match = re.search(r"(20[1-3][0-9])", filename)
    
#     fiscal_year = int(year_match.group(1)) if year_match else 2023

#     # 2. Extract company from filename tokens
#     # Replace separators with spaces
#     normalized = re.sub(r"[_\-\.]+", " ", stem)
#     tokens = normalized.split()

#     company_tokens = []
#     for token in tokens:
#         # Stop token collection once noise words or years appear
#         if token.lower() in KNOWN_NOISE_TOKENS or re.match(r"^20[1-3][0-9]$", token):
#             continue
#         company_tokens.append(token)

#     if company_tokens:
#         company_name = " ".join(company_tokens).strip().title()
#     else:
#         # Fallback to PDF Cover Text inspection
#         company_name = extract_company_from_cover_text(pdf_path)

#     return company_name, fiscal_year


# def ingest_all_pdfs(
#     pdf_dir: str = "./data/raw_pdfs",
#     chroma_db_path: str = "./data/chromadb",
#     collection_name: str = "esg_visual_reports"
# ):
#     if not torch.cuda.is_available():
#         print("[!] CUDA GPU not detected.")
#         return

#     gpu_name = torch.cuda.get_device_name(0)
#     print(f"[+] Starting Dynamic Ingestion Engine on GPU: {gpu_name}")

#     from chromadb.config import Settings
#     client = chromadb.PersistentClient(
#         path=chroma_db_path,
#         settings=Settings(
#             anonymized_telemetry=False,
#             allow_reset=True,
#             is_persistent=True
#         )
#     )
#     collection = client.get_or_create_collection(
#         name=collection_name,
#         metadata={"description": "Multimodal Temporal Auditor - Dynamic ESG Index"}
#     )

#     model_id = "vidore/colpali-v1.3-hf"
#     print(f"[+] Loading ColPali Model in native float16 ({model_id})...")

#     # Native float16 avoids bitsandbytes Byte tensor corruption in projection norm
#     processor = ColPaliProcessor.from_pretrained(model_id)
#     model = ColPaliForRetrieval.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         device_map={"": 0},
#         low_cpu_mem_usage=True,
#         attn_implementation="eager"
#     ).eval()

#     pdf_files = sorted(list(Path(pdf_dir).glob("*.pdf")))
#     if not pdf_files:
#         print(f"[!] No PDFs found in {pdf_dir}. Add files to begin.")
#         return

#     print(f"[+] Discovered {len(pdf_files)} PDF file(s) in {pdf_dir}\n")

#     for doc_idx, pdf_path in enumerate(pdf_files, 1):
#         company, fiscal_year = parse_metadata_dynamically(pdf_path)

#         print(f"[{doc_idx}/{len(pdf_files)}] Ingesting Document: {pdf_path.name}")
#         print(f"    ├─ Detected Company:     {company}")
#         print(f"    └─ Detected Fiscal Year: {fiscal_year}")

#         pdf = None
#         try:
#             pdf = pdfium.PdfDocument(pdf_path)
#             num_pages = len(pdf)
#             print(f"    ├─ Total Pages to Index: {num_pages}")

#             for page_idx in range(num_pages):
#                 page_num = page_idx + 1
#                 page = pdf[page_idx]
                
#                 # Render page at standard resolution
#                 image = page.render(scale=1.0).to_pil().convert("RGB")
#                 doc_id = f"{company.replace(' ', '_')}_{fiscal_year}_p{page_num}"

#                 # Process image through ColPali processor
#                 inputs = processor(images=image, return_tensors="pt")
#                 inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

#                 with torch.no_grad():
#                     outputs = model(**inputs)
#                     embeddings = outputs.embeddings if hasattr(outputs, "embeddings") else outputs[0]
#                     # Mean pool patch embeddings into single 1D vector (dim=128)
#                     pooled = torch.mean(embeddings.to(torch.float32), dim=1).squeeze(0).cpu().numpy().tolist()

#                 collection.upsert(
#                     ids=[doc_id],
#                     embeddings=[pooled],
#                     metadatas=[{
#                         "company": company,
#                         "fiscal_year": fiscal_year,
#                         "page_number": page_num,
#                         "source_file": pdf_path.name
#                     }],
#                     documents=[f"Page {page_num} of {company} {fiscal_year} Report ({pdf_path.name})"]
#                 )

#                 if page_num % 10 == 0 or page_num == num_pages:
#                     print(f"    ├─ Indexed page {page_num}/{num_pages} on CUDA...")

#                 del inputs, outputs, embeddings, pooled, image
#                 torch.cuda.empty_cache()

#             print(f"    └─ [✓] Successfully indexed {pdf_path.name}\n")

#         except Exception as e:
#             print(f"    └─ [X] Error indexing {pdf_path.name}: {e}\n")
#         finally:
#             if pdf is not None:
#                 pdf.close()
#             gc.collect()

#     print("[✓] Dynamic Multi-Company Ingestion Complete!")


# if __name__ == "__main__":
#     ingest_all_pdfs() 

import os
import re
import gc
from pathlib import Path
from typing import Tuple, List

import torch
import chromadb
from chromadb.config import Settings
import pypdfium2 as pdfium
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

# Prevent memory fragmentation on Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

KNOWN_NOISE_TOKENS = {
    "esg", "sustainability", "environmental", "progress", "report", "datasheet",
    "data", "factsheet", "annual", "csr", "update", "impact", "overview", "pdf"
}

AUDIT_KEYWORDS = [
    "scope 1", "scope 2", "scope 3", "ghg", "emissions", "carbon",
    "metric tons", "mwh", "gigawatt", "renewable", "net-zero", "target",
    "water", "waste", "baseline", "restatement", "kpi", "datasheet"
]


def parse_metadata_dynamically(pdf_path: Path) -> Tuple[str, int]:
    # Normalize filename: convert hyphens and underscores to spaces
    stem = pdf_path.stem.lower()
    clean_str = re.sub(r"[-_.]+", " ", stem)

    # 1. Direct 4-digit search for years 2015-2030 (no \b word boundary trap)
    year_match = re.search(r"(20[1-3][0-9])", clean_str)
    if year_match:
        fiscal_year = int(year_match.group(1))
    else:
        raise ValueError(f"Could not parse a 4-digit fiscal year from: {pdf_path.name}")

    # 2. Extract company name
    tokens = clean_str.split()
    company_tokens = [
        t for t in tokens 
        if t.lower() not in KNOWN_NOISE_TOKENS and not re.match(r"^20[1-3][0-9]$", t)
    ]
    company_name = " ".join(company_tokens).strip().title() if company_tokens else "Unknown"

    return company_name, fiscal_year

def get_relevant_page_indices(pdf: pdfium.PdfDocument) -> List[int]:
    """Scans raw text quickly to select only relevant data/appendix pages."""
    total_pages = len(pdf)
    relevant = []
    
    # Datasheets are typically in the back 30% of ESG reports
    datasheet_cutoff = int(total_pages * 0.70)

    for idx in range(total_pages):
        if idx >= datasheet_cutoff:
            relevant.append(idx)
            continue

        try:
            text = pdf[idx].get_textpage().get_text_range().lower()
            if any(kw in text for kw in AUDIT_KEYWORDS):
                relevant.append(idx)
        except Exception:
            relevant.append(idx)

    return relevant


def ingest_all_pdfs_ultralight(
    pdf_dir: str = "./data/raw_pdfs",
    chroma_db_path: str = "./data/chromadb",
    collection_name: str = "esg_visual_reports"
):
    if not torch.cuda.is_available():
        print("[!] CUDA GPU not detected.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"[+] Starting Ultralight Ingestion Engine on: {gpu_name}")

    client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True)
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Multimodal Temporal Auditor - Fast Ultralight Index"}
    )

    model_id = "vidore/colpali-v1.3-hf"
    print(f"[+] Loading ColPali in float16 ({model_id})...")

    processor = ColPaliProcessor.from_pretrained(model_id)
    model = ColPaliForRetrieval.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).eval()

    pdf_files = sorted(list(Path(pdf_dir).glob("*.pdf")))
    if not pdf_files:
        print(f"[!] No PDFs found in {pdf_dir}.")
        return

    print(f"[+] Found {len(pdf_files)} document(s) to index.\n")

    for doc_idx, pdf_path in enumerate(pdf_files, 1):
        company, fiscal_year = parse_metadata_dynamically(pdf_path)
        print(f"[{doc_idx}/{len(pdf_files)}] Reading: {pdf_path.name} ({company} - FY{fiscal_year})")

        pdf = None
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)

            # Fast page filter
            target_indices = get_relevant_page_indices(pdf)
            print(f"    ├─ Retained {len(target_indices)}/{total_pages} relevant pages")

            # Collect in memory and bulk-upsert to ChromaDB in chunks of 20
            buffer_ids = []
            buffer_embeddings = []
            buffer_metas = []
            buffer_docs = []

            for count, p_idx in enumerate(target_indices, 1):
                page_num = p_idx + 1
                page = pdf[p_idx]
                
                # Scale 0.5 cuts tensor dimensions in half, speeding up the forward pass 3x
                image = page.render(scale=0.5).to_pil().convert("RGB")
                
                # Constrain max dimension to 768px for memory safety
                if max(image.size) > 768:
                    image.thumbnail((768, 768), Image.Resampling.LANCZOS)

                # Process 1 page at a time (strictly prevents VRAM overflow / swapping)
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.embeddings if hasattr(outputs, "embeddings") else outputs[0]
                    # Mean-pool to 1D vector
                    pooled = torch.mean(embeddings.to(torch.float32), dim=1).squeeze(0).cpu().numpy().tolist()

                doc_id = f"{company.replace(' ', '_')}_{fiscal_year}_p{page_num}"
                buffer_ids.append(doc_id)
                buffer_embeddings.append(pooled)
                buffer_metas.append({
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "page_number": page_num,
                    "source_file": pdf_path.name
                })
                buffer_docs.append(f"Page {page_num} of {company} {fiscal_year} ({pdf_path.name})")

                # Flush to ChromaDB every 20 pages or on the last page
                if len(buffer_ids) >= 20 or count == len(target_indices):
                    collection.upsert(
                        ids=buffer_ids,
                        embeddings=buffer_embeddings,
                        metadatas=buffer_metas,
                        documents=buffer_docs
                    )
                    buffer_ids.clear()
                    buffer_embeddings.clear()
                    buffer_metas.clear()
                    buffer_docs.clear()

                del inputs, outputs, embeddings, pooled, image
                torch.cuda.empty_cache()

                if count % 10 == 0 or count == len(target_indices):
                    print(f"    ├─ Processed {count}/{len(target_indices)} pages...")

            print(f"    └─ [✓] Finished indexing {pdf_path.name}\n")

        except Exception as e:
            print(f"    └─ [X] Error processing {pdf_path.name}: {e}\n")
        finally:
            if pdf is not None:
                pdf.close()
            gc.collect()

    print("[✓] Dynamic Indexing Complete without VRAM thrashing!")


if __name__ == "__main__":
    ingest_all_pdfs_ultralight()