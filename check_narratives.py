import chromadb
import pypdfium2 as pdfium
import re
from pathlib import Path

client = chromadb.PersistentClient(path="./data/chromadb")
coll = client.get_collection("esg_visual_reports")

print("=" * 60)
print("1. CHECKING COMPANIES & YEARS STORED IN CHROMADB:")
all_data = coll.get(include=["metadatas"])
companies_years = {}
for m in all_data["metadatas"]:
    c = m["company"]
    y = m["fiscal_year"]
    companies_years.setdefault(c, set()).add(y)

for comp, years in companies_years.items():
    print(f"  • Company: '{comp}' -> Years: {sorted(list(years))}")
print("=" * 60)

# Check actual text on candidate pages for Apple and Amazon
for target_comp in ["Apple", "Amazon"]:
    print(f"\n2. CHECKING TEXT EXTRACTABILITY FOR {target_comp.upper()}:")
    matching_metas = [m for m in all_data["metadatas"] if target_comp.lower() in m["company"].lower()]
    
    if not matching_metas:
        print(f"  [X] No indexed pages found in ChromaDB matching '{target_comp}'!")
        continue

    # Sample one page from each year
    by_year = {}
    for m in matching_metas:
        by_year.setdefault(m["fiscal_year"], m)

    for y in sorted(by_year.keys()):
        meta = by_year[y]
        pdf_file = Path("./data/raw_pdfs") / meta["source_file"]
        if not pdf_file.exists():
            print(f"  [X] File not found: {pdf_file}")
            continue

        pdf = pdfium.PdfDocument(str(pdf_file))
        page_num = meta["page_number"]
        page_text = pdf[page_num - 1].get_textpage().get_text_range().strip()
        pdf.close()

        clean_snippet = " ".join(page_text.split()[:25])
        print(f"  FY{y} (Page {page_num} in {meta['source_file']}):")
        print(f"    Raw Text Length: {len(page_text)} chars")
        print(f"    Snippet: \"{clean_snippet}...\"")