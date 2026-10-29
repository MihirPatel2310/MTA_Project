import traceback
import chromadb
from main import AutoMultimodalTemporalAuditor as MultimodalTemporalAuditor

print("=" * 60)
print("1. DIRECT CHROMADB VERIFICATION")
print("=" * 60)

client = chromadb.PersistentClient(path="./data/chromadb")
collections = client.list_collections()
print(f"Collections found: {[c.name for c in collections]}")

if not any(c.name == "esg_visual_reports" for c in collections):
    print("[!] FATAL: Collection 'esg_visual_reports' DOES NOT EXIST in ./data/chromadb!")
    exit()

coll = client.get_collection("esg_visual_reports")
count = coll.count()
print(f"Total documents/pages stored: {count}")

if count == 0:
    print("[!] FATAL: ChromaDB collection is completely empty! You need to run: python -m src.ingestion")
    exit()

# Sample 5 metadata entries
sample = coll.get(limit=5, include=["metadatas"])
print("\nSample records stored in DB:")
for m in sample["metadatas"]:
    print(f"  • Company: '{m.get('company')}', FY: {m.get('fiscal_year')}, Page: {m.get('page_number')}")

print("\n" + "=" * 60)
print("2. TESTING 1 REAL AUDIT CALL VIA MAIN")
print("=" * 60)

auditor = MultimodalTemporalAuditor()

# Let's inspect what discover_all_companies_and_years returns:
if hasattr(auditor, "discover_all_companies_and_years"):
    discovered = auditor.discover_all_companies_and_years()
    print(f"auditor.discover_all_companies_and_years() -> {discovered}")
else:
    print("Method discover_all_companies_and_years not found on auditor.")
    discovered = {}

if not discovered:
    # Pick the first company from our DB sample
    sample_company = sample["metadatas"][0].get("company")
    sample_years = [2021, 2022, 2023, 2024]
else:
    sample_company = list(discovered.keys())[0]
    sample_years = discovered[sample_company]

print(f"\nAttempting test audit on Company: '{sample_company}' with Years: {sample_years}")

try:
    report = auditor.audit_company(
        company=sample_company,
        available_years=sample_years,
        query="Scope 1, Scope 2, Scope 3 greenhouse gas emissions"
    )
    print(f"\naudit_company returned type: {type(report)}")
    if report is None:
        print("[!] audit_company returned None!")
    else:
        print(f"[✓] SUCCESS! Report generated.")
        print(f"  • Extracted metrics count: {len(getattr(report, 'all_extracted_metrics', []))}")
        print(f"  • Restatements count: {len(getattr(report, 'restatements', []))}")
        print(f"  • Ambiguity results: {len(getattr(report, 'ambiguity_results', []))}")

except Exception as e:
    print(f"\n[!] CRASH inside audit_company:")
    traceback.print_exc()