import csv
import json
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "StockStreamTickersData.csv"

# Unofficial but widely used community API for AMFI scheme universe + NAV history.
# Returns a JSON array of objects: [{"schemeCode": 119551, "schemeName": "..."} ...]
MF_LIST_URL = "https://api.mfapi.in/mf"


def load_existing_rows(path: Path):
    if not path.exists():
        return [], set()

    # File is currently tab-separated but we tolerate comma too.
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not raw:
        return [], set()

    header = raw[0]
    delim = "\t" if "\t" in header else ","

    rows = []
    existing_symbols = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            company = (r.get("Company Name") or r.get("company name") or r.get("Company") or "").strip()
            symbol = (r.get("Symbol") or r.get("symbol") or r.get("Ticker") or "").strip()
            if not company or not symbol:
                continue
            rows.append({"Company Name": company, "Symbol": symbol})
            existing_symbols.add(symbol)

    return rows, existing_symbols


def fetch_mutual_fund_schemes():
    resp = requests.get(MF_LIST_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response; expected list of schemes")
    return data


def write_tsv(path: Path, rows):
    # Keep the file tab-separated (matches current file)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Company Name", "Symbol"], delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    existing_rows, existing_symbols = load_existing_rows(CSV_PATH)

    schemes = fetch_mutual_fund_schemes()
    added = 0
    for s in schemes:
        code = str(s.get("schemeCode", "")).strip()
        name = str(s.get("schemeName", "")).strip()
        if not code or not name or code.lower() == "none":
            continue

        symbol = f"MF:{code}"
        if symbol in existing_symbols:
            continue

        existing_rows.append({"Company Name": name, "Symbol": symbol})
        existing_symbols.add(symbol)
        added += 1

    # Stable-ish ordering: keep existing top section, then append all mutual funds (already appended).
    write_tsv(CSV_PATH, existing_rows)

    print(json.dumps({"csv": str(CSV_PATH), "added_mutual_funds": added, "total_rows": len(existing_rows)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

