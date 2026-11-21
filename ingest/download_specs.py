"""Download the two 3GPP Release 18 specifications into corpus/raw/.

Usage:
    python -m ingest.download_specs

Source: official 3GPP FTP archive. The .zip files are extracted and removed.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "corpus" / "raw"

SPECS = {
    "23501-ic0.docx": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/23501-ic0.zip",  # TS 23.501 V18.12.0
    "24501-ie0.docx": "https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/24501-ie0.zip",  # TS 24.501 V18.14.0
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SPECS.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"{name}: already present")
            continue
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            docx = [n for n in z.namelist() if n.lower().endswith(".docx")]
            if not docx:
                raise RuntimeError(f"No .docx in {url}")
            target.write_bytes(z.read(docx[0]))
        print(f"{name}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
