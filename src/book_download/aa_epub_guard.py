#!/usr/bin/env python3
"""
Book Download Research Component - EPUB Guard Helper
Robust EPUB download, validation, and format conversion utility
"""

from __future__ import annotations
import io
import os
import re
import subprocess
import sys
import urllib.request
import urllib.parse
import zipfile
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError

EPUB_MIMETYPE = b"application/epub+zip"
IPFS_GATEWAYS = (
    "https://ipfs.io/ipfs/{cid}?filename={fname}",
    "https://dweb.link/ipfs/{cid}?filename={fname}",
    "https://trustless-gateway.link/ipfs/{cid}?filename={fname}",
    "https://gateway.pinata.cloud/ipfs/{cid}?filename={fname}",
)

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124 Safari/537.36")

def _slug(s: str) -> str:
    """Create a filesystem-safe slug from a string."""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^A-Za-z0-9._ -]+", "", s)
    return s[:120] or "book"

def build_filename(title: str = "", author: str = "", ext: str = "epub", fallback: str = "download") -> str:
    """Build a clean filename from title and author."""
    base = " - ".join([_ for _ in (_slug(title), _slug(author)) if _]) or _slug(fallback)
    return f"{base}.{ext}"

def fetch_via_ipfs_cids(cids: Iterable[str], dest: Path, filename: str) -> Path:
    """Download file via IPFS CIDs, trying multiple gateways."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # try each CID across several gateways until one works
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Optional[Exception] = None
    backoff = 0.75
    for cid in cids:
        for tpl in IPFS_GATEWAYS:
            url = tpl.format(cid=cid, fname=urllib.parse.quote(filename))
            req = urllib.request.Request(url, headers={
                "User-Agent": _UA,
                "Accept": "*/*",
                "Connection": "close",
            })
            try:
                with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
                    f.write(r.read())
                tmp.replace(dest)
                return dest
            except HTTPError as e:
                # 403/429: backoff a bit then try next gateway/CID
                if e.code in (403, 429):
                    time.sleep(backoff)
                    backoff = min(backoff * 1.7, 5.0)
                last_err = e
                continue
            except URLError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
    if last_err:
        raise RuntimeError(f"All IPFS gateways failed: {last_err}")
    raise RuntimeError("No CIDs provided")

def sniff_format(path: Path) -> str:
    """Return 'epub', 'mobi', 'html', or 'unknown'."""
    with open(path, "rb") as f:
        head = f.read(4096)
    # EPUB = ZIP container (PK..) with special structure
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                if not names:
                    return "unknown"
                info = zf.infolist()[0]
                if info.filename != "mimetype":
                    return "zip"  # a ZIP, but not a valid EPUB layout
                if info.compress_type != zipfile.ZIP_STORED:
                    return "zip"
                with zf.open("mimetype") as m:
                    if m.read().strip() == EPUB_MIMETYPE:
                        return "epub"
                return "zip"
        except zipfile.BadZipFile:
            return "unknown"
    if b"BOOKMOBI" in head or b"FRM" in head[60:80]:  # crude but effective for MOBI/KF7/KF8
        return "mobi"
    if head.lstrip().lower().startswith(b"<!doctype") or head.lstrip().lower().startswith(b"<html"):
        return "html"
    return "unknown"

def ensure_valid_epub(path: Path, convert_mobi: bool = True, calibre_bin: str = "ebook-convert") -> Path:
    """Return path to a usable .epub. May rename or convert."""
    fmt = sniff_format(path)
    if fmt == "epub":
        # normalize extension if needed
        if path.suffix.lower() != ".epub":
            newp = path.with_suffix(".epub")
            path.rename(newp)
            return newp
        return path
    if fmt == "mobi":
        if convert_mobi:
            out = path.with_suffix(".epub")
            try:
                subprocess.run([calibre_bin, str(path), str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return out
            except FileNotFoundError:
                # calibre not installed; just rename to .mobi so the user can open it in a MOBI-capable reader
                newp = path.with_suffix(".mobi")
                path.rename(newp)
                return newp
        else:
            newp = path.with_suffix(".mobi")
            path.rename(newp)
            return newp
    if fmt == "html":
        raise ValueError("Downloaded HTML page instead of a book (likely an error/portal page from a mirror). Try another gateway.")
    if fmt == "zip":
        raise ValueError("ZIP file is not a valid EPUB (mimetype not first/uncompressed).")
    raise ValueError("Unknown or corrupted file; re-download from a different source.")

def download_from_metadata(meta: dict, out_dir: Path, prefer_title_author: bool = True, convert_mobi: bool = True) -> Path:
    """
    Minimal workflow:
    - build a nice filename from title/author (fallback to md5)
    - download via IPFS CIDs
    - sniff & fix extension, convert if MOBI
    - return final path
    """
    out_dir = Path(out_dir)
    title = meta.get("file_unified_data", {}).get("title_best", "") or ""
    author = meta.get("file_unified_data", {}).get("author_best", "") or ""
    md5 = (meta.get("identifiers_unified", {}) or {}).get("md5", ["file"])[0]
    cids = tuple((meta.get("file_unified_data", {}) or {}).get("ipfs_infos", []))
    cids = [d.get("ipfs_cid") for d in cids if d.get("ipfs_cid")]

    base = build_filename(title if prefer_title_author else "", author if prefer_title_author else "", ext="epub", fallback=md5)
    dest = out_dir / base
    tmp = dest.with_suffix(".download")
    # download to tmp name (so we don't mislead other code by extension)
    fetched = fetch_via_ipfs_cids(cids, tmp, base)
    final = ensure_valid_epub(fetched, convert_mobi=convert_mobi)
    # move final into out_dir with clean name if conversion changed suffix
    wanted = out_dir / build_filename(title, author, ext=final.suffix.lstrip("."), fallback=md5)
    if final != wanted:
        final = final.rename(wanted)
    return final

def test_epub_guard():
    """Test the EPUB guard functionality with sample metadata."""
    print("=== TESTING EPUB GUARD ===")
    
    # Sample metadata (Emma by Jane Austen)
    meta = {
        "file_unified_data": {
            "title_best": "Emma",
            "author_best": "Jane Austen",
            "ipfs_infos": [
                {"ipfs_cid": "QmfApP4c1A9YtDJor1TivVTLYpJpWkJB8BU55sUVQrHTuM"},
                {"ipfs_cid": "bafykbzacecjvuvv3oj6333etrjjnxjfutbsjyoy3bypux5slywuiyszp5uufy"},
            ],
        },
        "identifiers_unified": {"md5": ["05bb51653e0d905f9ff7c7df91258b02"]},
    }
    
    try:
        out = download_from_metadata(meta, Path("./test_downloads"))
        print(f"✓ Download successful: {out}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

if __name__ == "__main__":
    test_epub_guard()
