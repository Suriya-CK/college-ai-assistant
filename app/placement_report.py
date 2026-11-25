# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
"""Turn a college's placement documents (PDFs) into placement facts, without an LLM.

Placement portals often show recruiters only as logos and publish the real numbers as PDFs. Three layouts:

1. Table with campus and level:  S.No | Reg. No. | Name | Degree | Branch | College | UG/PG | Company
2. Table grouped by department:  DEPARTMENT heading, then  S.No | Reg. No. | Name | Company[, Company]
3. NIRF data (the national ranking submission every Indian college publishes, same format everywhere):
   per graduating year, students graduated, placed, median salary and higher studies.

Only companies, counts, campuses and departments are kept. Student names and register numbers are personal
data and are never stored, so the assistant can't be asked about individual students.
"""
import io
import re
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.config import Config
from app.storage import Database

logger = logging.getLogger(__name__)

# A table row starts with a serial number and a register number (6+ digits)
_ROW_START_RE = re.compile(r"^\s*\d+\s+\d{6,}\s")
# Layout 1 ends with "<College> <UG|PG> <Company>"
_ROW_END_RE = re.compile(r"\s([A-Z][A-Za-z.&-]{1,15})\s+(UG|PG)\s+(.+?)\s*$")
# Layout 2 ends with one company in capitals, or several separated by commas: "FOXCONN, QSPIDERS"
_CAPS_COMPANIES_RE = re.compile(r"\s([A-Z0-9][A-Z0-9&.()'-]*(?:\s*,\s*[A-Z0-9][A-Z0-9&.()'-]*)*)\s*$")
# A department heading in layout 2: a line in capitals with no digits ("MECHANICAL ENGINEERING")
_DEPARTMENT_RE = re.compile(r"^[A-Z][A-Z&,.()' -]{6,}$")
_NOT_DEPARTMENT_RE = re.compile(r"COLLEGE OF|UNIVERSITY|CELL|PLACEMENT|S\.NO|REGISTER|NUMBER|NAME|COMPANY", re.I)
_YEAR_RE = re.compile(r"(20\d\d)\s*[-–]\s*(20)?(\d\d)")
MIN_TABLE_ROWS = 5   # fewer rows than this is not a placement list (e.g. a contact table)
_NOT_A_COMPANY = {"", "-", "na", "n/a", "nil", "not placed", "higher studies"}

# NIRF "Placement & Higher Studies": one table per programme length ("UG [4 Years Program(s)]")
_NIRF_SECTION_RE = re.compile(
    r"((?:UG|PG)(?:-Integrated)?)\s*\[(\d+)\s*Years?\s*Program\(s\)\]\s*:?\s*Placement\s*&\s*higher\s*studies(.*?)"
    r"(?=(?:UG|PG)(?:-Integrated)?\s*\[\d+\s*Years?\s*Program|Ph\.?D\s*Student|Financial\s*Resources|$)", re.I | re.S)
# ... each row ends with: <graduating year> <graduated> <placed> <median salary>(<in words>) <higher studies>
_NIRF_ROW_RE = re.compile(r"(\d{4}-\d{2})\s+(\d+)\s+(\d+)\s+(\d+)\s*\(([^)]*)\)\s+(\d+)")


def read_pdf_text(data: bytes) -> str:
    """Text of every page of a PDF (the crawler stops at 40 pages; reports can be longer)."""
    from pypdf import PdfReader
    if not data.startswith(b"%PDF"):
        raise ValueError("Not a PDF file")
    return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)


def _clean_company(name: str) -> str:
    # "Alstom (Drive 2)" is a second recruitment drive by the same company
    return re.sub(r"\s*\((?:drive|round)\s*\d+\)\s*$", "", name, flags=re.I).strip(" .,-")


def _row_complete(row: str) -> bool:
    row = re.sub(r"\s+", " ", row)
    return bool(_ROW_END_RE.search(row) or _CAPS_COMPANIES_RE.search(row))


def _table_rows(text: str, strict: bool) -> List[Tuple[str, str]]:
    """(row text, department heading above it). A row may wrap over several lines (long branch names), so
    following lines are joined to it. strict: only join while the row is unfinished, so a signature line
    under the table ("Professor / EEE)") isn't read as a company (needed for layout 2, harmful for layout 1,
    where a half row can already look finished)."""
    rows: List[Tuple[str, str]] = []
    department = ""
    for line in text.splitlines():
        stripped = line.strip()
        if _ROW_START_RE.match(line):
            rows.append((stripped, department))
        elif strict and _DEPARTMENT_RE.match(stripped) and not _NOT_DEPARTMENT_RE.search(stripped):
            # Department headings only exist in layout 2; in layout 1 a wrapped "CEG UG VISA" looks just like one
            department = stripped.title().replace(" And ", " and ")
        elif (rows and stripped and not stripped.lower().startswith(("s.", "no", "reg. no"))
              and not (strict and (stripped.startswith("(") or _row_complete(rows[-1][0])))):
            rows[-1] = (rows[-1][0] + " " + stripped, rows[-1][1])
    return [(re.sub(r"\s+", " ", r), d) for r, d in rows]


def parse_rows(text: str) -> List[Dict[str, str]]:
    """Offers in a placement table, as {"college", "level", "department", "company", "student"}.
    "student" is only a row number (to count students with two offers once), never a name."""
    rows = _table_rows(text, strict=False)
    layout1 = sum(1 for r, _ in rows if _ROW_END_RE.search(r)) > len(rows) / 2
    if not layout1:
        rows = _table_rows(text, strict=True)

    parsed: List[Dict[str, str]] = []
    for i, (row, dept) in enumerate(rows):
        if layout1:                              # layout 1: campus and UG/PG columns
            m = _ROW_END_RE.search(row)
            if not m:
                continue
            companies, college, level = [m.group(3)], m.group(1), m.group(2)
        else:                                    # layout 2: company names in capitals at the end
            m = _CAPS_COMPANIES_RE.search(row)
            if not m:
                continue
            companies, college, level = m.group(1).split(","), "", ""
        for company in companies:
            company = _clean_company(company)
            if company.lower() not in _NOT_A_COMPANY:
                parsed.append({"college": college, "level": level, "department": dept,
                               "company": company, "student": str(i)})
    return parsed


def parse_nirf(text: str) -> List[Dict[str, Any]]:
    """NIRF placement rows: {"level", "years", "graduating_year", "graduated", "placed", "median", "higher"}."""
    flat = re.sub(r"\s+", " ", text)
    out = []
    for sec in _NIRF_SECTION_RE.finditer(flat):
        for m in _NIRF_ROW_RE.finditer(sec.group(3)):
            out.append({"level": sec.group(1).upper(), "years": int(sec.group(2)), "graduating_year": m.group(1),
                        "graduated": int(m.group(2)), "placed": int(m.group(3)), "median": int(m.group(4)),
                        "higher": int(m.group(6))})
    return out


def _company_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower().replace("&", "and"))


def report_year(title: str) -> str:
    """'PLACEMENT ... (2025 - 2026)' -> '2025-26'."""
    m = _YEAR_RE.search(title or "")
    return f"{m.group(1)}-{m.group(3)}" if m else ""


def _lakh(amount: int) -> str:
    return f"₹{amount / 100000:.2f}".rstrip("0").rstrip(".") + " lakh"


def summarise(rows: List[Dict[str, str]], year: str, report_name: str) -> Tuple[List[Dict[str, str]], str]:
    """(facts, page_text) for a placement table: one recruiter fact per company with its offer count and
    campuses or departments, a few placement statistics, and a readable list for search."""
    by_key: Dict[str, Counter] = defaultdict(Counter)       # company key -> spelling counts
    offers: Counter = Counter()
    groups: Dict[str, Counter] = defaultdict(Counter)
    by_campus = any(r["college"] for r in rows)
    label = "campus" if by_campus else "department"
    for r in rows:
        key = _company_key(r["company"])
        by_key[key][r["company"]] += 1
        offers[key] += 1
        group = r["college"] if by_campus else r.get("department", "")
        if group:
            groups[key][group] += 1
    name_of = {k: spellings.most_common(1)[0][0] for k, spellings in by_key.items()}
    when = f" ({year})" if year else ""
    kind = "Placement and internship offers" if "intern" in report_name.lower() else "Placement offers"
    students = len({r.get("student", str(i)) for i, r in enumerate(rows)})

    facts = []
    for key, n in offers.most_common():
        # "by campus" so CEG / MIT read as campuses (colleges), not as department abbreviations
        where = ", ".join(f"{g} {k}" for g, k in groups[key].most_common())
        facts.append({"category": "recruiter", "name": name_of[key],
                      "details": f"{n} offer{'s' if n != 1 else ''}{when}" + (f"; by {label}: {where}" if where else "")})

    level = Counter(r["level"] for r in rows if r["level"])
    per_group = Counter(r["college"] if by_campus else r.get("department", "") for r in rows)
    per_group.pop("", None)
    summary = f"{len(rows)} offers from {len(offers)} companies"
    if students != len(rows):
        summary += f", to {students} students"
    if level:
        summary += f" ({level['UG']} UG, {level['PG']} PG)"
    top = ", ".join(f"{name_of[k]} ({n})" for k, n in offers.most_common(10))
    facts.append({"category": "placement_stat", "name": f"{kind}{when}", "details": summary})
    if per_group:
        facts.append({"category": "placement_stat", "name": f"Offers by {label}{when}",
                      "details": ", ".join(f"{g} {n}" for g, n in per_group.most_common())})
    facts.append({"category": "placement_stat", "name": f"Top recruiters by offers{when}", "details": top})

    lines = [f"# Recruiting companies{when}", f"From the official report \"{report_name}\": {summary}."]
    if per_group:
        lines.append(f"Offers by {label}: " + ", ".join(f"{g} {n}" for g, n in per_group.most_common()) + ".")
    lines += ["", "## Companies and number of offers"]
    lines += [f"- {name_of[k]}: {n} offer{'s' if n != 1 else ''}"
              + (f" ({label}: {', '.join(groups[k])})" if groups[k] else "") for k, n in offers.most_common()]
    return facts, "\n".join(lines)


def summarise_nirf(rows: List[Dict[str, Any]], nirf_year: str) -> Tuple[List[Dict[str, str]], str]:
    """(facts, page_text) for NIRF placement data: one fact per programme level and graduating year."""
    src = f"NIRF {nirf_year}".strip()
    facts, lines = [], [f"# Placement statistics ({src})", "From the college's official NIRF data.", ""]
    for r in rows:
        prog = f"{r['level']} ({r['years']}-year programmes)"
        pct = f" ({100 * r['placed'] / r['graduated']:.0f}%)" if r["graduated"] else ""
        median = f"median salary {_lakh(r['median'])}" if r["median"] else "no median salary reported"
        details = (f"{r['placed']} of {r['graduated']} graduates placed{pct}; {median}; "
                   f"{r['higher']} went for higher studies ({src})")
        facts.append({"category": "placement_stat",
                      "name": f"{r['level']} placements, students graduating {r['graduating_year']}", "details": details})
        lines.append(f"- {prog}, graduating {r['graduating_year']}: {details}")
    return facts, "\n".join(lines)


def analyse_document(text: str, name: str) -> Optional[Dict[str, Any]]:
    """Recognise a placement table or NIRF data in a PDF's text. Returns None for any other document, else
    {"kind": "table"|"nirf", "facts", "page_text" (no student names), "page_title", "year", "rows", "companies"}.
    Used by the import-placements command and by the crawler for every PDF it finds."""
    nirf_rows = parse_nirf(text)
    if nirf_rows:
        found = re.search(r"20\d\d", name) or re.search(r"NIRF\D{0,10}(20\d\d)", text)
        nirf_year = re.sub(r"\D", "", found.group(0)) if found else ""
        facts, page_text = summarise_nirf(nirf_rows, nirf_year)
        return {"kind": "nirf", "facts": facts, "page_text": page_text, "year": nirf_year,
                "page_title": f"Placement statistics (NIRF {nirf_year})".replace(" )", ")").replace("( ", "("),
                "rows": len(nirf_rows), "companies": 0}
    rows = parse_rows(text)
    if len(rows) < MIN_TABLE_ROWS:
        return None
    year = report_year(name)
    facts, page_text = summarise(rows, year, name)
    return {"kind": "table", "facts": facts, "page_text": page_text, "year": year,
            "page_title": f"Placement report {year}: recruiting companies".replace("  ", " "),
            "rows": len(rows), "companies": sum(1 for f in facts if f["category"] == "recruiter")}


def import_placement_report(db: Database, college_id: int, source: str,
                            replace_recruiters: bool = True) -> Dict[str, Any]:
    """Read a placement document (URL or local PDF path) and store its facts for the college.

    replace_recruiters: for a placement table, drop the college's existing recruiter facts first. An official
    report is more reliable than what the AI picked out of web pages (which can mistake job adverts for recruiters).
    NIRF data never touches recruiter facts."""
    from app.crawler import CollegeCrawler

    if re.match(r"https?://", source):
        resp = requests.get(source, headers={"User-Agent": Config.USER_AGENT}, timeout=60)
        resp.raise_for_status()
        data, url = resp.content, source
        name = requests.utils.unquote(source.rsplit("/", 1)[-1])
    else:
        path = Path(source).expanduser()
        data, url, name = path.read_bytes(), path.resolve().as_uri(), path.name
    title = re.sub(r"\.pdf$", "", name, flags=re.I)
    doc = analyse_document(read_pdf_text(data), name)
    if not doc:
        raise ValueError("No placement data found. Expected a placement table (S.No | Reg. No. | Name | ... | "
                         "Company) or NIRF 'Placement & Higher Studies' data.")
    if doc["kind"] == "table" and replace_recruiters:
        db.delete_facts(college_id, "recruiter")
    added = save_document(db, college_id, url, doc, CollegeCrawler(db))
    logger.info(f"Placement document {title} ({doc['kind']}): {doc['rows']} rows, {doc['companies']} companies, "
                f"{added} new facts")
    return {"kind": doc["kind"], "rows": doc["rows"], "companies": doc["companies"], "facts_added": added,
            "year": doc["year"], "title": title}


def save_document(db: Database, college_id: int, url: str, doc: Dict[str, Any], crawler: Any,
                  save_page: bool = True) -> int:
    """Store an analysed document's facts, and its name-free summary as a searchable page (so questions like
    "Does Amazon recruit here?" find it). Returns how many facts were new."""
    db.delete_facts(college_id, "placement_stat", source_url=url)   # re-importing refreshes the numbers
    added = db.save_facts(college_id, doc["facts"], url)
    if save_page:
        page_id = db.save_page(college_id, url, doc["page_title"], doc["page_text"])
        db.delete_chunks_for_page(page_id)
        chunks = crawler.create_chunks(crawler.markdown_to_sections(doc["page_text"]), doc["page_title"])
        db.save_chunks_batch(college_id, page_id, url, doc["page_title"], chunks)
    return added
