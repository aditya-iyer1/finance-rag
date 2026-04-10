'''
pdf.loader.py

Functions:
- Load 10-K PDF Files
- Extract clean text
- Identify key section headers
- Return dictionary: {section_title: section_text}
'''


import fitz
import re
from typing import Dict, Optional

CANONICAL_10K_ITEM_TITLES = {
    "ITEM 1": "BUSINESS",
    "ITEM 1A": "RISK FACTORS",
    "ITEM 1B": "UNRESOLVED STAFF COMMENTS",
    "ITEM 1C": "CYBERSECURITY",
    "ITEM 2": "PROPERTIES",
    "ITEM 3": "LEGAL PROCEEDINGS",
    "ITEM 4": "MINE SAFETY DISCLOSURES",
    "ITEM 5": "MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES",
    "ITEM 6": "[RESERVED]",
    "ITEM 7": "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    "ITEM 7A": "QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
    "ITEM 8": "FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
    "ITEM 9": "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE",
    "ITEM 9A": "CONTROLS AND PROCEDURES",
    "ITEM 9B": "OTHER INFORMATION",
    "ITEM 9C": "DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPECTIONS",
    "ITEM 10": "DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE",
    "ITEM 11": "EXECUTIVE COMPENSATION",
    "ITEM 12": "SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND RELATED STOCKHOLDER MATTERS",
    "ITEM 13": "CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS, AND DIRECTOR INDEPENDENCE",
    "ITEM 14": "PRINCIPAL ACCOUNTANT FEES AND SERVICES",
    "ITEM 15": "EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
    "ITEM 16": "FORM 10-K SUMMARY",
}


def _clean_toc_artifacts(line: str) -> str:
    """Remove dot leaders and trailing page numbers from a candidate header line."""
    cleaned = re.sub(r'\.{2,}\s*\d+\s*$', '', line)
    cleaned = re.sub(r'(?:\s*\.\s*){2,}\s*\d*\s*$', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def _extract_section_title(item_part: str, content: str) -> str:
    """
    Extract a clean section title without swallowing the first sentence of body text.

    SEC filings often render as:
    - "Item 1A.\nRisk Factors"
    - "Item 1A.\nRisk Factors. . . . 23"
    - "Item 1A. Risk Factors Our business ..."
    """
    normalized_item = re.sub(r'\.+$', '', item_part).upper().strip()
    canonical_title = CANONICAL_10K_ITEM_TITLES.get(normalized_item)
    if canonical_title:
        return f"{normalized_item}. {canonical_title}"

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    first_line = _clean_toc_artifacts(lines[0]) if lines else ""

    if not first_line:
        return normalized_item

    title_candidate = first_line

    # If body text was merged onto the header line, keep only the header-like prefix.
    merged_header_match = re.match(
        r"^([A-Z][A-Za-z0-9&/,\-()' ]{0,120}?)(?=(?:\.\s+[A-Z][a-z])|(?:\s+[A-Z][a-z]+(?:\s+[a-z]{2,}){2,})|$)",
        title_candidate,
    )
    if merged_header_match:
        title_candidate = merged_header_match.group(1).strip(" .")

    # Extremely long "titles" are usually leaked body text. Fall back to a compact prefix.
    words = title_candidate.split()
    if len(words) > 8:
        title_candidate = " ".join(words[:8]).rstrip(" .")

    title_candidate = re.sub(r'\s+', ' ', title_candidate).strip(" .")
    if not title_candidate:
        return normalized_item

    return f"{normalized_item}. {title_candidate.upper()}"

def extract_raw_text_from_pdf(path: str) -> str:
	'''
	Extract raw text from a PDF file using PyMuPDF
	'''
	doc = fitz.open(path)
	text = ""
	for page in doc:
		text += page.get_text()
	return text

def split_into_sections(text: str) -> Dict[str, str]:
    # Pattern handles both formats:
    # - Tesla: "Item 1.\nBusiness\n4\nItem 1A." (title on separate line)
    # - JPMC: "Item 1.\nBusiness.    . . . . . . .\n1\nItem 1A." (title with dot leaders & page num)
    # The lookahead matches the next Item header or end of string
    pattern = re.compile(
        r"^(Item\s+\d{1,2}[A-C]?\.?).*?(\n|\Z)(.+?)"
        r"(?=^Item\s+\d{1,2}[A-C]?|\Z)",
        re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    matches = list(pattern.finditer(text))
    sections = {}

    covered_spans = []
    for i, match in enumerate(matches):
        item_part = match.group(1).strip()
        content = match.group(3).strip()
        title_line = _extract_section_title(item_part, content)
        
        covered_spans.append((match.start(), match.end()))

        if title_line in sections:
            title_line = f"{title_line} (part {i})"

        sections[title_line] = match.group(1).strip() + "\n" + content

    # Append leftover text (outside ITEMs)
    last_match_end = covered_spans[-1][1] if covered_spans else 0
    if last_match_end < len(text):
        leftover = text[last_match_end:].strip()
        if len(leftover) > 100:  # skip short footers or blanks
            sections["EXHIBITS / CERTIFICATIONS / MISC"] = leftover

    return sections

def parse_pdf_sections(path: str) -> Dict[str, str]:
	'''
	Full pipeline: load PDF, extract text, split into sections
	'''

	raw_text = extract_raw_text_from_pdf(path)
	sections = split_into_sections(raw_text)
	return sections
