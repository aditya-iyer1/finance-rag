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
        # Extract: "Item 1" or "Item 1A" (without the trailing dot for clean naming)
        item_part = match.group(1).strip()
        # Extract title from content (first line after item number)
        content = match.group(3).strip()
        first_line = content.split('\n')[0].strip() if content else ""
        # Clean title: remove dot leaders (". . . .") and page numbers like "32"
        title_clean = re.sub(r'[.\s]+\.\s*', ' ', first_line)  # "Business.    . . ." -> "Business"
        title_clean = re.sub(r'\s+\d+\s*$', '', title_clean)  # Remove trailing page numbers
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()  # Normalize whitespace
        # Combine: "Item 1. Business" (avoid double dots)
        if title_clean:
            # item_part is like "Item 1." or "Item 1A." - already has trailing dot
            title_line = f"{item_part} {title_clean}"
        else:
            title_line = item_part.rstrip('.')
        
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

