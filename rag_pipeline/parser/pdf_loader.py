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
    pattern = re.compile(r"(ITEM\s+\d+[A]?(?:\.\d+)?\.?\s+.+?)(?=\nITEM\s+\d+[A]?|\Z)", re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(text))
    sections = {}

    covered_spans = []
    for i, match in enumerate(matches):
        title_line = match.group(1).split('\n')[0].strip()
        content = match.group(1).strip()
        covered_spans.append((match.start(), match.end()))

        if title_line in sections:
            title_line = f"{title_line} (part {i})"

        sections[title_line] = content

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

