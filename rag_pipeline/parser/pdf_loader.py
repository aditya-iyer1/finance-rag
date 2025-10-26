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
	'''
	Split 10-K or financial document into meaninful sections based on SEC-style item headers.

	Returns:
		Dict of {section_header: section_text}
	'''

	# Regex patter to find 10-K item headers
	pattern = re.compile(r"(ITEM\s+\d+[A]?(?:\.\d+)?\.?\s+.+?)(?=\nITEM\s+\d+[A]?|\Z)", re.IGNORECASE | re.DOTALL)

	matches = list(pattern.finditer(text))
	sections = {}

	for i, match in enumerate(matches):
		title_line = match.group(1).split('\n')[0].strip()
		content = match.group(1).strip()

		if title_line in sections:
			title_line = f"{title_line} (part {i})"

		sections[title_line] = content

	return sections

def parse_pdf_sections(path: str) -> Dict[str, str]:
	'''
	Full pipeline: load PDF, extract text, split into sections
	'''

	raw_text = extract_raw_text_from_pdf(path)
	sections = split_into_sections(raw_text)
	return sections

