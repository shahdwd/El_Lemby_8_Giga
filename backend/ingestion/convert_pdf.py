"""
PDF to Canonical Markdown Converter — parses Egyptian law PDF files:
  1. Extract raw text from PDF
  2. Normalize Unicode presentation forms to standard Arabic
  3. Reverse visual line ordering to logical right-to-left (handling split words and layout anomalies)
  4. Segment text into canonical "# ARTICLE <n>" blocks (supporting various reversed layout formats)
  5. Save to backend/data/laws/

Run:
    python -m backend.ingestion.convert_pdf
"""

import os
import re
import sys
import unicodedata
from pathlib import Path
import pypdf

# Try to import normalize_digits from backend, fallback if run standalone
try:
    from backend.ingestion.normalize import normalize_digits
except ImportError:
    # Standard fallback digit mapping
    _ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
    _DIGIT_MAP = {c: str(i) for i, c in enumerate(_ARABIC_INDIC)}
    def normalize_digits(s: str) -> str:
        return "".join(_DIGIT_MAP.get(c, c) for c in s)

# Default paths
DEFAULT_PDF_PATH = r"C:\Users\shwae\Downloads\law-131-1948.pdf"
OUTPUT_DIR = Path("backend/data/laws")


def reverse_line_tokens(line: str) -> str:
    """
    Reverse visual line ordering to logical right-to-left.
    Pypdf extracts visual RTL text in left-to-right word order.
    Reversing the tokens restores the logical reading flow.
    """
    line = unicodedata.normalize('NFKC', line).strip()
    
    # Merge split "مادة" and "المادة" (any spacing inside the word due to layout engine splits)
    line = re.sub(r'\bم\s+ادة', 'مادة', line)
    line = re.sub(r'\bما\s+دة', 'مادة', line)
    line = re.sub(r'\bماد\s+ة', 'مادة', line)
    
    line = re.sub(r'\bالم\s+ادة', 'المادة', line)
    line = re.sub(r'\bالما\s+دة', 'المادة', line)
    line = re.sub(r'\bالماد\s+ة', 'المادة', line)
    
    if not line:
        return ""
        
    # Split by word characters and punctuation, preserving groupings
    tokens = re.findall(r'\w+|[^\w\s]', line)
    
    # Reverse tokens
    reversed_tokens = tokens[::-1]
    
    # Reconstruct the line
    return " ".join(reversed_tokens)


def clean_sentence_wrapping(text: str) -> str:
    """
    Heuristics to join words split across lines due to layout wrapping.
    For example: 'يس' at end of line and 'ري' at start of next line.
    """
    # Join common legal word splits observed in extraction
    replacements = {
        r'\bيس\s+ري\b': 'يسري',
        r'\bمبا\s+شرتها\b': 'مباشرتها',
        r'\bالاع\s+مال\b': 'الأعمال',
        r'\bالأع\s+مال\b': 'الأعمال',
        r'\bفي\s+ه\b': 'فيه',
        r'\bبمو\s+ته\b': 'بموته',
        r'\bالقا\s+نون\b': 'القانون',
        r'\bالاح\s+كام\b': 'الأحكام',
        r'\bالمد\s+نية\b': 'المدنية',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text


def parse_pdf_to_canonical_md(pdf_path: str, law_name: str, law_number: str, law_year: str) -> str:
    """Extract and parse PDF content into the canonical markdown format."""
    print(f"Opening PDF: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages to process: {total_pages}")
    
    # Step 1 & 2: Extract, normalize and reverse lines page-by-page
    all_lines = []
    for idx in range(total_pages):
        page = reader.pages[idx]
        text = page.extract_text()
        if not text:
            continue
            
        for line in text.splitlines():
            cleaned_line = reverse_line_tokens(line)
            if cleaned_line:
                all_lines.append(cleaned_line)
                
    full_text = "\n".join(all_lines)
    full_text = clean_sentence_wrapping(full_text)
    
    # Step 3: Segment into articles
    articles = []
    current_article = None
    
    # Regex to match article declarations in various orientations
    # 1. Starts with مادة/المادة (e.g. مادة 22)
    start_pattern1 = re.compile(r'^(مادة|المادة)\s*([\d\u0660-\u0669]+)(.*)$', re.IGNORECASE)
    # 2. Ends with مادة/المادة (e.g. ... – مادة 22)
    end_pattern1 = re.compile(r'^(.*?)(?:[–-]\s*)?(مادة|المادة)\s*([\d\u0660-\u0669]+)$', re.IGNORECASE)
    # 3. Starts with number مادة/المادة (e.g. 22 مادة) due to visual token reversal
    start_pattern2 = re.compile(r'^([\d\u0660-\u0669]+)\s*(مادة|المادة)(.*)$', re.IGNORECASE)
    # 4. Ends with number مادة/المادة (e.g. ... – 22 مادة)
    end_pattern2 = re.compile(r'^(.*?)(?:[–-]\s*)?([\d\u0660-\u0669]+)\s*(مادة|المادة)$', re.IGNORECASE)
    
    lines = full_text.splitlines()
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        m_start1 = start_pattern1.match(line_stripped)
        m_end1 = end_pattern1.match(line_stripped)
        m_start2 = start_pattern2.match(line_stripped)
        m_end2 = end_pattern2.match(line_stripped)
        
        if m_start1:
            # Save previous article
            if current_article:
                articles.append(current_article)
                
            art_num = normalize_digits(m_start1.group(2))
            art_suffix = m_start1.group(3).strip()
            current_article = {
                "header": f"{art_num} {art_suffix}".strip(),
                "lines": []
            }
        elif m_start2:
            # Save previous article
            if current_article:
                articles.append(current_article)
                
            art_num = normalize_digits(m_start2.group(1))
            art_suffix = m_start2.group(3).strip()
            current_article = {
                "header": f"{art_num} {art_suffix}".strip(),
                "lines": []
            }
        elif m_end1:
            # Save previous article
            if current_article:
                articles.append(current_article)
                
            art_num = normalize_digits(m_end1.group(3))
            art_text = m_end1.group(1).strip()
            current_article = {
                "header": art_num,
                "lines": []
            }
            if art_text:
                current_article["lines"].append(art_text)
        elif m_end2:
            # Save previous article
            if current_article:
                articles.append(current_article)
                
            art_num = normalize_digits(m_end2.group(2))
            art_text = m_end2.group(1).strip()
            current_article = {
                "header": art_num,
                "lines": []
            }
            if art_text:
                current_article["lines"].append(art_text)
        else:
            if current_article:
                current_article["lines"].append(line_stripped)
                
    if current_article:
        articles.append(current_article)
        
    print(f"Extracted {len(articles)} articles.")
    
    # Step 4: Write as canonical markdown
    md_content = []
    
    # YAML Frontmatter
    md_content.append("---")
    md_content.append(f"law_name: {law_name}")
    md_content.append(f"law_number: {law_number}")
    md_content.append(f"law_year: {law_year}")
    md_content.append("---")
    md_content.append("")
    
    for art in articles:
        md_content.append(f"# ARTICLE {art['header']}")
        md_content.append("")
        md_content.append("## TEXT")
        md_content.append("")
        
        # Join lines
        art_text = "\n".join(art["lines"])
        # Clean up formatting
        art_text = re.sub(r'\s+([،.؛:])', r'\1', art_text)
        art_text = re.sub(r'\(\s+', '(', art_text)
        art_text = re.sub(r'\s+\)', ')', art_text)
        
        md_content.append(art_text)
        md_content.append("")
        md_content.append("## REFERENCES")
        md_content.append("")
        md_content.append("None")
        md_content.append("")
        md_content.append("---")
        md_content.append("")
        
    return "\n".join(md_content)


def main():
    pdf_path = DEFAULT_PDF_PATH
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)
        
    # Metadata for law-131-1948 (Civil Law)
    law_name = "القانون رقم 131 لسنة 1948 بإصدار القانون المدني"
    law_number = "131"
    law_year = "1948"
    
    output_filename = "القانون_رقم_131_لسنة_1948_canonical.md"
    output_path = OUTPUT_DIR / output_filename
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    md_text = parse_pdf_to_canonical_md(pdf_path, law_name, law_number, law_year)
    
    print(f"Writing canonical markdown to {output_path}")
    output_path.write_text(md_text, encoding="utf-8")
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
