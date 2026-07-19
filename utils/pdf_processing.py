"""
Stage 1 - PDF Processing
========================
Extracts the raw text of a research paper PDF with PyMuPDF and then splits
that text into the standard sections of a scientific paper
(Abstract, Introduction, ..., References) using a simple heading heuristic.

Why a heuristic and not an ML model?
    Scientific papers almost always mark their sections with short heading
    lines such as "1 Introduction" or "5. Conclusion".  Detecting those
    lines with a regular expression is simple, fast, has no extra
    dependencies, and is easy to explain in a presentation.
"""

import os
import re

import fitz  # PyMuPDF - the library is imported under the name "fitz"

# The section names we try to detect.  Some common synonyms are mapped to
# one canonical name so the rest of the system only sees the 8 standard
# section labels required by the project description.
SECTION_SYNONYMS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related work",
    "related works": "related work",
    "background": "related work",
    "literature review": "related work",
    "methodology": "methodology",
    "method": "methodology",
    "methods": "methodology",
    "proposed method": "methodology",
    "approach": "methodology",
    "model": "methodology",
    "experiments": "experiments",
    "experiment": "experiments",
    "experimental setup": "experiments",
    "experimental results": "results",
    "results": "results",
    "results and discussion": "results",
    "evaluation": "results",
    "discussion": "results",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "references": "references",
    "bibliography": "references",
}

# A heading line looks like:  optional number ("1", "3.2", "IV.") + section name
# Examples that match: "Abstract", "1 Introduction", "5. Conclusion", "IV. RESULTS"
HEADING_PATTERN = re.compile(
    r"^\s*(?:[0-9]+\.?[0-9]*\.?|[IVX]+\.?)?\s*([A-Za-z][A-Za-z ]{2,40})\s*$"
)


def extract_text(pdf_path):
    """
    Purpose : Read every page of a PDF and return its full plain text.
    Inputs  : pdf_path (str) - path to the PDF file.
    Outputs : full_text (str) - the concatenated text of all pages.
    """
    document = fitz.open(pdf_path)
    pages = []
    for page in document:
        pages.append(page.get_text())  # plain-text extraction per page
    document.close()
    return "\n".join(pages)


def split_into_sections(full_text):
    """
    Purpose : Split the full text of a paper into its major sections.
    Inputs  : full_text (str) - output of extract_text().
    Outputs : sections (dict) - maps canonical section name -> section text.
              If no headings are found at all, everything is returned under
              the key "full text" so the pipeline never fails.
    """
    lines = full_text.split("\n")

    # Step 1: find the line numbers where a known section heading starts.
    boundaries = []  # list of (line_index, canonical_section_name)
    for i, line in enumerate(lines):
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        candidate = match.group(1).strip().lower()
        if candidate in SECTION_SYNONYMS:
            boundaries.append((i, SECTION_SYNONYMS[candidate]))

    # Fallback: no headings detected (e.g. a scanned or unusual PDF layout)
    if not boundaries:
        return {"full text": full_text}

    # Step 2: the text of a section runs from its heading line to the
    # line before the next heading.
    sections = {}
    for j, (start_line, name) in enumerate(boundaries):
        end_line = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(lines)
        text = "\n".join(lines[start_line + 1 : end_line]).strip()
        if not text:
            continue
        if name in sections:
            # Same canonical name found twice (e.g. "Results" and
            # "Discussion" both map to "results") -> just append.
            sections[name] += "\n" + text
        else:
            sections[name] = text

    # Keep any text that appears before the first detected heading
    # (usually the title and author list) under its own key.
    preamble = "\n".join(lines[: boundaries[0][0]]).strip()
    if preamble:
        sections["title & authors"] = preamble

    return sections


def process_pdf(pdf_path):
    """
    Purpose : Convenience function - complete Stage 1 for a single paper.
    Inputs  : pdf_path (str) - path to the PDF file.
    Outputs : (paper_name, sections)
              paper_name (str)  - the file name without ".pdf", used as the
                                  paper identifier everywhere in the system.
              sections (dict)   - canonical section name -> text.
    """
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    full_text = extract_text(pdf_path)
    sections = split_into_sections(full_text)
    return paper_name, sections
