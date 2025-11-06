import pathlib
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from bs4 import BeautifulSoup


class Corpus:
    def __init__(self, corpus_dir: str | pathlib.Path):

        self.corpus_dir = pathlib.Path(corpus_dir)
        self.files = list(self.corpus_dir.glob("*.xml"))
        self.data = {}

        if not self.files:
            raise FileNotFoundError(f"No .xml files found in {self.corpus_dir}")

        for file in self.files:
            self.data[file.name] = load_annotations(file)


def extract_text_from_xml(xml_file_path: str | Path) -> str:
    """Parse the XML file and extract the text content from the TEXT tag."""
    tree = ET.parse(Path(xml_file_path))
    root = tree.getroot()

    text_element = root.find("TEXT")
    return text_element.text.strip()


def compute_char_shift(original_text: str, truncated_text: str) -> dict:
    """
    Compute how character positions shift between original and truncated text. This is to fix the annotator's output .xml file after truncating the text.
    """
    shift_map = {}
    matcher = SequenceMatcher(None, original_text, truncated_text)

    truncated_pos = 0

    for opcode in matcher.get_opcodes():
        tag, i1, i2, j1, j2 = opcode

        if tag == "equal":
            for i in range(i1, i2):
                shift_map[i] = truncated_pos + (i - i1)
            truncated_pos += j2 - j1

        elif tag == "delete":
            # Characters that were removed
            for i in range(i1, i2):
                shift_map[i] = -1
        elif tag == "insert":
            # Move truncated_pos forward for new insertions
            truncated_pos += j2 - j1

        elif tag == "replace":
            # Handle replaced text by mapping deleted characters
            for i in range(i1, i2):
                shift_map[i] = -1
            truncated_pos += j2 - j1

    return shift_map


def find_all_occurrences(text: str, phrase: str) -> list:
    """Find all occurrences of a phrase in a text."""
    positions = []
    start = 0
    while start < len(text):
        start = text.find(phrase, start)
        if start == -1:
            break
        positions.append(start)
        start += len(phrase)
    return positions


def extract_sentence(text: str, span_start: int, span_end: int) -> str:
    """Break text into sentences using newlines or punctuation"""
    sentence_bounds = [
        (m.start(), m.end()) for m in re.finditer(r"[^.\n!?]+[.\n!?]+", text)
    ]

    for start_i, end_i in sentence_bounds:
        if start_i <= span_start < end_i:
            return text[start_i:end_i].strip()

    return text[span_start:span_end]


def load_annotations(file_path: str | Path) -> dict:
    """Extract annotations from the XML file."""
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")

    text = soup.find("TEXT").get_text()
    tags = soup.find("TAGS").find_all()

    data = defaultdict(list)

    for tag in tags:
        concept = tag.name
        attrs = dict(tag.attrs)

        if "spans" in attrs and "," not in attrs["spans"]:
            start, end = map(int, attrs["spans"].split("~"))
            attrs["sentence"] = extract_sentence(text, start, end)

        data[concept].append(attrs)

    return dict(data)


def parse_dtd_schema(dtd_path: str | pathlib.Path) -> dict:
    """Parse a DTD file and extract tag-attribute mappings."""
    with open(dtd_path, "r") as f:
        txt = f.read()

    schema = {}
    for line in txt.splitlines():
        if "<!ATTLIST" in line:
            parts = line.split()
            if len(parts) < 3:
                continue
            tag, attr = parts[1], parts[2]
            values = (
                [v.strip() for v in line.split("(")[1].split(")")[0].split("|")]
                if "(" in line and ")" in line
                else []
            )
            schema.setdefault(tag, {})[attr] = values

    return schema


def count_regex_patterns(pattern: str) -> int:
    """
    Count the number of top-level alternatives in a regex pattern.

    Each top-level alternative (separated by | at depth 0) is counted as a separate pattern.
    This handles nested parentheses, escaped characters, and character classes correctly.

    Args:
        pattern: The regex pattern string to count

    Returns:
        Number of top-level alternatives (patterns)

    Examples:
        >>> count_regex_patterns("\\b(word1|word2|word3)\\b")
        3
        >>> count_regex_patterns("\\b(lack of (supplies|resources)|shortage of (food|water))\\b")
        2
    """
    if not pattern or pattern.strip().startswith("//"):
        return 0

    pattern = pattern.strip()
    if not pattern:
        return 0

    # Remove word boundaries and anchors for counting if present. Handle patterns like \b(...)\b or ^(...)$
    if pattern.startswith("\\b(") and pattern.endswith(")\\b"):
        inner_pattern = pattern[3:-3]  # Remove \b( and )\b
    elif pattern.startswith("(") and pattern.endswith(")"):
        inner_pattern = pattern[1:-1]  # Remove ( and )
    else:
        inner_pattern = pattern

    # Count top-level | separators
    depth = 0
    in_char_class = False
    count = 1
    i = 0

    while i < len(inner_pattern):
        char = inner_pattern[i]

        if char == "\\":
            # Skip escaped character (next char is part of escape sequence)
            i += 2
            continue
        elif char == "[" and not in_char_class:
            # Enter character class
            in_char_class = True
        elif char == "]" and in_char_class:
            # Exit character class
            in_char_class = False
        elif char == "(" and not in_char_class:
            # Enter group (non-capturing groups like (?:...) also increase depth)
            depth += 1
        elif char == ")" and not in_char_class:
            depth -= 1
        elif char == "|" and depth == 0 and not in_char_class:
            # Top-level alternative separator
            count += 1

        i += 1

    return count


def count_patterns_from_file(file_path: str | Path) -> int:
    """
    Count total regex patterns from a file, counting each top-level alternative separately.

    Each non-empty, non-comment line is processed, and top-level alternatives within
    each line are counted separately.

    Args:
        file_path: Path to the regex pattern file

    Returns:
        Total number of patterns (top-level alternatives) in the file
    """
    file_path = Path(file_path)
    total_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                total_count += count_regex_patterns(line)

    return total_count


def compute_classification_metrics(tp: int, fp: int, fn: int) -> dict:
    """
    Calculate precision, recall, F1 score, and IAA.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iaa = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "iaa": iaa}
