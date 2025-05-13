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
