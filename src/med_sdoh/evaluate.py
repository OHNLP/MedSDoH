import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from med_sdoh.utils import Corpus, compute_classification_metrics, parse_dtd_schema


class Evaluation:
    def __init__(
        self,
        model_corpus: Corpus,
        gold_standard_corpus: Corpus,
        schema_path: str | pathlib.Path,
        overlap_ratio=0.5,
        excluded_concepts: list[str] = [],
    ):

        self.schema = parse_dtd_schema(schema_path)
        self.span, self.text = self._get_spans(model_corpus, gold_standard_corpus)
        self.overlap_ratio = overlap_ratio
        self.excluded_concepts = excluded_concepts
        self._micro_avg = None
        self._macro_avg = None
        self._errors = None

    @property
    def micro_avg(self) -> dict:
        if self._micro_avg is None:
            self._micro_avg = self._get_micro_avg()
        return self._micro_avg

    @property
    def macro_avg(self) -> pd.DataFrame:
        if self._macro_avg is None:
            self._macro_avg = self._get_macro_avg()
        return self._macro_avg

    @property
    def errors(self) -> pd.DataFrame:
        if self._errors is None:
            self._errors = self._get_errors()
        return self._errors

    def refresh(self):
        self._micro_avg = None
        self._macro_avg = None
        self._errors = None

    def _get_micro_avg(self) -> dict:
        """
        Compute micro-average precision, recall, F1 score, and IAA across all concepts.
        Returns:
            dict with precision, recall, f1, iaa
        """
        tp_total, fp_total, fn_total = 0, 0, 0

        print("TOTAL MICRO AVERAGE:")
        print("(Aggregating the contributions of all classes)")

        for doc in self.span:
            for concept in self.span[doc]:
                if concept in self.excluded_concepts:
                    continue

                ann1_spans = self.span[doc][concept]["ann1"]
                ann2_spans = self.span[doc][concept]["ann2"]

                tp, fp, fn, *_ = self._cal_matching_overlap_with_text(
                    ann1_spans, ann2_spans, doc, concept
                )

                tp_total += tp
                fp_total += fp
                fn_total += fn

        return compute_classification_metrics(tp_total, fp_total, fn_total)

    def _get_macro_avg(self) -> pd.DataFrame:

        concept_scores = {}

        # Aggregate TP, FP, FN by concept across all documents
        for doc in self.span:
            for concept in self.span[doc]:
                if concept in self.excluded_concepts:
                    continue

                ann1_spans = self.span[doc][concept]["ann1"]
                ann2_spans = self.span[doc][concept]["ann2"]

                tp, fp, fn, *_ = self._cal_matching_overlap_with_text(
                    ann1_spans, ann2_spans, doc, concept
                )

                if concept not in concept_scores:
                    concept_scores[concept] = {"tp": 0, "fp": 0, "fn": 0}
                concept_scores[concept]["tp"] += tp
                concept_scores[concept]["fp"] += fp
                concept_scores[concept]["fn"] += fn

        # Compute metrics per concept
        rows = []
        for concept, counts in concept_scores.items():
            tp_k, fp_k, fn_k = counts["tp"], counts["fp"], counts["fn"]
            nb_tags = tp_k + fp_k + fn_k

            if nb_tags == 0:
                print(f"No annotations found for concept: {concept}. Assigning NaN.")
                precision, recall, f1 = np.nan, np.nan, np.nan
            else:
                metrics = compute_classification_metrics(tp_k, fp_k, fn_k)
                precision = metrics["precision"]
                recall = metrics["recall"]
                f1 = metrics["f1"]

            rows.append(
                {
                    "concept_name": concept,
                    "tp": tp_k,
                    "fp": fp_k,
                    "fn": fn_k,
                    "nb_tags": nb_tags,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        # Convert to DataFrame
        df = pd.DataFrame(rows)
        df = df.sort_values("nb_tags", ascending=False)

        # Macro average
        macro_avg = {
            "concept_name": "Macro Average",
            "tp": df["tp"].sum(),
            "fp": df["fp"].sum(),
            "fn": df["fn"].sum(),
            "nb_tags": df["nb_tags"].sum(),
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "f1": df["f1"].mean(),
        }

        # Weighted average
        proportions = df["nb_tags"] / df["nb_tags"].sum()
        df["weighted_f1"] = df["f1"] * proportions
        df["weighted_precision"] = df["precision"] * proportions
        df["weighted_recall"] = df["recall"] * proportions

        weighted_avg = {
            "concept_name": "Weighted Average",
            "tp": df["tp"].sum(),
            "fp": df["fp"].sum(),
            "fn": df["fn"].sum(),
            "nb_tags": df["nb_tags"].sum(),
            "precision": None,
            "recall": None,
            "f1": None,
            "weighted_precision": df["weighted_precision"].sum(),
            "weighted_recall": df["weighted_recall"].sum(),
            "weighted_f1": df["weighted_f1"].sum(),
        }

        return pd.concat(
            [df, pd.DataFrame([macro_avg, weighted_avg])], ignore_index=True
        )

    def _get_errors(self) -> pd.DataFrame:
        """
        Return a DataFrame of span-level FP and FN errors with full text and concept name.
        """
        error_results = []

        for doc in self.span:
            for concept in self.span[doc]:
                ann1_spans = self.span[doc][concept]["ann1"]
                ann2_spans = self.span[doc][concept]["ann2"]

                (
                    *_,
                    fp_texts,
                    fn_texts,
                ) = self._cal_matching_overlap_with_text(
                    ann1_spans, ann2_spans, doc, concept
                )

                for error in fp_texts + fn_texts:
                    error_results.append(
                        {
                            "file": doc,
                            "concept": concept,
                            "span": error["span"],
                            "error_type": error["error_type"],
                            "text": error["text"],
                            "sentence": error["sentence"],
                        }
                    )

        df = pd.DataFrame(
            error_results,
            columns=["file", "concept", "span", "error_type", "text", "sentence"],
        )

        return df[~df["concept"].isin(self.excluded_concepts)].reset_index(drop=True)

    def _is_valid_annotation(self, attributes: dict) -> bool:
        """Ensure certainty is positive/confirmed and experiencer is patient."""
        return (
            attributes.get("certainty", "").lower() in {"positive", "confirmed"}
            and attributes.get("experiencer", "").lower() == "patient"
        )

    def _overlap(self, idx_a_str: str, ann2_spans: list[str]) -> bool:
        """Check if two spans overlap based on the given overlap ratio."""
        start_a, end_a = map(int, idx_a_str.split("~"))
        range_a = set(range(start_a, end_a))

        for idx_b_str in ann2_spans:
            start_b, end_b = map(int, idx_b_str.split("~"))
            range_b = set(range(start_b, end_b))

            iou = len(range_a & range_b) / len(
                range_a | range_b
            )  # intersection over union
            if iou >= self.overlap_ratio:
                return True

        return False

    def _cal_matching_overlap_with_text(self, ann1, ann2, doc, concept):
        tp, fp, fn = 0, 0, 0
        matched_ann2 = set()
        tp_texts, fp_texts, fn_texts = [], [], []

        ann1_data = self.text[doc][concept]["ann1"]
        ann2_data = self.text[doc][concept]["ann2"]

        # Process ann1 spans (model predictions)
        for span1 in ann1:
            entry1 = ann1_data.get(span1)
            if not entry1:
                continue

            is_valid1 = self._is_valid_annotation(entry1["attributes"])
            sentence1 = entry1["sentence"]
            found_match = False

            # Try to match against each gold (ann2) span
            for span2 in ann2:
                entry2 = ann2_data.get(span2)
                if not entry2:
                    continue

                is_valid2 = self._is_valid_annotation(entry2["attributes"])
                sentence2 = entry2["sentence"]

                if span1 == span2 or self._overlap(span1, [span2]):
                    # Found an overlapping span
                    found_match = True
                    matched_ann2.add(span2)

                    if is_valid1 and is_valid2:
                        # True Positive
                        tp += 1
                        tp_texts.append(
                            {
                                "text": entry1["attributes"].get("text", ""),
                                "span": span1,
                                "error_type": "TP",
                                "sentence": sentence1,
                            }
                        )
                    elif is_valid2 and not is_valid1:
                        # False Negative (model missed a valid gold annotation)
                        fn += 1
                        fn_texts.append(
                            {
                                "text": entry2["attributes"].get("text", ""),
                                "span": span2,
                                "error_type": "FN",
                                "sentence": sentence2,
                            }
                        )
                    break  # Stop checking after first match

            if not found_match and is_valid1:
                fp += 1
                fp_texts.append(
                    {
                        "text": entry1["attributes"].get("text", ""),
                        "span": span1,
                        "error_type": "FP",
                        "sentence": sentence1,
                    }
                )

        # Process ann2 spans that were not matched (additional FNs)
        for span2 in ann2:
            if span2 in matched_ann2:
                continue

            entry2 = ann2_data.get(span2)
            if not entry2:
                continue

            is_valid2 = self._is_valid_annotation(entry2["attributes"])
            sentence2 = entry2["sentence"]

            if is_valid2:
                fn += 1
                fn_texts.append(
                    {
                        "text": entry2["attributes"].get("text", ""),
                        "span": span2,
                        "error_type": "FN",
                        "sentence": sentence2,
                    }
                )

        return tp, fp, fn, tp_texts, fp_texts, fn_texts

    def _get_spans(self, model_corpus: Corpus, gold_standard_corpus: Corpus):
        span = defaultdict(dict)
        text = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for doc_name in model_corpus.data:
            if doc_name not in gold_standard_corpus.data:
                continue

            concepts_in_doc = set(model_corpus.data[doc_name]) | set(
                gold_standard_corpus.data[doc_name]
            )

            for concept in concepts_in_doc:
                model_tags = model_corpus.data[doc_name].get(concept, [])
                gold_tags = gold_standard_corpus.data[doc_name].get(concept, [])

                span[doc_name][concept] = {"ann1": [], "ann2": []}

                for tag in model_tags:
                    if "spans" in tag and "," not in tag["spans"]:
                        span_value = tag["spans"]
                        span[doc_name][concept]["ann1"].append(span_value)
                        text[doc_name][concept]["ann1"][span_value] = {
                            "sentence": tag.get("sentence", ""),
                            "attributes": tag,
                        }

                for tag in gold_tags:
                    if "spans" in tag and "," not in tag["spans"]:
                        span_value = tag["spans"]
                        span[doc_name][concept]["ann2"].append(span_value)
                        text[doc_name][concept]["ann2"][span_value] = {
                            "sentence": tag.get("sentence", ""),
                            "attributes": tag,
                        }

        return dict(span), dict(text)


if __name__ == "__main__":
    from med_sdoh.utils import Corpus

    model_corpus = Corpus("path/to/model_corpus")
    gold_standard_corpus = Corpus("path/to/gold_standard_corpus")
    schema_path = "path/to/schema.dtd"
    excluded_concepts = [
        "Sex_At_Birth",
        "Race_or_Ethnicity",
        "Sexual_Orientation",
        "Marital_Status",
    ]

    eval = Evaluation(
        model_corpus,
        gold_standard_corpus,
        schema_path,
        overlap_ratio=0.5,
        excluded_concepts=excluded_concepts,
    )

    print(eval.micro_avg)
    print(eval.macro_avg)
    print(eval.errors)
