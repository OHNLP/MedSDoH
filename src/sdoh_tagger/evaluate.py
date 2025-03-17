import csv
import glob
from os.path import expanduser

import pandas as pd
from bs4 import BeautifulSoup


class Corpus:
    def __init__(self, corpusDir, format="mae"):
        self.dir = corpusDir
        self.soup = {}
        if format == "mae":
            self.files = glob.glob(corpusDir + "/*.xml")
            self.load_MAE()
        if format == "brat":
            self.files = glob.glob(corpusDir + "/*.ann")
            self.load_Brat()

    def load_MAE(self, warn_and_continue=False):
        for p in self.files:
            name = p.split("/")[-1]
            ## standard:
            # name = p.split('/')[-1].split('_')[0]+'.xml'
            try:
                fp = open(p).read()
                soup = BeautifulSoup(fp, "xml")
                self.soup[name] = soup
            except:
                print("Error for file")

    def load_Brat(self, warn_and_continue=False):
        for p in self.files:
            ann_list = self.read_file_list(p)
            # name= '_'.join(p.split('/')[-1].split('_')[:4])
            ## standard:
            name = p.split("/")[-1].split("_")[0] + ".ann"

    def read_file_list(self, indir, d):
        opt_notes = []
        with open(indir, "rU") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=d)
            for row in spamreader:
                opt_notes += [row]
        return opt_notes


class Evaluation:
    def __init__(self, ann1Corpus, ann2Corpus, dtdDir, overlap_ratio):
        self.labels = {}
        self.schemaElementsAttr = self.load_dtd(dtdDir)
        self.spanCorpus, self.txtCorpus = self.get_spans(ann1Corpus, ann2Corpus)
        self.cpLevelOutput = []
        self.overlap_ratio = overlap_ratio

    def load_dtd(self, dtdDir):

        with open(dtdDir, "r") as f:
            txt = f.read()
        schemaElements = {}
        for i in txt.split("\n"):
            if "<!ATTLIST" in i:
                ele = i.split(" ")[1]
                schemaElements[ele] = {}
        for i in txt.split("\n"):
            if "<!ATTLIST" in i:
                ele = i.split(" ")[1]
                attr = i.split(" ")[2]
                if "(" in i and ")" in i:
                    attrList = [
                        sub_attr.strip()
                        for sub_attr in i.split("(")[1].split(")")[0].split("|")
                    ]
                else:
                    attrList = []
                if attr not in schemaElements[ele]:
                    schemaElements[ele][attr] = attrList
                else:
                    schemaElements[ele][attr] = schemaElements[ele][attr] + attrList
        return schemaElements

    def get_spans(self, ann1Corpus, ann2Corpus):
        spanCorpus, txtCorpus, attrCorpus = {}, {}, {}
        ann2Pool = [i.split("/")[-1] for i in ann2Corpus]
        for i in ann1Corpus:
            doc_name = i.split("/")[-1]
            if doc_name not in ann2Pool:
                continue
            spanCorpus[doc_name] = {}
            txtCorpus[doc_name] = {}
            soup = ann1Corpus[i]
            # for e in self.dtdElements
            for mae_concept in self.schemaElementsAttr:
                for item in soup.find_all(mae_concept):
                    mae_cp = item.name
                    try:
                        txt = item["text"]
                    except:
                        txt = ""
                    # get spans from MEA v0.9 and v2.0
                    if item.has_attr("spans"):
                        attrDict = {}
                        if "," in item["spans"]:
                            continue
                        key_ = doc_name + item.name + "ann1" + item["spans"]
                        attrList = self.schemaElementsAttr[mae_concept]
                        for at in attrList:
                            if item.has_attr(at):
                                if at not in attrDict:
                                    attrDict[at] = item[at]
                        try:
                            spanCorpus[doc_name][item.name]["ann1"].append(
                                item["spans"]
                            )
                        except KeyError:
                            spanCorpus[doc_name][item.name] = {
                                "ann1": [item["spans"]],
                                "ann2": [],
                            }
                        if key_ not in txtCorpus:
                            txtCorpus[key_] = [txt, attrDict]
                    elif item.has_attr("start"):
                        attrDict = {}
                        spans = item["start"] + "~" + item["end"]
                        key_ = doc_name + item.name + "ann1" + spans
                        attrList = self.schemaElementsAttr[mae_concept]
                        for at in attrList:
                            if item.has_attr(at):
                                if at not in attrDict:
                                    attrDict[at] = item[at]
                        try:
                            spanCorpus[doc_name][item.name]["ann1"].append(spans)
                        except KeyError:
                            spanCorpus[doc_name][item.name] = {
                                "ann1": [spans],
                                "ann2": [],
                            }
                        if key_ not in txtCorpus:
                            txtCorpus[key_] = [txt, attrDict]
        for i in ann2Corpus:
            doc_name = i.split("/")[-1]
            if doc_name not in spanCorpus:
                continue
            soup = ann2Corpus[i]
            for mae_concept in self.schemaElementsAttr:
                for item in soup.find_all(mae_concept):
                    try:
                        txt = item["text"]
                    except:
                        txt = ""
                    if item.has_attr("spans"):
                        if "," in item["spans"]:
                            continue
                        # spans = item['spans'].split('~')
                        txtCorpus[doc_name][item.name] = {"ann1": {}, "ann2": {}}
                        key_ = doc_name + item.name + "ann2" + item["spans"]
                        attrDict = {}
                        attrList = self.schemaElementsAttr[mae_concept]
                        for at in attrList:
                            if item.has_attr(at):
                                if at not in attrDict:
                                    attrDict[at] = item[at]
                        try:
                            spanCorpus[doc_name][item.name]["ann2"].append(
                                item["spans"]
                            )
                        except KeyError:
                            spanCorpus[doc_name][item.name] = {
                                "ann1": [],
                                "ann2": [item["spans"]],
                            }
                        if key_ not in txtCorpus:
                            txtCorpus[key_] = [txt, attrDict]
                    elif item.has_attr("start"):
                        spans = item["start"] + "~" + item["end"]
                        txtCorpus[doc_name][item.name] = {"ann1": {}, "ann2": {}}
                        key_ = doc_name + item.name + "ann2" + spans
                        attrDict = {}
                        attrList = self.schemaElementsAttr[mae_concept]
                        for at in attrList:
                            if item.has_attr(at):
                                if at not in attrDict:
                                    attrDict[at] = item[at]
                        try:
                            spanCorpus[doc_name][item.name]["ann2"].append(spans)
                        except KeyError:
                            spanCorpus[doc_name][item.name] = {
                                "ann1": [],
                                "ann2": [spans],
                            }
                        if key_ not in txtCorpus:
                            txtCorpus[key_] = [txt, attrDict]
        return spanCorpus, txtCorpus

    def get_micro_avg(self) -> tuple[float, float, float]:
        """Return micro average precision, recall, and F1 score."""
        tp_doc, fp_doc, fn_doc = 0, 0, 0
        print("TOTAL MICRO AVERAGE:")
        print("(Aggregation the contributions of all classes)")
        for doc in self.spanCorpus:
            for cp in self.spanCorpus[doc]:

                ann1_spans = self.spanCorpus[doc][cp]["ann1"]
                ann2_spans = self.spanCorpus[doc][cp]["ann2"]

                tp, fp, fn, *_ = self._cal_matching_overlap_with_text(
                    ann1_spans, ann2_spans, self.txtCorpus, doc, cp
                )
                tp_doc += tp
                fp_doc += fp
                fn_doc += fn
                self.cpLevelOutput += [[doc, cp, tp, fp, fn]]
        return self.calculate_performance(tp_doc, fp_doc, fn_doc)

    def get_macro_avg_per_category(self, excluded_categories=[]) -> pd.DataFrame:

        results = []
        tp_doc, fp_doc, fn_doc = 0, 0, 0

        for doc in self.spanCorpus:
            for cp in self.spanCorpus[doc]:

                ann1_spans = self.spanCorpus[doc][cp]["ann1"]
                ann2_spans = self.spanCorpus[doc][cp]["ann2"]

                tp, fp, fn, *_ = self._cal_matching_overlap_with_text(
                    ann1_spans, ann2_spans, self.txtCorpus, doc, cp
                )
                tp_doc += tp
                fp_doc += fp
                fn_doc += fn
                self.cpLevelOutput += [[doc, cp, tp, fp, fn]]

        cp_d = {}
        for cp in self.cpLevelOutput:
            if cp[1] not in cp_d:
                cp_d[cp[1]] = [cp[2], cp[3], cp[4]]
            else:
                cp_d[cp[1]] = [
                    cp_d[cp[1]][0] + cp[2],
                    cp_d[cp[1]][1] + cp[3],
                    cp_d[cp[1]][2] + cp[4],
                ]

        precision_macro, recall_macro, f1_macro = [], [], []
        for k in cp_d:
            if k in excluded_categories:  # skip excluded categories
                continue
            if cp_d[k][0] == 0:
                print("0 TP found in", k, "please double check your result")
                precision, recall, f1 = 0, 0, 0
            else:
                precision, recall, f1 = self.calculate_performance(
                    cp_d[k][0], cp_d[k][1], cp_d[k][2]
                )
            total_tags = int(
                cp_d[k][0] + cp_d[k][1] + cp_d[k][2]
            )  # Calculate # of tags
            results.append(
                {
                    "concept_name": k,
                    "tp": cp_d[k][0],
                    "fp": cp_d[k][1],
                    "fn": cp_d[k][2],
                    "nb_tags": total_tags,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
            precision_macro += [precision]
            recall_macro += [recall]
            f1_macro += [f1]

        # Make a dataframe and sort by # of tags
        df = pd.DataFrame(results)
        df = df.sort_values("nb_tags", ascending=False)

        # Get macro average
        macro_avg = {
            "concept_name": "",
            "tp": df["tp"].sum(),
            "fp": df["fp"].sum(),
            "fn": df["fn"].sum(),
            "nb_tags": df["nb_tags"].sum(),
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "f1": df["f1"].mean(),
        }
        return pd.concat([df, pd.DataFrame([macro_avg])], ignore_index=True)

    def print_mismatch(self, spanCorpus, txtCorpus, corpusDir):
        home = expanduser("~")
        txt = "annotation_file|concept_name|annotator|spans|agreement|text" + "\n"
        for doc in spanCorpus:
            for cp in spanCorpus[doc]:
                txt += self.print_cp_evidence(spanCorpus, doc, cp, txtCorpus)
        our_dir = corpusDir.replace("/input", "/output") + "/annotation_result.csv"
        with open(our_dir, "wb") as text_file:
            text_file.write(txt.encode("utf-8"))
        print("File saved at:", our_dir)

    def calculate_performance(self, tp, fp, fn) -> tuple[float, float, float]:
        try:
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            specificity = fn
            f1 = 2 * precision * recall / (precision + recall)
            iaa_ratio = tp / float(tp + fp + fn)
            return precision, recall, f1
        except:
            print("division zero error")

    def overlap(self, idx_a_str, ann2):
        for idx_b_str in ann2:
            idx_a = idx_a_str.split("~")
            idx_b = idx_b_str.split("~")
            aIndices = set(range(int(idx_a[0]), int(idx_a[1])))
            bIndices = set(range(int(idx_b[0]), int(idx_b[1])))
            overlap = len(aIndices & bIndices) / float(len(aIndices | bIndices))
            if overlap >= self.overlap_ratio:
                return True
        return False

    def cal_matching_exact(self, ann1, ann2):
        tp, fp = 0, 0
        for sp in ann1:
            if sp in ann2:
                tp += 1
            else:
                fp += 1
        fn = len(ann2) - tp
        return tp, fp, fn

    def is_valid_annotation(self, key: str) -> bool:
        """Ensure certainty is positive and experiencer is patient."""
        annotation_data = self.txtCorpus.get(key, ["", {}])
        attributes = annotation_data[1]  # Attributes dictionary
        return (
            attributes.get("certainty", "").lower() in ["positive", "confirmed"]
            and attributes.get("experiencer", "").lower() == "patient"
        )

    def _cal_matching_overlap_with_text(self, ann1, ann2, txtCorpus, doc, cp):
        tp, fp, fn = 0, 0, 0
        matched_ann2 = set()  # Track matched ann2 spans
        tp_texts, fp_texts, fn_texts = [], [], []

        # Process ann1 (checking TP and FP)
        for span1 in ann1:
            key1 = f"{doc}{cp}ann1{span1}"
            ann1_text = txtCorpus.get(key1, [""])[0]
            model_is_valid = self.is_valid_annotation(key1)
            found_match = False

            for span2 in ann2:  # gold standard
                key2 = f"{doc}{cp}ann2{span2}"
                ann2_text = txtCorpus.get(key2, [""])[0]
                ground_truth_valid = self.is_valid_annotation(key2)

                if span1 == span2 or self.overlap(span1, [span2]):
                    if model_is_valid and ground_truth_valid:
                        tp += 1
                        tp_texts.append(
                            {"text": ann1_text, "span": span1, "error_type": "TP"}
                        )
                        matched_ann2.add(span2)
                    elif (
                        ground_truth_valid
                        and not model_is_valid
                        and not any(
                            self.overlap(span, [span2])
                            for span in ann1
                            if self.is_valid_annotation(f"{doc}{cp}ann1{span}")
                        )
                    ):
                        fn += 1
                        if span2 not in [
                            e["span"] for e in fn_texts
                        ]:  # Prevent duplicate FNs
                            fn_texts.append(
                                {"text": ann2_text, "span": span2, "error_type": "FN"}
                            )
                    found_match = True
                    continue

            if (
                not found_match
                and model_is_valid
                and span1 not in [e["span"] for e in fp_texts]
            ):  # Avoid duplicate FP
                fp += 1
                fp_texts.append({"text": ann1_text, "span": span1, "error_type": "FP"})

        # Process ann2 (checking FN)
        for span2 in ann2:
            key2 = f"{doc}{cp}ann2{span2}"
            ann2_text = txtCorpus.get(key2, [""])[0]
            ground_truth_valid = self.is_valid_annotation(key2)

            if (
                span2 not in matched_ann2
                and ground_truth_valid
                and span2 not in [e["span"] for e in fn_texts]
            ):  # Prevent duplicate FN
                fn += 1
                fn_texts.append({"text": ann2_text, "span": span2, "error_type": "FN"})

        return tp, fp, fn, tp_texts, fp_texts, fn_texts

    def apply_transpose(self, attrDict, cp):
        for cp_ in self.schemaElementsAttr:
            if cp_ == cp:
                print(cp, self.schemaElementsAttr[cp])

    def print_cp_evidence(self, spanCorpus, doc, cp, txtCorpus):
        ann1 = spanCorpus[doc][cp]["ann1"]
        ann2 = spanCorpus[doc][cp]["ann2"]
        txt = ""
        tp, fp = 0, 0
        for sp in ann1:
            key_ = doc + cp + "ann1" + sp
            # self.apply_transpose(txtCorpus[key_][1], cp)
            att_txt = ""
            for m in txtCorpus[key_][1]:
                att_txt += m + ": " + txtCorpus[key_][1][m] + "|"
            if sp in ann2:
                tp += 1
                txt += (
                    doc
                    + "|"
                    + cp
                    + "|"
                    + "ann1"
                    + "|"
                    + sp
                    + "|"
                    + "agree"
                    + "|"
                    + txtCorpus[key_][0]
                    + "|"
                    + att_txt
                    + "\n"
                )
            elif self.overlap(sp, ann2):
                tp += 1
                txt += (
                    doc
                    + "|"
                    + cp
                    + "|"
                    + "ann1"
                    + "|"
                    + sp
                    + "|"
                    + "agree"
                    + "|"
                    + txtCorpus[key_][0]
                    + "|"
                    + att_txt
                    + "\n"
                )
            else:
                fp += 1
                txt += (
                    doc
                    + "|"
                    + cp
                    + "|"
                    + "ann1"
                    + "|"
                    + sp
                    + "|"
                    + "disagree"
                    + "|"
                    + txtCorpus[key_][0]
                    + "|"
                    + att_txt
                    + "\n"
                )
        for sp2 in ann2:
            key_ = doc + cp + "ann2" + sp2
            att_txt = ""
            for m in txtCorpus[key_][1]:
                att_txt += m + ": " + txtCorpus[key_][1][m] + "|"
            if sp2 in ann1:
                txt += (
                    doc
                    + "|"
                    + cp
                    + "|"
                    + "ann2"
                    + "|"
                    + sp2
                    + "|"
                    + "agree"
                    + "|"
                    + txtCorpus[key_][0]
                    + "|"
                    + att_txt
                    + "\n"
                )
            elif self.overlap(sp2, ann1):
                txt += (
                    doc
                    + "|"
                    + cp
                    + "|"
                    + "ann2"
                    + "|"
                    + sp2
                    + "|"
                    + "agree"
                    + "|"
                    + txtCorpus[key_][0]
                    + "|"
                    + att_txt
                    + "\n"
                )
            else:
                txt += (
                    doc
                    + "|"
                    + cp
                    + "|"
                    + "ann2"
                    + "|"
                    + sp2
                    + "|"
                    + "disagree"
                    + "|"
                    + txtCorpus[key_][0]
                    + "|"
                    + att_txt
                    + "\n"
                )
        fn = len(ann2) - tp
        return txt

    def get_errors_per_category(self):
        error_results = []

        for doc in self.spanCorpus:
            for cp in self.spanCorpus[doc]:
                ann1_spans = self.spanCorpus[doc][cp]["ann1"]
                ann2_spans = self.spanCorpus[doc][cp]["ann2"]

                text = self.txtCorpus.get(doc, {}).get(cp, ["Sentence not found"])
                error_sentence = text[0] if isinstance(text, list) else text

                for span in ann1_spans:
                    if span not in ann2_spans and not self.overlap(span, ann2_spans):
                        error_results.append(
                            {
                                "sentence": error_sentence,
                                "error_type": "FN",
                                "category": cp,
                            }
                        )

                for span in ann2_spans:
                    if span not in ann1_spans and not self.overlap(span, ann1_spans):
                        error_results.append(
                            {
                                "sentence": error_sentence,
                                "error_type": "FP",
                                "category": cp,
                            }
                        )

        df_errors = pd.DataFrame(
            error_results, columns=["sentence", "error_type", "category"]
        )

        return df_errors

    def get_errors_per_category_with_text(self) -> pd.DataFrame:
        error_results = []

        for doc in self.spanCorpus:
            for cp in self.spanCorpus[doc]:
                ann1_spans = self.spanCorpus[doc][cp]["ann1"]
                ann2_spans = self.spanCorpus[doc][cp]["ann2"]

                (
                    tp,
                    fp,
                    fn,
                    tp_texts,
                    fp_texts,
                    fn_texts,
                ) = self._cal_matching_overlap_with_text(
                    ann1_spans, ann2_spans, self.txtCorpus, doc, cp
                )

                for error in fp_texts + fn_texts:
                    error_results.append(
                        {
                            "file": doc,
                            "concept_name": cp,
                            "span": error["span"],
                            "error_type": error["error_type"],
                            "text": error["text"],
                        }
                    )

        df_errors = pd.DataFrame(
            error_results,
            columns=["file", "concept_name", "span", "error_type", "text"],
        )

        return df_errors


if __name__ == "__main__":

    model_corpus_dir = Corpus("../data/model_annotation", format="mae")
    gold_standard_corpus_dir = Corpus("../data/human_annotation", format="mae")
    dtd_path = "../models/MedSDoH/schema.dtd"
    excluded_categories = [
        "Sex_At_Birth",
        "Race_or_Ethnicity",
        "Sexual_Orientation",
        "Marital_Status",
    ]
    # Load the corpora
    model_corpus = Corpus(model_corpus_dir, format="mae")
    human_corpus = Corpus(gold_standard_corpus_dir, format="mae")

    eval = Evaluation(model_corpus.soup, human_corpus.soup, dtd_path, overlap_ratio=0.1)

    # Get evalutation performance
    df = eval.get_macro_avg_per_category(excluded_categories=excluded_categories)

    # Get erros per category
    df_errors = eval.get_errors_per_category_with_text().sort_values(
        by=["file", "concept_name"]
    )
    df_errors = df_errors[~df_errors["concept_name"].isin(excluded_categories)]
    df_errors.sort_values(by="concept_name")
