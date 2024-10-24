import json
import re
import unicodedata

from src.constants import NATOPS


class DatasetReader(object):
    def __init__(
        self,
        split,
        granularity=None,
    ):
        self.split = split
        self.split_name = split
        self.granularity = granularity

        self.claim_file = None
        self.retrieved_evidence_file = None
        self.proofver_file = None

    def get_sentence_content_from_id(self, sentence, sentence_searcher):
        doc_element = sentence_searcher.doc(sentence)  # sentence index
        if doc_element == None:
            return ""
        doc_content = json.loads(doc_element.raw())["contents"]
        return doc_content

    def process_sentence(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        # sentence = re.sub(" -LRB-", " ( ", sentence)
        # sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        return sentence

    def process_sentence_reverse(self, sentence):
        # sentence = re.sub("-", "--", sentence)
        sentence = re.sub(" \( ", " -LRB-", sentence)
        sentence = re.sub(" \)", "-RRB-", sentence)
        sentence = re.sub('"', "''", sentence)
        sentence = re.sub('"', "``", sentence)
        return sentence

    def process_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title

    def process_title_reverse(self, title):
        title = re.sub(" ", "_", title)
        title = re.sub(" \( ", " -LRB-", title)
        title = re.sub(" \)", "-RRB-", title)
        title = re.sub(":", "-COLON-", title)
        return title

    def get_sentence_predictions(self, num_docs=5):
        documents = {}

        with open(self.retrieved_evidence_file, "r") as in_file:
            lines = in_file.readlines()
            docs = []
            current_id = 1

            for i, line in enumerate(lines):
                line = line.strip()
                if int(line.split(" ")[0]) != current_id:
                    documents[current_id] = list(
                        dict.fromkeys(docs[:num_docs])
                    )  # consider only ten documents for sentence retrieval
                    docs = []
                    doc = " ".join(line.strip().split(" ")[2:-3])
                    docs.append(unicodedata.normalize("NFC", doc))
                    current_id = int(line.split(" ")[0])
                else:
                    doc = " ".join(line.strip().split(" ")[2:-3])
                    docs.append(unicodedata.normalize("NFC", doc))
                    current_id = int(line.split(" ")[0])
            if docs != []:
                documents[current_id] = list(dict.fromkeys(docs[:num_docs]))

        return documents

    def get_documents_predictions(self, prediction_file, num_docs=5, normalize=True):
        documents = {}

        with open(prediction_file, "r", encoding="utf-8") as in_file:
            lines = in_file.readlines()
            docs = []
            current_id = 1

            for i, line in enumerate(lines):
                line = line.strip()
                if int(line.split(" ")[0]) != current_id:  # (i % document_hits == 0) and (i != 0):
                    # documents.append(docs)
                    documents[current_id] = docs[:num_docs]  # consider only ten documents for sentence retrieval
                    docs = []
                    doc = " ".join(line.strip().split(" ")[2:-3])
                    if doc in set([x[0] for x in docs]):
                        continue
                    score = float(line.strip().split(" ")[-2])
                    if normalize:
                        docs.append((unicodedata.normalize("NFC", doc), score))
                    else:
                        docs.append((doc, score))
                    current_id = int(line.split(" ")[0])
                else:
                    doc = " ".join(line.strip().split(" ")[2:-3])
                    if doc in set([x[0] for x in docs]):
                        continue
                    score = float(line.strip().split(" ")[-2])
                    if normalize:
                        docs.append((unicodedata.normalize("NFC", doc), score))
                    else:
                        docs.append((doc, score))
                    current_id = int(line.split(" ")[0])
            if docs != []:
                documents[current_id] = docs[:num_docs]
        return documents

    def read_proofver_proofs(self):
        # Consider using split argument instead of class attribute
        natural_operations_sequences = {}
        proofs = {}
        if "validation" not in self.split_name:
            # Each id has multiple proofs
            with open(self.proofver_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line == "\n":
                        continue
                    content = line.split("\t")
                    if content[0] in natural_operations_sequences:
                        continue
                    proof = content[1].strip()
                    if int(content[0]) in natural_operations_sequences:
                        natural_operations_sequences[int(content[0])].append(proof)
                    else:
                        natural_operations_sequences[int(content[0])] = [proof]
            for id, sequences in natural_operations_sequences.items():
                proofs[id] = []
                for sequence in sequences:
                    matches = re.findall(
                        r"\{(.+?)\} \[(.+?)\] (.+?)", sequence
                    )  # Remove spaces due to manual formatting errors
                    proof = []
                    for match in matches:
                        match = (match[0].strip(), match[1].strip(), match[2].strip())
                        assert len(match) == 3, f"Does not match, got {match}"
                        assert match[2] in NATOPS
                        proof.append(match)
                    proofs[id].append(proof)
        else:
            print("MODE NOT KNOWN. ABORT...")
            sys.exit()
        return proofs
