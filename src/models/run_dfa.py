import argparse
import json
import os

from src.utils.util import ROOT_DIR
from src.constants import NATLOG_TRANSITION_MATRIX, NATOPS


def natlog_automaton(sequence):
    """
    (
        "eq",
        "gorward entailment",
        "negation",
        "alternation",
        "reverse entailment",
        "independent"
    ),
    (
        " =",
        " <",
        " !",
        " |",
        " >",
        " #"
    ),
    """

    current_status = "SUPPORTS"

    for relation in sequence:
        current_status = NATLOG_TRANSITION_MATRIX[current_status][relation]

    return current_status


def run_proofver_dfa(exp_name):
    input_path = os.path.join(ROOT_DIR, "exp_out", exp_name, "output_logs.csv")
    proofs = {}
    gold_proofs = {}
    with open(input_path, "r") as f_in:
        lines = f_in.readlines()
        sequence = []
        sequence_gold = []
        id = None
        for line in lines:
            if line == "\n":
                proofs[id] = sequence
                sequence = []
                gold_proofs[id] = sequence_gold
                sequence_gold = []
            else:
                content = line.strip().split("\t")
                id = int(content[0])
                pred_proof = content[6]
                sequence.append(pred_proof)
                gold_proof = content[4]
                sequence_gold.append(gold_proof)

    preds = {}
    gold = {}
    for key, proof in proofs.items():
        pred_verdict = natlog_automaton(proof)
        gold_verdict = natlog_automaton(gold_proofs[key])
        preds[key] = pred_verdict
        gold[key] = gold_verdict

    return [preds, gold]


def main():
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from src.data.data_reader import FeverDatasetReader

    parser = argparse.ArgumentParser(description="Converts FEVER jsonl wikipedia dump to anserini jsonl files.")
    parser.add_argument("--experiment_name", required=True, help="FEVER wiki-pages directory.")
    args = parser.parse_args()

    fever_data = FeverDatasetReader(split="validation")

    preds = []
    gold = []
    preds_dict, gold_dict = run_proofver_dfa(args.experiment_name)
    for qid, pred in preds_dict.items():
        preds.append(pred)
        gold.append(fever_data.labels[qid])

    conf_matrix = confusion_matrix(y_true=gold, y_pred=preds, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
    recall = recall_score(preds, gold, average="macro")
    precision = precision_score(preds, gold, average="macro")
    f1_score = f1_score(preds, gold, average="macro")
    print(classification_report(gold, preds))
    print(conf_matrix)
    print(recall, precision, f1_score)
    print([preds[i] == gold[i] for i in range(len(preds))].count(1) / len(preds))


if __name__ == "__main__":
    main()
