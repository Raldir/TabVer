import json
import os
import pickle
from functools import reduce

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.util import ROOT_DIR
from src.constants import (
    CLAIM_ID,
    CLAIM_SPAN_POS,
    LABEL,
    NATOPS,
    OP,
    PRED_PROB_LIST,
    PREDICTION,
)
from src.models.run_dfa import natlog_automaton


def load_feverous_numerical_challenges():
    input_file = os.path.join(ROOT_DIR, "data", "feverous", "feverous_dev_challenges.jsonl")
    challenges = set([])
    with open(input_file, "r") as f_in:
        for line in f_in.readlines():
            content = json.loads(line)
            if content["challenge"] == "Numerical Reasoning":
                challenges.add(content["id"])
    return challenges

def load_tabfact_challenges():
    input_file = os.path.join(ROOT_DIR, "data", "tabfact", "tabfact_dev_r2_ids.jsonl")
    with open(input_file, "r") as f_in:
        r2_ids = set(json.load(f_in))

    input_file = os.path.join(ROOT_DIR, "data", "tabfact", "tabfact_dev_r1_ids.jsonl")
    with open(input_file, "r") as f_in:
        r1_ids = set(json.load(f_in))        

    return [r1_ids, r2_ids]


class Evaluator:
    def __init__(self, config, datamodule):
        self.config = config
        self.datamodule = datamodule
        self.highest_acc = 0

    def _filter_neg_instances(self, span_dict):
        non_neg = []
        is_independence_natop = False
        gold_label = -1  # Set default gold label to negative answer
        natop_gold = "#"
        # Filter natops that predict negative token
        for natop, content in span_dict.items():
            pred = content[PREDICTION]
            gold = content[LABEL]
            if not gold == 0:  # No is on index 0
                if gold_label != -1:
                    print("THIS SHOULD NOT HAPPEN. DOUBLE NON_NEG GOLD LABEL")
                gold_label = gold
                natop_gold = NATOPS[content[OP]]
            if pred == 0:
                continue
            else:
                non_neg.append((natop, content))
        if not non_neg:
            is_independence_natop = True
            non_neg = span_dict.items()
        return [gold_label, natop_gold, non_neg, is_independence_natop]

    def _select_natop(self, span_num, non_neg, is_independence_natop, gold_label, natop_gold):
        max_prob = max([x[1][PRED_PROB_LIST] for x in non_neg])  # TODO MIN OR MAX???
        max_content = [x[1] for x in non_neg if x[1][PRED_PROB_LIST] == max_prob]
        max_index = [x[0] for x in non_neg if x[1][PRED_PROB_LIST] == max_prob][0]
        max_element = max_content[0]  # Get first item that adheres

        alignments = (
            self.datamodule.dataset_or.alignments[max_element[CLAIM_ID]]
            if max_element[CLAIM_ID] in self.datamodule.dataset_or.alignments
            else []
        )
        claim_span, evidence_span = alignments[max_element[CLAIM_SPAN_POS]]
        ev_spans = ["No", "Yes"]

        if is_independence_natop:
            natop_pred = "#"
        else:
            natop_pred = NATOPS[max_index]

        output = {
            "id": max_element[CLAIM_ID],
            "natop_gold": natop_gold,
            "natop_pred": natop_pred,
            "pred": max_element[PREDICTION],
            "prediction_prob": max_element[PRED_PROB_LIST],
            "gold": gold_label,
            "claim_span": claim_span,
            "pred_span": evidence_span,  # Use aligned evidence span here as supposed to
            "gold_span": ev_spans[gold_label],
            "binary_prediction": ev_spans[
                max_element[PREDICTION]
            ],  # Technically this is the prediction output, but just Yes No format so...
        }
        return output

    def _read_nli_predictions(self):
        predictions = {}
        file_path = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "probabilities.json")
        with open(file_path, "r") as f_in:
            lines = f_in.readlines()
            print("NUM LINES", len(lines))
            for i, line in enumerate(lines):
                qid, content = line.strip().split("\t")
                content = json.loads(content)
                predictions[int(qid)] = content
        return predictions

    def _select_proof(self, span_results, possible_parses, nli_prediction):

        if possible_parses:
            all_proofs = []
            predicted_verdict = []
            proof_scores = []
            proof_by_verdict = {}

            for parse in possible_parses:
                curr_proof = [span_results[x] for x in parse]
                all_proofs.append(curr_proof)
                proof_natops = [x["natop_pred"] for x in curr_proof]
                num_eles_wo_indep = len([x for x in proof_natops if x != "#"])

                proof_score = [
                    x["prediction_prob"] if x["natop_pred"] != "#" else self.config.dynamic_parsing_nei_threshold
                    for x in curr_proof
                ]

                proof_score = reduce(lambda x, y: x + y, proof_score) / len(curr_proof)

                pred_veracity = natlog_automaton(proof_natops)
                if self.config.num_classes == 2 and pred_veracity == "NOT ENOUGH INFO":
                    pred_veracity = "REFUTES"
                predicted_verdict.append(pred_veracity)
                proof_scores.append(proof_score)


            if not all([x == "NOT ENOUGH INFO" for x in predicted_verdict]):
                all_proofs_new = []
                predicted_verdict_new = []
                proof_scores_new = []
                for i in range(len(all_proofs)):
                    if predicted_verdict[i] != "NOT ENOUGH INFO":
                        all_proofs_new.append(all_proofs[i])
                        predicted_verdict_new.append(predicted_verdict[i])
                        proof_scores_new.append(proof_scores[i])
                all_proofs = all_proofs_new
                predicted_verdict = predicted_verdict_new
                proof_scores = proof_scores_new

                
            for i, proof in enumerate(all_proofs):
                if predicted_verdict[i] in proof_by_verdict:
                    proof_by_verdict[predicted_verdict[i]].append((i, proof_scores[i]))
                else:
                    proof_by_verdict[predicted_verdict[i]] = [(i, proof_scores[i])]

            for key, value in proof_by_verdict.items():
                proof_by_verdict[key] = sorted(value, key=lambda x: x[1], reverse=True)

            max_value = 0
            max_value_index = -1
            max_label = None
            for key in nli_prediction:
                if key not in proof_by_verdict:
                    continue
                score = proof_by_verdict[key][0][1] + nli_prediction[key]
                if score > max_value:
                    max_value = score
                    max_label = key
                    max_value_index = proof_by_verdict[key][0][0]

            best_proof = all_proofs[max_value_index]

            return best_proof

        else:
            keys = sorted(span_results.keys())
            proof = []
            for key in keys:
                proof.append(span_results[key])
            return proof

    def extract_natlog_proof(self, accumulated):
        proof_dict = {}  # Sort predictions in "claim -> claim_span -> preds"

        for i in range(len(accumulated[PREDICTION])):
            current_dict = {}
            for key in accumulated.keys():
                current_dict[key] = accumulated[key][i]

            if current_dict[CLAIM_ID] not in proof_dict:
                proof_dict[current_dict[CLAIM_ID]] = {}

            if current_dict[CLAIM_SPAN_POS] not in proof_dict[current_dict[CLAIM_ID]]:
                proof_dict[current_dict[CLAIM_ID]][current_dict[CLAIM_SPAN_POS]] = {current_dict[OP]: current_dict}
            else:
                proof_dict[current_dict[CLAIM_ID]][current_dict[CLAIM_SPAN_POS]][current_dict[OP]] = current_dict

        for claim_id in proof_dict:
            proof_dict[claim_id] = sorted(proof_dict[claim_id].items())

        claim_parsed_hierarchy = self.datamodule.dataset_or.claims_parsed_hierarchy
        nli_predictions = self._read_nli_predictions()

        generated_proofs = {}
        avg_proof_length = 0
        for claim_id, spans in proof_dict.items():
            proof = []
            span_results = {}
            claim_options = claim_parsed_hierarchy.get(claim_id, {})
            # print("CLAIM_OPTIONS", claim_options)

            for span_num, span in spans:
                gold_label, natop_gold, non_neg, is_independence_natop = self._filter_neg_instances(span)
                natop_probabilities = self._select_natop(
                    span_num, non_neg, is_independence_natop, gold_label, natop_gold
                )
                span_results[span_num] = natop_probabilities

            # print(claim_id)
            nli_prediction = nli_predictions[int(claim_id)] if int(claim_id) in nli_predictions else []
            proof = self._select_proof(span_results, claim_options, nli_prediction)

            avg_proof_length += len(proof)

            generated_proofs[claim_id] = proof

        print("Avg proof length: ", avg_proof_length / len(proof_dict))

        return generated_proofs

    def write_metric_content(self, generated_proofs, incorrect_ids):
        output_path = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "output_logs_qanatver.csv")
        # pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f_out:
            for key, proof in generated_proofs.items():
                for value in proof:
                    pred_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        value["id"],
                        value["claim_span"],
                        value["binary_prediction"],
                        value["gold_span"],
                        value["natop_gold"],
                        value["pred_span"],
                        value["natop_pred"],
                    )
                    f_out.write(pred_line)
                f_out.write("\n")
        output_path = os.path.join(
            ROOT_DIR, "exp_out", self.config.exp_name, "output_logs_errors_qanatver.csv"
        )
        # pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f_out:
            for key, proof in generated_proofs.items():
                if proof[0]["id"] in incorrect_ids:
                    for value in proof:
                        pred_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            value["id"],
                            value["claim_span"],
                            value["binary_prediction"],
                            value["gold_span"],
                            value["natop_gold"],
                            value["pred_span"],
                            value["natop_pred"],
                        )
                        f_out.write(pred_line)
                    f_out.write("Gold Verdict: {}\n".format(incorrect_ids[proof[0]["id"]]))
                    f_out.write("\n")

    def _plot_confusion_matrix(self, preds, gold, title):
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        conf_matrix = confusion_matrix(y_true=gold, y_pred=preds, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large")

        plt.xlabel("Predictions", fontsize=18)
        plt.ylabel("Actuals", fontsize=18)
        plt.title("Confusion Matrix", fontsize=18)
        plt.savefig(os.path.join("exp_out", self.config.exp_name, "{}.png".format(title)))

    def _compute_metrics_proof_level(self, generated_proofs, title, condition = None):
        preds = []
        gold = []
        incorrect_ids = {}
        for claim_id, proof in generated_proofs.items():
            if condition != None and claim_id not in condition:
                continue
            gold_veracity = self.datamodule.dataset_or.labels[claim_id]
            gold.append(gold_veracity)
            proof_natops = [x["natop_pred"] for x in proof]
            pred_veracity = natlog_automaton(proof_natops)
            if self.config.num_classes == 2:
                if pred_veracity == "NOT ENOUGH INFO":
                    pred_veracity = "REFUTES"
            preds.append(pred_veracity)
            if gold_veracity != pred_veracity:
                incorrect_ids[claim_id] = gold_veracity

        self._plot_confusion_matrix(preds, gold, title)

        recall = recall_score(preds, gold, average="macro")
        precision = precision_score(preds, gold, average="macro")
        f1 = f1_score(preds, gold, average="macro")
        acc = accuracy_score(preds, gold)

        print("Printing scores for: ", title)
        print(classification_report(gold, preds))

        return [acc, recall, precision, f1, incorrect_ids]

    def compute_metric(self, accumulated):
        results_span_level = {}

        generated_proofs = self.extract_natlog_proof(accumulated)

        if self.config.dataset == "feverous":
            numerical_challenges_ids = load_feverous_numerical_challenges()

            accuracy, recall, precision, f1, incorrect_ids = self._compute_metrics_proof_level(generated_proofs, "confusion_matrix_qanatver_numerical", condition = numerical_challenges_ids)

            print("Numerical Challenges performance")
            print("acc: {}, recall: {}, precision: {}, f1: {}".format(accuracy, recall, precision, f1))
            print("------------------------------------")

            output_path_numerical_challenges = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "dev_scores_qanatver_numerical.json")
            with open(output_path_numerical_challenges, "w") as f_out:
                f_out.write("{}\t".format(json.dumps({"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1})))

        elif self.config.dataset == "tabfact":
            # Distinguish between performance on R1 and R2 (R2 being more difficult)
            r1_ids, r2_ids = load_tabfact_challenges()

            if len(set(list(generated_proofs.keys())) & r1_ids) > 0: # Might be empty when debugging
                accuracy, recall, precision, f1, incorrect_ids = self._compute_metrics_proof_level(generated_proofs, "confusion_matrix_qanatver_numerical", condition = r1_ids)

                print("R1 Challenges performance")
                print("acc: {}, recall: {}, precision: {}, f1: {}".format(accuracy, recall, precision, f1))
                print("------------------------------------")

                output_path_numerical_challenges = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "dev_scores_qanatver_r1.json")
                with open(output_path_numerical_challenges, "w") as f_out:
                    f_out.write("{}\t".format(json.dumps({"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1})))

            if len(set(list(generated_proofs.keys())) & r2_ids) > 0: # Might be empty when debugging
                accuracy, recall, precision, f1, incorrect_ids = self._compute_metrics_proof_level(generated_proofs, "confusion_matrix_qanatver_numerical", condition = r2_ids)

                print("R2 Challenges performance")
                print("acc: {}, recall: {}, precision: {}, f1: {}".format(accuracy, recall, precision, f1))
                print("------------------------------------")

                output_path_numerical_challenges = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "dev_scores_qanatver_r2.json")
                with open(output_path_numerical_challenges, "w") as f_out:
                    f_out.write("{}\t".format(json.dumps({"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1})))


        accuracy, recall, precision, f1, incorrect_ids = self._compute_metrics_proof_level(generated_proofs, "confusion_matrix_qanatver")


        results_span_level["accuracy"] = accuracy
        results_span_level["recall"] = recall
        results_span_level["precision"] = precision
        results_span_level["f1"] = f1

        if results_span_level["accuracy"] > self.highest_acc and not self.config.use_cached_data:
            self.highest_acc = results_span_level["accuracy"]
            self.write_metric_content(generated_proofs, incorrect_ids)
            output_path = os.path.join(
                ROOT_DIR, "exp_out", self.config.exp_name, "model_output_qanatver.pkl"
            )
            output_raw_file = open(output_path, "wb")
            pickle.dump(accumulated, output_raw_file)
            output_raw_file.close()

        print("All performance")
        print("acc: {}, recall: {}, precision: {}, f1: {}".format(accuracy, recall, precision, f1))

        return results_span_level

    def run_cached_data(self):
        input_path = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "model_output.pkl")
        input_raw_file = open(input_path, "rb")
        accumulated = pickle.load(input_raw_file)
        input_raw_file.close()
        scores = self.compute_metric(accumulated)
        output_path = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "dev_scores_qanatver.json")
        with open(output_path, "w") as f_out:
            f_out.write("{}\t".format(json.dumps(scores)))
