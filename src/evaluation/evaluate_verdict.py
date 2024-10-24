import json
import os


from src.utils.util import ROOT_DIR
from src.constants import (
    CHOICES_SCORES_LIST,
    CLAIM_ID,
    LABEL,
    PREDICTION,
)

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class EvaluatorVerdict:
    def __init__(self, config, datamodule):
        self.config = config
        self.datamodule = datamodule

    def compute_metric(self, accumulated):
        ids = accumulated[CLAIM_ID]
        gold = accumulated[LABEL]
        prediction = accumulated[PREDICTION]
        prediction_probs = accumulated[CHOICES_SCORES_LIST]  # Probability for each answer choice

        if self.config.num_classes == 2:
            prediction = [x if x != 2 else 1 for x in accumulated[PREDICTION]]

        output_path = os.path.join(ROOT_DIR, "exp_out", self.config.exp_name, "probabilities")
        print(output_path)
        with open(output_path + ".json", "w") as f_out:
            for i in range(len(prediction)):
                curr_id = ids[i]
                prediction_prob = prediction_probs[i]
                curr_label_to_pred = {LABELS[i]: x for i, x in enumerate(prediction_prob)}
                f_out.write("{}\t{}\n".format(curr_id, json.dumps(curr_label_to_pred)))


        # Could also produce some scores for verdict QA part but not of interest here.
        return {"accuracy": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}
