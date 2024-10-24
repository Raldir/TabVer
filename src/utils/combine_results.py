import json
import os
import statistics
import sys

from src.utils.util import ROOT_DIR

def select_scores(scores_list, suffix):
    score = scores_list[-1]
    score = {key.replace("proof_", "") + suffix: value for key, value in score.items()}
    return score


if __name__ == "__main__":
    dataset = sys.argv[1]

    experiment_path = sys.argv[2]

    experiment_paths = os.path.join(ROOT_DIR, "exp_out", dataset, experiment_path)

    subfolders = [f.path for f in os.scandir(experiment_paths) if f.is_dir()]

    all_scores = []
    for folder in subfolders:
        if dataset == "feverous":
            in_paths = ["dev_scores_qanatver.json", "dev_scores_qanatver_numerical.json"]
        elif dataset == "tabfact":
            in_paths = ["dev_scores_qanatver.json", "dev_scores_qanatver_r1.json", "dev_scores_qanatver_r2.json"]

        scores = {}
        for in_path in in_paths:
            scores_path = os.path.join(folder, in_path)
            suffix = "" if len(in_path.split("_")) < 4 else "_{}".format(in_path.split("_")[3].split(".json")[0])
            with open(scores_path, "r") as f_in:
                score = json.loads(f_in.readlines()[0].strip())

                selected_scores = select_scores([score], suffix)

                scores.update(selected_scores)

        print(scores)
        all_scores.append(scores)
    


    avg_scores = {}
    variances = {}
    for selected_scores in all_scores:
        for key, value in selected_scores.items():
            if key not in avg_scores:
                avg_scores[key] = [value]
            else:
                avg_scores[key].append(value)
    for key, value in avg_scores.items():
        mean = statistics.mean(value)
        variance = statistics.stdev(value)
        avg_scores[key] = mean
        variances[key] = variance

    out_file_name = "avg_scores_qanatver.json"

    with open(os.path.join(experiment_paths, out_file_name), "w") as f_out:
        f_out.write("{}\n".format(json.dumps(avg_scores)))
        f_out.write("{}\n".format(json.dumps(variances)))

    # prisnt(subfolders)
