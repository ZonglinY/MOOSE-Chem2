import json, os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation
from Evaluation.evaluate import Evaluator

vague_cg_hyp_path = "./Data/processed_research_direction.json"
chem_annotation_path = "./Data/chem_research_2024_finegrained.xlsx"
preprocess_groundtruth_components_dir = "./Checkpoints/groundtruth_hyp_components_collection.json"
assert os.path.exists(preprocess_groundtruth_components_dir)

# 1: ChemCrow; 2. plain LLM
baseline_type = 2

model_name = "gpt-4o-mini"
api_type = 0
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
num_compare_times=5

if baseline_type == 1:
    # input path
    new_gene_hyp_path = "./Checkpoints/chemcrow_baseline_gene_hyp_collection_new_new_test.json"
    # output path
    new_final_scores_path = "./Checkpoints/chemcrow_baseline_final_scores_new_new_test.json"
elif baseline_type == 2:
    # input path
    new_gene_hyp_path = "./Checkpoints/chemcrow_baseline_gene_hyp_collection_new_new_test_baseline_type_{}.json".format(baseline_type)
    # output path
    new_final_scores_path = "./Checkpoints/chemcrow_baseline_final_scores_new_new_test_baseline_type_{}.json".format(baseline_type)
else:
    raise NotImplementedError("baseline_type: {} is not implemented".format(baseline_type))

bkg_q_list, dict_bkg2survey, dict_bkg2cg_hyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note = load_chem_annotation(chem_annotation_path)

with open(vague_cg_hyp_path, "r") as f:
    dict_bkg2cg_hyp = json.load(f)

evaluator = Evaluator(model_name, api_type, api_key, base_url, chem_annotation_path, preprocess_groundtruth_components_dir)

def evaluate_fg_hyp_with_chemcrow():
    with open(new_gene_hyp_path, "r") as f:
        new_gene_hyp_collection = json.load(f)
    assert len(new_gene_hyp_collection) == len(bkg_q_list)
    final_scores = []
    for bkg_id in range(len(bkg_q_list)):
        print("bkg_id: ", bkg_id)
        cur_gene_hyp = new_gene_hyp_collection[bkg_id]
        cur_average_compare_results = evaluator.check_one_generated_hyp_or_exp(bkg_id, cur_gene_hyp, type="hyp", num_compare_times=num_compare_times)
        final_scores.append(cur_average_compare_results)
        # calculate the average of the final scores, just to show the progress
        average_recall = np.mean(final_scores, axis=0)[4]
        print("average_recall: ", average_recall)
        # save while running, so that we can resume from the last saved checkpoint
        with open(new_final_scores_path, "w") as f:
            json.dump(final_scores, f)



if __name__ == "__main__":
    evaluate_fg_hyp_with_chemcrow()
