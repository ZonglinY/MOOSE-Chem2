import os, sys, json, argparse, builtins
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.hierarchy_greedy_utils import HGNode, HGTree, postprocess_reverse_ordered_pairwise_comparison_results
from Evaluation.evaluate import Evaluator
from Method.logging_utils import setup_logger
from Evaluation.pairwise_compare import PairwiseCompare
from Method.utils import load_chem_annotation
import numpy as np


# Input: 
#   file_path: str
#   hierarchy_id: int; the hierarchy_id of the hypothesis to be loaded
# Output: 
#   final_hypothesis: [str]
def load_final_hypothesis_from_HGTree(file_path, hierarchy_id):
    a = HGTree.load(file_path)
    final_hypothesis = []
    def search_tree_breadth_first(node, hierarchy_id):
        ttl_search_step = 0
        queue = [node]
        visited = set()
        while queue:
            node = queue.pop(0)
            for cur_init_point_id in range(len(node.full_generated_hyp)):
                # print("len(node.full_generated_hyp[cur_init_point_id]): ", len(node.full_generated_hyp[cur_init_point_id]))
                ttl_search_step += len(node.full_generated_hyp[cur_init_point_id])
            try:
                if node.hierarchy_id == hierarchy_id:
                    final_hypothesis.append(node.full_generated_hyp[-1][-1][0])
            except Exception as e:
                print("Warning: ", e)
            
            visited.add(node)
            for child in node.children:
                if child not in visited:
                    queue.append(child)
        print("ttl_search_step: ", ttl_search_step)
    search_tree_breadth_first(a.root, hierarchy_id)
    return final_hypothesis


# Input: 
#   file_path: str
#   hierarchy_id: int; the hierarchy_id of the hypothesis to be loaded
# Output: 
#   final_hypothesis: [str]
def load_final_hypothesis_from_HGTree_with_reasoning_steps(file_path, hierarchy_id):
    a = HGTree.load(file_path)
    def search_tree_breadth_first(node, hierarchy_id):
        final_hypothesis = []
        ttl_search_step = 0
        queue = [node]
        visited = set()
        while queue:
            node = queue.pop(0)
            for cur_init_point_id in range(len(node.full_generated_hyp)):
                # print("len(node.full_generated_hyp[cur_init_point_id]): ", len(node.full_generated_hyp[cur_init_point_id]))
                ttl_search_step += len(node.full_generated_hyp[cur_init_point_id])
            try:
                if node.hierarchy_id == hierarchy_id:
                    final_hypothesis.append(node.full_generated_hyp[-1][-1][0])
            except Exception as e:
                print("Warning: ", e)
            
            visited.add(node)
            for child in node.children:
                if child not in visited:
                    queue.append(child)
        print("ttl_search_step: ", ttl_search_step)
        return final_hypothesis, ttl_search_step
    final_hypothesis, ttl_search_step = search_tree_breadth_first(a.root, hierarchy_id)
    return final_hypothesis, ttl_search_step
    


# Input: 
#   file_path: str
# Output: 
#   final_hypothesis: [str]
def load_final_hypothesis_from_json(file_path):
    # data: [[hypothesis, reason], ...]
    with open(file_path, "r") as f:
        data = json.load(f)
    final_hypothesis = [data[-1][0]]
    return final_hypothesis


def load_final_hypothesis_from_json_with_reasoning_steps(file_path):
    # data: [[hypothesis, reasoning_steps], ...]
    with open(file_path, "r") as f:
        data = json.load(f)
    final_hypothesis = [data[-1][0]]
    ttl_search_step = len(data)
    return final_hypothesis, ttl_search_step


# Input: 
#   final_hypothesis: [hyp0, hyp1, ...]
# Output:
#   final_scores: [[precision, recall, f1, weighted_precision, weighted_recall, weighted_f1], ...]
#       len(final_scores) == len(final_hypothesis)
def evaluate_hyp(final_hypothesis, bkg_id, evaluator, type, num_compare_times):
    assert type in ["hyp", "exp"], f"type must be 'hyp' or 'exp', got {type}"

    final_scores = []
    for cur_gene_hyp in final_hypothesis:
        # cur_average_compare_results: [precision, recall, f1, weighted_precision, weighted_recall, weighted_f1]
        cur_average_compare_results = evaluator.check_one_generated_hyp_or_exp(bkg_id, cur_gene_hyp, type=type, num_compare_times=num_compare_times)
        final_scores.append(cur_average_compare_results)

    assert len(final_hypothesis) == len(final_scores), f"len(final_hypothesis): {len(final_hypothesis)}; len(final_scores): {len(final_scores)}"
    return final_scores


# Input:
#   which_exp: [bool, bool, bool]; [if perform hierarchy_greedy_5, if perform hierarchy_greedy_1, if perform greedy]
#   h5_exp_hierarchy_id: int; the hierarchy_id of the hypothesis to be loaded in hierarchy_greedy_5 
def load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id=4):
    # initialze
    final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = None, None, None

    # hierarchy_greedy: 5 hierarchy
    if which_exp[0]:
        file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        # Q: added this if statement temporarily
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            output_dir_postfix = "updated_prompt_feb_14"
            file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
            assert os.path.exists(file_path), f"File {file_path} does not exist."
        hierarchy_id = h5_exp_hierarchy_id
        # final_hypothesis_hierarchy_5: [hyp0, hyp1, ...]
        final_hypothesis_hierarchy_5 = load_final_hypothesis_from_HGTree(file_path, hierarchy_id)
    
    # hierarchy_greedy: 1 hierarchy
    if which_exp[1]:
        file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        # Q: added this if statement temporarily
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            output_dir_postfix = "updated_prompt_feb_14"
            file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
            assert os.path.exists(file_path), f"File {file_path} does not exist."
        hierarchy_id = 0
        # final_hypothesis_hierarchy_1: [hyp0, hyp1, ...]
        final_hypothesis_hierarchy_1 = load_final_hypothesis_from_HGTree(file_path, hierarchy_id)

    # greedy
    if which_exp[2]:
        # locam_minimum_threshold = int(locam_minimum_threshold) + 1
        locam_minimum_threshold = int(locam_minimum_threshold)
        file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_eval_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_bkgid_{bkg_id}_{output_dir_postfix}.json"
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_eval_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.json"
        # Q: added this if statement temporarily
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            output_dir_postfix = "updated_prompt_feb_14"
            file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_eval_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.json"
            assert os.path.exists(file_path), f"File {file_path} does not exist."
        # final_hypothesis_greedy: [hyp0, hyp1, ...]
        final_hypothesis_greedy = load_final_hypothesis_from_json(file_path)

    return final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy





def load_hypothesis_from_methods_with_reasoning_steps(bkg_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id=4):
    # initialze
    final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = None, None, None
    ttl_search_step_hierarchy_5, ttl_search_step_hierarchy_1, ttl_search_step_greedy = None, None, None

    # hierarchy_greedy: 5 hierarchy
    if which_exp[0]:
        file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        # Q: added this if statement temporarily
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            output_dir_postfix = "updated_prompt_feb_14"
            file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
            assert os.path.exists(file_path), f"File {file_path} does not exist."
        hierarchy_id = h5_exp_hierarchy_id
        # final_hypothesis_hierarchy_5: [hyp0, hyp1, ...]
        final_hypothesis_hierarchy_5, ttl_search_step_hierarchy_5 = load_final_hypothesis_from_HGTree_with_reasoning_steps(file_path, hierarchy_id)
    
    # hierarchy_greedy: 1 hierarchy
    if which_exp[1]:
        file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        # Q: added this if statement temporarily
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            output_dir_postfix = "updated_prompt_feb_14"
            file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_eval_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
            assert os.path.exists(file_path), f"File {file_path} does not exist."
        hierarchy_id = 0
        # final_hypothesis_hierarchy_1: [hyp0, hyp1, ...]
        final_hypothesis_hierarchy_1, ttl_search_step_hierarchy_1 = load_final_hypothesis_from_HGTree_with_reasoning_steps(file_path, hierarchy_id)

    # greedy
    if which_exp[2]:
        # locam_minimum_threshold = int(locam_minimum_threshold) + 1
        locam_minimum_threshold = int(locam_minimum_threshold)
        file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_eval_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_bkgid_{bkg_id}_{output_dir_postfix}.json"
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_eval_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.json"
        # Q: added this if statement temporarily
        if not os.path.exists(file_path) and if_generate_with_past_failed_hyp == 0:
            output_dir_postfix = "updated_prompt_feb_14"
            file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_eval_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.json"
            assert os.path.exists(file_path), f"File {file_path} does not exist."
        # final_hypothesis_greedy: [hyp0, hyp1, ...]
        final_hypothesis_greedy, ttl_search_step_greedy = load_final_hypothesis_from_json_with_reasoning_steps(file_path)

    return final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy, ttl_search_step_hierarchy_5, ttl_search_step_hierarchy_1, ttl_search_step_greedy



def check_average_search_step(start_id, end_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id):
    # check whether all files are there
    ttl_search_step_hierarchy_5_list, ttl_search_step_hierarchy_1_list, ttl_search_step_greedy_list = [], [], []
    for cur_bkg_id in range(start_id, end_id + 1):
        final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy, ttl_search_step_hierarchy_5, ttl_search_step_hierarchy_1, ttl_search_step_greedy = load_hypothesis_from_methods_with_reasoning_steps(cur_bkg_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)
        ttl_search_step_hierarchy_5_list.append(ttl_search_step_hierarchy_5)
        ttl_search_step_hierarchy_1_list.append(ttl_search_step_hierarchy_1)
        ttl_search_step_greedy_list.append(ttl_search_step_greedy)

    if which_exp[0]:    
        print("ttl_search_step_hierarchy_5:", np.mean(ttl_search_step_hierarchy_5_list), "±", np.std(ttl_search_step_hierarchy_5_list), "median:", np.median(ttl_search_step_hierarchy_5_list))
    if which_exp[1]:
        print("ttl_search_step_hierarchy_1:", np.mean(ttl_search_step_hierarchy_1_list), "±", np.std(ttl_search_step_hierarchy_1_list), "median:", np.median(ttl_search_step_hierarchy_1_list))
    if which_exp[2]:
        print("ttl_search_step_greedy:", np.mean(ttl_search_step_greedy_list), "±", np.std(ttl_search_step_greedy_list), "median:", np.median(ttl_search_step_greedy_list))




# results_compare_collection: [1/1.5/2, [[1/2, reason], ...]]
# compare_metric: in ["overall", "effectiveness", "novelty", "detailedness", "feasibility"]
def pairwise_compare_between_two_set_of_hypothesis(final_hypothesis_method1, final_hypothesis_method2, research_question, pairwise_compare, if_final_eval, compare_metric="overall", num_compare_times=5): 
    assert compare_metric in  ["overall", "effectiveness", "novelty", "detailedness", "feasibility"]

    results_compare_collection = []
    print(f"len(hyp_method1): {len(final_hypothesis_method1)}; len(hyp_method2): {len(final_hypothesis_method2)}")
    # compare h5 with h1
    for cur_hyp_m1 in final_hypothesis_method1:
        for cur_hyp_m2 in final_hypothesis_method2:
            ## response/response_reverse_order: [[1/2, reason], ...]
            # compare cur_hyp_m1 with cur_hyp_m2 (to avoid the position bias)
            response = pairwise_compare.compare(research_question, cur_hyp_m1, cur_hyp_m2, instruction_mode="same_hyp1_hyp2", hierarchy_level=None, if_final_eval=if_final_eval, if_no_unified_response=True, num_compare_times=num_compare_times, compare_metric=compare_metric)
            # compare cur_hyp_m2 with cur_hyp_m1 (swith positions to avoid the position bias)
            response_reverse_order = pairwise_compare.compare(research_question, cur_hyp_m2, cur_hyp_m1, instruction_mode="same_hyp1_hyp2", hierarchy_level=None, if_final_eval=if_final_eval, if_no_unified_response=True, num_compare_times=num_compare_times, compare_metric=compare_metric)
            response_reverse_order_reverse_back = postprocess_reverse_ordered_pairwise_comparison_results(response_reverse_order)
            ## final judgement
            full_response = response + response_reverse_order_reverse_back
            cur_scores = [d[0] for d in full_response]
            cur_average_score = sum(cur_scores) / len(cur_scores)
            if cur_average_score < 1.5:
                results_compare_collection.append([1, research_question, cur_hyp_m1, cur_hyp_m2, full_response])
            elif cur_average_score > 1.5:
                results_compare_collection.append([2, research_question, cur_hyp_m1, cur_hyp_m2, full_response])
            else:
                results_compare_collection.append([1.5, research_question, cur_hyp_m1, cur_hyp_m2, full_response])
            print("cur_average_score:", cur_average_score)
    return results_compare_collection


# Function:
#   compare hypothesis between methods
#   compare_metric: in ["overall", "effectiveness", "novelty", "detailedness", "feasibility"]
# Output:
#   results_compare_collection_h5_h1/results_compare_collection_h5_g/results_compare_collection_h1_g: [[1/2/1.5, reason], ...]
def pairwise_compare_h5_h1_g_with_one_example(bkg_id, research_question, which_exp, exp_model_name, exp_eval_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, if_generate_with_past_failed_hyp, if_print=True, compare_metric="overall", num_compare_times=5):
    # get final hypothesis from methods
    final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)

    print("comparing between hierarchy 5 and hierarchy 1")
    results_compare_collection_h5_h1 = pairwise_compare_between_two_set_of_hypothesis(final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, research_question, pairwise_compare, if_final_eval, compare_metric=compare_metric, num_compare_times=num_compare_times)
    # print("results_compare_collection_h5_h1:", results_compare_collection_h5_h1)
    print("comparing between hierarchy 5 and greedy")
    results_compare_collection_h5_g = pairwise_compare_between_two_set_of_hypothesis(final_hypothesis_hierarchy_5, final_hypothesis_greedy, research_question, pairwise_compare, if_final_eval, compare_metric=compare_metric, num_compare_times=num_compare_times)
    print("comparing between hierarchy 1 and greedy")
    results_compare_collection_h1_g = pairwise_compare_between_two_set_of_hypothesis(final_hypothesis_hierarchy_1, final_hypothesis_greedy, research_question, pairwise_compare, if_final_eval, compare_metric=compare_metric, num_compare_times=num_compare_times)

    if if_print:
        # print("results_compare_collection_h5_h1:", results_compare_collection_h5_h1)
        # print("results_compare_collection_h5_g:", results_compare_collection_h5_g)
        # print("results_compare_collection_h1_g:", results_compare_collection_h1_g)
        print("results_compare_collection_h5_h1:", [d[0] for d in results_compare_collection_h5_h1])
        print("results_compare_collection_h5_g:", [d[0] for d in results_compare_collection_h5_g])
        print("results_compare_collection_h1_g:", [d[0] for d in results_compare_collection_h1_g])
    return results_compare_collection_h5_h1, results_compare_collection_h5_g, results_compare_collection_h1_g


# Input:
# rlt_collection: [[[1/2, reason], ...]]
# Output:
# preference_to_1_ratio: [ratio_1, ratio_tie, ratio_2]
def calculate_average_preference(rlt_collection):
    cnt_1_win, cnt_2_win, cnt_tie = 0, 0, 0
    for cur_rlt in rlt_collection:
        for cur_pairwise_rlt in cur_rlt:
            if cur_pairwise_rlt[0] == 1:
                cnt_1_win += 1
            elif cur_pairwise_rlt[0] == 2:
                cnt_2_win += 1
            elif cur_pairwise_rlt[0] == 1.5:
                cnt_tie += 1
            else:
                raise Exception(f"cur_pairwise_rlt[0] must be 1 or 2 or 1.5, got {cur_pairwise_rlt[0]}")
    ttl_cnt = cnt_1_win + cnt_2_win + cnt_tie
    preference_to_1_ratio = [cnt_1_win / ttl_cnt, cnt_tie / ttl_cnt, cnt_2_win / ttl_cnt]
    return preference_to_1_ratio


# Include both start_id and end_id
def pairwise_compare_h5_h1_g_with_batch_examples(start_id, end_id, chem_annotation_path, which_exp, exp_model_name, exp_eval_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, pairwise_eval_model_name, pairwise_if_multiple_llm, if_generate_with_past_failed_hyp, if_print=True, if_save=False, compare_metric="overall", num_compare_times=5):
    # obtain groundtruth finegrained hypothesis and experiment
    bkg_q_list, dict_bkg2survey, dict_bkg2cg_hyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note = load_chem_annotation(chem_annotation_path)

    # check whether all files are there
    for cur_bkg_id in range(start_id, end_id + 1):
        final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = load_hypothesis_from_methods(cur_bkg_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)
    print("All files are there.")

    # h5_h1_collection: [bkg_id, pair_id, score/reason selection]
    h5_h1_collection, h5_g_collection, h1_g_collection = [], [], []
    for cur_bkg_id in range(start_id, end_id + 1):
        print("Processing bkg_id:", cur_bkg_id)
        cur_rq = bkg_q_list[cur_bkg_id]
        # cur_rlt_h5_h1/cur_rlt_h5_g/cur_rlt_h1_g: [[1/2, reason]]
        cur_rlt_h5_h1, cur_rlt_h5_g, cur_rlt_h1_g = pairwise_compare_h5_h1_g_with_one_example(cur_bkg_id, cur_rq, which_exp, exp_model_name, exp_eval_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, if_generate_with_past_failed_hyp, if_print, compare_metric=compare_metric, num_compare_times=num_compare_times)
        h5_h1_collection.append(cur_rlt_h5_h1)
        h5_g_collection.append(cur_rlt_h5_g)
        h1_g_collection.append(cur_rlt_h1_g)

    preference_to_1_ratio_h5_h1 = calculate_average_preference(h5_h1_collection)
    preference_to_1_ratio_h5_g = calculate_average_preference(h5_g_collection)
    preference_to_1_ratio_h1_g = calculate_average_preference(h1_g_collection)

    if if_print:
        print("preference_to_1_ratio_h5_h1: {}; preference_to_1_ratio_h5_g: {}; preference_to_1_ratio_h1_g: {}".format(preference_to_1_ratio_h5_h1, preference_to_1_ratio_h5_g, preference_to_1_ratio_h1_g))
    if if_save:
        compare_result_path = f"Analysis_Results/final_analysis_pairwise_compare_results_{start_id}_{end_id}_{exp_model_name}_{exp_eval_model_name}_{output_dir_postfix}_if_multiple_llm_{if_multiple_llm}_beam_size_branching_{beam_size_branching}_{if_use_vague_cg_hyp_as_input}_{h5_exp_hierarchy_id}_{pairwise_eval_model_name}_{pairwise_if_multiple_llm}_num_compare_times_{num_compare_times}_{compare_metric}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}.json"
        with open(compare_result_path, "w") as f:
            json.dump([[h5_h1_collection, h5_g_collection, h1_g_collection], [preference_to_1_ratio_h5_h1, preference_to_1_ratio_h5_g, preference_to_1_ratio_h1_g]], f, indent=4)
            print(f"Results have been saved to {compare_result_path}")
    return h5_h1_collection, h5_g_collection, h1_g_collection



# Function:
#   compare hypothesis between methods
#   compare_metric: in ["overall", "effectiveness", "novelty", "detailedness", "feasibility"]
#   which_exp: [bool, bool, bool]; [if perform hierarchy_greedy_5, if perform hierarchy_greedy_1, if perform greedy]
# Output:
#   results_compare_collection_h5_h1/results_compare_collection_h5_g/results_compare_collection_h1_g: [[1/2/1.5, reason], ...], or None (according to which_exp)
def pairwise_compare_multiple_llm_with_one_example(bkg_id, research_question, which_exp, exp_model_name, exp_eval_model_name, load_file_name_if_multiple_llm, exp_model_name_2, exp_eval_model_name_2, load_file_name_if_multiple_llm_2, output_dir_postfix, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, if_generate_with_past_failed_hyp, if_print=True, compare_metric="overall", num_compare_times=5):
    # get final hypothesis from methods

    final_hypothesis_hierarchy_5_multiple_llm_1, final_hypothesis_hierarchy_1_multiple_llm_1, final_hypothesis_greedy_multiple_llm_1 = load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name, exp_eval_model_name, load_file_name_if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)

    final_hypothesis_hierarchy_5_multiple_llm_2, final_hypothesis_hierarchy_1_multiple_llm_2, final_hypothesis_greedy_multiple_llm_2 = load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name_2, exp_eval_model_name_2, load_file_name_if_multiple_llm_2, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)


    results_compare_collection_h5_llm_2_1, results_compare_collection_h1_llm_2_1, results_compare_collection_g_llm_2_1 = None, None, None
    if which_exp[0]:
        print("comparing multiple llm 2 with multiple llm 1 in hierarchy 5")
        results_compare_collection_h5_llm_2_1 = pairwise_compare_between_two_set_of_hypothesis(final_hypothesis_hierarchy_5_multiple_llm_2, final_hypothesis_hierarchy_5_multiple_llm_1, research_question, pairwise_compare, if_final_eval, compare_metric=compare_metric, num_compare_times=num_compare_times)
    if which_exp[1]:
        print("comparing multiple llm 2 with multiple llm 1 in hierarchy 1")
        results_compare_collection_h1_llm_2_1 = pairwise_compare_between_two_set_of_hypothesis(final_hypothesis_hierarchy_1_multiple_llm_2, final_hypothesis_hierarchy_1_multiple_llm_1, research_question, pairwise_compare, if_final_eval, compare_metric=compare_metric, num_compare_times=num_compare_times)
    if which_exp[2]:
        print("comparing multiple llm 2 with multiple llm 1 in greedy")
        results_compare_collection_g_llm_2_1 = pairwise_compare_between_two_set_of_hypothesis(final_hypothesis_greedy_multiple_llm_2, final_hypothesis_greedy_multiple_llm_1, research_question, pairwise_compare, if_final_eval, compare_metric=compare_metric, num_compare_times=num_compare_times)

    if if_print:
        # print("results_compare_collection_h5_llm_2_1:", results_compare_collection_h5_llm_2_1)
        # print("results_compare_collection_h1_llm_2_1:", results_compare_collection_h1_llm_2_1)
        # print("results_compare_collection_g_llm_2_1:", results_compare_collection_g_llm_2_1)
        if which_exp[0]:
            print("results_compare_collection_h5_llm_2_1:", [d[0] for d in results_compare_collection_h5_llm_2_1])
        if which_exp[1]:
            print("results_compare_collection_h1_llm_2_1:", [d[0] for d in results_compare_collection_h1_llm_2_1])
        if which_exp[2]:
            print("results_compare_collection_g_llm_2_1:", [d[0] for d in results_compare_collection_g_llm_2_1])
    return results_compare_collection_h5_llm_2_1, results_compare_collection_h1_llm_2_1, results_compare_collection_g_llm_2_1



# Include both start_id and end_id
# exp_model_name: only used for file name matching & output file name
# pairwise_eval_model_name: use which model for pairwise evaluation
def pairwise_compare_multiple_llm_with_batch_examples(start_id, end_id, chem_annotation_path, which_exp, exp_model_name, exp_eval_model_name, load_file_name_if_multiple_llm, exp_model_name_2, exp_eval_model_name_2, load_file_name_if_multiple_llm_2, output_dir_postfix, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, pairwise_eval_model_name, pairwise_if_multiple_llm, if_generate_with_past_failed_hyp, if_print=True, if_save=False, compare_metric="overall", num_compare_times=5):
    # obtain groundtruth finegrained hypothesis and experiment
    bkg_q_list, dict_bkg2survey, dict_bkg2cg_hyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note = load_chem_annotation(chem_annotation_path)

    # check whether all files are there
    for cur_bkg_id in range(start_id, end_id + 1):
        final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = load_hypothesis_from_methods(cur_bkg_id, which_exp, exp_model_name, exp_eval_model_name, load_file_name_if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)
        final_hypothesis_hierarchy_5_2, final_hypothesis_hierarchy_1_2, final_hypothesis_greedy_2 = load_hypothesis_from_methods(cur_bkg_id, which_exp, exp_model_name_2, exp_eval_model_name_2, load_file_name_if_multiple_llm_2, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)
    print("All files are there.")

    # h5_llm_2_1_collection/h1_llm_2_1_collection/g_llm_2_1_collection: [[[1/2, reason], ...]]
    h5_llm_2_1_collection, h1_llm_2_1_collection, g_llm_2_1_collection = [], [], []
    for cur_bkg_id in range(start_id, end_id + 1):
        print("Processing bkg_id:", cur_bkg_id)
        cur_rq = bkg_q_list[cur_bkg_id]
        # cur_rlt_h5_h1/cur_rlt_h5_g/cur_rlt_h1_g: [[1/2, reason]]
        cur_rlt_h5_llm_2_1, cur_rlt_h1_llm_2_1, cur_rlt_g_llm_2_1 = pairwise_compare_multiple_llm_with_one_example(cur_bkg_id, cur_rq, which_exp, exp_model_name, exp_eval_model_name, load_file_name_if_multiple_llm, exp_model_name_2, exp_eval_model_name_2, load_file_name_if_multiple_llm_2, output_dir_postfix, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, if_generate_with_past_failed_hyp, if_print, compare_metric=compare_metric, num_compare_times=num_compare_times)
        h5_llm_2_1_collection.append(cur_rlt_h5_llm_2_1)
        h1_llm_2_1_collection.append(cur_rlt_h1_llm_2_1)
        g_llm_2_1_collection.append(cur_rlt_g_llm_2_1)

    # preference_to_1_ratio_h5_llm_2_1/preference_to_1_ratio_h1_llm_2_1/preference_to_1_ratio_g_llm_2_1: [ratio_1, ratio_tie, ratio_2], or None (according to which_exp)
    preference_to_1_ratio_h5_llm_2_1, preference_to_1_ratio_h1_llm_2_1, preference_to_1_ratio_g_llm_2_1 = None, None, None
    if which_exp[0]:
        preference_to_1_ratio_h5_llm_2_1 = calculate_average_preference(h5_llm_2_1_collection)
    if which_exp[1]:
        preference_to_1_ratio_h1_llm_2_1 = calculate_average_preference(h1_llm_2_1_collection)
    if which_exp[2]:
        preference_to_1_ratio_g_llm_2_1 = calculate_average_preference(g_llm_2_1_collection)

    if if_print:
        print("preference_to_1_ratio_h5_llm_2_1: {}; preference_to_1_ratio_h1_llm_2_1: {}; preference_to_1_ratio_g_llm_2_1: {}".format(preference_to_1_ratio_h5_llm_2_1, preference_to_1_ratio_h1_llm_2_1, preference_to_1_ratio_g_llm_2_1))
    if if_save:
        compare_result_path = f"Analysis_Results/final_analysis_pairwise_compare_results_between_multiple_llm_{start_id}_{end_id}_{exp_model_name}_{exp_eval_model_name}_{load_file_name_if_multiple_llm}_{exp_model_name_2}_{exp_eval_model_name_2}_{load_file_name_if_multiple_llm_2}_{output_dir_postfix}_beam_size_branching_{beam_size_branching}_{if_use_vague_cg_hyp_as_input}_{h5_exp_hierarchy_id}_{pairwise_eval_model_name}_{pairwise_if_multiple_llm}_num_compare_times_{num_compare_times}_{compare_metric}_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}.json"
        with open(compare_result_path, "w") as f:
            json.dump([[h5_llm_2_1_collection, h1_llm_2_1_collection, g_llm_2_1_collection], [preference_to_1_ratio_h5_llm_2_1, preference_to_1_ratio_h1_llm_2_1, preference_to_1_ratio_g_llm_2_1]], f, indent=4)
            print(f"Results have been saved to {compare_result_path}")
    return h5_llm_2_1_collection, h1_llm_2_1_collection, g_llm_2_1_collection








# Function:
#   compare hypothesis with groundtruth to calculate f1 score
# Output:
#   final_hypothesis_hierarchy_5_scores/final_hypothesis_hierarchy_1_scores/final_hypothesis_greedy_scores: [[precision, recall, f1, weighted_precision, weighted_recall, weighted_f1], ...]
def compare_h5_h1_g_with_groundtruth_one_example(if_eval, bkg_id, which_exp, exp_model_name, exp_eval_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, evaluator, num_compare_times, locam_minimum_threshold, h5_exp_hierarchy_id, if_generate_with_past_failed_hyp, if_print=True):
    # get final hypothesis from methods
    final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)

    final_hypothesis_hierarchy_5_scores, final_hypothesis_hierarchy_1_scores, final_hypothesis_greedy_scores = None, None, None
    # hierarchy_greedy: 5 hierarchy
    if if_eval:
        if which_exp[0]:
            final_hypothesis_hierarchy_5_scores = evaluate_hyp(final_hypothesis_hierarchy_5, bkg_id, evaluator, type="hyp", num_compare_times=num_compare_times)
    # hierarchy_greedy: 1 hierarchy
    if if_eval:
        if which_exp[1]:
            final_hypothesis_hierarchy_1_scores = evaluate_hyp(final_hypothesis_hierarchy_1, bkg_id, evaluator, type="hyp", num_compare_times=num_compare_times)
    # greedy
    if if_eval:
        if which_exp[2]:
            final_hypothesis_greedy_scores = evaluate_hyp(final_hypothesis_greedy, bkg_id, evaluator, type="hyp", num_compare_times=num_compare_times)

    # print
    if if_print:
        if which_exp[0]:
            print("final_hypothesis_hierarchy_5:", final_hypothesis_hierarchy_5)
        if which_exp[1]:
            print("final_hypothesis_hierarchy_1:", final_hypothesis_hierarchy_1)
        if which_exp[2]:
            print("final_hypothesis_greedy:", final_hypothesis_greedy)

        if which_exp[0]:
            print("len(final_hypothesis_hierarchy_5):", len(final_hypothesis_hierarchy_5))
        if which_exp[1]:
            print("len(final_hypothesis_hierarchy_1):", len(final_hypothesis_hierarchy_1))
        if which_exp[2]:
            print("len(final_hypothesis_greedy):", len(final_hypothesis_greedy))

        if if_eval:
            if which_exp[0]:
                print("final_hypothesis_hierarchy_5_scores:", final_hypothesis_hierarchy_5_scores)
            if which_exp[1]:
                print("final_hypothesis_hierarchy_1_scores:", final_hypothesis_hierarchy_1_scores)
            if which_exp[2]:
                print("final_hypothesis_greedy_scores:", final_hypothesis_greedy_scores)

    return final_hypothesis_hierarchy_5_scores, final_hypothesis_hierarchy_1_scores, final_hypothesis_greedy_scores
                






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=int, default=0, help="0: openai's API toolkit; 1: azure's API toolkit; 2: google's API toolkit")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="https://api.claudeshop.top/v1", help="base url for the API")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--preprocess_groundtruth_components_dir", type=str, default="./Checkpoints/groundtruth_hyp_components_collection.json", help="store the preprocessed groundtruth hypothesis's components (used for f1 score evaluation compared with generated hypothesis)")
    parser.add_argument("--log_dir", type=str, default="./Logs/analysis_recall.json")
    args = parser.parse_args()
    print(args)

    ## Setup logger
    logger = setup_logger(args.log_dir)
    # Redirect print to logger
    def custom_print(*args, **kwargs):
        message = " ".join(map(str, args))
        logger.info(message)
    # global print
    # print = custom_print
    builtins.print = custom_print
    # print(args)

    # we use ./Analysis_Results to store the analysis results (similar to ./Checkpoints)
    if os.path.exists("Analysis_Results") == False:
        os.mkdir("Analysis_Results")

    '''shared parameters for pairwise compare and f1 score compare (start)'''
    ## load/save file name parameters
    # number of times to compare the hypothesis to get average results: for both f1 scores and pairwise compare
    #   default value: '5' for pairwise evaluation, and '3' for f1 score evaluation
    num_compare_times = 3
    # Mainly used for file name matching.
    load_file_name_model_name = "gpt-4o-mini"
    load_file_name_eval_model_name = "gpt-4o-mini"
    # load_file_name_eval_model_name = "gemini-1.5-flash-latest"
    # Mainly used for file name matching. (pairwise compare) whether to use multiple llms for hypothesis gradient estimation. 0: single llm; 1: multiple same llms; 2: multiple different llms
    load_file_name_if_multiple_llm = 1
    # locam_minimum_threshold/beam_size_branching/if_use_vague_cg_hyp_as_input/output_dir_postfix: only used for file name matching
    # output_dir_postfix: updated_prompt_feb_14, updated_prompt_mar_29, test
    output_dir_postfix = "test"
    locam_minimum_threshold=2
    beam_size_branching=2
    if_use_vague_cg_hyp_as_input=1
    if_generate_with_past_failed_hyp=1
    
    ## general experiment parameters
    if_save = True
    if_print = True
    start_id, end_id = 0, 50
    # which_exp: [if perform hierarchy_greedy_5, if perform hierarchy_greedy_1, if perform greedy] (mainly used when pairwise_or_f1_compare_mode=1 and compare_mode=2)
    which_exp = [1, 0, 0]
    # h5_exp_hierarchy_id: the hierarchy_id of the hypothesis to be loaded in hierarchy_greedy_5
    h5_exp_hierarchy_id = 4
    # pairwise_or_f1_compare_mode: 1: pairwise compare; 2: f1 score; 3: check average search step
    pairwise_or_f1_compare_mode = 3
    '''shared parameters for pairwise compare and f1 score compare (end)'''

    ## pairwise compare
    if pairwise_or_f1_compare_mode == 1:
        '''shared parameters for pairwise compare (start)'''
        ## initialize pairwise evaluator
        # pairwise_eval_model_name = "gpt-4o"
        # pairwise_eval_model_name = "claude-3-5-sonnet-20241022"
        # pairwise_eval_model_name = "gemini-1.5-pro"
        # pairwise_eval_model_name = "gpt-4o-mini"
        # pairwise_eval_model_name = "claude-3-haiku-20240307"
        pairwise_eval_model_name = "gemini-1.5-flash-latest"
        pairwise_if_multiple_llm = 1
        pairwise_compare = PairwiseCompare(args.api_type, args.api_key, args.base_url, pairwise_eval_model_name, pairwise_if_multiple_llm)
        
        ## pairwise compare
        if_final_eval = True
        # compare_metric: in ["overall", "effectiveness", "novelty", "detailedness", "feasibility"]
        compare_metric_full_list = ["effectiveness", "detailedness", "novelty", "feasibility", "overall"]
        # compare_mode: 1: compare between hierarchy 5, hierarchy 1, and greedy; 2: compare between [claude, gpt4, gemini] and [gpt4, gpt4, gpt4]
        compare_mode = 2
        '''shared parameters for pairwise compare (end)'''


        '''parameters for pairwise compare (compare_mode 2) (start)'''
        load_file_name_model_name_2 = "gpt-4o-mini"
        load_file_name_eval_model_name_2 = "gpt-4o-mini"
        # load_file_name_eval_model_name_2 = "gemini-1.5-flash-latest"
        # load_file_name_if_multiple_llm_2 = 1
        load_file_name_if_multiple_llm_2 = 2
        '''parameters for pairwise compare (compare_mode 2) (end)'''

        for cur_compare_metric in compare_metric_full_list:
            print("cur_compare_metric: ", cur_compare_metric)
            if compare_mode == 1:
                # compare between hierarchy 5, hierarchy 1, and greedy
                # print("h5_exp_hierarchy_id: ", h5_exp_hierarchy_id)
                h5_h1_collection, h5_g_collection, h1_g_collection = pairwise_compare_h5_h1_g_with_batch_examples(start_id, end_id, args.chem_annotation_path, which_exp, load_file_name_model_name, load_file_name_eval_model_name, output_dir_postfix, load_file_name_if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, pairwise_eval_model_name=pairwise_eval_model_name, pairwise_if_multiple_llm=pairwise_if_multiple_llm, if_generate_with_past_failed_hyp=if_generate_with_past_failed_hyp, if_print=if_print, if_save=if_save, compare_metric=cur_compare_metric, num_compare_times=num_compare_times)
            elif compare_mode == 2:
                # compare between [claude, gpt4, gemini] and [gpt4, gpt4, gpt4]
                h5_llm_2_1_collection, h1_llm_2_1_collection, g_llm_2_1_collection = pairwise_compare_multiple_llm_with_batch_examples(start_id, end_id, args.chem_annotation_path, which_exp, load_file_name_model_name, load_file_name_eval_model_name, load_file_name_if_multiple_llm, load_file_name_model_name_2, load_file_name_eval_model_name_2, load_file_name_if_multiple_llm_2, output_dir_postfix, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, h5_exp_hierarchy_id, pairwise_eval_model_name=pairwise_eval_model_name, pairwise_if_multiple_llm=pairwise_if_multiple_llm, if_generate_with_past_failed_hyp=if_generate_with_past_failed_hyp, if_print=if_print, if_save=if_save, compare_metric=cur_compare_metric, num_compare_times=num_compare_times)
            else:
                raise Exception(f"compare_mode must be 1 or 2, got {compare_mode}")
    ## f1 score compare
    elif pairwise_or_f1_compare_mode == 2:
        '''parameters for f1 score compare (start)'''
        ## initialize f1 evaluator
        # evaluator_model_name = "gpt-4o"
        evaluator_model_name = "gpt-4o-mini"
        evaluator = Evaluator(evaluator_model_name, args.api_type, args.api_key, args.base_url, args.chem_annotation_path, args.preprocess_groundtruth_components_dir)

        # 0: only load data; 1: load data and evaluate
        if_eval = 1
        '''parameters for f1 score compare (end)'''
        if if_eval == 0:
            assert if_save == False

        f1_scores_collection = []
        for cur_bkg_id in range(start_id, end_id + 1):
            print("Processing bkg_id:", cur_bkg_id)
            # f1_scores: [final_hypothesis_hierarchy_5_scores, final_hypothesis_hierarchy_1_scores, final_hypothesis_greedy_scores]
            #   final_hypothesis_hierarchy_5_scores/final_hypothesis_hierarchy_1_scores/final_hypothesis_greedy_scores: [[precision, recall, f1, weighted_precision, weighted_recall, weighted_f1], ...]
            f1_scores = compare_h5_h1_g_with_groundtruth_one_example(if_eval, cur_bkg_id, which_exp, load_file_name_model_name, load_file_name_eval_model_name, output_dir_postfix, load_file_name_if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, evaluator, num_compare_times, locam_minimum_threshold, h5_exp_hierarchy_id, if_generate_with_past_failed_hyp, if_print=if_print)
            f1_scores_collection.append(f1_scores)
        
        if if_save:
            # added "which_exp" to the file name
            f1_result_path = f"Analysis_Results/final_analysis_f1_scores_results_{start_id}_{end_id}_{load_file_name_model_name}_{load_file_name_eval_model_name}_{output_dir_postfix}_{load_file_name_if_multiple_llm}_{locam_minimum_threshold}_beam_size_branching_{beam_size_branching}_{if_use_vague_cg_hyp_as_input}_{h5_exp_hierarchy_id}_{evaluator_model_name}_num_compare_times_{num_compare_times}_maxScoreEachComponent_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_{which_exp[0]}_{which_exp[1]}_{which_exp[2]}.json"
            with open(f1_result_path, "w") as f:
                json.dump(f1_scores_collection, f, indent=4)
                print(f"Results have been saved to {f1_result_path}")
    elif pairwise_or_f1_compare_mode == 3:
        check_average_search_step(start_id, end_id, which_exp, load_file_name_model_name, load_file_name_eval_model_name, load_file_name_if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)



    



    # ## get top matched score hypothesis from MOOSE-Chem
    # file_path = "Data/expert_eval_for_selected_hyp_in_exp_5.json"
    # # data: [[cur_hyp, cur_gdth_hyp, cnt_matched_insp, cur_matched_score, cur_matched_score_reason], ...]
    # with open(file_path, "r") as f:
    #     data = json.load(f)
    # print(data['0'][1])
    