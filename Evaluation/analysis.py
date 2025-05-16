import os, sys, json, argparse
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.hierarchy_greedy_utils import HGNode, HGTree
from Evaluation.evaluate import Evaluator
from Evaluation.pairwise_compare import PairwiseCompare
from Method.utils import load_chem_annotation



# Input: 
#   file_path: str
#   hierarchy_id: int; the hierarchy_id of the hypothesis to be loaded
# Output: 
#   final_hypothesis: [str]
def load_final_hypothesis_from_HGTree(file_path, hierarchy_id):
    a = HGTree.load(file_path)
    final_hypothesis = []
    def search_tree_breadth_first(node, hierarchy_id):
        queue = [node]
        visited = set()
        while queue:
            node = queue.pop(0)
            try:
                if node.hierarchy_id == hierarchy_id:
                    final_hypothesis.append(node.full_generated_hyp[-1][-1][0])
            except Exception as e:
                print("Warning: ", e)
            
            visited.add(node)
            for child in node.children:
                if child not in visited:
                    queue.append(child)
    search_tree_breadth_first(a.root, hierarchy_id)
    return final_hypothesis
    


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
def load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix):
    # initialze
    final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = None, None, None

    # hierarchy_greedy: 5 hierarchy
    if which_exp[0]:
        file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        # file_path = f"Checkpoints/hierarchical_greedy_5_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        hierarchy_id = 2
        # final_hypothesis_hierarchy_5: [hyp0, hyp1, ...]
        final_hypothesis_hierarchy_5 = load_final_hypothesis_from_HGTree(file_path, hierarchy_id)
    
    # hierarchy_greedy: 1 hierarchy
    if which_exp[1]:
        file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        # file_path = f"Checkpoints/hierarchical_greedy_1_{locam_minimum_threshold}_1_2_{exp_model_name}_{exp_model_name}_beam_compare_mode_0_beam_size_branching_{beam_size_branching}_num_init_for_EU_3_if_multiple_llm_{if_multiple_llm}_bkgid_{bkg_id}_{output_dir_postfix}.pkl"
        hierarchy_id = 0
        # final_hypothesis_hierarchy_1: [hyp0, hyp1, ...]
        final_hypothesis_hierarchy_1 = load_final_hypothesis_from_HGTree(file_path, hierarchy_id)

    # greedy
    if which_exp[2]:
        locam_minimum_threshold = int(locam_minimum_threshold) + 1
        file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{bkg_id}_{output_dir_postfix}.json"
        # file_path = f"Checkpoints/greedy_{locam_minimum_threshold}_1_{exp_model_name}_{exp_model_name}_if_multiple_llm_{if_multiple_llm}_bkgid_{bkg_id}_{output_dir_postfix}.json"
        # final_hypothesis_greedy: [hyp0, hyp1, ...]
        final_hypothesis_greedy = load_final_hypothesis_from_json(file_path)

    return final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy
    


# Function:
#   compare hypothesis between methods
# Output:
#   results_compare_collection_h5_h1/results_compare_collection_h5_g/results_compare_collection_h1_g: [[1/2, reason], ...]
def pairwise_compare_with_one_example(bkg_id, research_question, which_exp, exp_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, if_print=True):
    # get final hypothesis from methods
    final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix)

    def pairwise_compare_with_one_example_between_two_methods(final_hypothesis_method1, final_hypothesis_method2):
        results_compare_collection = []
        # compare h5 with h1
        for cur_hyp_m1 in final_hypothesis_method1:
            for cur_hyp_m2 in final_hypothesis_method2:
                # response: [1/2, reason]
                response = pairwise_compare.compare(research_question, cur_hyp_m1, cur_hyp_m2, instruction_mode="same_hyp1_hyp2", hierarchy_level=None, if_final_eval=if_final_eval)
                response[0] = int(response[0])
                assert response[0] == 1 or response[0] == 2
                results_compare_collection.append(response)
        return results_compare_collection

    results_compare_collection_h5_h1 = pairwise_compare_with_one_example_between_two_methods(final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1)
    results_compare_collection_h5_g = pairwise_compare_with_one_example_between_two_methods(final_hypothesis_hierarchy_5, final_hypothesis_greedy)
    results_compare_collection_h1_g = pairwise_compare_with_one_example_between_two_methods(final_hypothesis_hierarchy_1, final_hypothesis_greedy)

    if if_print:
        print("results_compare_collection_h5_h1:", [d[0] for d in results_compare_collection_h5_h1])
        print("results_compare_collection_h5_g:", [d[0] for d in results_compare_collection_h5_g])
        print("results_compare_collection_h1_g:", [d[0] for d in results_compare_collection_h1_g])
    return results_compare_collection_h5_h1, results_compare_collection_h5_g, results_compare_collection_h1_g


# Include both start_id and end_id
def pairwise_compare_with_batch_examples(start_id, end_id, chem_annotation_path, which_exp, exp_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, if_print=True):
    # obtain groundtruth finegrained hypothesis and experiment
    bkg_q_list, dict_bkg2survey, dict_bkg2cg_hyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note = load_chem_annotation(chem_annotation_path)

    # rlt_collection: [[[1/2, reason], ...]]
    def calculate_average_preference(rlt_collection):
        cnt_1, cnt_2 = 0, 0
        for cur_rlt in rlt_collection:
            for cur_pairwise_rlt in cur_rlt:
                if cur_pairwise_rlt[0] == 1:
                    cnt_1 += 1
                elif cur_pairwise_rlt[0] == 2:
                    cnt_2 += 1
                else:
                    raise Exception(f"cur_pairwise_rlt[0] must be 1 or 2, got {cur_pairwise_rlt[0]}")
        preference_to_1_ratio = cnt_1 / (cnt_1 + cnt_2)
        return preference_to_1_ratio


    h5_h1_collection, h5_g_collection, h1_g_collection = [], [], []
    for cur_bkg_id in range(start_id, end_id + 1):
        print("Processing bkg_id:", cur_bkg_id)
        cur_rq = bkg_q_list[cur_bkg_id]
        # cur_rlt_h5_h1/cur_rlt_h5_g/cur_rlt_h1_g: [[1/2, reason]]
        cur_rlt_h5_h1, cur_rlt_h5_g, cur_rlt_h1_g = pairwise_compare_with_one_example(cur_bkg_id, cur_rq, which_exp, exp_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, if_print)
        h5_h1_collection.append(cur_rlt_h5_h1)
        h5_g_collection.append(cur_rlt_h5_g)
        h1_g_collection.append(cur_rlt_h1_g)

    preference_to_1_ratio_h5_h1 = calculate_average_preference(h5_h1_collection)
    preference_to_1_ratio_h5_g = calculate_average_preference(h5_g_collection)
    preference_to_1_ratio_h1_g = calculate_average_preference(h1_g_collection)

    if if_print:
        print("preference_to_1_ratio_h5_h1: {}; preference_to_1_ratio_h5_g: {}; preference_to_1_ratio_h1_g: {}".format(preference_to_1_ratio_h5_h1, preference_to_1_ratio_h5_g, preference_to_1_ratio_h1_g))
    return h5_h1_collection, h5_g_collection, h1_g_collection


# Function:
#   compare hypothesis with groundtruth to calculate f1 score
# Output:
#   final_hypothesis_hierarchy_5_scores/final_hypothesis_hierarchy_1_scores/final_hypothesis_greedy_scores: [[precision, recall, f1, weighted_precision, weighted_recall, weighted_f1], ...]
def compare_h5_h1_g_with_groundtruth_one_example(if_eval, bkg_id, which_exp, exp_model_name, output_dir_postfix, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, evaluator, num_compare_times, locam_minimum_threshold, if_print=True):
    # get final hypothesis from methods
    final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = load_hypothesis_from_methods(bkg_id, which_exp, exp_model_name, if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix)

    final_hypothesis_hierarchy_5_scores, final_hypothesis_hierarchy_1_scores, final_hypothesis_greedy_scores = None, None, None
    # hierarchy_greedy: 5 hierarchy
    if if_eval:
        final_hypothesis_hierarchy_5_scores = evaluate_hyp(final_hypothesis_hierarchy_5, bkg_id, evaluator, type="hyp", num_compare_times=num_compare_times)
    # hierarchy_greedy: 1 hierarchy
    if if_eval:
        final_hypothesis_hierarchy_1_scores = evaluate_hyp(final_hypothesis_hierarchy_1, bkg_id, evaluator, type="hyp", num_compare_times=num_compare_times)
    # greedy
    if if_eval:
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
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="model name: gpt-4o/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--api_type", type=int, default=0, help="0: claude shop; 1: azure")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="https://api.claudeshop.top/v1", help="base url for the API")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--preprocess_groundtruth_components_dir", type=str, default="./Checkpoints/groundtruth_hyp_components_collection.json")
    parser.add_argument("--num_compare_times", type=int, default=5, help="(f1 scores) number of times to compare the hypothesis to get average results")
    parser.add_argument("--if_multiple_llm", type=int, default=1, help="(pairwise compare) whether to use multiple llms for hypothesis gradient estimation. 0: single llm; 1: multiple same llms; 2: multiple different llms")
    args = parser.parse_args()

    # initialize evaluator
    evaluator = Evaluator(args.model_name, args.api_type, args.api_key, args.chem_annotation_path, args.preprocess_groundtruth_components_dir)
    pairwise_compare = PairwiseCompare(args.api_type, args.api_key, args.base_url, args.model_name, args.if_multiple_llm)


    ## pairwise compare
    start_id, end_id = 3, 3
    output_dir_postfix = "updated_prompt_feb_13"
    locam_minimum_threshold=2
    beam_size_branching=1
    if_use_vague_cg_hyp_as_input=1
    which_exp = [1, 1, 1]
    if_final_eval = True

    h5_h1_collection, h5_g_collection, h1_g_collection = pairwise_compare_with_batch_examples(start_id, end_id, args.chem_annotation_path, which_exp, args.model_name, output_dir_postfix, args.if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, pairwise_compare, if_final_eval, locam_minimum_threshold, if_print=True)



    ## calculate f1 one bkg_id
    # if_eval = 0

    # bkg_id = 2
    # output_dir_postfix = "updated_prompt_feb_13"
    # locam_minimum_threshold=2
    # beam_size_branching=1
    # if_use_vague_cg_hyp_as_input=1
    # which_exp = [1, 1, 1]

    # f1_scores = compare_h5_h1_g_with_groundtruth_one_example(if_eval, bkg_id, which_exp, args.model_name, output_dir_postfix, args.if_multiple_llm, beam_size_branching, if_use_vague_cg_hyp_as_input, evaluator, args.num_compare_times, locam_minimum_threshold, if_print=True)

    
    # ## get top matched score hypothesis from MOOSE-Chem
    # file_path = "Data/expert_eval_for_selected_hyp_in_exp_5.json"
    # # data: [[cur_hyp, cur_gdth_hyp, cnt_matched_insp, cur_matched_score, cur_matched_score_reason], ...]
    # with open(file_path, "r") as f:
    #     data = json.load(f)
    # print(data['0'][1])
    

