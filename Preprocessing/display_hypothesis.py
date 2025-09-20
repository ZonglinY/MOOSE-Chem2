import os, sys, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Evaluation.analysis import load_final_hypothesis_from_HGTree, load_final_hypothesis_from_json


# used for match hypothesis file name
model_name="gpt-4o-mini"
eval_model_name="gpt-4o-mini"

locam_minimum_threshold=2
if_multiple_llm=1
output_dir_postfix="updated_prompt_mar_29_geophysics"

beam_compare_mode=0
beam_size_branching=2
num_init_for_EU=3
num_recom_trial_for_better_hyp=2
if_feedback=1
if_parallel=1
if_use_vague_cg_hyp_as_input=1
if_generate_with_example=1




def display_hypothesis(display_txt_file_path, if_hierarchical, hierarchy_id, start_bkg_id, end_bkg_id):

    ## load hypothesis
    final_hyp_list = []
    for cur_bkg_id in range(start_bkg_id, end_bkg_id + 1):

        # final_hypothesis: [hyp]
        if if_hierarchical == 1:
            num_hierarchy = 5
            hypothesis_path = f"./Checkpoints/hierarchical_greedy_{num_hierarchy}_{locam_minimum_threshold}_{if_feedback}_{num_recom_trial_for_better_hyp}_{model_name}_{eval_model_name}_beam_compare_mode_{beam_compare_mode}_beam_size_branching_{beam_size_branching}_num_init_for_EU_{num_init_for_EU}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{cur_bkg_id}_{output_dir_postfix}.pkl"
            cur_bkg_hypothesis = load_final_hypothesis_from_HGTree(hypothesis_path, hierarchy_id)
        else:
            hypothesis_path = f"./Checkpoints/greedy_{locam_minimum_threshold}_{if_feedback}_{model_name}_{eval_model_name}_if_multiple_llm_{if_multiple_llm}_if_use_vague_cg_hyp_as_input_{if_use_vague_cg_hyp_as_input}_bkgid_{cur_bkg_id}_{output_dir_postfix}.json"
            cur_bkg_hypothesis = load_final_hypothesis_from_json(hypothesis_path)
        
        final_hyp_list.extend(cur_bkg_hypothesis)

    ## display hypothesis
    with open(display_txt_file_path, "w") as f:
        for cur_id, cur_hypothesis in enumerate(final_hyp_list):
            f.write(f"Hypothsis {cur_id}:\n" + cur_hypothesis + "\n\n")
            f.write("\n")

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display the research background and output finegrained hypothesis")
    # parser.add_argument("--hypothesis_path", type=str, default="", help="the path to the output file of hierarchy_greedy.py or greedy.py")
    parser.add_argument("--display_txt_file_path", type=str, default="./finegrained_hyp.txt", help="the path to the output file of display_hypothesis.py")
    parser.add_argument("--if_hierarchical", type=int, default=1, help="if the output file is from hierarchy_greedy.py, 1 for yes, 0 for no")
    parser.add_argument("--hierarchy_id", type=int, default=4, help="the id of the hierarchy")
    parser.add_argument("--start_bkg_id", type=int, default=0, help="the id of the background")
    parser.add_argument("--end_bkg_id", type=int, default=4, help="the id of the background")
    args = parser.parse_args()

    # args.hypothesis_path = os.path.expanduser(args.hypothesis_path)
    args.display_txt_file_path = os.path.expanduser(args.display_txt_file_path)

    display_hypothesis(args.display_txt_file_path, args.if_hierarchical, args.hierarchy_id, args.start_bkg_id, args.end_bkg_id)
    print("Display hypothesis finished!")

