import os, sys, json
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Evaluation.analysis import load_hypothesis_from_methods
from Method.utils import load_chem_annotation



def select_hypothesis_for_expert_evaluation(if_save):

    ## load research question
    bkg_q, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note = load_chem_annotation("./Data/chem_research_2024_finegrained.xlsx")
    len_bkg_q = len(bkg_q)
    print("Total number of research questions: ", len_bkg_q)

    ttl_selected_hyp_3_to_rank = []
    ttl_selected_hyp_5_to_rank = []
    for cur_bkg_id in range(0, len_bkg_q):
        # set parameters to select files to extract hypothesis
        which_exp = [1, 1, 1]
        exp_model_name, exp_eval_model_name = "gpt-4o-mini", "gpt-4o-mini"
        if_multiple_llm = 1
        beam_size_branching = 2
        if_use_vague_cg_hyp_as_input = 1
        locam_minimum_threshold = 2
        output_dir_postfix = "updated_prompt_mar_29"
        h5_exp_hierarchy_id = 4
        if_generate_with_past_failed_hyp = 0
        # get final hypothesis from methods
        final_hypothesis_hierarchy_5, final_hypothesis_hierarchy_1, final_hypothesis_greedy = load_hypothesis_from_methods(cur_bkg_id, which_exp, exp_model_name, exp_eval_model_name, if_multiple_llm, 
        beam_size_branching, if_use_vague_cg_hyp_as_input, locam_minimum_threshold, output_dir_postfix, if_generate_with_past_failed_hyp, h5_exp_hierarchy_id)
        # selected_hyp_3_to_rank
        cur_selected_hyp_3_to_rank = [final_hypothesis_hierarchy_5[0], final_hypothesis_hierarchy_1[0], final_hypothesis_greedy[0]]
        ttl_selected_hyp_3_to_rank.append([cur_bkg_id, bkg_q[cur_bkg_id], cur_selected_hyp_3_to_rank])
        # selected_hyp_5_to_rank
        cur_selected_hyp_5_to_rank = [final_hypothesis_hierarchy_5[0], final_hypothesis_hierarchy_5[1], final_hypothesis_hierarchy_1[0], final_hypothesis_hierarchy_1[1], final_hypothesis_greedy[0]]
        ttl_selected_hyp_5_to_rank.append([cur_bkg_id, bkg_q[cur_bkg_id], cur_selected_hyp_5_to_rank])

    print("Total number of selected hypothesis for 3 to rank: ", len(ttl_selected_hyp_3_to_rank))
    print("Total number of selected hypothesis for 5 to rank: ", len(ttl_selected_hyp_5_to_rank))

    # save selected hypothesis
    if if_save:
        with open("expert_evaluation_3_to_rank.json", "w") as f:
            json.dump(ttl_selected_hyp_3_to_rank, f, indent=4)
        with open("expert_evaluation_5_to_rank.json", "w") as f:
            json.dump(ttl_selected_hyp_5_to_rank, f, indent=4)
        print("Selected hypothesis saved to expert_evaluation.json")
        write_selected_hyp_to_markdown_to_present(ttl_selected_hyp_3_to_rank, "expert_evaluation_3_to_rank")
        write_selected_hyp_to_markdown_to_present(ttl_selected_hyp_5_to_rank, "expert_evaluation_5_to_rank")
        print("Selected hypothesis saved to expert_evaluation_3_to_rank.md and expert_evaluation_5_to_rank.md")
    else:
        print("if_save is set to 0, not saving the selected hypothesis.")


def write_selected_hyp_to_markdown_to_present(selected_hyp_3_or_5, present_file_name):
    # write the id, question, and hypothesis to a txt file, where change to a new page for each id
    with open(present_file_name + ".md", "w", encoding="utf-8") as f:
        for idx, (cur_bkg_id, bkg_q, cur_selected_hyp_3_or_5_to_rank) in enumerate(selected_hyp_3_or_5, 1):
            f.write(f"# 🔍 Entry ID {cur_bkg_id}\n\n")
            
            f.write(f"### ❓ Research Question\n")
            f.write(f"{bkg_q}\n\n")
            
            f.write(f"### 🧪 Hypothesis Candidates\n\n")
            for i, hyp in enumerate(cur_selected_hyp_3_or_5_to_rank, 1):
                f.write(f"**Candidate {i}**\n")
                f.write(f"{hyp.strip()}\n\n")

            f.write("---\n\n")



if __name__ == "__main__":
    # set if save selected hypothesis
    if_save = 1
    select_hypothesis_for_expert_evaluation(if_save)
    