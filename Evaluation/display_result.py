import json, os, shutil

PAIRWISE_METRICS = ["effectiveness", "detailedness", "novelty", "feasibility", "overall"]


# ============================ Recall Score (start) ============================

# Function: first get the start and end id from the file name, and then calculate the average recall score for each file, and then calculate the weighted average recall score for all files
# file_names: [file_name1, file_name2, ...]
# which_exp: [H5, H1, Greedy]; 0/1. e.G., [1, 0, 0] means only H5's recall score is calculated
def show_recall_score_multiple_files(file_names, which_exp, which_num=4):

    assert len(which_exp) == 3, "which_exp must be a list of length 3"
    h5_recall_single_file_sum = 0
    h1_recall_single_file_sum = 0
    greedy_recall_single_file_sum = 0

    ttl_bkg_id_cnted = 0
    for file_name in file_names:
        start_id, end_id = parse_start_end_id_from_recall_file_name(file_name)
        h5_recall, h1_recall, greedy_recall = show_recall_score_single_file(file_name, which_exp, which_num)

        ttl_bkg_id_cnted += end_id - start_id + 1
        h5_recall_single_file_sum += h5_recall * (end_id - start_id + 1)
        h1_recall_single_file_sum += h1_recall * (end_id - start_id + 1)
        greedy_recall_single_file_sum += greedy_recall * (end_id - start_id + 1)
    print("ttl_bkg_id_cnted: ", ttl_bkg_id_cnted)

    h5_recall_multiple_file_average = h5_recall_single_file_sum / ttl_bkg_id_cnted
    h1_recall_multiple_file_average = h1_recall_single_file_sum / ttl_bkg_id_cnted
    greedy_recall_multiple_file_average = greedy_recall_single_file_sum / ttl_bkg_id_cnted
    print(f"h5_recall_multiple_file_average: {h5_recall_multiple_file_average:.5f}, h1_recall_multiple_file_average: {h1_recall_multiple_file_average:.5f}, greedy_recall_multiple_file_average: {greedy_recall_multiple_file_average:.5f}")
    return h5_recall_multiple_file_average, h1_recall_multiple_file_average, greedy_recall_multiple_file_average
        



# which_exp: [H5, H1, Greedy]; 0/1. e.G., [1, 0, 0] means only H5's recall score is calculated
# which_num: 0: precision, 1: recall, 2: f1, 3: weighted_precision, 4: weighted_recall, 5: weighted_f1
def show_recall_score_single_file(file_name, which_exp, which_num=4):
    # data: [[h5_scores, h1_scores, greedy_scores], ...]
    #   h5_scores/h1_scores/greedy_scores: [[precision, recall, f1, weighted_precision, weighted_recall, weighted_f1], ...]; can be multiple
    with open(file_name, 'r') as f:
        data = json.load(f)

    h5_recall_average = 0
    h1_recall_average = 0   
    greedy_recall_average = 0
    for i in range(len(data)):
        # h5 recall
        if which_exp[0]:
            for j in range(len(data[i][0])):
                h5_recall_average += data[i][0][j][which_num]
        # h1 recall
        if which_exp[1]:
            for j in range(len(data[i][1])):
                h1_recall_average += data[i][1][j][which_num]
        # greedy recall
        if which_exp[2]:
            for j in range(len(data[i][2])):
                greedy_recall_average += data[i][2][j][which_num]
    h5_recall_average /= len(data)
    h1_recall_average /= len(data)
    greedy_recall_average /= len(data)
    print("h5_recall_average: {:.2f}: h1_recall_average: {:.2f}, greedy_recall_average: {:.2f}".format(h5_recall_average, h1_recall_average, greedy_recall_average))
    return h5_recall_average, h1_recall_average, greedy_recall_average

# =============================== Recall Score (end) ===============================


# ============================ Pairwise (start) ============================

# Output:
#   pairwise_score (averaged): [[h5_h1_win, h5_h1_tie, h5_h1_lose], [h5_greedy_win, h5_greedy_tie, h5_greedy_lose], [h1_greedy_win, h1_greedy_tie, h1_greedy_lose]]
def display_pairwise_score_multiple_files(file_names, file_type):
    assert file_type in ["pairwise_Q1", "pairwise_Q3", "recall"]

    all_pairwise_score = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ttl_bkg_id_cnted = 0
    for file_name in file_names:
        start_id, end_id = parse_start_end_id_from_recall_file_name(file_name, file_type=file_type)
        with open(file_name, 'r') as f:
            data = json.load(f)
        # cur_pairwise_score: [[h5_score, h1_score, greedy_score], ...]
        cur_pairwise_score = data[1]
        assert len(cur_pairwise_score) == len(all_pairwise_score)
        for cur_pair_id in range(len(cur_pairwise_score)):
            # cur_pairwise_score[cur_pair_id]: [h5_score, h1_score, greedy_score] or None
            if cur_pairwise_score[cur_pair_id]:
                all_pairwise_score[cur_pair_id][0] += cur_pairwise_score[cur_pair_id][0] * (end_id - start_id + 1)
                all_pairwise_score[cur_pair_id][1] += cur_pairwise_score[cur_pair_id][1] * (end_id - start_id + 1)
                all_pairwise_score[cur_pair_id][2] += cur_pairwise_score[cur_pair_id][2] * (end_id - start_id + 1)
        ttl_bkg_id_cnted += end_id - start_id + 1
    print("\nttl_bkg_id_cnted: ", ttl_bkg_id_cnted)

    # calculate the average pairwise score
    for cur_pair_id in range(len(all_pairwise_score)):
        all_pairwise_score[cur_pair_id][0] = all_pairwise_score[cur_pair_id][0] / ttl_bkg_id_cnted
        all_pairwise_score[cur_pair_id][1] = all_pairwise_score[cur_pair_id][1] / ttl_bkg_id_cnted
        all_pairwise_score[cur_pair_id][2] = all_pairwise_score[cur_pair_id][2] / ttl_bkg_id_cnted
    # print("all_pairwise_score: ", all_pairwise_score)
    return all_pairwise_score


# ============================ Pairwise (end) ============================
    

# obtain the start_id and end_id from the recall file name
def parse_start_end_id_from_recall_file_name(file_name, file_type="recall"):
    if file_type == "recall":
        file_name = file_name.split("/")[-1]
        start_id = int(file_name.split("_")[5])
        end_id = int(file_name.split("_")[6])
    elif file_type == "pairwise_Q1":
        file_name = file_name.split("/")[-1]
        start_id = int(file_name.split("_")[5])
        end_id = int(file_name.split("_")[6])
    elif file_type == "pairwise_Q3":
        file_name = file_name.split("/")[-1]
        start_id = int(file_name.split("_")[8])
        end_id = int(file_name.split("_")[9])
    else:
        raise ValueError("file_type must be 'recall' or 'pairwise_Q1' or 'pairwise_Q3'")
    return start_id, end_id



# id_pairs: [[start_id1, end_id1], [start_id2, end_id2], ...]
# model_name_1 / eval_model_name_1 / if_multiple_llm_1 / model_name_2 / eval_model_name_2 / if_multiple_llm_2 / pairwise_eval_model_name / pairwise_if_multiple_llm: only used for pairwise_Q3
def synthesize_file_names_from_start_end_id_for_recall_files(start_end_id_pairs, num_compare_times=3, root_dir="./Analysis_Results/", file_type="recall", aspect_type="overall", model_name_1="gpt-4o-mini", eval_model_name_1="gpt-4o-mini", if_multiple_llm_1=1, model_name_2="gpt-4o-mini", eval_model_name_2="gpt-4o-mini", if_multiple_llm_2=1, pairwise_eval_model_name="gpt-4o-mini", pairwise_if_multiple_llm=1, if_generate_with_past_failed_hyp=0, which_exp=None, output_dir_postfix="updated_prompt_mar_29"):
    # check file_type
    assert file_type in ["recall", "pairwise_Q1", "pairwise_Q3"]
    # check aspect_type
    if file_type in ["pairwise_Q1", "pairwise_Q3"]:
        assert aspect_type in PAIRWISE_METRICS

    file_names = []
    for start_id, end_id in start_end_id_pairs:
        if file_type == "recall":
            # added "which_exp" to the file name
            cur_file = root_dir + f"/final_analysis_f1_scores_results_{start_id}_{end_id}_gpt-4o-mini_gpt-4o-mini_{output_dir_postfix}_{if_multiple_llm_1}_2_beam_size_branching_2_1_4_gpt-4o-mini_num_compare_times_{num_compare_times}_maxScoreEachComponent_if_generate_with_past_failed_hyp_{if_generate_with_past_failed_hyp}_{which_exp[0]}_{which_exp[1]}_{which_exp[2]}.json"
            if not os.path.exists(cur_file) and if_generate_with_past_failed_hyp == 0:
                cur_file = root_dir + f"/final_analysis_f1_scores_results_{start_id}_{end_id}_gpt-4o-mini_gpt-4o-mini_{output_dir_postfix}_{if_multiple_llm_1}_2_beam_size_branching_2_1_4_gpt-4o-mini_num_compare_times_{num_compare_times}_maxScoreEachComponent.json" # removed "if_generate_with_past_failed_hyp"
        elif file_type == "pairwise_Q1":
            cur_file = root_dir + f"/final_analysis_pairwise_compare_results_{start_id}_{end_id}_gpt-4o-mini_gpt-4o-mini_{output_dir_postfix}_if_multiple_llm_{if_multiple_llm_1}_beam_size_branching_2_1_4_gpt-4o-mini_1_num_compare_times_{num_compare_times}_{aspect_type}.json"
            # Q: 
            if not os.path.exists(cur_file):
                output_dir_postfix = "updated_prompt_feb_14"
                cur_file = root_dir + f"/final_analysis_pairwise_compare_results_{start_id}_{end_id}_gpt-4o-mini_gpt-4o-mini_{output_dir_postfix}_if_multiple_llm_{if_multiple_llm_1}_beam_size_branching_2_1_4_gpt-4o-mini_1_num_compare_times_{num_compare_times}_{aspect_type}.json"
        elif file_type == "pairwise_Q3":
            cur_file = root_dir + f"/final_analysis_pairwise_compare_results_between_multiple_llm_{start_id}_{end_id}_{model_name_1}_{eval_model_name_1}_{if_multiple_llm_1}_{model_name_2}_{eval_model_name_2}_{if_multiple_llm_2}_{output_dir_postfix}_beam_size_branching_2_1_4_{pairwise_eval_model_name}_{pairwise_if_multiple_llm}_num_compare_times_{num_compare_times}_{aspect_type}.json"
            # Q: 
            if not os.path.exists(cur_file):
                output_dir_postfix = "updated_prompt_feb_14"
                cur_file = root_dir + f"/final_analysis_pairwise_compare_results_between_multiple_llm_{start_id}_{end_id}_{model_name_1}_{eval_model_name_1}_{if_multiple_llm_1}_{model_name_2}_{eval_model_name_2}_{if_multiple_llm_2}_{output_dir_postfix}_beam_size_branching_2_1_4_{pairwise_eval_model_name}_{pairwise_if_multiple_llm}_num_compare_times_{num_compare_times}_{aspect_type}.json"
        else:
            raise ValueError("file_type must be 'recall' or 'pairwise_Q1' or 'pairwise_Q3'")
        assert os.path.exists(cur_file), f"File {cur_file} does not exist"
        file_names.append(cur_file)
    return file_names




if __name__ == "__main__":

    ## Shared parameters
    # root_dir = "./Analysis_Results_old/"
    root_dir = "./Analysis_Results/"
    # display_type: 1: pairwise score (Q1), 2: recall score (Q2), 3: pairwise score (Q3)
    display_type = 2
    # updated_prompt_mar_29
    output_dir_postfix = "updated_prompt_mar_29"


    ## Pairwise score
    if display_type == 1:
        ## parameters
        num_compare_times = 5
        start_end_id_pairs = [[0, 4], [5, 13], [14, 23], [24, 35], [36, 50]]
        # start_end_id_pairs = [[0, 4]]
        # start_end_id_pairs = [[5, 13]]
        # start_end_id_pairs = [[14, 23]]
        # start_end_id_pairs = [[24, 35]]
        # start_end_id_pairs = [[36, 50]]

        for cur_metric in PAIRWISE_METRICS:
            file_names = synthesize_file_names_from_start_end_id_for_recall_files(start_end_id_pairs, num_compare_times=num_compare_times, root_dir=root_dir, file_type="pairwise_Q1", aspect_type=cur_metric, output_dir_postfix=output_dir_postfix)
            cur_metric_all_pairwise_score = display_pairwise_score_multiple_files(file_names, file_type="pairwise_Q1")
            print("{}: \n{}".format(cur_metric, cur_metric_all_pairwise_score))
    # Recall score
    elif display_type == 2:
        ## parameters
        # which_num: 0: precision, 1: recall, 2: f1, 3: weighted_precision, 4: weighted_recall, 5: weighted_f1
        which_num = 1
        num_compare_times = 3
        start_end_id_pairs = [[0, 2], [3, 4], [5, 13], [14, 23], [24, 35], [36, 46], [47, 50]]
        # start_end_id_pairs = [[0, 50]]
        # start_end_id_pairs = [[0, 25], [26, 50]]
        ''' Table 2 parameters '''
        if_multiple_llm = 1
        if_generate_with_past_failed_hyp = 0
        which_exp = [1, 1, 1]  # [H5, H1, Greedy]; 0/1. e.G., [1, 0, 0] means only H5's recall score is calculated

        ''' Compare if_multiple_llm==0 and if_multiple_llm==1 parameters '''
        # if_multiple_llm = 0
        # which_exp = [1, 0, 0]  # [H5, H1, Greedy]; 0/1. e.G., [1, 0, 0] means only H5's recall score is calculated

        file_names = synthesize_file_names_from_start_end_id_for_recall_files(start_end_id_pairs, num_compare_times=num_compare_times, root_dir=root_dir, if_multiple_llm_1=if_multiple_llm, if_generate_with_past_failed_hyp=if_generate_with_past_failed_hyp, which_exp=which_exp, output_dir_postfix=output_dir_postfix)
        show_recall_score_multiple_files(file_names, which_exp, which_num=which_num)
    elif display_type == 3:
        ### parameters
        num_compare_times = 5
        ''' Table 3 parameters '''
        start_end_id_pairs = [[24, 35]]
        ''' Compare if_multiple_llm==0 and if_multiple_llm==1 parameters '''
        # start_end_id_pairs = [[5, 13], [14, 23], [36, 50]]
        # start_end_id_pairs = [[0, 4], [5, 13], [14, 23], [24, 35], [36, 50]]

        ''' Table 3 parameters '''
        ## ["gpt-4o-mini", 1]
        # ["gpt-4o-mini", "gpt-4o-mini", 1] v.s. ["gpt-4o-mini", "gpt-4o-mini", 2]
        # ["gpt-4o-mini", "gpt-4o-mini", 1] v.s. ["gpt-4o-mini", "gemini-1.5-flash-latest", 1]
        # ["gpt-4o-mini", "gemini-1.5-flash-latest", 1] v.s. ["gpt-4o-mini", "gpt-4o-mini", 2]
        ## ["gemini-1.5-flash-latest", 1]
        # ["gpt-4o-mini", "gemini-1.5-flash-latest", 1] v.s. ["gpt-4o-mini", "gpt-4o-mini", 1]
        # ["gpt-4o-mini", "gemini-1.5-flash-latest", 1] v.s. ["gpt-4o-mini", "gpt-4o-mini", 2]
        # ["gpt-4o-mini", "gpt-4o-mini", 1] v.s. ["gpt-4o-mini", "gpt-4o-mini", 2]
        
        ''' Compare if_multiple_llm==0 and if_multiple_llm==1 parameters '''
        ## ["gpt-4o-mini", 1]
        # ["gpt-4o-mini", "gpt-4o-mini", 0] v.s. ["gpt-4o-mini", "gpt-4o-mini", 1]

        pairwise_eval_model_name, pairwise_if_multiple_llm = ["gpt-4o-mini", 1]
        model_name_1, eval_model_name_1, if_multiple_llm_1 = ["gpt-4o-mini", "gpt-4o-mini", 1]
        model_name_2, eval_model_name_2, if_multiple_llm_2 = ["gpt-4o-mini", "gpt-4o-mini", 2]

        for cur_metric in PAIRWISE_METRICS:
            # file_names
            file_names = synthesize_file_names_from_start_end_id_for_recall_files(start_end_id_pairs, num_compare_times=num_compare_times, root_dir=root_dir, file_type="pairwise_Q3", aspect_type=cur_metric, model_name_1=model_name_1, eval_model_name_1=eval_model_name_1, if_multiple_llm_1=if_multiple_llm_1, model_name_2=model_name_2, eval_model_name_2=eval_model_name_2, if_multiple_llm_2=if_multiple_llm_2, pairwise_eval_model_name=pairwise_eval_model_name, pairwise_if_multiple_llm=pairwise_if_multiple_llm, output_dir_postfix=output_dir_postfix)
            # get the pairwise score
            cur_metric_all_pairwise_score = display_pairwise_score_multiple_files(file_names, file_type="pairwise_Q3")
            print("{}: \n{}".format(cur_metric, cur_metric_all_pairwise_score))