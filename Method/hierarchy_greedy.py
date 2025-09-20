import os, argparse, json, time, math, sys, re, builtins
import numpy as np
from openai import OpenAI, AzureOpenAI
import concurrent.futures
from google import genai
sys.stdout.reconfigure(encoding='utf-8')
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, instruction_prompts, llm_generation_while_loop, exchange_order_in_list
from Evaluation.pairwise_compare import PairwiseCompare
from Method.logging_utils import setup_logger
from Method.hierarchy_greedy_utils import HGNode, HGTree
from Method.hierarchy_greedy_utils import get_all_previous_hierarchy_hypothesis_prompt, find_the_best_hypothesis_among_list





class HierarchyGreedy(object):

    def __init__(self, args):
        self.args = args
        # Set API client
        # openai client
        if args.api_type == 0:
            self.client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        # azure client
        elif args.api_type == 1:
            self.client = AzureOpenAI(
                azure_endpoint = args.base_url, 
                api_key=args.api_key,  
                api_version="2024-06-01"
            )
        # google client
        elif args.api_type == 2:
            self.client = genai.Client(api_key=args.api_key)
        else:
            raise NotImplementedError
        # prepare pairwise comparison
        self.pairwise_compare = PairwiseCompare(args.api_type, args.eval_api_key, args.base_url, args.eval_model_name, if_multiple_llm=args.if_multiple_llm)
        # obtain groundtruth finegrained hypothesis and experiment from TOMATO-Chem2
        if self.args.if_use_custom_research_background_and_coarse_hyp == 0:
            self.bkg_q_list, self.dict_bkg2survey, self.dict_bkg2cg_hyp, self.dict_bkg2fg_hyp, self.dict_bkg2fg_exp, self.dict_bkg2note = load_chem_annotation(args.chem_annotation_path)
            # update dict_bkg2cg_hyp with the vague cg hypothesis
            if args.if_use_vague_cg_hyp_as_input == 1:
                assert os.path.exists(args.vague_cg_hyp_path)
                with open(args.vague_cg_hyp_path, "r") as f:
                    self.dict_bkg2cg_hyp = json.load(f)



    # Function: get fine-grained hypothesis for one research question WITH branching (beam size of hypothesis from one hierarchy)
    # Input
    #   cur_bkg_id: >= 0
    #   if_use_custom_research_background_and_coarse_hyp: 0 / 1
    # Output
    #   search_results_all_init: [search_results_init_0, search_results_init_1, ..., search_results_init_(num_init_for_EU), recombination_results_all_steps]
    #       search_results_init_0/1 / recombination_results_all_steps: [[hyp, reason], [hyp, reason], ...]
    def get_finegrained_hyp_for_one_research_question_Branching(self, cur_bkg_id):
        # basic input information
        if self.args.if_use_custom_research_background_and_coarse_hyp == 0:
            print("Loading data from TOMATA-Chem2 dataset...")
            research_question = self.bkg_q_list[cur_bkg_id]
            background_survey = self.dict_bkg2survey[research_question]
            input_cg_hyp = self.dict_bkg2cg_hyp[research_question]
        elif self.args.if_use_custom_research_background_and_coarse_hyp == 1:
            print("Loading data from custom research background...")
            # use the custom research background and coarse-grained hypothesis
            # custom_data: [[research_question, background_survey, input_cg_hyp], ...]
            with open(self.args.custom_research_background_and_coarse_hyp_path, "r") as f:
                custom_data = json.load(f)
                research_question, background_survey, input_cg_hyp = custom_data[cur_bkg_id]
        else:
            raise ValueError("Invalid cur_bkg_id: ", cur_bkg_id)
        print("Initial coarse-grained hypothesis: ", input_cg_hyp)

        # initialize the tree
        hgtree = HGTree(hierarchy_id=-1, base_hyp_reason=[input_cg_hyp, "The initial preliminary hypothesis."], next_hierarchy_hyp=[[input_cg_hyp, "The initial preliminary hypothesis."]])

        ## iterate over the hierarchy
        for cur_hierarchy_id in range(self.args.num_hierarchy):
            print("Hierarchy ID: ", cur_hierarchy_id)        

            ## find topk hypothesis (beam size) to enter the next phase of searching
            if cur_hierarchy_id == 0:
                # the first hierarchy
                topk_hypothesis = []
                for cur_beam_id in range(self.args.beam_size_branching):
                    topk_hypothesis.append([input_cg_hyp, "The initial preliminary hypothesis.", hgtree.root])
                # topk_hypothesis_last_success_time = topk_hypothesis
            else:
                # not the first hierarchy: collect the hypothesis candidates from the previous hierarchy and select top k
                # topk_hypothesis: [[hyp, reason, node], [hyp, reason, node], ...]
                if_success_find_topk, topk_hypothesis = hgtree.find_the_top_k_hypothesis_to_enter_a_hierarchy_and_set_next_hierarchy_hyp_to_nodes(research_question, background_survey, self.args.beam_size_branching, cur_hierarchy_id, self.pairwise_compare, compare_mode=self.args.beam_compare_mode)
                assert if_success_find_topk == True
                # assert isinstance(if_success_find_topk, bool)
                # if if_success_find_topk:
                #     # topk_hypothesis_last_success_time: used to continue the search in the current hierarchy if no better hypothesis can be found in the previous hierarchy (topk_hypothesis_last_success_time might from the previous previous hierarchy); update it only when the previous hierarchy has successfully found topk hypothesis
                #     topk_hypothesis_last_success_time = topk_hypothesis
                # else:
                #     print("No better hypothesis can be searched in the previous hierarchy, so we continue from the best hypothesis in the last hierarchy that has successfully found topk hypothesis.")
                #     topk_hypothesis = topk_hypothesis_last_success_time

            ## phase of searching from k hypothesis (in k beams) in one hierarchy: iterate over the topk hypothesis
            # cur_hyp_collection_in_topk: [hyp, reason, node]
            # Parallelize the loop using ThreadPoolExecutor
            if self.args.if_parallel == 1:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            self.process_each_branch,
                            cur_beam_id,
                            cur_hyp_collection_in_topk,
                            cur_hierarchy_id,
                            hgtree,
                            background_survey,
                            input_cg_hyp,
                            research_question
                        )
                        for cur_beam_id, cur_hyp_collection_in_topk in enumerate(topk_hypothesis)
                    ]
                    ## previous code, might not stop when KeyboardInterrupt
                    # for future in concurrent.futures.as_completed(futures):
                    #     try:
                    #         future.result()
                    #     except Exception as e:
                    #         print(f"Error in parallel task: {e}")
                    while futures:
                        try:
                            # 使用短超时检查完成的任务
                            done_futures = []
                            for future in list(futures):
                                try:
                                    future.result(timeout=2.0)
                                    done_futures.append(future)
                                except concurrent.futures.TimeoutError:
                                    continue
                                except Exception as e:
                                    print(f"Error in parallel task: {e}")
                                    done_futures.append(future)
                            
                            # 移除已完成的任务
                            for future in done_futures:
                                futures.remove(future)
                                
                            if not done_futures:
                                time.sleep(0.5)  # 短暂休息，允许KeyboardInterrupt
                                
                        except KeyboardInterrupt:
                            print("检测到中断信号，取消剩余任务...")
                            for future in futures:
                                future.cancel()
                            raise
            else:
                for cur_beam_id, cur_hyp_collection_in_topk in enumerate(topk_hypothesis):
                    self.process_each_branch(cur_beam_id, cur_hyp_collection_in_topk, cur_hierarchy_id, hgtree, background_survey, input_cg_hyp, research_question)

        ## save the search tree results after finishing all hierarchies (since if the file already exists, this code will skip this bkg_id: the file is there only when the search is completed)
        if self.args.if_save == 1:
            hgtree.save(self.args.output_dir)

    
    # cur_hyp_collection_in_topk: [hyp, reason, node]
    # The original non-parallelized loop:
    #   for cur_beam_id, cur_hyp_collection_in_topk in enumerate(topk_hypothesis):
    # Output: no return value (but directly update the search tree)
    def process_each_branch(self, cur_beam_id, cur_hyp_collection_in_topk, cur_hierarchy_id, hgtree, background_survey, input_cg_hyp, research_question):
        print("\tBranch ID: ", cur_beam_id)
        print("\tDeveloping based on the hypothesis: ", cur_hyp_collection_in_topk[0])
        ## initialize cur_search_node
        cur_search_node = HGNode(hierarchy_id=cur_hierarchy_id, base_hyp_reason=cur_hyp_collection_in_topk[:2])
        ## get cur_survey_with_additional_info, which contains the corresponding best hypothesis from all the previous hierarchies of this top; we need it to enter the hypothesis search phase
        if cur_hierarchy_id == 0:
            # get parent node
            parent_node_for_cur_search_node = hgtree.root
            # add cur_search_node to the parent node (mutual update)
            parent_node_for_cur_search_node.add_child(cur_search_node)
            cur_search_node.set_parent(parent_node_for_cur_search_node)
            # initialize input information
            corresponding_best_hyp_from_previous_hierarchies = None
            prev_hierarchy_gene_fg_hyp = None
        else:
            # get parent node; parent_node_for_cur_search_node: the parent node of cur_search_node
            parent_node_for_cur_search_node = cur_hyp_collection_in_topk[2]
            # add cur_search_node to the parent node (mutual update)
            parent_node_for_cur_search_node.add_child(cur_search_node)
            cur_search_node.set_parent(parent_node_for_cur_search_node)
            # initialize input information
            # corresponding_best_hyp_from_previous_hierarchies: [best_hyp_from_hierarchy_0, best_hyp_from_hierarchy_1, ...]; best_hyp_from_hierarchy_0/1: str
            corresponding_best_hyp_from_previous_hierarchies = cur_search_node.find_best_hyp_in_all_previous_hierarchies()
            # prev_hierarchy_gene_fg_hyp: best hypothesis from the previous hierarchy
            prev_hierarchy_gene_fg_hyp = corresponding_best_hyp_from_previous_hierarchies[-1]
        cur_survey_with_additional_info = get_all_previous_hierarchy_hypothesis_prompt(background_survey, input_cg_hyp, corresponding_best_hyp_from_previous_hierarchies, cur_hierarchy_id)
        ## searching of one hypothesis over one hierarchy
        search_results_all_init = self.search_over_one_hierarchy(research_question, cur_survey_with_additional_info, input_cg_hyp, cur_hierarchy_id, cur_beam_id, prev_hierarchy_gene_fg_hyp)
        ## integrate results from one hierarchy into the final results
        if len(search_results_all_init) > 0:
            # update cur_search_node
            cur_search_node.replace_full_generated_hyp(search_results_all_init)
            print(f"\tThe best hyp of this branch: {search_results_all_init[-1][-1][0]}\n\tReasoning process of this best hyp: {search_results_all_init[-1][-1][1]}")
        else:
            assert cur_hierarchy_id > 0
            print("INFO: No better hypothesis can be searched in this branch than the best hypothesis from its corresponding branch in the previous hierarchy.")




    # Function: given inputs, try EU with args.num_init_for_EU initial points, iterative each initial point until local minimum or max_search_step
    # Output
    #   search_results_all_init: [search_results_init_0, search_results_init_1, ..., search_results_init_(num_init_for_EU), recombination_results_all_steps]
    #       search_results_init_x / recombination_results_all_steps: [[hyp, reason], [hyp, reason], ...]
    def search_over_one_hierarchy(self, cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_branch_id, prev_hierarchy_gene_fg_hyp):
        ## try with |args.num_init_for_EU| initial points, to search for |args.max_search_step| local minimums
        # search_results_all_init: [cur_init_search_results, ..., recombination_results_all_steps]
        #   cur_init_search_results / recombination_results_all_steps: [[hyp, reason], [hyp, reason], ...]
        search_results_all_init = []        

        ## Parallelize the above loop using ThreadPoolExecutor
        if self.args.if_parallel == 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ProcessPoolExecutor if it's CPU-bound
                # Submit tasks for each initialization ID
                futures = [
                    executor.submit(
                        self.search_along_one_search_line,
                        cur_q,
                        cur_survey,
                        cur_cg_hyp,
                        cur_hierarchy_id,
                        cur_branch_id,
                        cur_init_id,
                        prev_hierarchy_gene_fg_hyp,
                        'normal_search'  # Pass 'prompt_type' directly
                    )
                    for cur_init_id in range(self.args.num_init_for_EU)
                ]
                ## previous code, might not stop when KeyboardInterrupt
                # # Collect results as they complete
                # for future in concurrent.futures.as_completed(futures):
                #     try:
                #         cur_init_search_results = future.result()
                #         if len(cur_init_search_results) > 0:
                #             search_results_all_init.append(cur_init_search_results)
                #     except Exception as e:
                #         print(f"Error in processing: {e}")
                while futures:
                    try:
                        done_futures = []
                        for future in list(futures):
                            try:
                                cur_init_search_results = future.result(timeout=2.0)
                                if len(cur_init_search_results) > 0:
                                    search_results_all_init.append(cur_init_search_results)
                                done_futures.append(future)
                            except concurrent.futures.TimeoutError:
                                continue
                            except Exception as e:
                                print(f"Error in processing: {e}")
                                done_futures.append(future)
                        
                        for future in done_futures:
                            futures.remove(future)
                            
                        if not done_futures:
                            time.sleep(0.5)
                            
                    except KeyboardInterrupt:
                        print("检测到中断信号，取消剩余任务...")
                        for future in futures:
                            future.cancel()
                        raise
        else:
            for cur_init_id in range(self.args.num_init_for_EU):
                print("\t\tInitial point ID: ", cur_init_id)
                # cur_init_search_results: [[hyp, reason], [hyp, reason], ...]
                cur_init_search_results = self.search_along_one_search_line(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_branch_id, cur_init_id, prev_hierarchy_gene_fg_hyp, prompt_type='normal_search')
                if len(cur_init_search_results) > 0:
                    search_results_all_init.append(cur_init_search_results)

        ## final_results
        if len(search_results_all_init) > 0:
            print("\tRecombination from local minimums...")
            # try num_recom_trial_for_better_hyp times to find better hypothesis than the local minimums
            if_success_recombination = False
            for cur_try_id in range(self.args.num_recom_trial_for_better_hyp):
                print("\tRecombination trial ID: ", cur_try_id)
                recombination_results_all_steps, best_local_minimum_hyp, best_local_minimum_id = self.recombination_from_local_minimums(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, search_results_all_init, cur_branch_id)
                if len(recombination_results_all_steps) > 0:
                    # recombinations can lead to better hypothesis
                    print("INFO: found better hypothesis through recombination.")
                    # reorder the search results from several initial points (the best local minimum hypothesis is the last one)
                    cur_best_local_minimum_hyp = search_results_all_init.pop(best_local_minimum_id)
                    search_results_all_init.append(cur_best_local_minimum_hyp)
                    # append the recombination results
                    search_results_all_init.append(recombination_results_all_steps)
                    if_success_recombination = True
                    break
                else:
                    # if the recombination can't lead to better hypothesis, then the best hypothesis is the best local minimum hypothesis
                    print("INFO: can't find better hypothesis through recombination in trial {}.".format(cur_try_id))
            if if_success_recombination == False:
                print("INFO: no better hypothesis found through recombination, use the best local minimum hypothesis.")
                assert isinstance(best_local_minimum_hyp, str)
                search_results_all_init.append([[best_local_minimum_hyp, "The best local minimum hypothesis."]])
        else:
            # not a single local minimum can be found in this hierarchy (the potential local minimum are all worse than the hypothesis from the previous hierarchy)
            print("INFO: no better searched hypothesis for this hierarchy than the previous hierarchy.")
        
        return search_results_all_init
    


    # Input:
    #    search_results_all_init: [cur_init_search_results_0, cur_init_search_results_1, ..., search_results_init_(num_init_for_EU)]
    #       cur_init_search_results: [[hyp, reason], [hyp, reason], ...]
    # Output:
    #    recombination_results_all_steps: [[hyp, reason], [hyp, reason], ...]
    #    best_local_minimum_hyp: str
    def recombination_from_local_minimums(self, cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, search_results_all_init, cur_branch_id):
        ## recombination between the local minimums of each initial point
        # the local minimums of each initial point: local_minimums: [[hyp, reason], [hyp, reason], ...]
        local_minimums = [cur_init_search_results[-1] for cur_init_search_results in search_results_all_init]
        # rank the local minimums
        hierarchy_level = cur_hierarchy_id if self.args.num_hierarchy > 1 else None
        # use 5 num_compare_times to ensure the best hypothesis is selected
        best_local_minimum_id = find_the_best_hypothesis_among_list(cur_q, cur_survey, local_minimums, self.pairwise_compare, hierarchy_level=hierarchy_level, num_compare_times=5)
        assert isinstance(local_minimums[best_local_minimum_id], list), f"Expected list, got: {type(local_minimums[best_local_minimum_id])}, value: {local_minimums[best_local_minimum_id]}"
        best_local_minimum_hyp = local_minimums[best_local_minimum_id][0]
        # turn local_minimums into prompt
        local_minimums_prompt = "\nNext research hypothesis candidate: ".join([local_minimum[0] for local_minimum in local_minimums])
        # only recombine when there are at least two local minimums
        if len(local_minimums) >= 2:
            # recombination
            # recombination_results_all_steps: [[hyp, reason], [hyp, reason], ...]
            # merge local_minimums_prompt with cur_survey
            cur_survey = cur_survey + "\n\nNext we introducte the research hypothesis candidates where we can leverage their advantages and avoid their disadvantages to form a new hypothesis: " + local_minimums_prompt
            recombination_results_all_steps = self.search_along_one_search_line(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_branch_id, cur_init_id=0, prev_hierarchy_gene_fg_hyp=best_local_minimum_hyp, prompt_type='recombination_search')
            return recombination_results_all_steps, best_local_minimum_hyp, best_local_minimum_id
        elif len(local_minimums) == 1:
            assert best_local_minimum_id == 0
            # "The only local minimum hypothesis found."
            recombination_results_all_steps = [[local_minimums[0][0], "Only one hypothesis is found in this branch."]]
            return recombination_results_all_steps, best_local_minimum_hyp, best_local_minimum_id
        else:
            raise ValueError("Invalid number of local minimums: ", local_minimums)
    


    # Input
    #   cur_q / cur_survey / cur_cg_hyp: text
    #   prev_hierarchy_gene_fg_hyp: None / text
    #   prompt_type: 'normal_search' or 'recombination_search'
    # Output
    #   cur_init_search_results: [[hyp, reason], [hyp, reason], ...]
    def search_along_one_search_line(self, cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_branch_id, cur_init_id, prev_hierarchy_gene_fg_hyp, prompt_type):
        assert prompt_type == 'normal_search' or prompt_type == 'recombination_search'
        if prompt_type == 'normal_search':
            print("\t\tInitial point ID: ", cur_init_id)
        cur_init_search_results = []
        prev_step_gene_fg_hyp = None
        for cur_search_step_id in range(self.args.max_search_step):
            # print("\t\t\tSearch step ID: ", cur_search_step_id)
            print(f"cur_hierarchy_id: {cur_hierarchy_id}, cur_branch_id: {cur_branch_id}, cur_init_id: {cur_init_id}, cur_search_step_id: {cur_search_step_id}, prompt_type: {prompt_type}")
            structured_gene, selection_reason, if_continue_search = self.one_step_greedy_search_strictly_better_than_previous(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_init_id, prev_hierarchy_gene_fg_hyp, prev_step_gene_fg_hyp, cur_search_step_id, self.args.locam_minimum_threshold, prompt_type=prompt_type)
            # local minimum detection
            if if_continue_search == False:
                print("INFO: Early stopping: this search might have already reached a local minimum.")
                break
            # save the results of this search step
            cur_init_search_results.append(structured_gene)
            assert structured_gene is not None
            prev_step_gene_fg_hyp = structured_gene[0]
            # print("current hyp: ", prev_step_gene_fg_hyp)
        return cur_init_search_results
    

    # Input
    # cur_hierarchy_id: 0/1/2/3/4
    # Output
    # structured_gene: [hypothesis, reason]
    # selection_reason: [selection (1/2), reason]
    # if_continue_search: early stopping: this search might have already reached a local minimum; when if_continue_search == False, the returned structured_gene should be overlooked (since it is not better than the previous one)
    def one_step_greedy_search_strictly_better_than_previous(self, cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_init_id, prev_hierarchy_gene_fg_hyp, this_hierarchy_prev_step_gene_fg_hyp, cur_search_step_id, locam_minimum_threshold, prompt_type):
        ## determine the prev step pf hyp to compare
        if cur_search_step_id == 0:
            assert this_hierarchy_prev_step_gene_fg_hyp is None
            prev_hyp_to_compare = prev_hierarchy_gene_fg_hyp
        else:
            assert this_hierarchy_prev_step_gene_fg_hyp is not None
            prev_hyp_to_compare = this_hierarchy_prev_step_gene_fg_hyp
        ## start search
        if_better=False
        cnt_search_single_step=0
        if_continue_search=True
        # past_failed_hyp: [[hypothesis, reason], ...]
        if self.args.if_generate_with_past_failed_hyp == 1:
            past_failed_hyp = []
            # back up the original survey to advoid add the same past failed hypothesis to the survey multiple times
            cur_survey_ori = cur_survey
        while not if_better:
            # add past failed hypothesis to the survey to mimic in-context RL
            if self.args.if_generate_with_past_failed_hyp == 1:
                if len(past_failed_hyp) > 0:
                    prompt_past_failed_hyp = ""
                    for i in range(len(past_failed_hyp)):
                        prompt_past_failed_hyp += "The {}th previous hypothesis that is not better than the base hypothesis is: ".format(i+1) + past_failed_hyp[i][0] + "\n" + "The reason is: " + past_failed_hyp[i][1] + "\n"
                    prompt_past_failed_hyp += "\nBelow are some previous updated hypotheses that are not better than the base hypothesis and the corresponding reasons (The reason might mention Research hypothesis candidate 1 and Research hypothesis candidate 2. Out of them, Research hypothesis candidate 1 is the base hypothesis, and Research hypothesis candidate 2 is the not better updated hypothesis), you may be aware of them: " + prompt_past_failed_hyp
                    cur_survey = cur_survey_ori + prompt_past_failed_hyp
                    print("Added past failed hypothesis to the survey to mimic in-context RL")
            # structured_gene: [hypothesis, reason]
            if self.args.if_feedback == 1:
                structured_gene = self.one_step_greedy_search_with_feedback(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_init_id, prev_hierarchy_gene_fg_hyp, this_hierarchy_prev_step_gene_fg_hyp, prompt_type)
            else:
                structured_gene = self.one_step_greedy_search(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_init_id, prev_hierarchy_gene_fg_hyp, this_hierarchy_prev_step_gene_fg_hyp, prompt_type)
            print("\t\t\t\tcurrent hyp: ", structured_gene[0])
            # selection_reason: [selection (1/2), reason]
            # when this_hierarchy_prev_step_gene_fg_hyp == None, the previous fine-grained hypothesis is not generated, thus use the coarse-grained hypothesis for comparison
            # cur_hierarchy_level is not None only when num_hierarchy > 1
            cur_hierarchy_level = cur_hierarchy_id if self.args.num_hierarchy > 1 else None
            selection_reason = self.pairwise_compare.compare(cur_q, prev_hyp_to_compare, structured_gene[0], instruction_mode="strict_to_hyp2", hierarchy_level=cur_hierarchy_level)
            if selection_reason[0] == 2:
                # the new hypothesis is better
                if_better=True
                cnt_search_single_step=0
            else:
                print("\t\t\t\tThe new hypothesis is not better than the previous one, try again... \n")
                # print("The new hypothesis is not better than the previous one, try again... \nReason: {}".format(selection_reason[1]))
                if self.args.if_generate_with_past_failed_hyp == 1:
                    past_failed_hyp.append([structured_gene[0], selection_reason[1]])
                    print("Collected past failed hypothesis to mimic in-context RL")
            cnt_search_single_step+=1
            if if_better == False and cnt_search_single_step >= locam_minimum_threshold:
                if_continue_search=False
                break
        return structured_gene, selection_reason, if_continue_search
    


    # Output
    #   structured_gene: [updated hypothesis, reason]
    def one_step_greedy_search_with_feedback(self, cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_init_id, prev_hierarchy_gene_fg_hyp, this_hierarchy_prev_step_gene_fg_hyp, prompt_type):
        ## one step greedy search (find one d)
        # structured_gene_original_optimization: [hypothesis, reason]
        structured_gene_original_optimization = self.one_step_greedy_search(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_init_id, prev_hierarchy_gene_fg_hyp, this_hierarchy_prev_step_gene_fg_hyp, prompt_type)

        ## feedback and update
        # previous hypothsis
        if this_hierarchy_prev_step_gene_fg_hyp is None:
            if prev_hierarchy_gene_fg_hyp is None:
                assert isinstance(cur_cg_hyp, str)
                prev_hyp = cur_cg_hyp
            else:
                assert isinstance(prev_hierarchy_gene_fg_hyp, str)
                prev_hyp = prev_hierarchy_gene_fg_hyp
        else:
            assert isinstance(this_hierarchy_prev_step_gene_fg_hyp, str)
            prev_hyp = this_hierarchy_prev_step_gene_fg_hyp

        # structured_gene_updated_hyp: [updated hypothesis, reason]
        structured_gene_updated_hyp = self.feedback_to_hyp_and_update_hyp(cur_q, cur_survey, cur_hierarchy_id, prev_hyp, structured_gene_original_optimization[0], structured_gene_original_optimization[1])
        return structured_gene_updated_hyp

    
    # Output
    #   structured_gene: [updated hypothesis, reason]
    def feedback_to_hyp_and_update_hyp(self, cur_q, cur_survey, cur_hierarchy_id, prev_hyp, cur_hyp, cur_hyp_reason):
        # print("\t\t\traw hyp: ", cur_hyp)
        ## feedback
        # prompts
        if self.args.num_hierarchy == 5:
            prompts = instruction_prompts("validity_clarity_feedback_to_hyp", assist_info=cur_hierarchy_id)
        elif self.args.num_hierarchy == 1:
            prompts = instruction_prompts("validity_clarity_feedback_to_hyp")
        else:
            raise ValueError("Invalid num_hierarchy: ", self.args.num_hierarchy)
        assert len(prompts) == 6
        full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + prev_hyp + prompts[3] + cur_hyp + prompts[4] + cur_hyp_reason + prompts[5]
        # generate finegrained hypothesis
        # feedback: str
        feedback = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=False, temperature=1.0, api_type=self.args.api_type)
        # print("\t\t\tfeedback: ", feedback)
        ## update hypothesis
        # prompts
        prompts = instruction_prompts("update_hyp_based_on_feedback")
        assert len(prompts) == 6
        full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + prev_hyp + prompts[3] + cur_hyp + prompts[4] + feedback + prompts[5]
        # generate finegrained hypothesis
        # structured_gene: [[hypothesis, reason]]
        structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Revised Hypothesis:'], temperature=1.0, api_type=self.args.api_type)
        structured_gene = exchange_order_in_list(structured_gene)
        assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
        # print("\t\t\trefined hyp from feedback: ", structured_gene[0][0])
        return structured_gene[0]
    


    # Function
    # get the search result of one step greedy search
    # cur_survey: str; it should include the coarse-grained hypothesis if cur_hierarchy_id == 0; it should include the coarse-grained hypothesis and the best hypothesis from the previous hierarchy if cur_hierarchy_id > 0; it should include the local minimum hypotheses if prompt_type == 'recombination_search'
    # Input
    # cur_q/cur_survey/cur_cg_hyp/prev_step_gene_fg_hyp: str
    # prompt_type: 'normal_search' or 'recombination_search'; 'normal_search': normal search step; 'recombination_search': recombination step
    # Output
    # structured_gene: [hypothesis, reason]
    def one_step_greedy_search(self, cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, cur_init_id, prev_hierarchy_gene_fg_hyp, this_hierarchy_prev_step_gene_fg_hyp, prompt_type):
        assert prompt_type == 'normal_search' or prompt_type == 'recombination_search'
        # initialize prev_hierarchy_gene_fg_hyp
        if prev_hierarchy_gene_fg_hyp == None:
            # if hypothesis from the previous hierarchy is not provided, then this is the first hierarchy, and we use the coarse-grained hypothesis as the hypothesis from the previous hierarchy
            assert cur_hierarchy_id == 0
            assert isinstance(cur_cg_hyp, str)
            prev_hierarchy_gene_fg_hyp = cur_cg_hyp
        # prompts
        if this_hierarchy_prev_step_gene_fg_hyp is None:
            # the first search step
            if prompt_type == 'normal_search':
                if self.args.num_hierarchy == 5:
                    prompts = instruction_prompts("hierarchy_greedy_search_five_hierarchy_first_step", assist_info=[cur_hierarchy_id, self.args.if_generate_with_example])
                elif self.args.num_hierarchy == 1:
                    prompts = instruction_prompts("greedy_search_first_step", assist_info=[cur_hierarchy_id, self.args.if_generate_with_example])
                else:
                    raise ValueError("Invalid num_hierarchy: ", self.args.num_hierarchy)
                assert len(prompts) == 3
                # full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + cur_cg_hyp + prompts[3] + prev_hierarchy_gene_fg_hyp + prompts[4]
                # full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + prev_hierarchy_gene_fg_hyp + prompts[3]
                full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] 
            elif prompt_type == 'recombination_search':
                prompts = instruction_prompts("recombination_first_step", assist_info=[cur_hierarchy_id])
                assert len(prompts) == 3
                full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2]
            else:
                raise ValueError("Invalid prompt type: ", prompt_type)
        else:
            # the following search steps (with previous search results)
            if prompt_type == 'normal_search':
                if self.args.num_hierarchy == 5:
                    prompts = instruction_prompts("hierarchy_greedy_search_five_hierarchy_following_step", assist_info=[cur_hierarchy_id, self.args.if_generate_with_example])
                elif self.args.num_hierarchy == 1:
                    prompts = instruction_prompts("greedy_search_following_step", assist_info=[cur_hierarchy_id, self.args.if_generate_with_example])
                else:
                    raise ValueError("Invalid num_hierarchy: ", self.args.num_hierarchy)
                assert len(prompts) == 4
                # full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + cur_cg_hyp + prompts[3] + prev_hierarchy_gene_fg_hyp + prompts[4] + this_hierarchy_prev_step_gene_fg_hyp + prompts[5]
                # full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + prev_hierarchy_gene_fg_hyp + prompts[3] + this_hierarchy_prev_step_gene_fg_hyp + prompts[4]
                full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + this_hierarchy_prev_step_gene_fg_hyp + prompts[3]
            elif prompt_type == 'recombination_search':
                prompts = instruction_prompts("recombination_following_step", assist_info=[cur_hierarchy_id])
                assert len(prompts) == 4
                full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + this_hierarchy_prev_step_gene_fg_hyp + prompts[3]
            else:
                raise ValueError("Invalid prompt type: ", prompt_type)
        # print("\t\t\tfull_prompt: ", full_prompt)
            

        # generate finegrained hypothesis
        # structured_gene: [[hypothesis, reason], ...]
        structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Revised Hypothesis:'], temperature=1.0, api_type=self.args.api_type)
        structured_gene = exchange_order_in_list(structured_gene)
        assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
        cur_hyp = structured_gene[0][0]
        # print("\t\t\tcurrent hyp: ", cur_hyp)
        return structured_gene[0]








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Greedy search for fine-grained hypothesis generation')
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="model name: gpt-4o/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--eval_model_name", type=str, default="gpt-4o", help="model name for evaluation: gpt-4o/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--api_type", type=int, default=0, help="0: openai's API toolkit; 1: azure's API toolkit; 2: Gemini")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--eval_api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="", help="base url for the API")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--bkg_id", type=int, default=0, help="background research question id; can be the id for both TOMATO-Chem2 and custom inputs; 0~N: use the N-th background research question in the chem_annotation_path or custom_research_background_and_coarse_hyp_path")
    parser.add_argument("--output_dir", type=str, default="./Checkpoints/hypothesis_evaluation_results.json")
    parser.add_argument("--if_save", type=int, default=0, help="whether save grouping results")
    parser.add_argument("--max_search_step", type=int, default=60, help="maximum search steps")
    parser.add_argument("--locam_minimum_threshold", type=int, default=5, help="local minimum threshold")
    parser.add_argument("--num_init_for_EU", type=int, default=5, help="how many initial points to start search with EU")
    parser.add_argument("--num_hierarchy", type=int, default=5, help="number of hierarchical levels to search")
    parser.add_argument("--if_feedback", type=int, default=0, help="for each fundamental search step, whether to provide feedback to the hypothesis and update the hypothesis with the feedback")
    parser.add_argument("--num_recom_trial_for_better_hyp", type=int, default=5, help="number of trials for recombination to find better hypothesis than the local minimums")
    parser.add_argument("--beam_size_branching", type=int, default=1, help="how many hypotheses from one hierarchy will enter the next hierarchy")
    parser.add_argument("--beam_compare_mode", type=int, default=0, help="0: directly select the recombination hypotheses in the previous hierarchy to enter the next hierarchy; 1: select topk of [recombination hypotheses, best local minimum hypotheses] to enter the next hierarchy")
    parser.add_argument("--if_parallel", type=int, default=1, help="whether to use parallel computing")
    parser.add_argument("--if_multiple_llm", type=int, default=0, help="whether to use multiple llms for hypothesis gradient estimation. 0: single llm; 1: multiple same llms; 2: multiple different llms")
    parser.add_argument("--if_use_vague_cg_hyp_as_input", type=int, default=0, help="whether to use processed vague coarse-grained hypothesis as input (by Data_Processing/input_hyp_processing.py)")
    parser.add_argument("--vague_cg_hyp_path", type=str, default="./Data/processed_research_direction.json", help="store processed vague coarse-grained hypothesis")
    parser.add_argument("--if_generate_with_example", type=int, default=1, help="during optimization, whether to use hypothesis example in the prompt to generate hypothesis in each step")
    parser.add_argument("--if_generate_with_past_failed_hyp", type=int, default=0, help="during optimization, whether to use past failed hypothesis in the prompt to generate hypothesis in each step")
    parser.add_argument("--if_use_custom_research_background_and_coarse_hyp", type=int, default=0, help="whether to use custom research question & background survey & coarse-grained hypothesis; 0: use the background research question in the chem_annotation_path; 1: use custom research question in custom_research_background_and_coarse_hyp_path")
    parser.add_argument("--custom_research_background_and_coarse_hyp_path", type=str, default="./custom_research_background_and_coarse_hyp.json", help="if bkg_id == -1, then use this path to load custom research question and background survey and coarse-grained hypothesis; in a format of json file, with keys: 'research_question', 'background_survey', 'coarse_grained_hypothesis'; if bkg_id != -1, then this path will be ignored")
    args = parser.parse_args()

    assert args.if_save in [0, 1]
    assert args.num_init_for_EU >= 1
    assert args.num_hierarchy in [1, 5]
    assert args.if_feedback in [0, 1]
    assert args.beam_compare_mode in [0, 1]
    assert args.if_parallel in [0, 1]
    assert args.if_multiple_llm in [0, 1, 2]
    assert args.if_use_vague_cg_hyp_as_input in [0, 1]
    # we need if_generate_with_example to be 1
    assert args.if_generate_with_example in [1]
    assert args.if_generate_with_past_failed_hyp in [0, 1]
    assert args.if_use_custom_research_background_and_coarse_hyp in [0, 1]

    ## Setup logger
    logger = setup_logger(args.output_dir)
    # Redirect print to logger
    def custom_print(*args, **kwargs):
        message = " ".join(map(str, args))
        logger.info(message)
    # global print
    # print = custom_print
    builtins.print = custom_print
    print(args)

    ## if use custom research question & background survey & coarse-grained hypothesis   
    if args.if_use_custom_research_background_and_coarse_hyp == 0:
        assert args.chem_annotation_path != ""
        assert os.path.exists(args.chem_annotation_path)
    elif args.if_use_custom_research_background_and_coarse_hyp == 1:
        assert args.custom_research_background_and_coarse_hyp_path != ""
        assert os.path.exists(args.custom_research_background_and_coarse_hyp_path)
    else:
        raise ValueError("Invalid if_use_custom_research_background_and_coarse_hyp: ", args.if_use_custom_research_background_and_coarse_hyp)

    ## start running the framework
    if os.path.exists(args.output_dir):
        print("Warning: {} already exists.".format(args.output_dir))
    else:
        ## start hierarchical greedy search
        print(f"Start hierarchical greedy search... num_hierarchy: {args.num_hierarchy}")
        start_time = time.time()
        greedy = HierarchyGreedy(args)
        # get fine-grained hypothesis for one research question
        greedy.get_finegrained_hyp_for_one_research_question_Branching(args.bkg_id)
        duration = (time.time() - start_time) / 60
        print("Finished hierarchical greedy search. Duration: {:.2f} minutes".format(duration))
