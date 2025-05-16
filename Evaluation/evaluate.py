import os, argparse, json, time, copy, math, sys, re
import numpy as np
from openai import OpenAI, AzureOpenAI
import concurrent.futures
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, evaluation_instruction_prompts, llm_generation_while_loop

class Evaluator(object):

    def __init__(self, model_name, api_type, api_key, chem_annotation_path, preprocess_groundtruth_components_dir=None):
        # assign values
        self.model_name = model_name
        self.api_type = api_type
        self.api_key = api_key
        # set OpenAI API key
        if self.api_type == 0:
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.claudeshop.top/v1")
        else:
            raise NotImplementedError
        # obtain groundtruth finegrained hypothesis and experiment
        self.bkg_q_list, self.dict_bkg2survey, self.dict_bkg2cg_hyp, self.dict_bkg2fg_hyp, self.dict_bkg2fg_exp, self.dict_bkg2note = load_chem_annotation(chem_annotation_path)   
        # load from preprocess_groundtruth_components_dir
        if isinstance(preprocess_groundtruth_components_dir, str) and os.path.exists(preprocess_groundtruth_components_dir):
            with open(preprocess_groundtruth_components_dir, "r") as f:
                # self.groundtruth_hyp_components_collection = [[groundtruth_hyp_components0], [groundtruth_hyp_components1], ...]
                self.groundtruth_hyp_components_collection = json.load(f)
        else:
            self.groundtruth_hyp_components_collection = None
                    
    

    # Output:
    #   average_compare_results: [precision, recall, f1, weighted_precision, weighted_recall, weighted_f1]
    def check_one_generated_hyp_or_exp(self, cur_bkg_id, gene_hyp, type="hyp", num_compare_times=3):
        assert type in ["hyp", "exp"]
        assert isinstance(gene_hyp, str)
        print("Checking hypothesis for background id: ", cur_bkg_id)
        # obtain groundtruth finegrained hypothesis (cur_bkg_fg_hyp) and groundtruth finegrained experiment (cur_bkg_fg_exp)
        cur_bkg_q = self.bkg_q_list[cur_bkg_id]
        if type == "hyp":
            groundtruth_hyp = self.dict_bkg2fg_hyp[cur_bkg_q]
        elif type == "exp":
            groundtruth_hyp = self.dict_bkg2fg_exp[cur_bkg_q]
        else:
            raise Exception("Invalid type: ", type)
        # break groundtruth_hyp and gene_hyp into components
        if self.groundtruth_hyp_components_collection is not None and len(self.groundtruth_hyp_components_collection) > cur_bkg_id:
            groundtruth_hyp_components = self.groundtruth_hyp_components_collection[cur_bkg_id]
        else:
            groundtruth_hyp_components = self.break_finegrained_hyp_or_exp(groundtruth_hyp, type)
        gene_hyp_components = self.break_finegrained_hyp_or_exp(gene_hyp, type)
        # compare
        num_max_trial = 10
        for cur_trial in range(num_max_trial):
            ## not parallel version
            # compare_results_collection = []
            # for cur_compare_id in range(num_compare_times):
            #     # cur_compare_results: [precision, recall, f1, weighted_precision, weighted_recall, weighted_f1]
            #     cur_compare_results = self.calculate_precision_recall_f1(groundtruth_hyp_components, gene_hyp_components, type)
            #     compare_results_collection.append(cur_compare_results)
            ## parallel version
            with concurrent.futures.ThreadPoolExecutor() as executor:
                compare_results_collection = list(executor.map(
                    lambda _: self.calculate_precision_recall_f1(groundtruth_hyp_components, gene_hyp_components, type), 
                    range(num_compare_times)
                ))
            # calculate std of weighted_f1 to check whether there is any outlier in the comparison results
            weighted_f1_collection = [x[5] for x in compare_results_collection]
            std_weighted_f1 = np.std(weighted_f1_collection)
            if std_weighted_f1 <= 0.08:
                break
            else:
                print("Warning: large std for weighted_f1, might have outliers, retrying...; std_weighted_f1: {}; weighted_f1_collection: {}".format(std_weighted_f1, weighted_f1_collection))
        # calculate average results
        average_compare_results = np.mean(compare_results_collection, axis=0)
        print("compare_results_collection: ", compare_results_collection)
        print("average_compare_results: ", average_compare_results)
        return average_compare_results
    


    # break finegrained hypothesis into components
    # finegrained_hyp: text
    # splitted_components: [component0, component1, ...]
    def break_finegrained_hyp_or_exp(self, finegrained_hyp, type="hyp"):
        assert type in ["hyp", "exp"]
        # prompts
        first_prompts = evaluation_instruction_prompts('break_finegrained_hyp_or_exp', assist_info=type)
        refine_prompts = evaluation_instruction_prompts('break_finegrained_hyp_or_exp_refine', assist_info=type)
        assert len(first_prompts) == 2
        assert len(refine_prompts) == 3
        # first step
        full_prompt_first = first_prompts[0] + finegrained_hyp + first_prompts[1]
        components_try_1 = llm_generation_while_loop(full_prompt_first, self.model_name, self.client, if_structured_generation=False, temperature=0.0, api_type=self.api_type)
        # Make sure the components are successfully splitted
        if_successful_split = False
        while not if_successful_split:
            try:
                # refine the components
                full_prompt_second = refine_prompts[0] + finegrained_hyp + refine_prompts[1] + components_try_1 + refine_prompts[2]
                # structured_components: [[component0, reasoning_process0], [component1, reasoning_process1], ...]
                structured_components = llm_generation_while_loop(full_prompt_second, self.model_name, self.client, if_structured_generation=True, template=['Id of the component:', 'Component:'], temperature=0.0, api_type=self.api_type, restructure_output_model_name=self.model_name)
                # clean structured_components
                splitted_components = [cur_d[1] for cur_d in structured_components]
                # reasoning_process usually contain the details of the component, while the component itself usually is only a brief name
                # splitted_components = [cur_d[0] + '. ' + cur_d[1] for cur_d in structured_components]
                if_successful_split = True
            except:
                print("Unsuccessful splitting, retrying...")
                continue
        return splitted_components



    # groundtruth_hyp_components/gene_hyp_components: output of break_finegrained_hyp_or_exp()
    # groundtruth_hyp_components/gene_hyp_components: [component0, component1, ...]
    def calculate_precision_recall_f1(self, groundtruth_hyp_components, gene_hyp_components, type="hyp"):
        assert type in ["hyp", "exp"]
        len_groundtruth, len_gene = len(groundtruth_hyp_components), len(gene_hyp_components)
        def get_text_components_from_list(components):
            text_components = ["Component id " + str(cur_id) + ": " + cur_component for cur_id, cur_component in enumerate(components)]
            text_components = "\t".join(text_components)
            return text_components
        text_groundtruth_hyp_components = get_text_components_from_list(groundtruth_hyp_components)
        text_gene_hyp_components = get_text_components_from_list(gene_hyp_components)
        # function: input groundtruth and gene components, output precision, recall, and f1 (two modes: if_groundtruth_oriented=True/False)
        def get_f1_from_components(if_groundtruth_oriented=True):
            assert if_groundtruth_oriented in [True, False]
            # prompts
            first_prompts = evaluation_instruction_prompts('compare_components_from_gt_and_gene', assist_info=[type, if_groundtruth_oriented])
            refine_prompts = evaluation_instruction_prompts('compare_components_from_gt_and_gene_refine', assist_info=[type, if_groundtruth_oriented])
            assert len(first_prompts) == 3
            assert len(refine_prompts) == 4
            # first step to compare
            full_prompt_first = first_prompts[0] + text_groundtruth_hyp_components + first_prompts[1] + text_gene_hyp_components + first_prompts[2]
            compare_try_1 = llm_generation_while_loop(full_prompt_first, self.model_name, self.client, if_structured_generation=False, temperature=0.0, api_type=self.api_type)
            # refine the comparison
            full_prompt_second = refine_prompts[0] + text_groundtruth_hyp_components + refine_prompts[1] + text_gene_hyp_components + refine_prompts[2] + compare_try_1 + refine_prompts[3]
            # structured_compare: [[component0, level0], [component1, level1], ...]
            structured_compare = llm_generation_while_loop(full_prompt_second, self.model_name, self.client, if_structured_generation=True, template=['Covered component:', 'Covered level:'], temperature=0.0, api_type=self.api_type, restructure_output_model_name=self.model_name)
            # print("structured_compare: ", structured_compare)
            # clean structured_compare
            structured_compare_cleaned = []
            for cur_id, cur_d in enumerate(structured_compare):
                cur_d = [x.strip('-').strip() for x in cur_d]
                # check the extracted hypothesis
                if ("none" in cur_d[0].lower() or "not covered" in cur_d[0].lower()) and len(cur_d[0]) < 10:
                    continue
                # extract and check the extracted evaluation level
                idx_all_lvl = []
                for cur_lvl_id in range(0, 4):
                    # find the first index of each level (1~4), and use the smallest one as the level
                    idx = cur_d[1].find(str(cur_lvl_id))
                    if idx != -1:
                        idx_all_lvl.append(idx)
                    else:
                        idx_all_lvl.append(999)
                if min(idx_all_lvl) == 999:
                    cur_d_lvl = 0
                    print("Warning: no level found, set to 0: ", cur_d[1])
                else:
                    # cur_d_lvl: [0, 1, 2, 3]
                    cur_d_lvl = str(np.argmin(idx_all_lvl))
                structured_compare_cleaned.append([cur_d[0], cur_d_lvl])
            # print("\n\ncompare_try_1: ", compare_try_1)
            # print("\n\nstructured_compare: ", structured_compare)
            print("\n\nstructured_compare_cleaned: ", structured_compare_cleaned)
            return structured_compare_cleaned
        ## calculate precision, recall, and f1
        # structured_compare_cleaned_groudtruth_oriented / structured_compare_cleaned_gene_oriented: [[component0, level0], [component1, level1], ...]
        structured_compare_cleaned_groudtruth_oriented = get_f1_from_components(if_groundtruth_oriented=True)
        structured_compare_cleaned_gene_oriented = get_f1_from_components(if_groundtruth_oriented=False)
        # recall, precision, f1 (only count when level > 0)
        recall = len([int(x[1]) for x in structured_compare_cleaned_groudtruth_oriented if int(x[1]) > 0]) / len_groundtruth
        precision = len([int(x[1]) for x in structured_compare_cleaned_gene_oriented if int(x[1]) > 0]) / len_gene
        # weighted recall, weighted precision, weighted f1
        weighted_recall = sum([int(x[1]) for x in structured_compare_cleaned_groudtruth_oriented]) / (len_groundtruth * 3)
        weighted_precision = sum([int(x[1]) for x in structured_compare_cleaned_gene_oriented]) / (len_gene * 3)
        print(f"len_groundtruth: {len_groundtruth}; len_gene: {len_gene}")
        def calculate_f1(precision, recall):
            if recall != 0 and precision != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            return f1
        f1 = calculate_f1(precision, recall)
        weighted_f1 = calculate_f1(weighted_precision, weighted_recall)
        with open("./Checkpoints/intermediate_generation.json", "w") as f:
            json.dump([groundtruth_hyp_components, gene_hyp_components, structured_compare_cleaned_groudtruth_oriented, structured_compare_cleaned_gene_oriented], f)
        return [precision, recall, f1, weighted_precision, weighted_recall, weighted_f1]




    # prepare groundtruth hypothesis components for start_bkg_id and end_bkg_id and all bkg_id in between
    # Input:
    #   start_bkg_id, end_bkg_id: int; both inclusive
    #   type: str; "hyp" or "exp"
    #   save_path: if specified (not None) and if_save == 1, save the results to the path
    #   if_continue_from_previous: continue from the previous results
    def preparing_groundtruth_hyp_components(self, start_bkg_id, end_bkg_id, type="hyp", save_path=None, if_save=False, if_continue_from_previous=False):
        assert type in ["hyp", "exp"]
        assert save_path is None or isinstance(save_path, str)
        assert if_save in [True, False]
        assert if_continue_from_previous in [True, False]
        # groundtruth_hyp_components_collection: [[groundtruth_hyp_components0], [groundtruth_hyp_components1], ...]
        groundtruth_hyp_components_collection = []
        # continue from the previous results
        if if_continue_from_previous:
            assert self.groundtruth_hyp_components_collection is not None
            groundtruth_hyp_components_collection = copy.deepcopy(self.groundtruth_hyp_components_collection)
            start_bkg_id = len(groundtruth_hyp_components_collection)
        # start looping
        for cur_bkg_id in range(start_bkg_id, end_bkg_id+1):
            print("\nPreparing groundtruth hypothesis components for background id: ", cur_bkg_id)
            cur_bkg_q = self.bkg_q_list[cur_bkg_id]
            if type == "hyp":
                groundtruth_hyp = self.dict_bkg2fg_hyp[cur_bkg_q]
            elif type == "exp":
                groundtruth_hyp = self.dict_bkg2fg_exp[cur_bkg_q]
            else:
                raise Exception("Invalid type: ", type)
            cur_groundtruth_hyp_components = self.break_finegrained_hyp_or_exp(groundtruth_hyp, type)
            print("len(cur_groundtruth_hyp_components): ", len(cur_groundtruth_hyp_components))
            groundtruth_hyp_components_collection.append(cur_groundtruth_hyp_components)
            # save to save_path
            if if_save and save_path is not None:
                with open(save_path, "w") as f:
                    json.dump(groundtruth_hyp_components_collection, f)
        # print("groundtruth_hyp_components_collection: ", groundtruth_hyp_components_collection)
        return groundtruth_hyp_components_collection






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="model name: gpt-4o/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--api_type", type=int, default=0, help="0: claude shop; 1: azure")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--preprocess_groundtruth_components_dir", type=str, default="./Checkpoints/groundtruth_hyp_components_collection.json")
    parser.add_argument("--num_compare_times", type=int, default=5, help="number of times to compare the hypothesis to get average results")
    args = parser.parse_args()

    evaluator = Evaluator(args.model_name, args.api_type, args.api_key, args.chem_annotation_path, args.preprocess_groundtruth_components_dir)


    ## preparing_groundtruth_hyp_components
    # start_bkg_id, end_bkg_id = 0, 50
    # if_save = False
    # if_continue_from_previous = True
    # evaluator.preparing_groundtruth_hyp_components(start_bkg_id, end_bkg_id, type="hyp", save_path=args.preprocess_groundtruth_components_dir, if_save=if_save, if_continue_from_previous=if_continue_from_previous)


    ## test evaluate with one hypothesis
    bkg_id = 1
    gene_hyp = ""
    evaluator.check_one_generated_hyp_or_exp(bkg_id, gene_hyp, type="hyp", num_compare_times=args.num_compare_times)


    