import os, argparse, json, time, copy, math, sys, re
import numpy as np
from openai import OpenAI, AzureOpenAI
import concurrent.futures
from google import genai
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, evaluation_instruction_prompts, llm_generation_while_loop, get_first_number_from_string

class Evaluator(object):

    def __init__(self, model_name, api_type, api_key, base_url, chem_annotation_path, preprocess_groundtruth_components_dir=None):
        # assign values
        self.model_name = model_name
        self.api_type = api_type
        self.api_key = api_key
        # Set API client
        # openai client
        if api_type == 0:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        # azure client
        elif api_type == 1:
            self.client = AzureOpenAI(
                azure_endpoint = base_url, 
                api_key=api_key,  
                api_version="2024-06-01"
            )
        # google client
        elif api_type == 2:
            self.client = genai.Client(api_key=api_key)
        else:
            raise NotImplementedError
        # obtain groundtruth finegrained hypothesis and experiment
        self.bkg_q_list, self.dict_bkg2survey, self.dict_bkg2cg_hyp, self.dict_bkg2fg_hyp, self.dict_bkg2fg_exp, self.dict_bkg2note = load_chem_annotation(chem_annotation_path)   
        # load from preprocess_groundtruth_components_dir
        if isinstance(preprocess_groundtruth_components_dir, str) and os.path.exists(preprocess_groundtruth_components_dir):
            with open(preprocess_groundtruth_components_dir, "r") as f:
                # self.groundtruth_hyp_components_collection = [[groundtruth_hyp_components0], [groundtruth_hyp_components1], ...]
                self.groundtruth_hyp_components_collection = json.load(f)
            print("Loaded groundtruth_hyp_components_collection from: ", preprocess_groundtruth_components_dir)
        else:
            self.groundtruth_hyp_components_collection = None
            print("No groundtruth_hyp_components_collection loaded.")
                    
    

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
        average_compare_results = average_compare_results.tolist()
        print("compare_results_collection: ", compare_results_collection)
        print("average_compare_results: ", average_compare_results)
        return average_compare_results
    

    # break finegrained hypothesis into paragraphs, and then break each paragraph into components
    # finegrained_hyp: text
    # splitted_components: [component0, component1, ...]
    def break_finegrained_hyp_or_exp(self, finegrained_hyp, type="hyp"):

        finegrained_hyp_paragraphs = finegrained_hyp.split("\n")
        ttl_components = []
        for cur_paragraph in finegrained_hyp_paragraphs:
            if cur_paragraph.strip() == "":
                continue
            cur_components = self.break_finegrained_hyp_or_exp_one_paragraph(cur_paragraph, type)
            ttl_components.extend(cur_components)
        ttl_components_no_superficial_repetition = list(set(ttl_components))
        print("\nttl_components: ", ttl_components_no_superficial_repetition)
        return ttl_components_no_superficial_repetition

    


    # break finegrained hypothesis into components
    # finegrained_hyp: text
    # splitted_components: [component0, component1, ...]
    def break_finegrained_hyp_or_exp_one_paragraph(self, finegrained_hyp, type="hyp"):
        assert type in ["hyp", "exp"]
        # prompts
        first_prompts = evaluation_instruction_prompts('break_finegrained_hyp_or_exp', assist_info=type)
        refine_prompts = evaluation_instruction_prompts('break_finegrained_hyp_or_exp_refine', assist_info=type)
        assert len(first_prompts) == 2
        assert len(refine_prompts) == 3
        # first try
        full_prompt_first = first_prompts[0] + finegrained_hyp + first_prompts[1]
        components_try_1 = llm_generation_while_loop(full_prompt_first, self.model_name, self.client, if_structured_generation=False, temperature=0.0, api_type=self.api_type)
        # second try
        if_successful_split = False
        while not if_successful_split:
            try:
                # refine the components
                full_prompt_second = refine_prompts[0] + finegrained_hyp + refine_prompts[1] + components_try_1 + refine_prompts[2]
                # structured_components: [[component0, reasoning_process0], [component1, reasoning_process1], ...]
                structured_components = llm_generation_while_loop(full_prompt_second, self.model_name, self.client, if_structured_generation=True, template=['Id of the component:', 'Component:'], temperature=0.0, restructure_output_model_name=self.model_name, api_type=self.api_type)
                # clean structured_components
                splitted_components = [cur_d[1] for cur_d in structured_components]
                # reasoning_process usually contain the details of the component, while the component itself usually is only a brief name
                # splitted_components = [cur_d[0] + '. ' + cur_d[1] for cur_d in structured_components]
                if_successful_split = True
            except:
                print("Unsuccessful splitting, retrying...")
                continue
        splitted_components_no_superficial_repetition = list(set(splitted_components))
        print("splitted_components_paragraph: ", splitted_components_no_superficial_repetition)
        if len(splitted_components) != len(splitted_components_no_superficial_repetition):
            print(f"Warning: superficial repetition during break_components. len(splitted_components): {len(splitted_components)}; len(splitted_components_no_superficial_repetition): {len(splitted_components_no_superficial_repetition)}")
        return splitted_components_no_superficial_repetition


    # function: measure the covered degree of the gene_hyp_components compared to the groundtruth_hyp_components
    def measure_covered_degree_between_lists_of_components(self, if_groundtruth_oriented, text_groundtruth_hyp_components, text_gene_hyp_components, type="hyp"):
        assert if_groundtruth_oriented in [True, False]
        # prompts
        first_prompts = evaluation_instruction_prompts('compare_components_from_gt_and_gene', assist_info=[type, if_groundtruth_oriented])
        refine_prompts = evaluation_instruction_prompts('compare_components_from_gt_and_gene_refine', assist_info=[type, if_groundtruth_oriented])
        assert len(first_prompts) == 3
        assert len(refine_prompts) == 4
        # first try to compare
        full_prompt_first = first_prompts[0] + text_groundtruth_hyp_components + first_prompts[1] + text_gene_hyp_components + first_prompts[2]
        compare_try_1 = llm_generation_while_loop(full_prompt_first, self.model_name, self.client, if_structured_generation=False, temperature=0.0, api_type=self.api_type)
        # second try to compare
        full_prompt_second = refine_prompts[0] + text_groundtruth_hyp_components + refine_prompts[1] + text_gene_hyp_components + refine_prompts[2] + compare_try_1 + refine_prompts[3]
        # structured_compare: [[component0, level0], [component1, level1], ...]
        structured_compare = llm_generation_while_loop(full_prompt_second, self.model_name, self.client, if_structured_generation=True, template=['Covered component:', 'Covered level:'], temperature=0.0, restructure_output_model_name=self.model_name, api_type=self.api_type)
        # print("structured_compare: ", structured_compare)
        # clean structured_compare
        structured_compare_cleaned = {}
        for cur_id, cur_d in enumerate(structured_compare):
            cur_d = [x.strip('-').strip() for x in cur_d]
            # check the extracted hypothesis
            if ("none" in cur_d[0].lower() and len(cur_d[0]) < 15) or "not covered" in cur_d[0].lower() or "n/a" in cur_d[0].lower():
                continue
            # extract and check the extracted evaluation level
            cur_d_lvl = get_first_number_from_string(cur_d[1])
            if cur_d_lvl not in ['0', '1', '2', '3']:
                print("Warning: invalid level: {} for component: {}".format(cur_d_lvl, cur_d[0]))
                cur_d_lvl = 0
            # only when matched level > 0, the component is considered as a valid component
            if int(cur_d_lvl) > 0:
                # keep the highest level if there are duplicated components
                if cur_d[0] in structured_compare_cleaned:
                    print("Warning: duplicated component: {}. previous level: {}, this level: {}".format(cur_d[0], structured_compare_cleaned[cur_d[0]], cur_d_lvl))
                    if int(cur_d_lvl) > int(structured_compare_cleaned[cur_d[0]]):
                        structured_compare_cleaned[cur_d[0]] = cur_d_lvl
                else:
                    structured_compare_cleaned[cur_d[0]] = cur_d_lvl
        structured_compare_cleaned = [[k, v] for k, v in structured_compare_cleaned.items()]
        # print("\n\ncompare_try_1: ", compare_try_1)
        print("\n\nstructured_compare: ", structured_compare)
        print("\nstructured_compare_cleaned: ", structured_compare_cleaned)
        return structured_compare_cleaned
    

    # mode: 1: only component; 2: component and level
    def get_text_components_from_list(self, components, mode=1):
        # components: [component0, component1, ...]
        if mode == 1:
            # get rid of the prefix "Component id \d+: " in the component
            components = [re.sub(r'^Component id \d+: ', '', item) for item in components]
            text_components = ["Component id " + str(cur_id) + ": " + cur_component for cur_id, cur_component in enumerate(components)]
            text_components = "\n".join(text_components)
        # components: [[component0, level0], [component1, level1], ...]
        elif mode == 2:
            # get rid of the prefix "Component id \d+: " in the component
            components = [[re.sub(r'^Component id \d+: ', '', item[0]), item[1]] for item in components]
            text_components = ["Component id " + str(cur_id) + ": " + cur_component[0] + "; Covered level: " + cur_component[1] for cur_id, cur_component in enumerate(components)]
            text_components = "\n".join(text_components)
        else:
            raise NotImplementedError
        return text_components
    

    # Use LLM to get rid of repeated components
    # Input
    #   matched_components: [[component0, level0], [component1, level1], ...]
    # Output
    #   structured_components_no_repetition: [[component0, level0], [component1, level1], ...]
    def get_rid_of_repeated_components(self, matched_components):
        len_matched_components_original = len(matched_components)
        # first check whether there are exact duplicated components; if so, keep the highest level
        matched_components_dict = {}
        for cur_component in matched_components:
            if cur_component[0] in matched_components_dict:
                if int(cur_component[1]) > int(matched_components_dict[cur_component[0]]):
                    matched_components_dict[cur_component[0]] = cur_component[1]
            else:
                matched_components_dict[cur_component[0]] = cur_component[1]
        matched_components = [[k, v] for k, v in matched_components_dict.items()]            
        
        # use LLM to get rid of semantically repeated components
        # text_matched_components: text
        text_matched_components = self.get_text_components_from_list(matched_components, mode=2)
        # prompts
        first_prompts = evaluation_instruction_prompts('get_rid_of_repeated_components', assist_info=None)
        # refine_prompts = evaluation_instruction_prompts('get_rid_of_repeated_components_refine', assist_info=None)
        assert len(first_prompts) == 2
        # assert len(refine_prompts) == 3
        # only try once
        full_prompt = first_prompts[0] + text_matched_components + first_prompts[1]
        # structured_components: [[component0, level0], [component1, level1], ...]
        structured_components_no_repetition = llm_generation_while_loop(full_prompt, self.model_name, self.client, if_structured_generation=True, template=['Component:', 'Covered level:'], temperature=0.0, restructure_output_model_name=self.model_name, api_type=self.api_type)

        # use get_first_number_from_string to check the level
        for cur_id, cur_d in enumerate(structured_components_no_repetition):
            cur_d_lvl = get_first_number_from_string(cur_d[1])
            if cur_d_lvl not in ['0', '1', '2', '3']:
                print("Warning: invalid level: {} for component: {}".format(cur_d_lvl, cur_d[0]))
                cur_d_lvl = 0
            structured_components_no_repetition[cur_id][1] = cur_d_lvl
            
        # only keep the components with level > 0
        structured_components_no_repetition = [[cur_d[0], cur_d[1]] for cur_d in structured_components_no_repetition if int(cur_d[1]) > 0]
        # print("\n\nstructured_components_no_repetition: ", structured_components_no_repetition)

        if len(set([len_matched_components_original, len(matched_components), len(structured_components_no_repetition)])) != 1:
            print("Warning: len_matched_components_original: {}; len_matched_components_no_superficial_repetition: {}; len_structured_components_no_repetition: {}".format(len_matched_components_original, len(matched_components), len(structured_components_no_repetition)))
        return structured_components_no_repetition
        

    


    # groundtruth_hyp_components/gene_hyp_components: output of break_finegrained_hyp_or_exp()
    # groundtruth_hyp_components/gene_hyp_components: [component0, component1, ...]
    def calculate_precision_recall_f1(self, groundtruth_hyp_components, gene_hyp_components, type="hyp"):
        assert type in ["hyp", "exp"]
        len_groundtruth, len_gene = len(groundtruth_hyp_components), len(gene_hyp_components)
        
        text_groundtruth_hyp_components = self.get_text_components_from_list(groundtruth_hyp_components)
        num_gene_hyp_segments = math.ceil(len_gene / 3)
        structured_compare_cleaned_groudtruth_oriented = []
        for cur_segment_id in range(num_gene_hyp_segments):
            cur_gene_hyp_components = gene_hyp_components[cur_segment_id*3 : min((cur_segment_id+1)*3, len_gene)]
            text_gene_hyp_components = self.get_text_components_from_list(cur_gene_hyp_components)
            ## calculate precision, recall, and f1
            # cur_structured_compare_cleaned_groudtruth_oriented: [[component0, level0], [component1, level1], ...]
            cur_structured_compare_cleaned_groudtruth_oriented = self.measure_covered_degree_between_lists_of_components(if_groundtruth_oriented=True, text_groundtruth_hyp_components=text_groundtruth_hyp_components, text_gene_hyp_components=text_gene_hyp_components, type=type)
            structured_compare_cleaned_groudtruth_oriented.extend(cur_structured_compare_cleaned_groudtruth_oriented)  
        print("\nstructured_compare_cleaned_groudtruth_oriented: ", structured_compare_cleaned_groudtruth_oriented)
        structured_compare_cleaned_groudtruth_oriented_no_repetition = self.get_rid_of_repeated_components(structured_compare_cleaned_groudtruth_oriented)
        print("\nstructured_compare_cleaned_groudtruth_oriented_no_repetition: ", structured_compare_cleaned_groudtruth_oriented_no_repetition)
        # structured_compare_cleaned_gene_oriented = self.measure_covered_degree_between_lists_of_components(if_groundtruth_oriented=False, text_groundtruth_hyp_components=text_groundtruth_hyp_components, text_gene_hyp_components=text_gene_hyp_components, type=type)
        structured_compare_cleaned_gene_oriented = structured_compare_cleaned_groudtruth_oriented_no_repetition
        # recall, precision, f1 (only count when level > 0)
        recall = len([int(x[1]) for x in structured_compare_cleaned_groudtruth_oriented_no_repetition if int(x[1]) > 0]) / len_groundtruth
        precision = len([int(x[1]) for x in structured_compare_cleaned_gene_oriented if int(x[1]) > 0]) / len_gene
        # weighted recall, weighted precision, weighted f1
        weighted_recall = sum([int(x[1]) for x in structured_compare_cleaned_groudtruth_oriented_no_repetition]) / (len_groundtruth * 3)
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
        # # intermidiate results for debugging
        # for i in range(10000):
        #     debug_file = "./evaluate_components_for_debugging_{}.json".format(i)
        #     if not os.path.exists(debug_file):
        #         with open(debug_file, "w") as f:
        #             json.dump([groundtruth_hyp_components, gene_hyp_components, structured_compare_cleaned_groudtruth_oriented_no_repetition], f)
        #         break
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
            print(f"len(cur_groundtruth_hyp_components): {len(cur_groundtruth_hyp_components)}")
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
    parser.add_argument("--api_type", type=int, default=1, help="0: openai's API toolkit; 1: azure's API toolkit; 2: google's API toolkit")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="", help="base url for the API")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--preprocess_groundtruth_components_dir", type=str, default="./Checkpoints/groundtruth_hyp_components_collection.json")
    parser.add_argument("--num_compare_times", type=int, default=5, help="number of times to compare the hypothesis to get average results")
    args = parser.parse_args()

    evaluator = Evaluator(args.model_name, args.api_type, args.api_key, args.base_url, args.chem_annotation_path, args.preprocess_groundtruth_components_dir)


    ## preparing_groundtruth_hyp_components
    start_bkg_id, end_bkg_id = 0, 50
    # start_bkg_id, end_bkg_id = 33, 33
    if_save = True
    if_continue_from_previous = False
    evaluator.preparing_groundtruth_hyp_components(start_bkg_id, end_bkg_id, type="hyp", save_path=args.preprocess_groundtruth_components_dir, if_save=if_save, if_continue_from_previous=if_continue_from_previous)


    ## test evaluate with one hypothesis
    # bkg_id = 1
    # gene_hyp = ""
    # evaluator.check_one_generated_hyp_or_exp(bkg_id, gene_hyp, type="hyp", num_compare_times=args.num_compare_times)


    