import os, argparse, json, time, copy, math, sys, re
import numpy as np
from openai import OpenAI, AzureOpenAI
from google import genai
sys.stdout.reconfigure(encoding='utf-8')
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, instruction_prompts, llm_generation_while_loop, exchange_order_in_list
from Evaluation.pairwise_compare import PairwiseCompare
from Method.logging_utils import setup_logger

class Greedy(object):

    def __init__(self, args):
        self.args = args
        ## Set API client
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
        # obtain groundtruth finegrained hypothesis and experiment
        if self.args.if_use_custom_research_background_and_coarse_hyp == 0:
            self.bkg_q_list, self.dict_bkg2survey, self.dict_bkg2cg_hyp, self.dict_bkg2fg_hyp, self.dict_bkg2fg_exp, self.dict_bkg2note = load_chem_annotation(args.chem_annotation_path)  
            # update dict_bkg2cg_hyp with the vague cg hypothesis
            if args.if_use_vague_cg_hyp_as_input == 1:
                assert os.path.exists(args.vague_cg_hyp_path)
                with open(args.vague_cg_hyp_path, "r") as f:
                    self.dict_bkg2cg_hyp = json.load(f) 



    def get_finegrained_hyp_for_one_research_question(self, cur_bkg_id):
        # basic input information
        if self.args.if_use_custom_research_background_and_coarse_hyp == 0:
            print("Loading data from TOMATA-Chem2 dataset...")
            cur_q = self.bkg_q_list[cur_bkg_id]
            cur_survey = self.dict_bkg2survey[cur_q]
            cur_cg_hyp = self.dict_bkg2cg_hyp[cur_q]
        elif self.args.if_use_custom_research_background_and_coarse_hyp == 1:
            print("Loading data from custom research background...")
            # use the custom research background and coarse-grained hypothesis
            # custom_data: [[research_question, background_survey, input_cg_hyp], ...]
            with open(self.args.custom_research_background_and_coarse_hyp_path, "r") as f:
                custom_data = json.load(f)
                cur_q, cur_survey, cur_cg_hyp = custom_data[cur_bkg_id]
        else:
            raise ValueError("Invalid cur_bkg_id: ", cur_bkg_id)
        print("Initial coarse-grained hypothesis: ", cur_cg_hyp)

        # input_cg_hypothesis_prompt
        input_cg_hypothesis_prompt = "\n\nThe coarse-grained hypothesis is: {}\nThe coarse-grained hypothesis is proposed by a student and has not been verified by experiments. ".format(cur_cg_hyp) 
        # merge input_cg_hypothesis_prompt into cur_survey
        cur_survey += input_cg_hypothesis_prompt

        # search for 10 times
        # full_results: [[hypothesis, reason], ...]
        full_results = []
        prev_step_gene_fg_hyp = None
        for cur_search_step_id in range(self.args.max_search_step):
            print("Search step ID: ", cur_search_step_id)
            structured_gene, selection_reason, if_continue_search = self.one_step_greedy_search_strictly_better_than_previous(cur_q, cur_survey, cur_cg_hyp, prev_step_gene_fg_hyp, cur_search_step_id, self.args.locam_minimum_threshold)
            # local minimum detection
            if if_continue_search == False:
                print("INFO: Early stopping: this search might have already reached a local minimum.")
                break
            # save the search results
            full_results.append(structured_gene)
            if structured_gene is not None:
                prev_step_gene_fg_hyp = structured_gene[0]
                # print("\tcurrent hyp: ", prev_step_gene_fg_hyp)
            else:
                break

        # save the search results after all search steps
        if self.args.if_save == 1:
            with open(self.args.output_dir, 'w') as f:
                json.dump(full_results, f, indent=4)



    # Output
    # structured_gene: [hypothesis, reason]
    # selection_reason: [selection (1/2), reason]
    # if_continue_search: early stopping: this search might have already reached a local minimum
    def one_step_greedy_search_strictly_better_than_previous(self, cur_q, cur_survey, cur_cg_hyp, prev_step_gene_fg_hyp, cur_search_step_id, locam_minimum_threshold):
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
                structured_gene = self.one_step_greedy_search_with_feedback(cur_q, cur_survey, cur_cg_hyp, prev_step_gene_fg_hyp)
            else:
                structured_gene = self.one_step_greedy_search(cur_q, cur_survey, cur_cg_hyp, prev_step_gene_fg_hyp)
            print("\tcurrent hyp: ", structured_gene[0])
            # selection_reason: [selection (1/2), reason]
            # when prev_step_gene_fg_hyp == None, the previous fine-grained hypothesis is not generated, thus use the coarse-grained hypothesis for comparison
            if cur_search_step_id == 0:
                assert prev_step_gene_fg_hyp is None
                selection_reason = self.pairwise_compare.compare(cur_q, cur_cg_hyp, structured_gene[0], instruction_mode="strict_to_hyp2", hierarchy_level=None)
            else:
                assert prev_step_gene_fg_hyp is not None
                selection_reason = self.pairwise_compare.compare(cur_q, prev_step_gene_fg_hyp, structured_gene[0], instruction_mode="strict_to_hyp2", hierarchy_level=None)
            if selection_reason[0] == 2:
                # the new hypothesis is better
                if_better=True
                cnt_search_single_step=0
            else:
                print("\tThe new hypothesis is not better than the previous one, try again... ")
                # print("\nThe new hypothesis is not better than the previous one, try again... \nReason: {}".format(selection_reason[1]))
                if self.args.if_generate_with_past_failed_hyp == 1:
                    past_failed_hyp.append([structured_gene[0], selection_reason[1]])
                    print("Collected past failed hypothesis to mimic in-context RL")
            cnt_search_single_step+=1
            if cnt_search_single_step >= locam_minimum_threshold:
                if_continue_search=False
                break
        return structured_gene, selection_reason, if_continue_search



    # Output
    # structured_gene_updated_hyp: [hypothesis, reason]
    def one_step_greedy_search_with_feedback(self, cur_q, cur_survey, cur_cg_hyp, prev_step_gene_fg_hyp):
        # one step greedy search (find one d)
        # structured_gene_original_optimization: [hypothesis, reason]
        structured_gene_original_optimization = self.one_step_greedy_search(cur_q, cur_survey, cur_cg_hyp, prev_step_gene_fg_hyp)

        ## feedback and update
        # previous hypothsis
        if prev_step_gene_fg_hyp is None:
            assert isinstance(cur_cg_hyp, str)
            prev_hyp = cur_cg_hyp
        else:
            assert isinstance(prev_step_gene_fg_hyp, str)
            prev_hyp = prev_step_gene_fg_hyp

        # structured_gene_updated_hyp: [updated hypothesis, reason]
        structured_gene_updated_hyp = self.feedback_to_hyp_and_update_hyp(cur_q, cur_survey, prev_hyp, structured_gene_original_optimization[0], structured_gene_original_optimization[1])
        return structured_gene_updated_hyp


    # Output
    #   structured_gene: [updated hypothesis, reason]
    def feedback_to_hyp_and_update_hyp(self, cur_q, cur_survey, prev_hyp, cur_hyp, cur_hyp_reason):
        # print("\traw hyp: ", cur_hyp)
        ## feedback
        # prompts
        prompts = instruction_prompts("validity_clarity_feedback_to_hyp")
        assert len(prompts) == 6
        full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + prev_hyp + prompts[3] + cur_hyp + prompts[4] + cur_hyp_reason + prompts[5]
        # generate finegrained hypothesis
        # feedback: str
        feedback = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=False, temperature=1.0, api_type=self.args.api_type)
        # print("\tfeedback: ", feedback)
        ## update hypothesis
        # prompts
        prompts = instruction_prompts("update_hyp_based_on_feedback")
        assert len(prompts) == 6
        full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + prev_hyp + prompts[3] + cur_hyp + prompts[4] + feedback + prompts[5]
        # generate finegrained hypothesis
        # structured_gene: [hypothesis, reason]
        structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Revised Hypothesis:'], temperature=1.0, api_type=self.args.api_type)
        structured_gene = exchange_order_in_list(structured_gene)
        assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
        # print("\trefined hyp from feedback: ", structured_gene[0][0])
        return structured_gene[0]
    


    # Function
    # get the search result of one step greedy search
    # Input
    # cur_q/cur_survey/cur_cg_hyp/prev_step_gene_fg_hyp: str
    #       cur_survey: it should include the coarse-grained hypothesis
    # Output
    # structured_gene: [hypothesis, reason]
    def one_step_greedy_search(self, cur_q, cur_survey, cur_cg_hyp, prev_step_gene_fg_hyp=None):
        # prompts
        if prev_step_gene_fg_hyp is None:
            # the first search step
            prompts = instruction_prompts("greedy_search_first_step", assist_info=[None, self.args.if_generate_with_example])
            assert len(prompts) == 3
            # full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + cur_cg_hyp + prompts[3]
            full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2]
        else:
            # the following search steps (with previous search results)
            prompts = instruction_prompts("greedy_search_following_step", assist_info=[None, self.args.if_generate_with_example])
            assert len(prompts) == 4
            # full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + cur_cg_hyp + prompts[3] + prev_step_gene_fg_hyp + prompts[4]
            full_prompt = prompts[0] + cur_q + prompts[1] + cur_survey + prompts[2] + prev_step_gene_fg_hyp + prompts[3]
        # print("\tfull_prompt: ", full_prompt)
        # generate finegrained hypothesis
        # structured_gene: [[hypothesis, reason], ...]
        # checking the format of structured_gene
        while True:
            structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Revised Hypothesis:'], temperature=1.0, api_type=self.args.api_type)
            structured_gene = exchange_order_in_list(structured_gene)
            if len(structured_gene) == 1 and len(structured_gene[0]) == 2:
                break
            else:
                print("Warning: incorrect structured_gene format, try again... structured_gene: ", structured_gene)
        return structured_gene[0]

        






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Greedy search for fine-grained hypothesis generation')
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="model name: gpt-4o/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--eval_model_name", type=str, default="gpt-4o", help="model name for evaluation: gpt-4o/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--api_type", type=int, default=1, help="0: openai's API toolkit; 1: azure's API toolkit")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--eval_api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="", help="base url for the API")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--bkg_id", type=int, default=0, help="background research question id")
    parser.add_argument("--output_dir", type=str, default="./Checkpoints/hypothesis_evaluation_results.json")
    parser.add_argument("--if_save", type=int, default=0, help="whether save grouping results")
    parser.add_argument("--max_search_step", type=int, default=10, help="maximum search steps")
    parser.add_argument("--locam_minimum_threshold", type=int, default=5, help="local minimum threshold")
    parser.add_argument("--if_feedback", type=int, default=0, help="for each fundamental search step, whether to provide feedback to the hypothesis and update the hypothesis with the feedback")
    parser.add_argument("--if_multiple_llm", type=int, default=0, help="whether to use multiple llms for hypothesis gradient estimation. 0: single llm; 1: multiple same llms; 2: multiple different llms")
    parser.add_argument("--if_use_vague_cg_hyp_as_input", type=int, default=0, help="whether to use processed vague coarse-grained hypothesis as input (by Data_Processing/input_hyp_processing.py)")
    parser.add_argument("--vague_cg_hyp_path", type=str, default="./Data/processed_research_direction.json", help="store processed vague coarse-grained hypothesis")
    parser.add_argument("--if_generate_with_example", type=int, default=1, help="during optimization, whether to use hypothesis example in the prompt to generate hypothesis in each step")
    parser.add_argument("--if_generate_with_past_failed_hyp", type=int, default=0, help="during optimization, whether to use past failed hypothesis in the prompt to generate hypothesis in each step")
    parser.add_argument("--if_use_custom_research_background_and_coarse_hyp", type=int, default=0, help="whether to use custom research question & background survey & coarse-grained hypothesis; 0: use the background research question in the chem_annotation_path; 1: use custom research question in custom_research_background_and_coarse_hyp_path")
    parser.add_argument("--custom_research_background_and_coarse_hyp_path", type=str, default="./custom_research_background_and_coarse_hyp.json", help="if bkg_id == -1, then use this path to load custom research question and background survey and coarse-grained hypothesis; in a format of json file, with keys: 'research_question', 'background_survey', 'coarse_grained_hypothesis'; if bkg_id != -1, then this path will be ignored")
    args = parser.parse_args()

    assert args.if_save in [0, 1]
    assert args.if_feedback in [0, 1]
    assert args.if_multiple_llm in [0, 1, 2]
    assert args.if_use_vague_cg_hyp_as_input in [0, 1]
    # we need if_generate_with_example to be 1
    assert args.if_generate_with_example in [1]
    assert args.if_generate_with_past_failed_hyp in [0, 1]
    assert args.api_type in [0, 1, 2]
    assert args.if_use_custom_research_background_and_coarse_hyp in [0, 1]

    ## Setup logger
    logger = setup_logger(args.output_dir)
    # Redirect print to logger
    def custom_print(*args, **kwargs):
        message = " ".join(map(str, args))
        logger.info(message)
    global print
    print = custom_print
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

    if os.path.exists(args.output_dir):
        print("Warning: {} already exists.".format(args.output_dir))
    else:
        ## Start greedy search
        print("Start greedy search...")
        start_time = time.time()
        greedy = Greedy(args)
        # get fine-grained hypothesis for one research question
        greedy.get_finegrained_hyp_for_one_research_question(args.bkg_id)
        duration = (time.time() - start_time) / 60
        print("Finished: duration: {:.2f} minutes".format(duration))