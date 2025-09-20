import os, argparse, json, time, copy, math, sys, re
import numpy as np
from openai import OpenAI, AzureOpenAI
from google import genai
import concurrent.futures
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import evaluation_instruction_prompts, llm_generation_while_loop, exchange_order_in_list

class PairwiseCompare(object):
    def __init__(self, api_type, api_key, base_url, model_name="claude35S", if_multiple_llm=0):
        assert if_multiple_llm in [0, 1, 2]
        self.api_type = api_type
        ## Set API client
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
        # default is self.model_name, but can be overwritten by eval_model_name provided in the compare_single_llm() function; 
        #   also used for unified preference from multiple llms
        #   if if_multiple_llm == 0/1, then use self.model_name to compare
        self.model_name = model_name
        self.if_multiple_llm = if_multiple_llm


    # Input:
    #   research_question/hypothesis1/hypothesis2: str
    #   instruction_mode: str; "strict_to_hyp2" or "same_hyp1_hyp2"
    #   hierarchy_level: "0, 1, 2, 3, 4" if it is from hierarchy greedy and 5 hierarchies; or "None" if it is not from hierarchy greedy, or from hierarchy greedy but with only 1 hierarchy (which is equivalent to the greedy search + self-consistency)
    #   if_final_eval: bool; if we are hypothesis2 is a more updated version of hypothesis1, then we can set if_final_eval=False; else we can set if_final_eval=True
    #   if_no_unified_response: bool; if we don't need to unify the response from multiple llms, then we can set if_no_unified_response=True (in this case, the response will be a list of three responses from three llms)
    #   num_compare_times: int; the number of times to compare the same pair of hypotheses
    #   compare_metric: str; "overall", "effectiveness", "novelty", "detailedness", "feasibility"
    # return: 
    #   if_no_unified_response == False: [selection (1/2), reason]
    #   if_no_unified_response == True: [[selection (1/2), reason], ...]
    def compare(self, research_question, hypothesis1, hypothesis2, instruction_mode="strict_to_hyp2", hierarchy_level=None, if_final_eval=False, if_no_unified_response=False, num_compare_times=3, compare_metric="overall"):
        assert self.if_multiple_llm in [0, 1, 2]
        assert instruction_mode == "strict_to_hyp2" or instruction_mode == "same_hyp1_hyp2"
        assert compare_metric in ["overall", "effectiveness", "novelty", "detailedness", "feasibility"]
        def get_llm_response(cur_eval_model_name):
            return self.compare_single_llm(research_question, hypothesis1, hypothesis2, instruction_mode, hierarchy_level, eval_model_name=cur_eval_model_name, if_final_eval=if_final_eval, compare_metric=compare_metric)
        # print("hierarchy_level: ", hierarchy_level)
        if self.if_multiple_llm == 0:
            # usually means we need to return more than one pairwise evaluation results; when if_multiple_llm == 0, it probably not during search, but during ranking local found minima 
            if if_no_unified_response:
                if not num_compare_times > 1:
                    print(f"Warning: if_no_unified_response is set to True, but num_compare_times is not greater than 1: {num_compare_times}")
                eval_models = [self.model_name for i in range(num_compare_times)]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    llm_response_collection = list(executor.map(get_llm_response, eval_models))
                return llm_response_collection
            else:
                # if num_compare_times > 1:
                #     print(f"Warning: if_no_unified_response is set to False, but num_compare_times is greater than 1: {num_compare_times}")
                # use the default self.model_name to compare
                return self.compare_single_llm(research_question, hypothesis1, hypothesis2, instruction_mode, hierarchy_level, if_final_eval=if_final_eval, compare_metric=compare_metric)
        elif self.if_multiple_llm == 1 or self.if_multiple_llm == 2:
            ## use multiple LLMs to compare (gpt4o, claude35S, gemini)
            # not parallelized code
            # llm_response_collection = []
            # for cur_eval_model_name in ["claude35S", "gpt4o", "gemini15P"]:
            #     cur_response = self.compare_single_llm(research_question, hypothesis1, hypothesis2, instruction_mode, hierarchy_level, eval_model_name=cur_eval_model_name, if_final_eval=if_final_eval, compare_metric=compare_metric)
            #     llm_response_collection.append(cur_response)
            # parallelized code
            llm_response_collection = []
            # put gpt-4o in the middle to alleviate overly relying on the first model's preference, since in practice, gpt-4o usually think gpt-4o's preference is the best, when putting gpt-4o in the first place (however I don't know whether put gpt-4o in the middle can alleviate this issue)
            if self.if_multiple_llm == 1:
                eval_models = [self.model_name for i in range(num_compare_times)]
            elif self.if_multiple_llm == 2:
                assert num_compare_times == 3, print("num_compare_times: ", num_compare_times)
                if self.model_name == "gpt-4o" or "gemini-1.5-pro" in self.model_name or "claude-3-5-sonnet" in self.model_name:
                    eval_models = ["claude-3-5-sonnet-20241022", "gpt-4o", "gemini-1.5-pro"]
                elif self.model_name == "gpt-4o-mini" or "gemini-1.5-flash" in self.model_name or "claude-3-haiku" in self.model_name:
                    eval_models = ["claude-3-haiku-20240307", "gpt-4o-mini", "gemini-1.5-flash-latest"]
                else:
                    raise ValueError("Wrong model_name: ", self.model_name)
            else:
                raise ValueError("Wrong if_multiple_llm value: ", self.if_multiple_llm)
            print("eval_models: ", eval_models)
            # print("starting pairwise compare in parallel...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                llm_response_collection = list(executor.map(get_llm_response, eval_models))
            # print("ended pairwise compare in parallel...")
            # directly reply the three responses without unifying the response
            if if_no_unified_response:
                # llm_response_collection: [[selection (1/2), reason], ...]
                return llm_response_collection
            ## if the three responses have the same preference, then don't need to unify the response and return the first response
            preference_collection = [int(cur_response[0]) for cur_response in llm_response_collection]
            if max(preference_collection) == min(preference_collection):
                print("preference_collection: ", preference_collection)
                return llm_response_collection[0]
            ## diverse opinions exist, unify the response
            if compare_metric == "overall" and if_final_eval == True:
                # print("pairwise compare between final hypothesis")
                prompts = evaluation_instruction_prompts("pairwise_compare_between_final_hyp_unify_response", assist_info=[instruction_mode])
            elif compare_metric == "overall" and if_final_eval == False:
                # print("pairwise compare for search")
                prompts = evaluation_instruction_prompts("pairwise_compare_unify_response", assist_info=[instruction_mode])
            elif compare_metric == "effectiveness":
                prompts = evaluation_instruction_prompts("pairwise_compare_effectiveness_unify_response", assist_info=[instruction_mode])
            elif compare_metric == "novelty":
                prompts = evaluation_instruction_prompts("pairwise_compare_novelty_unify_response", assist_info=[instruction_mode])
            elif compare_metric == "detailedness":
                prompts = evaluation_instruction_prompts("pairwise_compare_detailedness_unify_response", assist_info=[instruction_mode])
            elif compare_metric == "feasibility":
                prompts = evaluation_instruction_prompts("pairwise_compare_feasibility_unify_response", assist_info=[instruction_mode])
            else:
                raise ValueError("Wrong compare_metric: ", compare_metric)
            assert len(prompts) == 5
            ## multiple_llms_response_prompt
            multiple_llms_response_prompt = ""
            for cur_llm_response_id, cur_llm_response in enumerate(llm_response_collection):
                assert len(cur_llm_response) == 2
                multiple_llms_response_prompt += f"Expert {cur_llm_response_id + 1}'s preference: research hypothesis candidate {cur_llm_response[0]}; Expert {cur_llm_response_id + 1}'s reason: {cur_llm_response[1]}\n"  
            full_prompt = prompts[0] + research_question + prompts[1] + hypothesis1 + prompts[2] + hypothesis2 + prompts[3] + multiple_llms_response_prompt + prompts[4]
            if_correct_format = False
            while not if_correct_format:
                # response: [[reason, selection (1/2)]]
                response = llm_generation_while_loop(full_prompt, self.model_name, self.client, if_structured_generation=True, template=['Reasoning process:', 'Selection of research hypothesis candidate:'], temperature=0.0, api_type=self.api_type)
                # response: [[selection (1/2), reason]]
                response = exchange_order_in_list(response)
                if len(response) > 1:
                    print("Warning: multuple response, only take the first one. len(response): ", len(response), "response: ", response)
                    response = [response[0]]
                assert len(response) == 1 and len(response[0]) == 2, print("len(response): ", len(response), "response: ", response)
                if not (("1" in response[0][0] or "2" in response[0][0]) and not ("1" in response[0][0] and "2" in response[0][0])):
                    # print("Try again: response[0][0] should contain '1' or '2' but not both. response[0][0]: ", response[0][0])
                    continue
                if "1" in response[0][0]:
                    response[0][0] = 1
                elif "2" in response[0][0]:
                    response[0][0] = 2 
                else:
                    raise ValueError("Wrong output format of pairwise comparison.")
                if response[0][0] == 1 or response[0][0] == 2:
                    if_correct_format = True
                else:
                    print(f"Wrong output format of pairwise comparison: {response}, try again...")
            print("preference_collection: ", [preference_collection, response[0][0]])
            return response[0]
        else:
            raise ValueError("Wrong if_multiple_llm value: ", self.if_multiple_llm)


    
    # Input:
    #   research_question/hypothesis1/hypothesis2: str
    #   instruction_mode: str; "strict_to_hyp2" or "same_hyp1_hyp2"
    #   hierarchy_level: "0, 1, 2, 3, 4" if it is from hierarchy greedy; or "None" if it is not from hierarchy greedy, or if it is a general comparison
    # return: [selection (1/2), reason]
    #   if_final_eval: bool; if during searching or during final evaluation
    #   compare_metric: "overall", "effectiveness", "novelty", "detailedness", "feasibility"
    def compare_single_llm(self, research_question, hypothesis1, hypothesis2, instruction_mode="strict_to_hyp2", hierarchy_level=None, eval_model_name=None, if_final_eval=False, compare_metric="overall"):
        assert isinstance(research_question, str)
        assert instruction_mode == "strict_to_hyp2" or instruction_mode == "same_hyp1_hyp2"
        assert eval_model_name == None or isinstance(eval_model_name, str)
        assert compare_metric in ["overall", "effectiveness", "novelty", "detailedness", "feasibility"]
        # default eval_model_name is self.model_name, but can be overwritten by eval_model_name
        if eval_model_name == None:
            eval_model_name = self.model_name
        if isinstance(hypothesis1, str) and isinstance(hypothesis2, str):
            # prompts
            if compare_metric == "overall" and if_final_eval == True:
                # print("pairwise compare between final hypothesis")
                prompts = evaluation_instruction_prompts("pairwise_compare_between_final_hyp", assist_info=[instruction_mode])
            elif compare_metric == "overall" and if_final_eval == False:
                # print("pairwise compare for search")
                prompts = evaluation_instruction_prompts("pairwise_compare", assist_info=[instruction_mode, hierarchy_level])
            elif compare_metric == "effectiveness":
                prompts = evaluation_instruction_prompts("pairwise_compare_effectiveness", assist_info=[instruction_mode])
            elif compare_metric == "novelty":
                prompts = evaluation_instruction_prompts("pairwise_compare_novelty", assist_info=[instruction_mode])
            elif compare_metric == "detailedness":
                prompts = evaluation_instruction_prompts("pairwise_compare_detailedness", assist_info=[instruction_mode])
            elif compare_metric == "feasibility":
                prompts = evaluation_instruction_prompts("pairwise_compare_feasibility", assist_info=[instruction_mode])
            else:
                raise ValueError("Wrong compare_metric: ", compare_metric)
            assert len(prompts) == 4
            full_prompt = prompts[0] + research_question + prompts[1] + hypothesis1 + prompts[2] + hypothesis2 + prompts[3]
            if_correct_format = False
            for cur_exp_id in range(500):
                # response: [[reason, selection (1/2)]]
                # print("full_prompt: ", full_prompt)
                response = llm_generation_while_loop(full_prompt, eval_model_name, self.client, if_structured_generation=True, template=['Reasoning process:', 'Selection of research hypothesis candidate:'], temperature=0.0, api_type=self.api_type)
                # response: [[selection (1/2), reason]]
                response = exchange_order_in_list(response)
                # print("response: ", response)
                if len(response) > 1:
                    print("Warning: multuple response, only take the first one. len(response): ", len(response), "response: ", response)
                    response = [response[0]]
                assert len(response) == 1 and len(response[0]) == 2, print("len(response): ", len(response), "response: ", response)
                # print("response[0][0]: ", response[0][0])
                # sometimes response[0][0] is "Candidate 1", but not just "1" or "2"
                # response[0][0] = int(response[0][0])
                # response[0][0] should contain "1" or "2" but not both
                if not (("1" in response[0][0] or "2" in response[0][0]) and not ("1" in response[0][0] and "2" in response[0][0])):
                    # print("Warning: response[0][0] should contain '1' or '2' but not both. response[0][0]: ", response[0][0])
                    # print("Warning: response[0][0] should contain '1' or '2' but not both.")
                    pass
                idx_1 = response[0][0].find("1")
                idx_2 = response[0][0].find("2")
                idx_1 = 1000 if idx_1 < 0 else idx_1
                idx_2 = 1000 if idx_2 < 0 else idx_2
                if idx_1 < idx_2:
                    response[0][0] = 1
                elif idx_2 < idx_1:
                    response[0][0] = 2 
                else:
                    assert idx_1 == 1000 and idx_2 == 1000, print(f"idx_1: {idx_1}; idx_2: {idx_2}")
                    print("Warning: response[0][0] contain neither '1' nor '2', compare again... \nresponse[0][0]: ", response[0][0])
                    continue
                if response[0][0] == 1 or response[0][0] == 2:
                    if_correct_format = True
                    break
                else:
                    print(f"Wrong output format of pairwise comparison: {response}, try again...")
        # if one of the hypotheses is not generated
        elif isinstance(hypothesis1, str) or isinstance(hypothesis2, str):
            if isinstance(hypothesis1, str):
                return [1, "The first hypothesis is not generated."]
            else:
                return [2, "The second hypothesis is not generated."]
        else:
            raise ValueError("Both hypotheses are not generated.")
        # print("final pick is chosen: ", response[0])
        return response[0]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise comparison")
    parser.add_argument("--api_type", type=int, default=0, help="0: openai's API toolkit; 1: azure's API toolkit; 2: google's API toolkit")
    parser.add_argument("--api_key", type=str, default=None, help="")
    parser.add_argument("--base_url", type=str, default="", help="base url for the API")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name. claude35S/gpt-4o")
    parser.add_argument("--eval_example_path", type=str, default="./Checkpoints/eval_example.json", help="Evaluation example path")
    args = parser.parse_args()

    # test
    pairwise_compare = PairwiseCompare(args.api_type, args.api_key, args.base_url, args.model_name)
    # load evaluation example
    with open(args.eval_example_path, "r", encoding="utf-8") as f:
        research_question, hypothesis1, hypothesis2 = json.load(f)
    # compare: same_hyp1_hyp2 / strict_to_hyp2
    response = pairwise_compare.compare(research_question, hypothesis1, hypothesis2, instruction_mode="same_hyp1_hyp2", hierarchy_level=None)
    print(response)