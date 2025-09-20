import argparse, os, sys, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, llm_generation_while_loop, exchange_order_in_list
from Baselines.fundamental_assumption_utils import instruction_prompts
from openai import OpenAI, AzureOpenAI
from google import genai

class VerifyAssumption(object):

    def __init__(self, args):
        self.args = args
        # Add a lock for thread-safe file operations
        self.file_lock = threading.Lock()
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
        # load data
        self.bkg_q_list, self.dict_bkg2survey, self.dict_bkg2cg_hyp, self.dict_bkg2fg_hyp, self.dict_bkg2fg_exp, self.dict_bkg2note = load_chem_annotation(args.chem_annotation_path)  
        with open(args.vague_cg_hyp_path, "r") as f:
            self.dict_bkg2cg_hyp = json.load(f) 
        # load or initialize the lists
        self.load_or_initialize_lists()

    def load_or_initialize_lists(self):
        self.general_concept_list_path = "./Baselines/general_concept_list.json"
        self.replaced_general_concept_list_path = "./Baselines/replaced_general_concept_list.json"
        self.cg_hyp_with_replaced_general_concept_list_path = "./Baselines/cg_hyp_with_replaced_general_concept_list.json"
        self.specific_concept_list_path = "./Baselines/specific_concept_list.json"
        self.hyp_with_specific_concept_list_path = "./Baselines/hyp_with_specific_concept_list.json"
        self.hyp_with_specific_concept_score_list_path = "./Baselines/hyp_with_specific_concept_score_list.json"
        ## Load or initialize the lists
        # general_concept_list: {bkg_id: [general_concept1, general_concept2]}
        if os.path.exists(self.general_concept_list_path):
            with open(self.general_concept_list_path, "r") as f:
                self.general_concept_list = json.load(f)
                print("Loaded general_concept_list")
        else:
            self.general_concept_list = {}
        # replaced_general_concept_list: {bkg_id: {general_concept1: [replaced_general_concept1]}}
        if os.path.exists(self.replaced_general_concept_list_path):
            with open(self.replaced_general_concept_list_path, "r") as f:
                self.replaced_general_concept_list = json.load(f)
                print("Loaded replaced_general_concept_list")
        else:
            self.replaced_general_concept_list = {}
        # cg_hyp_with_replaced_general_concept_list: {bkg_id: {general_concept1: {replaced_general_concept1: cg_hyp_with_replaced_general_concept1}}}
        if os.path.exists(self.cg_hyp_with_replaced_general_concept_list_path):
            with open(self.cg_hyp_with_replaced_general_concept_list_path, "r") as f:
                self.cg_hyp_with_replaced_general_concept_list = json.load(f)
                print("Loaded cg_hyp_with_replaced_general_concept_list")
        else:
            self.cg_hyp_with_replaced_general_concept_list = {}
        # specific_concept_list: {bkg_id: {general_concept1/replaced_general_concept1: [specific_concept1, specific_concept2, specific_concept3]}}
        if os.path.exists(self.specific_concept_list_path):
            with open(self.specific_concept_list_path, "r") as f:
                self.specific_concept_list = json.load(f)
                print("Loaded specific_concept_list")
        else:
            self.specific_concept_list = {}
        # hyp_with_specific_concept_list: {bkg_id: {general_concept1/replaced_general_concept1: {specific_concept1: hyp_with_specific_concept}}}
        if os.path.exists(self.hyp_with_specific_concept_list_path):
            with open(self.hyp_with_specific_concept_list_path, "r") as f:
                self.hyp_with_specific_concept_list = json.load(f)
                print("Loaded hyp_with_specific_concept_list")
        else:
            self.hyp_with_specific_concept_list = {}
        # hyp_with_specific_concept_score_list: {bkg_id: {general_concept1/replaced_general_concept1: {specific_concept1: score}}}
        if os.path.exists(self.hyp_with_specific_concept_score_list_path):
            with open(self.hyp_with_specific_concept_score_list_path, "r") as f:
                self.hyp_with_specific_concept_score_list = json.load(f)
                print("Loaded hyp_with_specific_concept_score_list")
        else:
            self.hyp_with_specific_concept_score_list = {}

    def save_lists(self):
        with self.file_lock:
            if self.general_concept_list:
                with open(self.general_concept_list_path, "w") as f:
                    json.dump(self.general_concept_list, f)
            if self.replaced_general_concept_list:
                with open(self.replaced_general_concept_list_path, "w") as f:
                    json.dump(self.replaced_general_concept_list, f)
            if self.cg_hyp_with_replaced_general_concept_list:
                with open(self.cg_hyp_with_replaced_general_concept_list_path, "w") as f:
                    json.dump(self.cg_hyp_with_replaced_general_concept_list, f)
            if self.specific_concept_list:
                with open(self.specific_concept_list_path, "w") as f:
                    json.dump(self.specific_concept_list, f)
            if self.hyp_with_specific_concept_list:
                with open(self.hyp_with_specific_concept_list_path, "w") as f:
                    json.dump(self.hyp_with_specific_concept_list, f)
            if self.hyp_with_specific_concept_score_list:
                with open(self.hyp_with_specific_concept_score_list_path, "w") as f:
                    json.dump(self.hyp_with_specific_concept_score_list, f)

    def _process_single_extract_general_concept(self, cur_bkg_id):
        """Process a single background question for general concept extraction"""
        bkg_id_key = str(cur_bkg_id)
        if bkg_id_key in self.general_concept_list:
            return None
            
        cur_bkg_q = self.bkg_q_list[cur_bkg_id]
        cur_cg_hyp = self.dict_bkg2cg_hyp[cur_bkg_q]
        cur_prompt = instruction_prompts("extract_general_concept")
        full_prompt = cur_prompt[0] + cur_bkg_q + cur_prompt[1] + cur_cg_hyp + cur_prompt[2]
        structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'General Concepts:'], temperature=1.0, api_type=self.args.api_type)
        assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
        cur_concepts = structured_gene[0][1]
        cur_concepts = cur_concepts.strip().split(",")
        cur_concepts = [cur_concept.strip() for cur_concept in cur_concepts]
        print(f"Background {cur_bkg_id} concepts: {cur_concepts}")
        return bkg_id_key, cur_concepts

    def extract_general_concept(self, max_workers=4):
        """Parallel version of extract_general_concept"""
        bkg_ids_to_process = []
        for cur_bkg_id in range(len(self.bkg_q_list)):
            bkg_id_key = str(cur_bkg_id)
            if bkg_id_key not in self.general_concept_list:
                bkg_ids_to_process.append(cur_bkg_id)
        
        if not bkg_ids_to_process:
            print("All general concepts already extracted")
            return
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_bkg_id = {executor.submit(self._process_single_extract_general_concept, bkg_id): bkg_id 
                               for bkg_id in bkg_ids_to_process}
            
            for future in as_completed(future_to_bkg_id):
                result = future.result()
                if result is not None:
                    bkg_id_key, cur_concepts = result
                    with self.file_lock:
                        self.general_concept_list[bkg_id_key] = cur_concepts
                    self.save_lists()

    def _process_single_synthesize_new_hyp_with_replaced_general_concept(self, cur_bkg_id):
        """Process a single background question for synthesizing new hypotheses with replaced concepts"""
        bkg_id_key = str(cur_bkg_id)
        if bkg_id_key in self.replaced_general_concept_list:
            return None
            
        # Initialize the lists
        with self.file_lock:
            if bkg_id_key not in self.replaced_general_concept_list:
                self.replaced_general_concept_list[bkg_id_key] = {}
            if bkg_id_key not in self.cg_hyp_with_replaced_general_concept_list:
                self.cg_hyp_with_replaced_general_concept_list[bkg_id_key] = {}
                for cur_concept in self.general_concept_list[bkg_id_key]:
                    self.cg_hyp_with_replaced_general_concept_list[bkg_id_key][cur_concept] = {}

        cur_bkg_q = self.bkg_q_list[cur_bkg_id]
        cur_cg_hyp = self.dict_bkg2cg_hyp[cur_bkg_q]
        cur_concepts = self.general_concept_list[bkg_id_key]
        
        results = {}
        for cur_concept in cur_concepts:
            # propose a new concept to replace the current general concept
            cur_prompt = instruction_prompts("propose_new_concept_to_replace_general_concept")
            full_prompt = cur_prompt[0] + cur_bkg_q + cur_prompt[1] + cur_cg_hyp + cur_prompt[2] + cur_concept + cur_prompt[3]
            structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'New Concept:'], temperature=1.0, api_type=self.args.api_type)
            assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
            cur_new_concept = structured_gene[0][1]
            print(f"Background {cur_bkg_id}, concept {cur_concept} -> new concept: {cur_new_concept}")
            
            # synthesize a new hypothesis with the replaced general concept
            cur_prompt = instruction_prompts("synthesize_new_hyp_with_replaced_general_concept")
            full_prompt = cur_prompt[0] + cur_bkg_q + cur_prompt[1] + cur_cg_hyp + cur_prompt[2] + cur_concept + cur_prompt[3] + cur_new_concept + cur_prompt[4]
            structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'New Hypothesis:'], temperature=1.0, api_type=self.args.api_type)
            assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
            cur_new_hyp = structured_gene[0][1]
            print(f"Background {cur_bkg_id}, new hypothesis: {cur_new_hyp}")
            
            results[cur_concept] = (cur_new_concept, cur_new_hyp)
        
        return bkg_id_key, results

    def synthesize_new_hyp_with_replaced_general_concept(self, max_workers=4):
        """Parallel version of synthesize_new_hyp_with_replaced_general_concept"""
        bkg_ids_to_process = []
        for cur_bkg_id in range(len(self.bkg_q_list)):
            bkg_id_key = str(cur_bkg_id)
            if bkg_id_key not in self.replaced_general_concept_list:
                bkg_ids_to_process.append(cur_bkg_id)
        
        if not bkg_ids_to_process:
            print("All replaced general concepts already processed")
            return
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_bkg_id = {executor.submit(self._process_single_synthesize_new_hyp_with_replaced_general_concept, bkg_id): bkg_id 
                               for bkg_id in bkg_ids_to_process}
            
            for future in as_completed(future_to_bkg_id):
                result = future.result()
                if result is not None:
                    bkg_id_key, results = result
                    with self.file_lock:
                        for cur_concept, (cur_new_concept, cur_new_hyp) in results.items():
                            self.replaced_general_concept_list[bkg_id_key][cur_concept] = [cur_new_concept]
                            self.cg_hyp_with_replaced_general_concept_list[bkg_id_key][cur_concept][cur_new_concept] = cur_new_hyp
                    self.save_lists()

    
    def get_ori_and_replaced_general_concepts(self, cur_bkg_id):
        print("replaced_general_concept_list: ", self.replaced_general_concept_list)
        # Convert cur_bkg_id to string since JSON keys are strings
        bkg_id_key = str(cur_bkg_id)
        cur_ori_general_concepts = list(self.replaced_general_concept_list[bkg_id_key].keys())
        # cur_replaced_general_concepts: [['conductive polymer', 'bb'], ['solvent effects', 'solvent effects']], we want to make it to be ['conductive polymer', 'bb', 'solvent effects', 'solvent effects']
        cur_replaced_general_concepts = list(self.replaced_general_concept_list[bkg_id_key].values())
        cur_replaced_general_concepts = [cur_replaced_general_concept for cur_replaced_general_concept_list in cur_replaced_general_concepts for cur_replaced_general_concept in cur_replaced_general_concept_list]
        print("cur_ori_general_concepts: ", cur_ori_general_concepts)
        print("cur_replaced_general_concepts: ", cur_replaced_general_concepts)
        cur_ori_and_replaced_general_concepts = cur_ori_general_concepts + cur_replaced_general_concepts
        return cur_ori_and_replaced_general_concepts
    

    def _process_single_naming_specific_concept(self, cur_bkg_id):
        """Process a single background question for naming specific concepts"""
        bkg_id_key = str(cur_bkg_id)
        if bkg_id_key in self.specific_concept_list:
            return None
            
        # Initialize the lists
        with self.file_lock:
            if bkg_id_key not in self.specific_concept_list:
                self.specific_concept_list[bkg_id_key] = {}
        
        cur_bkg_q = self.bkg_q_list[cur_bkg_id]
        cur_cg_hyp = self.dict_bkg2cg_hyp[cur_bkg_q]
        cur_ori_and_replaced_general_concepts = self.get_ori_and_replaced_general_concepts(cur_bkg_id)
        
        results = {}
        for cur_concept in cur_ori_and_replaced_general_concepts:
            # propose a list of specific concepts that belong to the general concept and are the most relevant to the research question from that general concept
            cur_prompt = instruction_prompts("naming_specific_concept")
            full_prompt = cur_prompt[0] + cur_bkg_q + cur_prompt[1] + cur_cg_hyp + cur_prompt[2] + cur_concept + cur_prompt[3]
            structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Specific Concepts:'], temperature=1.0, api_type=self.args.api_type)
            assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
            cur_specific_concepts = structured_gene[0][1]
            print(f"Background {cur_bkg_id}, concept {cur_concept} -> specific concepts: {cur_specific_concepts}")
            cur_specific_concepts = cur_specific_concepts.strip().split(",")
            cur_specific_concepts = [cur_specific_concept.strip() for cur_specific_concept in cur_specific_concepts]
            # include the original general concept in the specific concepts
            cur_specific_concepts = [cur_concept] + cur_specific_concepts
            results[cur_concept] = cur_specific_concepts
        
        return bkg_id_key, results

    def naming_specific_concept(self, max_workers=4):
        """Parallel version of naming_specific_concept"""
        bkg_ids_to_process = []
        for cur_bkg_id in range(len(self.bkg_q_list)):
            bkg_id_key = str(cur_bkg_id)
            if bkg_id_key not in self.specific_concept_list:
                bkg_ids_to_process.append(cur_bkg_id)
        
        if not bkg_ids_to_process:
            print("All specific concepts already named")
            return
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_bkg_id = {executor.submit(self._process_single_naming_specific_concept, bkg_id): bkg_id 
                               for bkg_id in bkg_ids_to_process}
            
            for future in as_completed(future_to_bkg_id):
                result = future.result()
                if result is not None:
                    bkg_id_key, results = result
                    with self.file_lock:
                        for cur_concept, cur_specific_concepts in results.items():
                            self.specific_concept_list[bkg_id_key][cur_concept] = cur_specific_concepts
                    self.save_lists()


    def _process_single_synthesize_new_hyp_with_specific_concept(self, cur_bkg_id):
        """Process a single background question for synthesizing new hypotheses with specific concepts"""
        bkg_id_key = str(cur_bkg_id)
        if bkg_id_key in self.hyp_with_specific_concept_list:
            return None
            
        cur_ori_and_replaced_general_concepts = self.get_ori_and_replaced_general_concepts(cur_bkg_id)
        # Initialize the lists
        with self.file_lock:
            if bkg_id_key not in self.hyp_with_specific_concept_list:
                self.hyp_with_specific_concept_list[bkg_id_key] = {}
                for cur_concept in cur_ori_and_replaced_general_concepts:
                    self.hyp_with_specific_concept_list[bkg_id_key][cur_concept] = {}
        
        cur_bkg_q = self.bkg_q_list[cur_bkg_id]
        cur_cg_hyp = self.dict_bkg2cg_hyp[cur_bkg_q]
        
        results = {}
        for cur_concept in cur_ori_and_replaced_general_concepts:
            cur_specific_concepts = self.specific_concept_list[bkg_id_key][cur_concept]
            results[cur_concept] = {}
            for cur_specific_concept in cur_specific_concepts:
                if cur_specific_concept == cur_concept:
                    results[cur_concept][cur_specific_concept] = cur_cg_hyp
                    continue
                # synthesize a new hypothesis with the specific concept
                cur_prompt = instruction_prompts("synthesize_new_hyp_with_specific_concept")
                full_prompt = cur_prompt[0] + cur_bkg_q + cur_prompt[1] + cur_cg_hyp + cur_prompt[2] + cur_concept + cur_prompt[3] + cur_specific_concept + cur_prompt[4]
                structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'New Hypothesis:'], temperature=1.0, api_type=self.args.api_type)
                assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
                cur_new_hyp = structured_gene[0][1]
                print(f"Background {cur_bkg_id}, concept {cur_concept}, specific {cur_specific_concept} -> new hypothesis: {cur_new_hyp}")
                results[cur_concept][cur_specific_concept] = cur_new_hyp
        
        return bkg_id_key, results

    def synthesize_new_hyp_with_specific_concept(self, max_workers=4):
        """Parallel version of synthesize_new_hyp_with_specific_concept"""
        bkg_ids_to_process = []
        for cur_bkg_id in range(len(self.bkg_q_list)):
            bkg_id_key = str(cur_bkg_id)
            if bkg_id_key not in self.hyp_with_specific_concept_list:
                bkg_ids_to_process.append(cur_bkg_id)
        
        if not bkg_ids_to_process:
            print("All specific concept hypotheses already synthesized")
            return
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_bkg_id = {executor.submit(self._process_single_synthesize_new_hyp_with_specific_concept, bkg_id): bkg_id 
                               for bkg_id in bkg_ids_to_process}
            
            for future in as_completed(future_to_bkg_id):
                result = future.result()
                if result is not None:
                    bkg_id_key, results = result
                    with self.file_lock:
                        for cur_concept, concept_results in results.items():
                            for cur_specific_concept, cur_new_hyp in concept_results.items():
                                self.hyp_with_specific_concept_list[bkg_id_key][cur_concept][cur_specific_concept] = cur_new_hyp
                    self.save_lists()


    def _process_single_scoring_each_hyp_with_specific_concept(self, cur_bkg_id):
        """Process a single background question for scoring hypotheses with specific concepts"""
        bkg_id_key = str(cur_bkg_id)
        if bkg_id_key in self.hyp_with_specific_concept_score_list:
            return None
            
        cur_ori_and_replaced_general_concepts = self.get_ori_and_replaced_general_concepts(cur_bkg_id)
        # Initialize the lists
        with self.file_lock:
            if bkg_id_key not in self.hyp_with_specific_concept_score_list:
                self.hyp_with_specific_concept_score_list[bkg_id_key] = {}
                for cur_concept in cur_ori_and_replaced_general_concepts:
                    self.hyp_with_specific_concept_score_list[bkg_id_key][cur_concept] = {}
        
        cur_bkg_q = self.bkg_q_list[cur_bkg_id]
        cur_cg_hyp = self.dict_bkg2cg_hyp[cur_bkg_q]
        
        results = {}
        for cur_concept in cur_ori_and_replaced_general_concepts:
            cur_specific_concepts = self.specific_concept_list[bkg_id_key][cur_concept]
            results[cur_concept] = {}
            for cur_specific_concept in cur_specific_concepts:
                cur_new_hyp = self.hyp_with_specific_concept_list[bkg_id_key][cur_concept][cur_specific_concept]
                cur_prompt = instruction_prompts("scoring_each_hyp_with_specific_concept")
                full_prompt = cur_prompt[0] + cur_bkg_q + cur_prompt[1] + cur_new_hyp + cur_prompt[2]
                structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Score:'], temperature=1.0, api_type=self.args.api_type)
                assert len(structured_gene) == 1 and len(structured_gene[0]) == 2
                cur_score = structured_gene[0][1]
                print(f"Background {cur_bkg_id}, concept {cur_concept}, specific {cur_specific_concept} -> score: {cur_score}")
                results[cur_concept][cur_specific_concept] = cur_score
        
        return bkg_id_key, results

    def scoring_each_hyp_with_specific_concept(self, max_workers=4):
        """Parallel version of scoring_each_hyp_with_specific_concept"""
        bkg_ids_to_process = []
        for cur_bkg_id in range(len(self.bkg_q_list)):
            bkg_id_key = str(cur_bkg_id)
            if bkg_id_key not in self.hyp_with_specific_concept_score_list:
                bkg_ids_to_process.append(cur_bkg_id)
        
        if not bkg_ids_to_process:
            print("All hypothesis scores already computed")
            return
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_bkg_id = {executor.submit(self._process_single_scoring_each_hyp_with_specific_concept, bkg_id): bkg_id 
                               for bkg_id in bkg_ids_to_process}
            
            for future in as_completed(future_to_bkg_id):
                result = future.result()
                if result is not None:
                    bkg_id_key, results = result
                    with self.file_lock:
                        for cur_concept, concept_results in results.items():
                            for cur_specific_concept, cur_score in concept_results.items():
                                self.hyp_with_specific_concept_score_list[bkg_id_key][cur_concept][cur_specific_concept] = cur_score
                    self.save_lists()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify fundamental assumption')
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_type", type=int, default=0, help="0: openai's API toolkit; 1: azure's API toolkit")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="", help="base url for the API")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--vague_cg_hyp_path", type=str, default="./Data/processed_research_direction.json", help="store processed vague coarse-grained hypothesis")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    args = parser.parse_args()

    verify_assumption = VerifyAssumption(args)
    # verify_assumption.extract_general_concept(max_workers=args.max_workers)
    # verify_assumption.synthesize_new_hyp_with_replaced_general_concept(max_workers=args.max_workers)
    # verify_assumption.naming_specific_concept(max_workers=args.max_workers)
    # verify_assumption.synthesize_new_hyp_with_specific_concept(max_workers=args.max_workers)
    verify_assumption.scoring_each_hyp_with_specific_concept(max_workers=args.max_workers)
