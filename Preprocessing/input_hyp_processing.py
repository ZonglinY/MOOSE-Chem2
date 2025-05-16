import os, sys, argparse, json, re
from openai import OpenAI, AzureOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, preprocessing_instruction_prompts, llm_generation_while_loop


def process_note(input_string):
    # Step 1: Split by ";"
    parts = input_string.split(";")

    # Step 2: Process each part
    processed_parts = []
    for part in parts:
        # Remove "insp X:" where X can be multiple numbers separated by "/" or have a space before the number
        cleaned_part = re.sub(r'insp\s*[\d/]+:\s*', '', part)  
        # Remove "(ref id: X)", "(ref id X)", "(ref id: X/Y/Z)", or "(ref id X/Y/Z)" 
        cleaned_part = re.sub(r'\(?ref id:?[\s\d/]+\)?', '', cleaned_part)  
        cleaned_part = cleaned_part.strip()  # Remove extra spaces
        if cleaned_part:
            processed_parts.append(cleaned_part)
    return processed_parts


def process_coarse_grained_hypothesis(input_file_path, output_file_path, api_key, base_url, model_name):
    # obtain groundtruth finegrained hypothesis and experiment
    bkg_q_list, dict_bkg2survey, dict_bkg2cg_hyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note = load_chem_annotation(input_file_path)  
    # prepare API client
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompts = preprocessing_instruction_prompts("preprocess_cg_hyp_to_research_direction")
    refine_prompts = preprocessing_instruction_prompts("preprocess_cg_hyp_to_research_direction_refine")
    assert len(prompts) == 4

    research_direction_collection = {}
    for cur_bkg_id in range(len(bkg_q_list)):
        print(f"Processing {cur_bkg_id+1}/{len(bkg_q_list)}")
        cur_q = bkg_q_list[cur_bkg_id]
        cur_cg_hyp = dict_bkg2cg_hyp[cur_q]
        cur_note = dict_bkg2note[cur_q]

        cur_note_prompt = ""
        cur_note = process_note(cur_note)
        for i, note in enumerate(cur_note):
            cur_note_prompt += f"{i+1}: {note}\n"
        print("cur_note_prompt:", cur_note_prompt)

        ## first round of generation
        full_prompt = prompts[0] + cur_q + prompts[1] + cur_cg_hyp + prompts[2] + cur_note_prompt + prompts[3]
        # structured_gene: [[reason, research direction]]
        structured_gene = llm_generation_while_loop(full_prompt, model_name, client, if_structured_generation=True, template=['Reasoning Process:', 'Research direction:'], temperature=1.0)
        cur_research_direction = structured_gene[0][1]
        ## second round of generation
        full_prompt = refine_prompts[0] + cur_q + refine_prompts[1] + cur_cg_hyp + refine_prompts[2] + cur_note_prompt + refine_prompts[3] + cur_research_direction + refine_prompts[4]
        structured_gene = llm_generation_while_loop(full_prompt, model_name, client, if_structured_generation=True, template=['Reasoning Process:', 'Research direction:'], temperature=1.0)
        cur_research_direction_refine = structured_gene[0][1]

        print("cur_research_direction:", cur_research_direction)
        print("cur_research_direction_refine:", cur_research_direction_refine)
        research_direction_collection[cur_q] = cur_research_direction_refine

    with open(output_file_path, "w") as f:
        json.dump(research_direction_collection, f, indent=4)
        print(f"Coarse-grained hypothesis has been processed and saved to {output_file_path}.")








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="https://api.claudeshop.top/v1", help="base url for the API")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024_finegrained.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--output_dir", type=str, required=True, help="The path to the output file.")
    args = parser.parse_args()
    
    process_coarse_grained_hypothesis(args.chem_annotation_path, args.output_dir, args.api_key, args.base_url, args.model_name)