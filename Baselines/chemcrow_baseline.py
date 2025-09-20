from chemcrow.agents import ChemCrow
import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, llm_generation
from openai import OpenAI

vague_cg_hyp_path = "./Data/processed_research_direction.json"
chem_annotation_path = "./Data/chem_research_2024_finegrained.xlsx"
model_name = "gpt-4o-mini"
api_type = 0
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# 1: ChemCrow; 2. plain LLM
baseline_type = 2
# output path
new_gene_hyp_path = "./Checkpoints/chemcrow_baseline_gene_hyp_collection_new_new_test_baseline_type_{}.json".format(baseline_type)

if baseline_type == 1:
    chem_model = ChemCrow(model=model_name, temp=0.1, streaming=False)
elif baseline_type == 2:
    client = OpenAI(api_key=api_key, base_url=base_url)
    temperature = 0.8
else:
    raise NotImplementedError("baseline_type: {} is not implemented".format(baseline_type))


bkg_q_list, dict_bkg2survey, dict_bkg2cg_hyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note = load_chem_annotation(chem_annotation_path)

with open(vague_cg_hyp_path, "r") as f:
    dict_bkg2cg_hyp = json.load(f)




def generate_fg_hyp_with_chemcrow():

    gene_hyp_collection = []
    for bkg_id in range(len(bkg_q_list)):
        print("=========== bkg_id: ", bkg_id, "===========")
        cur_bkg_q = bkg_q_list[bkg_id]
        cur_cg_hyp = dict_bkg2cg_hyp[cur_bkg_q]
        print("cur_bkg_q: ", cur_bkg_q)
        print("cur_cg_hyp: ", cur_cg_hyp)
        cur_prompt = "Research question: " + cur_bkg_q + "\n" + "Below is a a preliminary coarse-grained research hypothesis for the research question, please help to make modifications into the coarse-grained hypothesis, to make it an effective and complete fine-grained hypothesis: " + cur_cg_hyp + "\n" + "Now, please generate a fine-grained hypothesis that contains every methodological and experimental details for the research question. "

        if baseline_type == 1:
            cur_gene = chem_model.run(cur_prompt)
        elif baseline_type == 2:
            cur_gene = llm_generation(cur_prompt, model_name, client, temperature=temperature, api_type=api_type)
        else:
            raise NotImplementedError("baseline_type: {} is not implemented".format(baseline_type))
        print("==="*10)
        print("cur_gene: ", cur_gene)
        print("==="*10)
        gene_hyp_collection.append(cur_gene)

    with open(new_gene_hyp_path, "w") as f:
        json.dump(gene_hyp_collection, f)



if __name__ == "__main__":
    generate_fg_hyp_with_chemcrow()




