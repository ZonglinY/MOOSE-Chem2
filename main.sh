# Function mode:
#   0: framework inference
#   1: pairwise comparison between two manually set hypotheses (for simple testing)
#   2: reference-based evaluation of a manually set hypothesis (for simple testing) & breaking ground truth hypothesis into their components
#   3: systematic evaluation of the generated hypotheses in terms of pairwise comparison or reference-based evaluation. 
#       Need to adjust the parameters in main() in ./Evaluation/analysis.py
#   4: preprocess coarse-grained ground truth hypothesis to more coarse one as research direction (used as input to MOOSE-Chem2)
#   5: copilot setting: obtain inputs (custom research background and research direction) for MOOSE-Chem2 and dump them into formatted file
#   6: copilot setting: display the generated fine-grained hypotheses in txt
function_mode=3
custom_research_background_and_coarse_hyp_path=./custom_research_background_and_coarse_hyp.json

model_name=gpt-4o-mini
eval_model_name=gpt-4o-mini
# api_type: 0: OpenAI; 1: Azure; 2: Gemini
api_type=
api_key=
eval_api_key=
base_url=


if [[ ${function_mode} -eq 0 ]]; then
    ## hyper-parameters
    if_hierarchical=1
    num_hierarchy=5
    locam_minimum_threshold=2
    if_multiple_llm=1
    bkg_id=0
    if_use_custom_research_background_and_coarse_hyp=1
    # updated_prompt_feb_14 / updated_prompt_mar_29 / test
    output_dir_postfix="test"

    beam_compare_mode=0
    beam_size_branching=3
    num_init_for_EU=3
    num_recom_trial_for_better_hyp=2
    if_feedback=1
    if_parallel=1
    if_use_vague_cg_hyp_as_input=1
    if_generate_with_example=1
    if_generate_with_past_failed_hyp=0
    
    ## framework inference
    if [[ ${if_hierarchical} -eq 0 ]]; then
        python -u ./Method/greedy.py \
            --bkg_id ${bkg_id} \
            --api_type ${api_type} --api_key ${api_key} --eval_api_key ${eval_api_key} --base_url ${base_url} \
            --model_name ${model_name} --eval_model_name ${eval_model_name} \
            --output_dir ./Checkpoints/greedy_${locam_minimum_threshold}_${if_feedback}_${model_name}_${eval_model_name}_if_multiple_llm_${if_multiple_llm}_if_use_vague_cg_hyp_as_input_${if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_${if_generate_with_past_failed_hyp}_bkgid_${bkg_id}_${output_dir_postfix}.json \
            --if_save 1 \
            --max_search_step 150 --locam_minimum_threshold ${locam_minimum_threshold} --if_feedback ${if_feedback} \
            --if_multiple_llm ${if_multiple_llm} --if_use_vague_cg_hyp_as_input ${if_use_vague_cg_hyp_as_input} --if_generate_with_example ${if_generate_with_example} \
            --if_generate_with_past_failed_hyp ${if_generate_with_past_failed_hyp} \
            --if_use_custom_research_background_and_coarse_hyp ${if_use_custom_research_background_and_coarse_hyp} \
            --custom_research_background_and_coarse_hyp_path ${custom_research_background_and_coarse_hyp_path}
    elif [[ ${if_hierarchical} -eq 1 ]]; then
        python -u ./Method/hierarchy_greedy.py \
            --bkg_id ${bkg_id} \
            --api_type ${api_type} --api_key ${api_key} --eval_api_key ${eval_api_key} --base_url ${base_url} \
            --model_name ${model_name} --eval_model_name ${eval_model_name} \
            --output_dir ./Checkpoints/hierarchical_greedy_${num_hierarchy}_${locam_minimum_threshold}_${if_feedback}_${num_recom_trial_for_better_hyp}_${model_name}_${eval_model_name}_beam_compare_mode_${beam_compare_mode}_beam_size_branching_${beam_size_branching}_num_init_for_EU_${num_init_for_EU}_if_multiple_llm_${if_multiple_llm}_if_use_vague_cg_hyp_as_input_${if_use_vague_cg_hyp_as_input}_if_generate_with_past_failed_hyp_${if_generate_with_past_failed_hyp}_bkgid_${bkg_id}_${output_dir_postfix}.pkl \
            --if_save 1 \
            --max_search_step 150 --locam_minimum_threshold ${locam_minimum_threshold} --if_feedback ${if_feedback} \
            --num_hierarchy ${num_hierarchy} --beam_compare_mode ${beam_compare_mode} --beam_size_branching ${beam_size_branching} \
            --num_init_for_EU ${num_init_for_EU} --num_recom_trial_for_better_hyp ${num_recom_trial_for_better_hyp} --if_parallel ${if_parallel} \
            --if_multiple_llm ${if_multiple_llm} --if_use_vague_cg_hyp_as_input ${if_use_vague_cg_hyp_as_input} --if_generate_with_example ${if_generate_with_example} \
            --if_generate_with_past_failed_hyp ${if_generate_with_past_failed_hyp} \
            --if_use_custom_research_background_and_coarse_hyp ${if_use_custom_research_background_and_coarse_hyp} \
            --custom_research_background_and_coarse_hyp_path ${custom_research_background_and_coarse_hyp_path}
    fi
elif [[ ${function_mode} -eq 1 ]]; then
    ## compare a pair of manually set hypotheses for simple testing
    python -u ./Evaluation/pairwise_compare.py --model_name ${model_name} --api_type ${api_type} --api_key ${api_key} --base_url ${base_url} \
        --eval_example_path ./Checkpoints/Backup/eval_example.json
elif [[ ${function_mode} -eq 2 ]]; then
    ## evaluate the generated hypotheses with the ground truth
    python -u ./Evaluation/evaluate.py --model_name ${model_name} --api_type ${api_type} --api_key ${api_key} --base_url ${base_url} \
        --preprocess_groundtruth_components_dir ./Checkpoints/groundtruth_hyp_components_collection.json \
        --num_compare_times 5
elif [[ ${function_mode} -eq 3 ]]; then
    ## analysis
    # --model_name ${model_name}
    python -u ./Evaluation/analysis.py --api_type ${api_type} --api_key ${api_key} --base_url ${base_url} \
        --preprocess_groundtruth_components_dir ./Checkpoints/groundtruth_hyp_components_collection.json
elif [[ ${function_mode} -eq 4 ]]; then
    ## preprocess coarse-grained ground truth hypothesis
    python -u ./Preprocessing/input_hyp_processing.py --model_name ${model_name} --api_type ${api_type} --api_key ${api_key} --base_url ${base_url} \
        --output_dir ./Data/processed_research_direction.json
elif [[ ${function_mode} -eq 5 ]]; then
    ## dump custom research background and coarse hypothesis
    python -u ./Preprocessing/custom_research_background_dumping.py --if_load_from_moosechem_ranking_file 1 \
        --moosechem_ranking_file_path ~/MOOSE-Chem/Geo/evaluation_GPT4o-mini.json \
        --custom_research_background_and_coarse_hyp_path ${custom_research_background_and_coarse_hyp_path}
elif [[ ${function_mode} -eq 6 ]]; then
    ## display the generated finegrained hypotheses in txt
    python -u ./Preprocessing/display_hypothesis.py --if_hierarchical 1 --hierarchy_id 4 \
        --start_bkg_id 0 --end_bkg_id 4  --display_txt_file_path ./finegrained_hyp.txt

fi
