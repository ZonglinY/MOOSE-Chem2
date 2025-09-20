if_hierarchical=1
num_hierarchy=5
locam_minimum_threshold=2

beam_compare_mode=0
beam_size_branching=2
num_init_for_EU=3
num_recom_trial_for_better_hyp=2
if_feedback=1
if_parallel=1
if_use_vague_cg_hyp_as_input=1
if_generate_with_example=1
if_generate_with_past_failed_hyp=1
if_multiple_llm=1


model_name=gpt-4o-mini
eval_model_name=gpt-4o-mini
api_type=
api_key=
eval_api_key=
base_url=
output_dir_postfix="test"
if_use_custom_research_background_and_coarse_hyp=0
custom_research_background_and_coarse_hyp_path=./custom_research_background_and_coarse_hyp.json



for bkg_id in {0..50}
do 
    echo "\n\nEntering loop for bkg_id: $bkg_id"
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
done