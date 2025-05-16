#!/bin/bash

eval_mode=3
if_hierarchical=5
num_hierarchy=1
locam_minimum_threshold=3

beam_compare_mode=0
beam_size_branching=3
num_init_for_EU=3
num_recom_trial_for_better_hyp=2
if_feedback=1
if_parallel=1
if_multiple_llm=1
if_use_vague_cg_hyp_as_input=1

# gpt-4o
# gpt-4o-mini
model_name=gpt-4o
eval_model_name=gpt-4o
api_key=sk-
# only used by hierarchical greedy
eval_api_key=sk-
base_url=


output_dir_postfix=""
bkg_id=0


if [[ ${eval_mode} -eq 0 ]]; then
    ## framework inference
    if [[ ${if_hierarchical} -eq 0 ]]; then
        python -u ./Method/greedy.py \
            --bkg_id ${bkg_id} \
            --api_type 0 --api_key ${api_key} --eval_api_key ${eval_api_key} --base_url ${base_url} \
            --model_name ${model_name} --eval_model_name ${eval_model_name} \
            --output_dir ./Checkpoints/greedy_${locam_minimum_threshold}_${if_feedback}_${model_name}_${eval_model_name}_if_multiple_llm_${if_multiple_llm}_if_use_vague_cg_hyp_as_input_${if_use_vague_cg_hyp_as_input}_bkgid_${bkg_id}_${output_dir_postfix}.json \
            --if_save 1 \
            --max_search_step 150 --locam_minimum_threshold ${locam_minimum_threshold} --if_feedback ${if_feedback} \
            --if_multiple_llm ${if_multiple_llm} --if_use_vague_cg_hyp_as_input ${if_use_vague_cg_hyp_as_input}
    elif [[ ${if_hierarchical} -eq 1 ]]; then
        python -u ./Method/hierarchy_greedy.py \
            --bkg_id ${bkg_id} \
            --api_type 0 --api_key ${api_key} --eval_api_key ${eval_api_key} --base_url ${base_url} \
            --model_name ${model_name} --eval_model_name ${eval_model_name} \
            --output_dir ./Checkpoints/hierarchical_greedy_${num_hierarchy}_${locam_minimum_threshold}_${if_feedback}_${num_recom_trial_for_better_hyp}_${model_name}_${eval_model_name}_beam_compare_mode_${beam_compare_mode}_beam_size_branching_${beam_size_branching}_num_init_for_EU_${num_init_for_EU}_if_multiple_llm_${if_multiple_llm}_if_use_vague_cg_hyp_as_input_${if_use_vague_cg_hyp_as_input}_bkgid_${bkg_id}_${output_dir_postfix}.pkl \
            --if_save 1 \
            --max_search_step 150 --locam_minimum_threshold ${locam_minimum_threshold} --if_feedback ${if_feedback} \
            --num_hierarchy ${num_hierarchy} --beam_compare_mode ${beam_compare_mode} --beam_size_branching ${beam_size_branching} \
            --num_init_for_EU ${num_init_for_EU} --num_recom_trial_for_better_hyp ${num_recom_trial_for_better_hyp} --if_parallel ${if_parallel} \
            --if_multiple_llm ${if_multiple_llm} --if_use_vague_cg_hyp_as_input ${if_use_vague_cg_hyp_as_input}
    fi
elif [[ ${eval_mode} -eq 1 ]]; then
    ## compare a pair of manually set hypotheses for simple testing
    python -u ./Evaluation/pairwise_compare.py --model_name ${model_name} --api_type 0 --api_key ${api_key} \
        --eval_example_path ./Checkpoints/eval_example.json
elif [[ ${eval_mode} -eq 2 ]]; then
    ## evaluate the generated hypotheses with the ground truth
    python -u ./Evaluation/evaluate.py --model_name ${model_name} --api_type 0 --api_key ${api_key}\
        --preprocess_groundtruth_components_dir ./Checkpoints/groundtruth_hyp_components_collection.json \
        --num_compare_times 5
elif [[ ${eval_mode} -eq 3 ]]; then
    ## analysis
    # --model_name ${model_name}
    python -u ./Evaluation/analysis.py --api_type 0 --api_key ${api_key}\
        --preprocess_groundtruth_components_dir ./Checkpoints/groundtruth_hyp_components_collection.json \
        --num_compare_times 5
elif [[ ${eval_mode} -eq 4 ]]; then
    ## preprocess coarse-grained ground truth hypothesis
    python -u ./Preprocessing/input_hyp_processing.py --model_name ${model_name} --api_key ${api_key} --base_url ${base_url} \
        --output_dir ./Data/processed_research_direction.json
        
fi
