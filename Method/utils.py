import os, re, json, random, time, math, sys
import pandas as pd
from google.genai import types
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.discipline_specific_prompt_chemistry import hyp_example_hierarchy_1, hyp_example_hierarchy_2, hyp_example_hierarchy_3, hyp_example_hierarchy_4, hyp_example_hierarchy_5, DISCIPLINE, INTRODUCTION_OF_HIERARCHIES, HIERARCHY_LIST, DESCRIPTION_HIERARCHY_LIST, FEEDBACK_HIERARCHY_VALIDITY_CLARITY, FEEDBACK_NON_HIERARCHY_VALIDITY_CLARITY, EXAMPLE_DESCRIPTION_FINAL_HYPOTHESIS, EXAMPLE_UPDATE_NEW_HIERARCHY
# from Method.discipline_specific_prompt_geophysics import hyp_example_hierarchy_1, hyp_example_hierarchy_2, hyp_example_hierarchy_3, hyp_example_hierarchy_4, hyp_example_hierarchy_5, DISCIPLINE, INTRODUCTION_OF_HIERARCHIES, HIERARCHY_LIST, DESCRIPTION_HIERARCHY_LIST, FEEDBACK_HIERARCHY_VALIDITY_CLARITY, FEEDBACK_NON_HIERARCHY_VALIDITY_CLARITY, EXAMPLE_DESCRIPTION_FINAL_HYPOTHESIS, EXAMPLE_UPDATE_NEW_HIERARCHY

# HYPTHESIS_GENERATION_CUSTOM_GUIDE: is added to every prompt involving hypothesis generation
HYPTHESIS_GENERATION_CUSTOM_GUIDE = '''
Please formulate a valid, feasible, novel, detailed, and constructive hypothesis, primarily emphasizing the methodology and mechanistic design. Each step in your hypothesis should be clear, precise, and free from ambiguity. The expected performance or potential impact of the hypothesis is not the main focus and should be mentioned minimally.
The generated hypothesis must not exceed 800 words, but it can be shorter if conciseness doesn't sacrifice essential details (normally 800 words should be more than enough to describe the essential idea and essential details of a hypothesis). The hypothesis must remain concise yet comprehensive, while avoiding unnecessary verbosity or redundant explanations of common scientific knowledge. If your initial hypothesis exceeds 800 words, try to compress it until it meets this constraint without omitting any critical information.
'''



## Function: used by load_chem_annotation() and load_chem_annotation_with_feedback(); used to recover background_survey_strict and background_question_strict
# background_strict_raw: a list of the raw background survey, some of them are "NA"; when it is "NA", we should find its component in background_normal
# background_normal: a list of the normal background survey, no "NA"
# background_strict_raw_nan_indicator: a list of boolean values indicating whether the corresponding background_strict_raw is "NA"
def recover_raw_background(background_strict_raw, background_normal, background_strict_raw_nan_indicator):
    background_strict = []
    for cur_survey_id, cur_survey in enumerate(background_strict_raw):
        if background_strict_raw_nan_indicator[cur_survey_id]:
            cur_value = background_normal[cur_survey_id].strip()
            background_strict.append(cur_value)
        else:
            cur_survey = cur_survey.strip()
            # this assertion is to make sure the content is not variants of "NA"
            assert len(cur_survey) > 10
            cur_value = cur_survey
            background_strict.append(cur_value)
    return background_strict


# load xlsx annotations, bkg question -> inspirations
# bkg_q: [bq0, bq1, ...]
# dict_bkg2insp: {'bq0': [insp0, insp1, ...], 'bq1': [insp0, insp1, ...], ...}
# dict_bkg2survey: {'bq0': survey0, 'bq1': survey1, ...}
def load_chem_annotation(chem_annotation_path, if_use_strict_survey_question=1, if_use_background_survey=1):
    assert if_use_strict_survey_question in [0, 1]
    assert if_use_background_survey in [0, 1]
    if if_use_background_survey == 0:
        print("Warning: Not Using Survey.")
    ## load chem_research.xlsx to know the groundtruth inspirations
    chem_annotation = pd.read_excel(chem_annotation_path, 'Overall')
    nan_values = chem_annotation.isna()
    bkg_survey = list(chem_annotation[chem_annotation.columns[4]])
    # some of the components are "NA"; if it is NA, we should find its component in bkg_survey
    bkg_survey_strict_raw = list(chem_annotation[chem_annotation.columns[5]])
    # print("bkg_survey_strict_raw: ", bkg_survey_strict_raw)
    bkg_survey_strict = recover_raw_background(bkg_survey_strict_raw, bkg_survey, nan_values[chem_annotation.columns[5]])
    bkg_q = list(chem_annotation[chem_annotation.columns[6]])
    # some of the components are "NA"; if it is NA, we should find its component in bkg_q
    bkg_q_strict_raw = list(chem_annotation[chem_annotation.columns[7]])
    bkg_q_strict = recover_raw_background(bkg_q_strict_raw, bkg_q, nan_values[chem_annotation.columns[7]])
    insp1 = list(chem_annotation[chem_annotation.columns[9]])
    insp2 = list(chem_annotation[chem_annotation.columns[11]])
    insp3 = list(chem_annotation[chem_annotation.columns[13]])
    groundtruthHyp = list(chem_annotation[chem_annotation.columns[15]])
    reasoningprocess = list(chem_annotation[chem_annotation.columns[17]])
    note = list(chem_annotation[chem_annotation.columns[18]])
    finegrained_hyp = list(chem_annotation[chem_annotation.columns[19]])
    finegrained_exp = list(chem_annotation[chem_annotation.columns[20]])
    ## determine which version of survey and question to use
    if if_use_strict_survey_question:
        bkg_survey = bkg_survey_strict
        bkg_q = bkg_q_strict
    ## start looping for collection
    dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2reasoningprocess = {}, {}, {}, {}, {}
    dict_bkg2idx, dict_idx2bkg = {}, {}
    dict_bkg2fg_hyp, dict_bkg2fg_exp = {}, {}
    for cur_b_id, cur_b in enumerate(bkg_q):
        # update bkg_q to remove leading and trailing spaces
        cur_b = cur_b.strip()
        bkg_q[cur_b_id] = cur_b
        ## dict_bkg2insp
        cur_b_insp = []
        # insp1
        if nan_values[chem_annotation.columns[9]][cur_b_id] == False:
            cur_b_insp.append(insp1[cur_b_id].strip())
        # insp2
        if nan_values[chem_annotation.columns[11]][cur_b_id] == False:
            cur_b_insp.append(insp2[cur_b_id].strip())
        # insp3
        if nan_values[chem_annotation.columns[13]][cur_b_id] == False:
            cur_b_insp.append(insp3[cur_b_id].strip())
        dict_bkg2insp[cur_b] = cur_b_insp
        ## dict_bkg2survey
        if if_use_background_survey:
            assert nan_values[chem_annotation.columns[4]][cur_b_id] == False
            dict_bkg2survey[cur_b] = bkg_survey[cur_b_id].strip()
        else:
            dict_bkg2survey[cur_b] = "Survey not provided. Please overlook the survey."
        ## dict_bkg2groundtruthHyp
        assert nan_values[chem_annotation.columns[15]][cur_b_id] == False
        dict_bkg2groundtruthHyp[cur_b] = groundtruthHyp[cur_b_id].strip()
        ## dict_bkg2reasoningprocess
        assert nan_values[chem_annotation.columns[17]][cur_b_id] == False
        dict_bkg2reasoningprocess[cur_b] = reasoningprocess[cur_b_id].strip()
        ## dict_bkg2note
        assert nan_values[chem_annotation.columns[18]][cur_b_id] == False
        dict_bkg2note[cur_b] = note[cur_b_id].strip()
        ## dict_bkg2idx, dict_idx2bkg
        dict_bkg2idx[cur_b] = cur_b_id
        dict_idx2bkg[cur_b_id] = cur_b
        ## dict_bkg2fg_hyp, dict_bkg2fg_exp
        dict_bkg2fg_hyp[cur_b] = finegrained_hyp[cur_b_id]
        dict_bkg2fg_exp[cur_b] = finegrained_exp[cur_b_id]
    # return bkg_q, dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2idx, dict_idx2bkg, dict_bkg2reasoningprocess
    return bkg_q, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2fg_hyp, dict_bkg2fg_exp, dict_bkg2note



# Call Openai API,k input is prompt, output is response
def llm_generation(prompt, model_name, client, temperature=1.0, api_type=0):
    # print("prompt: ", prompt)
    if "claude-3-haiku" in model_name:
        max_tokens = 4096
    else:
        max_tokens = 8192
    cnt_max_trials = 1
    # start inference util we get generation
    for cur_trial in range(cnt_max_trials):
        try:
            if api_type in [0, 1]:
                completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ]
                )
                generation = completion.choices[0].message.content.strip()
            # google client
            elif api_type == 2:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                generation = response.text.strip()
            else:
                raise NotImplementedError
            break
        except Exception as e:
            print("API Error occurred: ", e)
            time.sleep(0.25)
            if cur_trial == cnt_max_trials - 1:
                raise Exception("Failed to get generation after {} trials because of API error: {}.".format(cnt_max_trials, e))
    # print("generation: ", generation)
    return generation



# =============================================================================
# DEPRECATED: Old extraction functions (commented out)
# These functions are replaced by the new marker-based extraction functions
# that work with both reasoning and non-reasoning LLMs
# =============================================================================

'''
# gene: (generated) text; '#' and '*' will be removed from gene, since they are assumed to be generated by LLM as markdown format --- this format can result in not exact match between the title extracted from generation and the groundtruth title in the benchmark
# template: ['Title:', 'Reason:']
# structured_gene: [[Title, Reason], ...]
def get_structured_generation_from_raw_generation(gene, template):
    # use .strip("#") to remove the '#' or "*" in the gene (the '#' or "*" is usually added by the LLM as a markdown format); used to match text (eg, title)
    gene = re.sub("[#*]", "", gene).strip()
    assert len(template) == 2, print("template: ", template)
    # # some times generation will capitalize the first letter of the template, so we use the lower case for both generation and template to match: not adopting it since it might influence chemistry terms (e.g., Fe2+ -> fe2+)
    # gene = gene.lower()
    # template = [item.lower() for item in template]
    # the template might not appear in the first sentence of the gene, get rid of noise sentences before the first template[0]
    if not gene.startswith(template[0]):
        gene_split = gene.split('\n')
        # if the gene is not starting with the title, the second paragraph in gene_split might be the title
        gene_split = [item for item in gene_split if item.strip() != ""]
        assert len(gene_split) >= 2, print("gene_split: ", gene_split)
        # iterate to find the first template[0] in gene_split
        for id_line, line in enumerate(gene_split):
            if gene_split[id_line].find(template[0]) > 0 and gene_split[id_line].find(template[0]) < 15:
                gene_split_split = gene_split[id_line].split(template[0])
                assert len(gene_split_split) == 2, print("gene_split_split: ", gene_split_split)
                gene_split[id_line] = template[0] + gene_split_split[1]
            if gene_split[id_line].startswith(template[0]):
                gene = '\n'.join(gene_split[id_line:])
                break
        # assert gene.startswith(template[0]), print("gene: ", gene)
        assert gene.startswith(template[0])
    # structured_gene: [[title, reason], [title, reason], ...]
    structured_gene = []
    gene_split = gene.split(template[0])
    # split to every title block, including one title and one reason
    for cur_gs in gene_split:
        # split the one title and one reason
        cur_gs = cur_gs.strip()
        if cur_gs == "":
            continue
        cur_gs_split = cur_gs.split(template[1])
        # deal with unexpected situations
        if len(cur_gs_split) > 2:
            # if there are more than one template[1] in cur_gs, we prefer the one with prefix as '\n' (since it matches more as the designed format)
            cur_gs_split = cur_gs.split('\n' + template[1])
            # by preferring the one with prefix as '\n' still do not work, so we simply concatenate the rest of the elements other than the first element
            if len(cur_gs_split) > 2:
                cur_gs_split = [cur_gs_split[0], '\n'.join(cur_gs_split[1:])]
            # in case none of the template[1] is with prefix as '\n'
            elif len(cur_gs_split) == 1:
                cur_gs_split = cur_gs.split(template[1])
                cur_gs_split = [cur_gs_split[0], '\n'.join(cur_gs_split[1:])]
        # assert len(cur_gs_split) == 2, print("cur_gs_split: ", cur_gs_split)
        assert len(cur_gs_split) == 2
        # strip every elements in cur_gs_split
        for i in range(len(cur_gs_split)):
            cur_gs_split[i] = cur_gs_split[i].strip().strip(";").strip()
        structured_gene.append(cur_gs_split)
    return structured_gene





def get_structured_generation_from_raw_generation_by_llm(gene, template, client, temperature, model_name, api_type):
    assert isinstance(gene, str), print("type(gene): ", type(gene))
    # use .strip("#") to remove the '#' or "*" in the gene (the '#' or "*" is usually added by the LLM as a markdown format); used to match text (eg, title)
    gene = re.sub("[#*]", "", gene).strip()
    assert len(template) == 2, print("template: ", template)
    # In your answer, please only mention the words in the template when use it as a template. For example, if the template is ['Hypothesis:', 'Reasoning Process:'], then your answer should not contain 'Analysis of the Hypothesis:', since it also contain 'Hypothesis:'.
    # Whenever there are information in the passage related to the template, please restructure the information into the template format;
    prompt = "You are a helpful assistant.\nPlease help to organize the following passage into a structured format, following the template. When restructure the passage with the template, please try not to rephrase but to use the original information in the passage (to avoid information distortion). If the template is only about a subset of information in the passage, you can extract only that subset of information to fill the template. If there is no such information for the template in the passage, please still output the exact template first, and fill the content for the template as 'None'. \n\nThe passage is: \n" + gene + f"\n\nThe template is: \n{template[0]} \n{template[1]} \n. Now, please restructure the passage strictly with the template (literally strictly, e.g., the case style of the template should also remain the same when used to restructure the passage)."
    # print("prompt: ", prompt)
    
    # while loop to make sure there will be one successful generation
    max_trials = 10
    for cur_trial in range(max_trials):
        try:
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            # print("generation (in): ", generation)
            structured_gene = get_structured_generation_from_raw_generation(generation, template=template)
            # print("structured_gene (in): ", structured_gene)
            return structured_gene
        except Exception as e:
            if temperature < 2.0:
                temperature += 0.25
            # Q: do not change to more powerful model, since different users might have different model_name (even for the same model)
            # if temperature >= 0.7:
            #     model_name = "gpt-4o"
            # if the format of feedback is wrong, try again in the while loop
            print("generation (in): ", generation)
            print("template: ", template)
            print("Exception (in): {}, try again..".format(repr(e)))
            print(f"update temperature to {temperature} and use {model_name} for extraction in case new generation can be successful..")
    # print("structured_gene: ", structured_gene)
    raise Exception("Failed to restructure the passage with the template after {} trials.".format(max_trials))


    



## Function:
#   llm inference with the prompt + guarantee to reply a structured generation accroding to the template (guarantee by the while loop)
#   gene_format_constraint: [id of structured gene to comply with the constraint, constraint (['Yes', 'No'], where the content in the id of structured gene should be inside the constraint)]
#   if_only_return_one_structured_gene_component: True or False; most of the time structured_gene will only have one component (eg, [[hyp, reasoning process]]). When it is True, this function will only return the first element of structured_gene. If it is set to true and structured_gene has more than one component, a warning will be raised
#   restructure_output_model_name: the model name used to extract structured generation if the original generation does not match the template. It is set in case some used model (model_name) is not powerful enough to follow the template, and in this case we can still extract the desired structured generation by using a more powerful model (restructure_output_model_name) to extract the structured generation from the original generation
def llm_generation_while_loop(prompt, model_name, client, if_structured_generation=False, template=None, gene_format_constraint=None, if_only_return_one_structured_gene_component=False, temperature=1.0, restructure_output_model_name=None, api_type=0):
    # assertions
    assert if_structured_generation in [True, False]
    if if_structured_generation:
        assert template is not None
    if restructure_output_model_name == None:
        restructure_output_model_name = model_name
    else:
        if restructure_output_model_name != model_name:
            print(f"Warning: restructure_output_model_name is set to {restructure_output_model_name}, which is different from model_name: {model_name}.")

    # while loop to make sure there will be one successful generation
    cnt_max_trials = 5
    generation = None
    for cur_trial in range(cnt_max_trials):
        try:
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            # print("generation: ", generation)
            # structured_gene
            if if_structured_generation:
                # structured_gene: [[title, reason], [title, reason], ...]
                # try with template matching first, if not work, try llm to formulate the generation according to the template; if not work again, then probably it is the problem of the original generation, then try llm_generation() again
                try:
                    # print("Using a template matching method to extract information from the LLM's generation")
                    structured_gene = get_structured_generation_from_raw_generation(generation, template=template)
                except:
                    # print("Information to be extracted by an LLM from the LLM's generation")
                    structured_gene = get_structured_generation_from_raw_generation_by_llm(generation, template=template, client=client, temperature=temperature, model_name=restructure_output_model_name, api_type=api_type)
                if gene_format_constraint != None:
                    assert len(gene_format_constraint) == 2, print("gene_format_constraint: ", gene_format_constraint)
                    # we use structured_gene[0] here since most of the time structured_gene will only have one component (eg, [[hyp, reasoning process]])
                    assert structured_gene[0][gene_format_constraint[0]].strip() in gene_format_constraint[1], print("structured_gene[0][gene_format_constraint[0]].strip(): {}; gene_format_constraint[1]: {}".format(structured_gene[0][gene_format_constraint[0]].strip(), gene_format_constraint[1]))
                # print("structured_gene: ", structured_gene)
            break
        except Exception as e:
            # if the format of feedback is wrong, try again in the while loop
            print("generation: ", generation)
            print("AssertionError: {}, try again..".format(repr(e)))
            if cur_trial == cnt_max_trials - 1:
                raise Exception("Failed to get generation after {} trials because of Error: {}.".format(cnt_max_trials, e))

    # structured_gene
    if if_structured_generation:
        if if_only_return_one_structured_gene_component:
            if len(structured_gene) > 1:
                print("Warning: structured_gene has more than one component: ", structured_gene)
            return structured_gene[0]
        else:
            return structured_gene
    else:
        return generation
'''
    

# -------------------------------------------------------------
# New extraction functions for marker-based extraction
# -------------------------------------------------------------

def extract_between_markers(source: str, label_regex: str):
    """Return text between '<label> starts' and '<label> ends'.

    Parameters
    ----------
    source : str
        The raw LLM response.
    label_regex : str
        A *REGEX* describing the label (e.g. 'Research\\s*question' or
        'Inspiration\\s+1'). It should NOT contain the 'starts'/'ends'
        keywords; they are added internally.

    Returns
    -------
    str | None
        The extracted content with compacted whitespace, or None if the
        pattern is not found.
    """
    # Remove markdown emphasis to simplify matching.
    plain = re.sub(r'[\*_]+', '', source)

    pattern = rf'{label_regex}\s*starts\s*:?\s*([\s\S]+?)\s*{label_regex}\s*ends'

    m = re.search(pattern, plain, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    content = m.group(1).strip()
    return content if content else None


def extract_reasoning_model_content(text):
    """
    Extract the actual answer from reasoning models that use <think> or <answer> tags.
    Also removes thinking patterns for cleaner extraction.
    Returns the content after the thinking section.
    
    Compatible with various reasoning models including DeepSeek-R1, o1, etc.
    """
    import re
    
    # Keep original text in case we need to return it
    original_text = text
    
    # First, check if this is DeepSeek-R1 format with <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        # Found answer tags - this is DeepSeek-R1 format
        text = answer_match.group(1).strip()
    else:
        # Handle malformed responses where content comes after </think>
        after_think_match = re.search(r'</think>\s*\n(.+)', text, re.DOTALL | re.IGNORECASE)
        if after_think_match:
            # Extract everything after </think>
            text = after_think_match.group(1).strip()
        else:
            # Remove any <think> content (complete or partial)
            patterns_to_remove = [
                r'<think>.*?</think>',  # Complete think tags
                r'<think>.*$',  # Unclosed think tag at end
                r'^.*</think>',  # Everything up to and including </think>
                r'</think>?\s*$',  # Trailing closing tag (with optional typo)
                r'</think>',  # Any remaining closing tags
                r'<think>',  # Any remaining opening tags
            ]
            
            cleaned_text = text
            for pattern in patterns_to_remove:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
            text = cleaned_text
    
    # Check if we have marked sections (e.g., "**Field starts:**")
    has_marked_sections = bool(re.search(r'\*\*\w+.*(?:starts|ends)\*\*', text))
    
    if not has_marked_sections:
        # Only apply cleanup if we don't have marked sections
        # Remove common thinking/explanation patterns at the end
        thinking_patterns = [
            r',\s*as\s+requested\.\s*$',
            r'\.\s*This\s+precisely.*$',
            r'\.\s*I\'ll\s+.*$',
            r'\.\s*Let\s+me\s+.*$',
            r'without\s+additional\s+commentary.*$',
        ]
        
        for pattern in thinking_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Final cleanup
    text = text.strip()
    
    # Return cleaned text, or original if cleaning resulted in empty string
    return text if text else original_text


def extract_field(text, field_name, expected_type='text', strict_extraction=False):
    """Universal field extraction with type awareness.
    
    Args:
        text: The LLM response text
        field_name: The field to extract (e.g., "Hypothesis", "Answer", "Redundant")
        expected_type: 'text', 'bool'/'yes_no', 'number', etc.
        strict_extraction: If True, only use marker extraction (no fallbacks)
    
    Returns:
        Extracted value in appropriate type, or None if extraction fails
    """
    import re
    
    # Clean up reasoning model output (remove <think> tags)
    cleaned_text = extract_reasoning_model_content(text)
    
    # Try marker extraction first (works for both types of models after cleaning)
    result = extract_between_markers(cleaned_text, field_name)
    
    # If strict extraction and no marker found, return None
    if strict_extraction and not result:
        return None
    
    # Process based on expected type
    if expected_type in ['bool', 'yes_no', 'boolean']:
        if result:
            result_lower = result.lower().strip()
            if result_lower in ['yes', 'true', '1', 'correct', 'valid']:
                return True
            elif result_lower in ['no', 'false', '0', 'incorrect', 'invalid']:
                return False
        
        # Simple fallback: check start of cleaned text
        if not strict_extraction:
            text_lower = cleaned_text.lower().strip()[:100]
            if any(word in text_lower for word in ['yes', 'true', 'correct']):
                return True
            if any(word in text_lower for word in ['no', 'false', 'incorrect']):
                return False
        return None
    
    elif expected_type == 'number':
        if result:
            numbers = re.findall(r'\d+', result)
            if numbers:
                return int(numbers[0])
        
        # Simple fallback: look for number in cleaned text
        if not strict_extraction:
            numbers = re.findall(r'\b(\d+)\b', cleaned_text[:200])
            if numbers:
                return int(numbers[0])
        return None
    
    else:  # Default to text extraction
        if result:
            return result.strip()
        
        # Simple fallback: look for pattern "field_name: value"
        if not strict_extraction:
            escaped_field = re.escape(field_name)
            # Try simple colon pattern
            pattern = rf"{escaped_field}[:\s]+(.+?)(?:\n|$)"
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Remove common markers
                value = re.sub(r'^\*+|\*+$', '', value).strip()
                value = re.sub(r'^["\']|["\']$', '', value).strip()
                return value
        
        return None


# Helper function for extracting multiple numbered field pairs
def llm_generation_with_multiple_extractions(prompt, model_name, client, repeating_field_pattern, max_items=10, temperature=1.0, api_type=0, max_retries=3):
    """
    Generate LLM response and extract multiple numbered field pairs.
    Useful for extracting lists like Title 1/Reason 1, Title 2/Reason 2, etc.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: The model to use
        client: The API client
        repeating_field_pattern: List of tuples defining the pattern that repeats for each numbered item
                                 [(field_name, expected_type), ...]
                                 e.g., [("Title", "text"), ("Reason", "text")] 
                                 will extract Title 1/Reason 1, Title 2/Reason 2, etc.
        max_items: Maximum number of items to try extracting (default 10)
        temperature: Temperature for generation
        api_type: API type (0=OpenAI, 1=Azure, 2=Google)
        max_retries: Maximum number of retries if extraction fails
    
    Returns:
        List of extracted item dictionaries, e.g., [{"Title": "...", "Reason": "..."}, ...]
    """
    for attempt in range(max_retries):
        try:
            # Generate response
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            
            # Extract numbered field pairs
            extracted_items = []
            for i in range(1, max_items + 1):
                item = {}
                all_fields_found = True
                
                for field_name, expected_type in repeating_field_pattern:
                    # Try to extract this numbered field
                    value = extract_field(generation, f"{field_name} {i}", expected_type=expected_type, strict_extraction=True)
                    if value is None:
                        all_fields_found = False
                        break
                    item[field_name] = value
                
                # If we found all fields for this item, add it
                if all_fields_found:
                    extracted_items.append(item)
                else:
                    # Stop looking for more items once we miss one
                    break
            
            # If we got at least one complete item, consider it successful
            if extracted_items:
                return extracted_items
            
            # If this was the last attempt, return empty list
            if attempt == max_retries - 1:
                print(f"Warning: No complete items could be extracted after {max_retries} attempts.")
                return []
            
            # Otherwise, retry with slightly higher temperature
            temperature = min(temperature + 0.1, 1.5)
            print(f"No items extracted (attempt {attempt + 1}/{max_retries}), retrying...")
            
        except Exception as e:
            print(f"Error during extraction attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return []
    
    return []


# New function to replace llm_generation_while_loop with simpler extraction
def llm_generation_with_extraction(prompt, model_name, client, expected_fields=None, temperature=1.0, api_type=0, max_retries=15):
    """
    Generate LLM response and extract structured fields using marker-based extraction.
    Includes retry logic if extraction fails.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: The model to use
        client: The API client
        expected_fields: List of tuples [(field_name, expected_type), ...]
                        e.g., [("Title", "text"), ("Reason", "text")]
                        Supported types: "text", "number", "yes_no" (or "bool"/"boolean")
        temperature: Temperature for generation
        api_type: API type (0=OpenAI, 1=Azure, 2=Google)
        max_retries: Maximum number of retries if extraction fails (default 15)
    
    Returns:
        If expected_fields is provided: dict with extracted fields
        Otherwise: raw generation text
    """
    # If no expected fields, just return raw generation
    if not expected_fields:
        generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
        return generation
    
    # Validate expected_fields format
    if not isinstance(expected_fields, list):
        raise ValueError("expected_fields must be a list of tuples [(field_name, expected_type), ...]")
    for field_info in expected_fields:
        if not isinstance(field_info, tuple) or len(field_info) != 2:
            raise ValueError(f"Each field must be a tuple of (field_name, expected_type). Got: {field_info}")
        field_name, expected_type = field_info
        if expected_type not in ['text', 'number', 'yes_no', 'bool', 'boolean']:
            raise ValueError(f"Unsupported expected_type '{expected_type}' for field '{field_name}'. "
                           f"Supported types: 'text', 'number', 'yes_no' (or 'bool'/'boolean')")
    
    # Try extraction with retries
    for attempt in range(max_retries):
        try:
            # Generate response
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            
            # Extract fields
            extracted = {}
            all_fields_extracted = True
            
            for field_name, expected_type in expected_fields:
                value = extract_field(generation, field_name, expected_type=expected_type, strict_extraction=True)
                extracted[field_name] = value
                
                # Check if extraction failed (None or empty for text fields)
                if value is None or (expected_type == 'text' and value == ""):
                    all_fields_extracted = False
                    print(f"Warning: Failed to extract field '{field_name}' (attempt {attempt + 1}/{max_retries})")
            
            # If all required fields were extracted successfully, return
            if all_fields_extracted:
                return extracted
            
            # If this was the last attempt, return what we got
            if attempt == max_retries - 1:
                print(f"Warning: Some fields could not be extracted after {max_retries} attempts.")
                return extracted
            
            # Otherwise, retry with slightly higher temperature
            temperature = min(temperature + 0.1, 1.5)
            
        except Exception as e:
            print(f"Error during extraction attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                # On final attempt, return empty dict for all fields
                return {field_name: None for field_name, _ in expected_fields}
    
    # Should not reach here, but return empty dict as fallback
    return {field_name: None for field_name, _ in expected_fields}


# =============================================================================
# Compatibility wrapper for old code using llm_generation_while_loop
# This wrapper converts old-style calls to use the new extraction functions
# =============================================================================

def llm_generation_while_loop(prompt, model_name, client, if_structured_generation=False, template=None, gene_format_constraint=None, if_only_return_one_structured_gene_component=False, temperature=1.0, restructure_output_model_name=None, api_type=0):
    """
    Compatibility wrapper for old code that uses llm_generation_while_loop.
    Converts to use the new llm_generation_with_extraction function.
    """
    if not if_structured_generation:
        # Simple generation without extraction
        return llm_generation_with_extraction(prompt, model_name, client, expected_fields=None, temperature=temperature, api_type=api_type)
    
    # Convert template to expected_fields format
    if template and len(template) == 2:
        # Map old template format to new expected_fields format
        # Old format: ['Reasoning Process:', 'Revised Hypothesis:']
        # New format: [("Reasoning Process", "text"), ("Revised Hypothesis", "text")]
        field1_name = template[0].rstrip(':').strip()
        field2_name = template[1].rstrip(':').strip()
        expected_fields = [(field1_name, "text"), (field2_name, "text")]
        
        # Call new extraction function
        result = llm_generation_with_extraction(prompt, model_name, client, expected_fields=expected_fields, temperature=temperature, api_type=api_type, max_retries=15)
        
        # Convert result to old format [[field1_value, field2_value]]
        if result and field1_name in result and field2_name in result:
            structured_gene = [[result[field1_name], result[field2_name]]]
            
            # Apply gene_format_constraint if provided
            if gene_format_constraint is not None:
                if len(gene_format_constraint) == 2:
                    constraint_idx = gene_format_constraint[0]
                    constraint_values = gene_format_constraint[1]
                    if structured_gene[0][constraint_idx].strip() not in constraint_values:
                        raise Exception(f"Constraint not satisfied: {structured_gene[0][constraint_idx].strip()} not in {constraint_values}")
            
            # Return based on if_only_return_one_structured_gene_component
            if if_only_return_one_structured_gene_component:
                return structured_gene[0]
            else:
                return structured_gene
        else:
            # Fallback to empty structure if extraction failed
            if if_only_return_one_structured_gene_component:
                return ["", ""]
            else:
                return [["", ""]]
    else:
        # Unsupported template format, fall back to simple generation
        return llm_generation_with_extraction(prompt, model_name, client, expected_fields=None, temperature=temperature, api_type=api_type)


# Input:
#   input_list: [[item0, item1], [item0, item1], ...]
# Output:
#   output_list: [[item1, item0], [item1, item0], ...]
def exchange_order_in_list(input_list):
    output_list = []
    for cur_input_list in input_list:
        if isinstance(cur_input_list, list):
            assert len(cur_input_list) == 2
            output_list.append(cur_input_list[::-1])
        elif isinstance(cur_input_list, str):
            assert len(input_list) == 2
            output_list = input_list[::-1]
            break
        else:
            raise ValueError("Invalid input type. Expected list or string.")
    return output_list


def get_first_number_from_string(input_string):
    if isinstance(input_string, str):
        match = re.search(r'\d+', input_string)
        if match:
            return match.group()
        else:
            return None
    else:
        return None





# A collection of prompts for different modules
def instruction_prompts(module_name, assist_info=None):
    num_major_details_tradition = "eight"
    if module_name == "greedy_search_first_step":
        # assist_info: [cur_hierarchy_id, if_generate_with_example]
        assert len(assist_info) == 2
        assert assist_info[0] in [None, 0] and assist_info[1] in [0, 1], print("Exception: assist_info: ", assist_info)

        if assist_info[1] == 1:
            assist_example = f"Taking an organic {DISCIPLINE} example, the final fine-grained hypothesis we want to reach should look like this example: \n" + hyp_example_hierarchy_5 + "\nIn this example, there are five major components, and sufficient details and experimental conditions for each major component."
        else:
            assist_example = ""

        # check whether assist_example is consistent with assist_info[1] (if_generate_with_example)
        assert assist_example == "" if assist_info[1] == 0 else assist_example != "" and len(assist_example) > 0, print("assist_example: ", assist_example)
            
        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, and a preliminary coarse-grained research hypothesis for the research question, please help to make modifications into the coarse-grained hypothesis, to make it one step closer to a more effective and more complete fine-grained hypothesis. Please ensure the improved hypothesis directly addresses the given research question, rather than shifting focus to a different question in a similar domain. \
                   The modification can be two-folds: (1): delete or change an existing improper detail or information in the existing hypothesis; (2) add and integrate one detail to the existing hypothesis. If you choose to add a detail, do not simply append new information to the existing hypothesis. Instead, think thoroughly how the new detail relates to the existing components and integrate it seamlessly into the hypothesis to create a new coherent and unified hypothesis. In addition if you choose to add a detail to a general information, if the corresponding general information is correct, you should try to keep the corresponding general information in the updated hypothesis and also mention the details, instead of replacing the general information with the details. In this way, it would be much easier for scientists to understand both the general infomration/structure and the details from your generated hypothesis. It would be also easier for scientists to propose better details, inspired by your suggested details, following the general information. \
                   Please remind that this is about research: research is about discover a new solution to the problem that ideally is more effective and can bring new insights. Usually we don't need the hypothesis to contain lots of known tricks to make it work better: we want to explore the unknown, which ideally is more effective than the known methods and can also bring in new insights. Therefore, a research hypothesis is usually about a small set (usually less than {num_major_details_tradition}) of major components (and lots of details on how to implement these major components), which overall composes a novel and complete solution to the research question, which potentially can bring in new insights. Hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable, as we only want to know exactly what are the key components that fundamentally make the hypothesis work. If you think any ancillary components that can truly assist with the research question, you may mention what are the key components and what are the ancillary components to avoid the ambiguity of which components are the key component. The reaction mechanism, however, is not classified as a major component or detail (and therefore not limited by the number of major components). Instead, a novel and valid reaction mechanism can be a good source of insights. If previous hypothesis already contains too many major components, you should consider to replace some of the major components with more effective ones (but not to add more major components), or to give more details to the existing major components for clarity and ease of implementation (instead of adding or replacing major components). \
                   Experimental feasibility is also an important secondary consideration. In many cases, {DISCIPLINE} labs may lack the resources or setup to test hypotheses that are overly complex in terms of procedures or equipment requirements. Therefore, when two hypotheses are equally effective and scientifically valuable, we prefer the one that is simpler and more feasible to implement experimentally. However, this does not mean reducing the level of scientific detail. In fact, part of this refinement task is to add fine-grained conceptual or mechanistic details to improve clarity and completeness. These scientific details are not considered contributors to experimental complexity. Please continue enriching the hypothesis with meaningful and well-integrated details, while keeping in mind that unnecessary procedural complexity should be avoided when possible. \
                   {assist_example} \
                   {HYPTHESIS_GENERATION_CUSTOM_GUIDE} \
                   The research question is: \n", "\nThe survey is: \n", "\nNow please help to make modifications into the coarse-grained hypothesis, to make it one step closer to a more effective and more complete fine-grained hypothesis. Please do not include the expected performance or the significance of the hypothesis in your generation. For readability and clarity, please structure your revised hypothesis into coherent (several) paragraphs.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Revised Hypothesis starts:** [your complete revised hypothesis] **Revised Hypothesis ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this revised hypothesis] **Reasoning Process ends**"]
    elif module_name == "greedy_search_following_step":
        # assist_info: [cur_hierarchy_id, if_generate_with_example]
        assert len(assist_info) == 2
        assert assist_info[0] in [None, 0] and assist_info[1] in [0, 1], print("Exception: assist_info: ", assist_info)

        if assist_info[1] == 1:
            assist_example = f"Taking an organic {DISCIPLINE} example, the final fine-grained hypothesis we want to reach should look like this example: \n" + hyp_example_hierarchy_5 + "\nIn this example, there are five major components, and sufficient details and experimental conditions for each major component."
        else:
            assist_example = ""

        # check whether assist_example is consistent with assist_info[1] (if_generate_with_example)
        assert assist_example == "" if assist_info[1] == 0 else assist_example != "" and len(assist_example) > 0, print("assist_example: ", assist_example)

        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, a preliminary coarse-grained research hypothesis for the research question, and a previous (relatively more) fine-grained hypothesis derived from the coarse-grained hypothesis by a student, please help to make modifications into the fine-grained hypothesis, to make it one step closer to a more effective and more complete fine-grained hypothesis. Please ensure the improved hypothesis directly addresses the given research question, rather than shifting focus to a different question in a similar domain. \
                   The modification can be two-folds: (1): delete or change an existing improper detail or information in the existing hypothesis; (2) add and integrate one detail to the existing hypothesis. If you choose to add a detail, do not simply append new information to the existing hypothesis. Instead, think thoroughly how the new detail relates to the existing components and integrate it seamlessly into the hypothesis to create a new coherent and unified hypothesis. In addition if you choose to add a detail to a general information, if the corresponding general information is correct, you should try to keep the corresponding general information in the updated hypothesis and also mention the details, instead of replacing the general information with the details. In this way, it would be much easier for scientists to understand both the general infomration/structure and the details from your generated hypothesis. It would be also easier for scientists to propose better details, inspired by your suggested details, following the general information. \
                   Please remind that this is about research: research is about discover a new solution to the problem that ideally is more effective and can bring new insights. Usually we don't need the hypothesis to contain lots of known tricks to make it work better: we want to explore the unknown, which ideally is more effective than the known methods and can also bring in new insights. Therefore, a research hypothesis is usually about a small set (usually less than {num_major_details_tradition}) of major components (and lots of details on how to implement these major components), which overall composes a novel and complete solution to the research question, which potentially can bring in new insights. Hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable, as we only want to know exactly what are the key components that fundamentally make the hypothesis work. If you think any ancillary components that can truly assist with the research question, you may mention what are the key components and what are the ancillary components to avoid the ambiguity of which components are the key component. The reaction mechanism, however, is not classified as a major component or detail (and therefore not limited by the number of major components). Instead, a novel and valid reaction mechanism can be a good source of insights. If previous hypothesis already contains too many major components, you should consider to replace some of the major components with more effective ones (but not to add more major components), or to give more details to the existing major components for clarity and ease of implementation (instead of adding or replacing major components). \
                   Experimental feasibility is also an important secondary consideration. In many cases, {DISCIPLINE} labs may lack the resources or setup to test hypotheses that are overly complex in terms of procedures or equipment requirements. Therefore, when two hypotheses are equally effective and scientifically valuable, we prefer the one that is simpler and more feasible to implement experimentally. However, this does not mean reducing the level of scientific detail. In fact, part of this refinement task is to add fine-grained conceptual or mechanistic details to improve clarity and completeness. These scientific details are not considered contributors to experimental complexity. Please continue enriching the hypothesis with meaningful and well-integrated details, while keeping in mind that unnecessary procedural complexity should be avoided when possible. \
                   {assist_example} \
                   {HYPTHESIS_GENERATION_CUSTOM_GUIDE} \
                   The research question is: \n", "\nThe survey is: \n", "\nThe previous fine-grained hypothesis is: \n", "\nNow please help to make modifications into the previous fine-grained hypothesis, to make it one step closer to a more effective and more complete fine-grained hypothesis. Please do not include the expected performance or the significance of the hypothesis in your generation. For readability and clarity, please structure your revised hypothesis into coherent (several) paragraphs.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Revised Hypothesis starts:** [your complete revised hypothesis] **Revised Hypothesis ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this revised hypothesis] **Reasoning Process ends**"]
    elif module_name == "hierarchy_greedy_search_five_hierarchy_first_step":
        # assist_info: [cur_hierarchy_id, if_generate_with_example]
        assert len(assist_info) == 2
        assert assist_info[0] in [0, 1, 2, 3, 4] and assist_info[1] in [0, 1], print("Exception: assist_info: ", assist_info)

        # assist_instruction, assist_example
        assist_example = ""
        if assist_info[0] == 0:
            assist_instruction = HIERARCHY_LIST[0]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[0]               
        elif assist_info[0] == 1:
            assist_instruction = HIERARCHY_LIST[1]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[1]
        elif assist_info[0] == 2:
            assist_instruction = HIERARCHY_LIST[2]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[2]
        elif assist_info[0] == 3:
            assist_instruction = HIERARCHY_LIST[3]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[3]
        elif assist_info[0] == 4:
            assist_instruction = HIERARCHY_LIST[4]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[4]
        else:
            raise NotImplementedError
        
        # check whether assist_example is consistent with assist_info[1] (if_generate_with_example)
        assert assist_example == "" if assist_info[1] == 0 else assist_example != "" and len(assist_example) > 0, print("assist_example: ", assist_example)

        # assist_hierarchy_notice
        if assist_info[0] > 0:
            assist_hierarchy_notice = f"When we search for hypothesis in an hierarchy, if we are updating/adding/removing a general/specific component, we should also update the corresponding reaction mechanism in the hypothesis (the new reaction mechanism should be more effective than the previous one judged from {DISCIPLINE} principles). In general we should maintain the information found in the previous hierarchies (if we are not updating them) to keep current hypothesis more complete. {EXAMPLE_UPDATE_NEW_HIERARCHY}"
        else:
            assist_hierarchy_notice = ""

        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, and a research hypothesis from the previous hierarchy, please help to make modifications into the hypothesis from the previous hierarchy, to make it one step closer to a more effective and more complete fine-grained hypothesis in this hierarchy. Please ensure the improved hypothesis directly addresses the given research question, rather than shifting focus to a different question in a similar domain. \
                   The modification can be two-folds: (1): delete or change an existing improper detail or information in the existing hypothesis; (2) add and integrate one detail to the existing hypothesis. If you choose to add a detail, do not simply append new information to the existing hypothesis. Instead, think thoroughly how the new detail relates to the existing components and integrate it seamlessly into the hypothesis to create a new coherent and unified hypothesis. In addition if you choose to add a detail to a general information, if the corresponding general information is correct, you should try to keep the corresponding general information in the updated hypothesis and also mention the details, instead of replacing the general information with the details. In this way, it would be much easier for scientists to understand both the general infomration/structure and the details from your generated hypothesis. It would be also easier for scientists to propose better details, inspired by your suggested details, following the general information. \
                   Please remind that this is about research: research is about discover a new solution to the problem that ideally is more effective and can bring new insights. Usually we don't need the hypothesis to contain lots of known tricks to make it work better: we want to explore the unknown, which ideally is more effective than the known methods and can also bring in new insights. Therefore, a research hypothesis is usually about a small set (usually less than {num_major_details_tradition}) of major components (and lots of details on how to implement these major components), which overall composes a novel and complete solution to the research question, which potentially can bring in new insights. Hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable, as we only want to know exactly what are the key components that fundamentally make the hypothesis work. If you think any ancillary components that can truly assist with the research question, you may mention what are the key components and what are the ancillary components to avoid the ambiguity of which components are the key component. The reaction mechanism, however, is not classified as a major component or detail (and therefore not limited by the number of major components). Instead, a novel and valid reaction mechanism can be a good source of insights. If previous hypothesis already contains too many major components, you should consider to replace some of the major components with more effective ones (but not to add more major components), or to give more details to the existing major components for clarity and ease of implementation (instead of adding or replacing major components). \
                   Experimental feasibility is also an important secondary consideration. In many cases, {DISCIPLINE} labs may lack the resources or setup to test hypotheses that are overly complex in terms of procedures or equipment requirements. Therefore, when two hypotheses are equally effective and scientifically valuable, we prefer the one that is simpler and more feasible to implement experimentally. However, this does not mean reducing the level of scientific detail. In fact, part of this refinement task is to add fine-grained conceptual or mechanistic details to improve clarity and completeness. These scientific details are not considered contributors to experimental complexity. Please continue enriching the hypothesis with meaningful and well-integrated details, while keeping in mind that unnecessary procedural complexity should be avoided when possible. \
                   Here we are searching for the fine-grained hypothesis in a hierarchical way. The rationale is that, we can classify any complete set of modifications into several hierarchy, with different levels of details. If we do not search in a hierarchical way, we need to consider all the available details in all hierarchy levels for each search step, which (1) has a very high complexity, and (2) first search a low-level detail might largely influence the following search of a high-level detail: it might stuck in one high-level detail corresponding to the already searched low-level detail without considering the other low-level details corresponding to other high-level details, making the search process stuck in a local minumum at the beginning. \
                   {INTRODUCTION_OF_HIERARCHIES}\
                   {assist_hierarchy_notice} \
                   Here we want to search in the {assist_info[0]} hierarchy: please make modifications into the research hypothesis from the previous hierarchy in terms of {assist_instruction}, to make it one step closer to a more effective and more complete fine-grained hypothesis. Please only search for modifications inside the current hierarchy and do not search for modifications inside the next hierarchy. It is fine to delete or modify information from the previous hierarchy. \
                   Please retain and incorporate relevant information from previous hierarchies, if present and scientifically appropriate, when refining the hypothesis in the current hierarchy. Each refinement should build upon earlier levels where valid, while revising or replacing any incorrect or suboptimal elements to ensure continuity and completeness. {EXAMPLE_DESCRIPTION_FINAL_HYPOTHESIS} These elements should be explicitly integrated into a coherent, mechanistically consistent hypothesis that clearly reflects how they interact to support the proposed reaction pathway. \
                   {assist_example} \
                   {HYPTHESIS_GENERATION_CUSTOM_GUIDE} \
                   The research question is: \n", "\nThe survey is: \n", "\nNow please help to make modifications into the hypothesis from the previous hierarchy, to make it one step closer to a more effective and more complete fine-grained hypothesis in this hierarchy. Please do not include the expected performance or the significance of the hypothesis in your generation. For readability and clarity, please structure your revised hypothesis into coherent (several) paragraphs.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Revised Hypothesis starts:** [your complete revised hypothesis] **Revised Hypothesis ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this revised hypothesis] **Reasoning Process ends**"]
    elif module_name == "hierarchy_greedy_search_five_hierarchy_following_step":
        # assist_info: [cur_hierarchy_id, if_generate_with_example]
        assert len(assist_info) == 2
        assert assist_info[0] in [0, 1, 2, 3, 4] and assist_info[1] in [0, 1], print("Exception: assist_info: ", assist_info)

        # assist_instruction, assist_example
        assist_example = ""
        if assist_info[0] == 0:
            assist_instruction = HIERARCHY_LIST[0]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[0]
        elif assist_info[0] == 1:
            assist_instruction = HIERARCHY_LIST[1]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[1]
        elif assist_info[0] == 2:
            assist_instruction = HIERARCHY_LIST[2]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[2]
        elif assist_info[0] == 3:
            assist_instruction = HIERARCHY_LIST[3]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[3]
        elif assist_info[0] == 4:
            assist_instruction = HIERARCHY_LIST[4]
            if assist_info[1] == 1:
                assist_example = DESCRIPTION_HIERARCHY_LIST[4]
        else:
            raise NotImplementedError
        
        # check whether assist_example is consistent with assist_info[1] (if_generate_with_example)
        assert assist_example == "" if assist_info[1] == 0 else assist_example != "" and len(assist_example) > 0, print("assist_example: ", assist_example)

        # assist_hierarchy_notice
        if assist_info[0] > 0:
            assist_hierarchy_notice = f"When we search for hypothesis in an hierarchy, if we are updating/adding/removing a general/specific component, we should also update the corresponding reaction mechanism in the hypothesis (the new reaction mechanism should be more effective than the previous one judged from {DISCIPLINE} principles). In general we should maintain the information found in the previous hierarchies (if we are not updating them) to keep current hypothesis more complete. {EXAMPLE_UPDATE_NEW_HIERARCHY}"
        else:
            assist_hierarchy_notice = ""

        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, a research hypothesis from the previous hierarchy, and a preliminary fine-grained research hypothesis developed based on the research hypothesis from the previous hierarchy proposed by a student, please help to make modifications into the preliminary fine-grained hypothesis, to make it one step closer to a more effective and more complete fine-grained hypothesis in this hierarchy. Please ensure the improved hypothesis directly addresses the given research question, rather than shifting focus to a different question in a similar domain. \
                   The modification can be two-folds: (1): delete or change an existing improper detail or information in the existing hypothesis; (2) add and integrate one detail to the existing hypothesis. If you choose to add a detail, do not simply append new information to the existing hypothesis. Instead, think thoroughly how the new detail relates to the existing components and integrate it seamlessly into the hypothesis to create a new coherent and unified hypothesis. In addition if you choose to add a detail to a general information, if the corresponding general information is correct, you should try to keep the corresponding general information in the updated hypothesis and also mention the details, instead of replacing the general information with the details. In this way, it would be much easier for scientists to understand both the general infomration/structure and the details from your generated hypothesis. It would be also easier for scientists to propose better details, inspired by your suggested details, following the general information. \
                   Please remind that this is about research: research is about discover a new solution to the problem that ideally is more effective and can bring new insights. Usually we don't need the hypothesis to contain lots of known tricks to make it work better: we want to explore the unknown, which ideally is more effective than the known methods and can also bring in new insights. Therefore, a research hypothesis is usually about a small set (usually less than {num_major_details_tradition}) of major components (and lots of details on how to implement these major components), which overall composes a novel and complete solution to the research question, which potentially can bring in new insights. Hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable, as we only want to know exactly what are the key components that fundamentally make the hypothesis work. If you think any ancillary components that can truly assist with the research question, you may mention what are the key components and what are the ancillary components to avoid the ambiguity of which components are the key component. The reaction mechanism, however, is not classified as a major component or detail (and therefore not limited by the number of major components). Instead, a novel and valid reaction mechanism can be a good source of insights. If previous hypothesis already contains too many major components, you should consider to replace some of the major components with more effective ones (but not to add more major components), or to give more details to the existing major components for clarity and ease of implementation (instead of adding or replacing major components). \
                   Experimental feasibility is also an important secondary consideration. In many cases, {DISCIPLINE} labs may lack the resources or setup to test hypotheses that are overly complex in terms of procedures or equipment requirements. Therefore, when two hypotheses are equally effective and scientifically valuable, we prefer the one that is simpler and more feasible to implement experimentally. However, this does not mean reducing the level of scientific detail. In fact, part of this refinement task is to add fine-grained conceptual or mechanistic details to improve clarity and completeness. These scientific details are not considered contributors to experimental complexity. Please continue enriching the hypothesis with meaningful and well-integrated details, while keeping in mind that unnecessary procedural complexity should be avoided when possible. \
                   Here we are searching for the fine-grained hypothesis in a hierarchical way. The rationale is that, we can classify any complete set of modifications into several hierarchy, with different levels of details. If we do not search in a hierarchical way, we need to consider all the available details in all hierarchy levels for each search step, which (1) has a very high complexity, and (2) first search a low-level detail might largely influence the following search of a high-level detail: it might stuck in one high-level detail corresponding to the already searched low-level detail without considering the other low-level details corresponding to other high-level details, making the search process stuck in a local minumum at the beginning. \
                   {INTRODUCTION_OF_HIERARCHIES}\
                   {assist_hierarchy_notice} \
                   Here we want to search in the {assist_info[0]} hierarchy: please make modifications into the preliminary fine-grained hypothesis in terms of {assist_instruction}, to make it one step closer to a more effective and more complete fine-grained hypothesis. Please only search for modifications inside the current hierarchy and do not search for modifications inside the next hierarchy. It is fine to delete or modify information from the previous hierarchy. \
                   Please retain and incorporate relevant information from previous hierarchies, if present and scientifically appropriate, when refining the hypothesis in the current hierarchy. Each refinement should build upon earlier levels where valid, while revising or replacing any incorrect or suboptimal elements to ensure continuity and completeness. {EXAMPLE_DESCRIPTION_FINAL_HYPOTHESIS} These elements should be explicitly integrated into a coherent, mechanistically consistent hypothesis that clearly reflects how they interact to support the proposed reaction pathway. \
                   {assist_example} \
                   {HYPTHESIS_GENERATION_CUSTOM_GUIDE} \
                   The research question is: \n", "\nThe survey is: \n", "\nThe preliminary fine-grained hypothesis is: \n", "\nNow please help to make modifications into the preliminary fine-grained hypothesis, to make it one step closer to a more effective and more complete fine-grained hypothesis in this hierarchy. Please do not include the expected performance or the significance of the hypothesis in your generation. For readability and clarity, please structure your revised hypothesis into coherent (several) paragraphs.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Revised Hypothesis starts:** [your complete revised hypothesis] **Revised Hypothesis ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this revised hypothesis] **Reasoning Process ends**"]
    elif module_name == "recombination_first_step":
        assert assist_info[0] >= 0 and assist_info[0] <= 4, print("assist_info: ", assist_info)
        if assist_info[0] > 0:
            assist_hierarchy_notice = f"When we search for hypothesis in an hierarchy, if we are updating/adding/removing a general/specific component, we should also update the corresponding reaction mechanism in the hypothesis (the new reaction mechanism should be more effective than the previous one judged from {DISCIPLINE} principles). In general we should maintain the information found in the previous hierarchies (if we are not updating them) to keep current hypothesis more complete. {EXAMPLE_UPDATE_NEW_HIERARCHY}"
        else:
            assist_hierarchy_notice = ""
        # "\nOne of the good research hypothesis candidates is: \n", 
        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, several proposed research hypothesis candidates to refer to, please try your best to leverage the advantages of the hypothesis candidates and avoid their disadvantages to formulate a research hypothesis that is more effective than the existing proposed hypothesis candidates to the research question. Please ensure the improved hypothesis directly addresses the given research question, rather than shifting focus to a different question in a similar domain. You should rely on your own knowledge and understanding of {DISCIPLINE} to help this task. \
                   In research, a research hypothesis is usually about a small set (usually less than {num_major_details_tradition}) of major components (and sufficient details on how to implement these major components), which overall composes a novel and complete solution to the research question, which potentially can bring in new insights. Hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable. The focus should be on identifying the key components that are essential for the hypothesis to work, along with any supplementary components that enhance or support the primary components.  \
                   In this task, you are not required to add numerous additional details to the existing hypothesis candidates (although you are supposed to keep/remain a similar level of details in your answer). Instead, aim to formulate a more effective hypothesis that retains fewer than {num_major_details_tradition} major components by leveraging the strengths of the major components in the provided candidates and if needed, incorporating your knowledge and understanding of {DISCIPLINE} to compose the strengths in a reasonable and understandable way. The reaction mechanism is usually a good source of insights and is not classified as a major component or detail. Instead, it provides high-level guidance for selecting the major components and their details. If several reaction mechanisms are mentioned in the hypothesis candidates, try to select/compose one set of reaction mechanism in your answer that is more effective than the mechanisms in the existing candidates.  \
                   {assist_hierarchy_notice} \
                   {HYPTHESIS_GENERATION_CUSTOM_GUIDE} \
                   The research question is: \n", "\nThe survey is: \n", "\nNow, please formulate a more effective research hypothesis by utilizing the advantages of the hypothesis candidates while addressing their disadvantages and considering the student's preliminary trial hypothesis. By 'utilizing the advantages,' the aim is not merely to aggregate appealing components, but to thoughtfully integrate strengths in a way that effectively addresses the research question. Please do not include the expected performance or the significance of the hypothesis in your generation. For readability and clarity, please structure your revised hypothesis into coherent (several) paragraphs.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Revised Hypothesis starts:** [your complete revised hypothesis] **Revised Hypothesis ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this revised hypothesis] **Reasoning Process ends**"]
    elif module_name == "recombination_following_step":
        assert assist_info[0] >= 0 and assist_info[0] <= 4, print("assist_info: ", assist_info)
        if assist_info[0] > 0:
            assist_hierarchy_notice = f"When we search for hypothesis in an hierarchy, if we are updating/adding/removing a general/specific component, we should also update the corresponding reaction mechanism in the hypothesis (the new reaction mechanism should be more effective than the previous one judged from {DISCIPLINE} principles). In general we should maintain the information found in the previous hierarchies (if we are not updating them) to keep current hypothesis more complete. {EXAMPLE_UPDATE_NEW_HIERARCHY}"
        else:
            assist_hierarchy_notice = ""
        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, several proposed research hypothesis candidates to refer to, please try your best to leverage the advantages of the hypothesis candidates and avoid their disadvantages to formulate a research hypothesis that is more effective than the existing proposed hypothesis candidates to the research question. Please ensure the improved hypothesis directly addresses the given research question, rather than shifting focus to a different question in a similar domain. You should rely on your own knowledge and understanding of {DISCIPLINE} to help this task. A student has made a preliminary trial to formulate a better research hypothesis, you may refer to it to improve it, or you can leave it and formulate a new one for better hypothesis. \
                   In research, a research hypothesis is usually about a small set (usually less than {num_major_details_tradition}) of major components (and sufficient details on how to implement these major components), which overall composes a novel and complete solution to the research question, which potentially can bring in new insights. Hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable. The focus should be on identifying the key components that are essential for the hypothesis to work, along with any supplementary components that enhance or support the primary components.  \
                   In this task, you are not required to add numerous additional details to the existing hypothesis candidates (although you are supposed to keep/remain a similar level of details in your answer). Instead, aim to formulate a more effective hypothesis that retains fewer than {num_major_details_tradition} major components by leveraging the strengths of the major components in the provided candidates and if needed, incorporating your knowledge and understanding of {DISCIPLINE} to compose the strengths in a reasonable and understandable way. The reaction mechanism is usually a good source of insights and is not classified as a major component or detail. Instead, it provides high-level guidance for selecting the major components and their details. If several reaction mechanisms are mentioned in the hypothesis candidates, try to select/compose one set of reaction mechanism in your answer that is more effective than the mechanisms in the existing candidates.  \
                   {assist_hierarchy_notice} \
                   {HYPTHESIS_GENERATION_CUSTOM_GUIDE} \
                   The research question is: \n", "\nThe survey is: \n", "\nThe preliminary trial by the student is: \n", "\nNow, please formulate a more effective research hypothesis by utilizing the advantages of the hypothesis candidates while addressing their disadvantages and considering the student's preliminary trial hypothesis. By 'utilizing the advantages,' the aim is not merely to aggregate appealing components, but to thoughtfully integrate strengths in a way that effectively addresses the research question. Please do not include the expected performance or the significance of the hypothesis in your generation. For readability and clarity, please structure your revised hypothesis into coherent (several) paragraphs.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Revised Hypothesis starts:** [your complete revised hypothesis] **Revised Hypothesis ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this revised hypothesis] **Reasoning Process ends**"]
    elif module_name == "validity_clarity_feedback_to_hyp":
        assert assist_info == None or (assist_info >= 0 and assist_info <= 4), print("assist_info: ", assist_info)
        if assist_info == 0:
            assist_instruction = FEEDBACK_HIERARCHY_VALIDITY_CLARITY[0]
        elif assist_info == 1:
            assist_instruction = FEEDBACK_HIERARCHY_VALIDITY_CLARITY[1]
        elif assist_info == 2:
            assist_instruction = FEEDBACK_HIERARCHY_VALIDITY_CLARITY[2]
        elif assist_info == 3:
            assist_instruction = FEEDBACK_HIERARCHY_VALIDITY_CLARITY[3]
        elif assist_info == 4:
            assist_instruction = FEEDBACK_HIERARCHY_VALIDITY_CLARITY[4]
        elif assist_info == None:
            assist_instruction = FEEDBACK_NON_HIERARCHY_VALIDITY_CLARITY[0]
        else:
            raise NotImplementedError
        # For example, the added information could be about 'add xxx component can ensure xxx across a wide temperature range'. In this example, the specific temperature range is not specified and you should point it out. Also in this example, the use of word 'ensure' is too strong and might not be valid (more soft word is preferred)
        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, a research hypothesis in the previous step, and the updated hypothesis based on the previous hypothesis, please provide feedback on the validity and clarity of the updated hypothesis. Please ensure that your feedback remains focused on improving the hypothesis in a way that directly addresses the given research question, without diverging into related but different questions. \
                   In terms of validity, the updated hypothesis has not been tested with real experiments, so please help with the scientists to identify any potential issues in the updated hypothesis that might make it ineffective or not feasible to the research question. The validity should be judged based on fundamental {DISCIPLINE} principles rather than the description provided in the text. The clarity aspect mainly refers to whether a hypothesis is not clear, underspecified, or any component of the hypothesis is not well integrated. Please provide feedback to the hypothesis in terms of the validity and clarity aspects. \
                   {assist_instruction}  \
                   Here are some notes you should consider when you determine the feedback on the validity and clarity of the updated hypothesis: hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable. The focus should be on identifying the key components that are essential for the hypothesis to work, along with any supplementary components that enhance or support the primary components. If you think any contents in the updated hypothesis are not very relevant and not helpful to the research question, your feedback should include how to delete the irrelevant information and possibly keep the good parts of the them (if there is any). In addition to the feedback to the unnecessary components, in general you should focus on those fundamental key components in the hypothesis that work for the research question to provide feedback in terms of how to make them more valid and effective / more clear and specified. If you think the current hypothesis is not complete, or it needs other necessary components for a better solution for the research question, you may also provide feedback or suggestions on it. Maybe think of this question when providing feedback: is this hypothesis valid / effective enough for the research question, or is the hypothesis specified enough in terms of our interested aspects? If not, then fundamentally why not? And how should we modify / refine the hypothesis to fill the gap?  \
                   Please rely on your own knowledge and understanding in {DISCIPLINE} to provide feedback. If you find any problems in the updated hypothesis, please also provide suggestions on how to improve the hypothesis. \
                   The research question is: \n", "\nThe survey is: \n", "\nThe previous hypothesis is: \n", "\nThe updated hypothesis is: \n", "\nThe rational of the updated hypothesis proposed by a student (which might not be reliable enough) is: \n", "\nNow please provide feedback on the validity (whether effective) and clarity (whether specified) of the updated hypothesis, and also give your suggestions on how to make it better to address your mentioned problems. Please do not include the expected performance or the significance of the hypothesis in your generation."]
    elif module_name == "update_hyp_based_on_feedback":
        prompts = [f"You are assisting with scientist's research. Given their research question, a survey on the past methods for the research question, a research hypothesis in the previous step, the updated hypothesis based on the previous hypothesis, and feedback to the updated hypothesis in terms of validity (whether effective) and clarity (whether clear, specified, and integrated), please help to improve the updated hypothesis based on the feedback. Please ensure the improved hypothesis directly addresses the given research question, rather than shifting focus to a different question in a similar domain. Please remind that hypotheses that include an excessive number of irrelevant or unnecessary major components, which do not contribute to addressing the research question, are less favorable. Instead, we want to identify the fundamental key points that can help the research hypothesis to better address the research question. \
                   If you add details to the updated hypothesis to make it more clear, please (1) make sure the details are well integrated with the existing hypothesis, i.e., how are the new details related to the reaction mechanism and the original contents in the original hypothesis, and how can it help with the original hypothesis; and (2) try to keep the corresponding general information in the previous hypothesis in the updated hypothesis while adding the specified information (if they are proper and corect), especially when the corresponding general information is about the reaction mechanism of the hypothesis. Keeping the corresponding general information that is proper and correct is beneficial, since they can help scientists to understand the structure of the hypothesis and why the details are needed in terms of its relation to the general structure. If you think the corresponding general information (e.g, reaction mechamism, and general components) has flaws and need to be updated, you may also make modifications to them. \
                   {HYPTHESIS_GENERATION_CUSTOM_GUIDE} \
                   The research question is: \n", "\nThe survey is: \n", "\nThe previous hypothesis is: \n", "\nThe updated hypothesis is: \n", "\nThe feedback to the updated hypothesis in terms of validity and clarity is: \n", "\nNow please help to refine the updated hypothesis based on the feedback. Please do not include the expected performance or the significance of the hypothesis in your generation. For readability and clarity, please structure your revised hypothesis into coherent (several) paragraphs.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Revised Hypothesis starts:** [your complete revised hypothesis] **Revised Hypothesis ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this revised hypothesis] **Reasoning Process ends**"]
    else:
        raise NotImplementedError

    return prompts




# A collection of prompts for different modules
def evaluation_instruction_prompts(module_name, assist_info=None):
    num_major_details_tradition = "eight"
    if module_name == "break_finegrained_hyp_or_exp":
        assert assist_info == "hyp" or assist_info == "exp"
        if assist_info == "hyp":
            info = "research hypothesis"
        else:
            info = "experiment design"
        prompts = [f"""
                    You are assisting scientists by carefully breaking down a {DISCIPLINE} hypothesis into explicitly defined **technical solution components only**. 

                    Here, a **'component'** strictly refers to explicitly described technical aspects directly stated in the hypothesis itself, specifically:

                    - **Explicitly named chemicals or materials.**
                    - **Clearly stated functional groups.**
                    - **Explicitly described chemical reactions or reaction mechanisms** (including explicitly defined chemical intermediates or reaction steps).
                    - **Explicitly detailed synthesis or fabrication methods, procedures, or structural designs.**
                    - **Explicitly named catalysts, ligands, or additives.**
                    - **Clearly stated reaction conditions** (such as solvents, bases, temperatures, durations, atmospheres).

                    Carefully follow **ALL** the following instructions when identifying components:

                    ### (1) **ONLY INCLUDE explicitly defined technical content from the hypothesis itself:**

                    - **Explicitly stated chemicals or materials** (e.g., "[EMIM][DCA] ionic liquid").
                    - **Explicitly stated catalysts, ligands, additives** (e.g., "acetylacetonate (acac) ligand").
                    - **Clearly described chemical reaction steps or mechanisms** (e.g., "transmetallation forming aryl-Ni(II) intermediate").
                    - **Detailed synthesis or fabrication steps explicitly described** (e.g., "UV polymerization with 80 wt% [EMIM][DCA]").
                    - **Clearly defined structural elements** explicitly described (e.g., "Radiative Cooling Layer made of hierarchically porous poly(vinylidene-co-hexafluoropropene)").
                    - **Explicitly stated reaction conditions** (exact solvents, bases, temperatures, durations, and atmospheres. The number must be specified if applied, else it should not be counted as a component. When adopting it as an component, you must include the specified number in your answer).

                    ### (2) **Always attach a concise ancillary description to each component:**

                    - For **each identified technical component**, explicitly attach a brief and concise description clearly stating its specific technical role or purpose **as explicitly defined in the hypothesis**.
                    - Example format:
                    - "**Acetylacetonate (acac) ligand:** stabilizes nickel catalyst and enhances coordination with weak native functional groups."
                    - "**[EMIM][DCA] ionic liquid:** provides high ionic conductivity and thermal stability."
                    - **Never include ancillary descriptions as separate, independent components.**

                    ### (3) **STRICTLY EXCLUDE all of the following:**

                    - **Experimental outcomes, performance evaluations, or numerical measurements** (e.g., efficiencies, voltages, currents, Seebeck coefficients, power densities, stability durations).
                    - **Experimental validations, radical experiments, or computational/calculated results** (e.g., radical clock, TEMPO, NMR confirmations, Raman spectroscopy confirmations, DFT calculations).
                    - **General descriptions of benefits, advantages, or conceptual explanations** (e.g., "improves efficiency," "enhances stability," "ensures high performance," "enables sustainable conversion").
                    - **Descriptions of scalability, flexibility, or integration potential** (e.g., "units are scalable," "integration to achieve high voltage").
                    - **Applications, potential uses, or practical scenarios** (e.g., "useful for drug discovery," "powers wearable devices").
                    - **Broad conceptual mechanisms or vague, general statements** (e.g., "utilizes environmental radiation," "operates in diverse weather conditions," "maintains stable output").

                    ### (4) **Each component MUST represent exactly ONE distinct explicitly defined technical element:**

                    - If multiple chemicals, reaction steps, synthesis methods, or reaction conditions are explicitly mentioned together, carefully separate each into **individual distinct components**.
                    - **Never merge multiple distinct technical elements into one component.**

                    ---

                    ### **In summary (use this as a quick checklist):**

                    ✅ **INCLUDE (with attached concise descriptions):**  
                    - Chemicals, materials, functional groups  
                    - Catalysts, ligands, additives  
                    - Detailed reaction mechanisms/intermediates  
                    - Explicit synthesis/fabrication steps  
                    - Structural designs clearly stated  
                    - Explicit reaction conditions

                    ❌ **EXCLUDE:**  
                    - Experimental outcomes, numerical data, or characterizations  
                    - Validations or computational results  
                    - General benefits, applications, scalability, or flexibility  
                    - Conceptual or vague statements not explicitly chemical or technical

                    Now, carefully break down the following hypothesis into explicitly defined, strictly technical components, precisely following these comprehensive instructions. Present each component clearly and strictly in the following format: 'Id of the component: \nComponent: \n'. The {info} is:
                    """, 
                    "\nNow please answer the question."]
    elif module_name == "break_finegrained_hyp_or_exp_refine":
        assert assist_info == "hyp" or assist_info == "exp"
        if assist_info == "hyp":
            info = "research hypothesis"
        else:
            info = "experiment design"
        prompts = [f"""
                   You are assisting scientists by carefully breaking down a {DISCIPLINE} hypothesis into explicitly defined **technical solution components only**. 

                    Here, a **'component'** strictly refers to explicitly described technical aspects directly stated in the hypothesis itself, specifically:

                    - **Explicitly named chemicals or materials.**
                    - **Clearly stated functional groups.**
                    - **Explicitly described chemical reactions or reaction mechanisms** (including explicitly defined chemical intermediates or reaction steps).
                    - **Explicitly detailed synthesis or fabrication methods, procedures, or structural designs.**
                    - **Explicitly named catalysts, ligands, or additives.**
                    - **Clearly stated reaction conditions** (such as solvents, bases, temperatures, durations, atmospheres).

                    Carefully follow **ALL** the following instructions when identifying components:

                    ### (1) **ONLY INCLUDE explicitly defined technical content from the hypothesis itself:**

                    - **Explicitly stated chemicals or materials** (e.g., "[EMIM][DCA] ionic liquid").
                    - **Explicitly stated catalysts, ligands, additives** (e.g., "acetylacetonate (acac) ligand").
                    - **Clearly described chemical reaction steps or mechanisms** (e.g., "transmetallation forming aryl-Ni(II) intermediate").
                    - **Detailed synthesis or fabrication steps explicitly described** (e.g., "UV polymerization with 80 wt% [EMIM][DCA]").
                    - **Clearly defined structural elements** explicitly described (e.g., "Radiative Cooling Layer made of hierarchically porous poly(vinylidene-co-hexafluoropropene)").
                    - **Explicitly stated reaction conditions** (exact solvents, bases, temperatures, durations, and atmospheres. The number must be specified if applied, else it should not be counted as a component. When adopting it as an component, you must include the specified number in your answer).

                    ### (2) **Always attach a concise ancillary description to each component:**

                    - For **each identified technical component**, explicitly attach a brief and concise description clearly stating its specific technical role or purpose **as explicitly defined in the hypothesis**.
                    - Example format:
                    - "**Acetylacetonate (acac) ligand:** stabilizes nickel catalyst and enhances coordination with weak native functional groups."
                    - "**[EMIM][DCA] ionic liquid:** provides high ionic conductivity and thermal stability."
                    - **Never include ancillary descriptions as separate, independent components.**

                    ### (3) **STRICTLY EXCLUDE all of the following:**

                    - **Experimental outcomes, performance evaluations, or numerical measurements** (e.g., efficiencies, voltages, currents, Seebeck coefficients, power densities, stability durations).
                    - **Experimental validations, radical experiments, or computational/calculated results** (e.g., radical clock, TEMPO, NMR confirmations, Raman spectroscopy confirmations, DFT calculations).
                    - **General descriptions of benefits, advantages, or conceptual explanations** (e.g., "improves efficiency," "enhances stability," "ensures high performance," "enables sustainable conversion").
                    - **Descriptions of scalability, flexibility, or integration potential** (e.g., "units are scalable," "integration to achieve high voltage").
                    - **Applications, potential uses, or practical scenarios** (e.g., "useful for drug discovery," "powers wearable devices").
                    - **Broad conceptual mechanisms or vague, general statements** (e.g., "utilizes environmental radiation," "operates in diverse weather conditions," "maintains stable output").

                    ### (4) **Each component MUST represent exactly ONE distinct explicitly defined technical element:**

                    - If multiple chemicals, reaction steps, synthesis methods, or reaction conditions are explicitly mentioned together, carefully separate each into **individual distinct components**.
                    - **Never merge multiple distinct technical elements into one component.**

                    ---

                    ### **In summary (use this as a quick checklist):**

                    ✅ **INCLUDE (with attached concise descriptions):**  
                    - Chemicals, materials, functional groups  
                    - Catalysts, ligands, additives  
                    - Detailed reaction mechanisms/intermediates  
                    - Explicit synthesis/fabrication steps  
                    - Structural designs clearly stated  
                    - Explicit reaction conditions

                    ❌ **EXCLUDE:**  
                    - Experimental outcomes, numerical data, or characterizations  
                    - Validations or computational results  
                    - General benefits, applications, scalability, or flexibility  
                    - Conceptual or vague statements not explicitly chemical or technical

                    Now, a novice student has attempted to break down the above hypothesis into components but has likely made errors against the provided instructions. Common mistakes made by the student include (1) incorrectly identifying general background information, descriptions of the research task, experimental outcomes, test results, numerical measurements, performance evaluations, benefits, or other non-technical details as components; (2) forgetting to include any specified chemicals, reaction mechanisms, and experiment settings as components. Please carefully review and refine the student's answer by clearly removing any mistakenly included non-technical components and ensuring each remaining component explicitly meets all the previously defined criteria for a technical solution component. Please help the student to improve his answer by providing a corrected and improved breakdown according to the original detailed instructions. Please answer with the following format: 'Id of the component: \nComponent: \n'. The {info} is: 
                    """, 
                    f"The student's answer to break the {info} into components is: ", "\nNow please answer the question."]
    elif module_name == "compare_components_from_gt_and_gene":
        assert len(assist_info) == 2
        assert assist_info[0] == "hyp" or assist_info[0] == "exp"
        assert assist_info[1] == True or assist_info[1] == False
        if assist_info[0] == "hyp":
            info = "research hypothesis"
        else:
            info = "experiment design"
        if assist_info[1] == True:
            oriented_info = "groundtruth"
            oriented_reverse_info = "machine generated"
        else:
            oriented_info = "machine generated"
            oriented_reverse_info = "groundtruth"
        prompts = [f'''
                   Please compare the components from the groundtruth research hypothesis with the components from the machine-generated research hypothesis. We aim to determine how well each groundtruth hypothesis component is covered by the machine-generated hypothesis components.

                    Specifically, evaluate each component in the groundtruth hypothesis against the provided machine-generated hypothesis components and assign a 'covered degree' based strictly on the following levels:

                    Level 3 (Exact match): For specific chemical used: The pair of components is specified and nearly identical in terms of technical details and intended function. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified and nearly identical (difference less than 10%) in terms of the numbers.

                    Level 2 (Highly related): For specific chemical used: The pair of components is specified and very closely related, sharing substantial technical similarity and partially overlapping functions. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified and very similar (difference less than 20%) in terms of the numbers.

                    Level 1 (Moderately related): For specific chemical used: Either (1) the pair of components lacks specificity but is conceptually similar at a general level, or (2) they are specified but have medium-level technical or functional relevance, contributing positively but indirectly. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified and moderately similar (difference less than 30%) in terms of the numbers.

                    Level 0 (Not related): For specific chemical used: Either (1) the pair of components lacks specificity and relevance, or (2) the pair is clearly specified but unrelated or minimally related. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified but have more than 30% difference in terms of the numbers, or they are not all specified.

                    When assigning the 'covered degree', follow these guidelines strictly:

                    - Each component in the groundtruth hypothesis must be individually evaluated exactly as provided. Do NOT combine, split, or modify components.

                    - Components from the machine-generated hypothesis are used exclusively to assess coverage of groundtruth components.

                    - Ignore components not containing technical content.
                   
                    Please answer with the following format: 'Covered component: \nCovered level: \n'. Please use the content of the component as the 'covered component', but not use an component id to refer without any content of the conpotent. \nThe groundtruth components are: \n
                   ''', "\nThe machine generated components are: \n", f"\nNow please help us analyze which components in the {oriented_info} {info} are covered by the {oriented_reverse_info} {info} in which degree. Please evaluate the 'covered degree' of one component for only one time."]
    elif module_name == "compare_components_from_gt_and_gene_refine":
        assert len(assist_info) == 2
        assert assist_info[0] == "hyp" or assist_info[0] == "exp"
        assert assist_info[1] == True or assist_info[1] == False
        if assist_info[0] == "hyp":
            info = "research hypothesis"
        else:
            info = "experiment design"
        if assist_info[1] == True:
            oriented_info = "groundtruth"
            oriented_reverse_info = "machine generated"
        else:
            oriented_info = "machine generated"
            oriented_reverse_info = "groundtruth"
        prompts = [f'''
                   Please compare the components from the groundtruth research hypothesis with the components from the machine-generated research hypothesis. We aim to determine how well each groundtruth hypothesis component is covered by the machine-generated hypothesis components.

                    Specifically, evaluate each component in the groundtruth hypothesis against the provided machine-generated hypothesis components and assign a 'covered degree' based strictly on the following levels:

                    Level 3 (Exact match): For specific chemical used: The pair of components is specified and nearly identical in terms of technical details and intended function. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified and nearly identical (difference less than 10%) in terms of the numbers.

                    Level 2 (Highly related): For specific chemical used: The pair of components is specified and very closely related, sharing substantial technical similarity and partially overlapping functions. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified and very similar (difference less than 20%) in terms of the numbers.

                    Level 1 (Moderately related): For specific chemical used: Either (1) the pair of components lacks specificity but is conceptually similar at a general level, or (2) they are specified but have medium-level technical or functional relevance, contributing positively but indirectly. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified and moderately similar (difference less than 30%) in terms of the numbers.

                    Level 0 (Not related): For specific chemical used: Either (1) the pair of components lacks specificity and relevance, or (2) the pair is clearly specified but unrelated or minimally related. For specific experimental conditions (e.g., temperature, pressure): The pair of components is specified but have more than 30% difference in terms of the numbers, or they are not all specified.

                    When assigning the 'covered degree', follow these guidelines strictly:

                    - Each component in the groundtruth hypothesis must be individually evaluated exactly as provided. Do NOT combine, split, or modify components.

                    - Components from the machine-generated hypothesis are used exclusively to assess coverage of groundtruth components.

                    - Ignore components not containing technical content.
                   
                    Please answer with the following format: 'Covered component: \nCovered level: \n'. Please use the content of the component as the 'covered component', but not use an component id to refer without any content of the conpotent. \nThe groundtruth components are: \n"
                   ''', "\nThe machine generated components are: \n", "\nThe student's analysis is: \n", f"\nNow please help use analyze which components in the {oriented_info} {info} are covered by the {oriented_reverse_info} {info} in which degree. Please evaluate the 'covered degree' of one component for only one time."]
    elif module_name == "get_rid_of_repeated_components":
        prompts = [f'''
                    We have decomposed {DISCIPLINE} research hypotheses into technical components to compare the student-proposed hypothesis against the groundtruth hypothesis. Another student previously compared these components, identifying overlaps, and assigned each an "overlap degree" (numeric values; higher is better).

                    However, the student frequently repeats identical or highly similar components, even if they're not adjacent. Your task is to:

                    1. Remove all repeated or highly similar components, even if they appear non-adjacently. Each unique component must appear only once.

                    2. Merge repeated components into one, following these guidelines strictly:

                    - If repeated components contain very similar information, keep only the version with the highest numeric overlap degree.

                    - If one repeated component has clearly more detailed or richer information, keep this richer component, even if its overlap degree is numerically lower. Prioritize information richness over numeric degree in such cases.

                    3. Maintain the original numeric "overlap degree" from the retained component exactly as-is.

                    - Never output "None" for the overlap degree.

                    - Always preserve the numeric degree value from the selected retained component.

                    4. Ensure no duplicates or repetitions remain in your final output.

                    If there are no repeated or similar components, simply return the original components with their numeric overlap degrees unchanged.
                   
                    Please answer with the following format: 'Id of the component: \nComponent: \n'. The student's preliminary comparison is: 
                    ''', "\nNow please answer the question."]
    elif module_name == "get_rid_of_repeated_components_refine":
        prompts = [f'''
                    We have decomposed {DISCIPLINE} research hypotheses into technical components to compare the student-proposed hypothesis against the groundtruth hypothesis. Another student previously compared these components, identifying overlaps, and assigned each an "overlap degree" (numeric values; higher is better).

                    However, the student frequently repeats identical or highly similar components, even if they're not adjacent. Your task is to:

                    1. Remove all repeated or highly similar components, even if they appear non-adjacently. Each unique component must appear only once.

                    2. Merge repeated components into one, following these guidelines strictly:

                    - If repeated components contain very similar information, keep only the version with the highest numeric overlap degree.

                    - If one repeated component has clearly more detailed or richer information, keep this richer component, even if its overlap degree is numerically lower. Prioritize information richness over numeric degree in such cases.

                    3. Maintain the original numeric "overlap degree" from the retained component exactly as-is.

                    - Never output "None" for the overlap degree.

                    - Always preserve the numeric degree value from the selected retained component.

                    4. Ensure no duplicates or repetitions remain in your final output.

                    If there are no repeated or similar components, simply return the original components with their numeric overlap degrees unchanged.
                   
                    Please answer with the following format: 'Id of the component: \nComponent: \n'. Now that someone has checked the preliminary comparison, and proposed an updated version. Please refer to both the original comparison result and the updated result, try to fix any problem inside (if there is any) and then give your refined answer of the comparison. The student's preliminary comparison is: 
                    ''', "\nThe updated comparison is: ", "\nNow please answer the question."]
    # update pairwise_compare_prev_update's prompt with gpt-4o
    elif module_name == "pairwise_compare":
        assert len(assist_info) == 2
        assert assist_info[0] == "strict_to_hyp2" or assist_info[0] == "same_hyp1_hyp2" 
        assert assist_info[1] in [0, 1, 2, 3, 4] or assist_info[1] == None
        # strict_to_hyp2_info
        if assist_info[0] == "strict_to_hyp2":
            strict_to_hyp2_info = "We should be careful when making modifications to research hypothesis candidate 1: since once we accept the modifications (Research hypothesis candidate 2), it might be hard to revert them back." 
        else:
            strict_to_hyp2_info = ""
        # hierarchy_info
        hierarchy_info = "Here we roughly classify all possible contents in a fine-grained hypothesis into five hierarchies: (1) Mechanism of the Reaction: Describes how the reaction proceeds at a conceptual level, focusing on electron flow, bond formation and breaking, and any intermediates or transition states involved. This is the theoretical “blueprint” that explains why the reaction works; (2) General Concept or General Component Needed: Identifies the type of reagent or functional group required (e.g., “a strong acid,” “a Lewis base,” “an activated aromatic ring”) without committing to a specific chemical. It outlines the broader roles that are necessary for the mechanism to proceed; (3) Specific Components for the General Concept: Narrows down from the general category to a particular substance (e.g., “concentrated HCl” for a strong acid, “benzene” for an aromatic ring). This makes the reaction hypothesis testable by specifying which chemicals fulfill the roles; (4) Full Details of the Specific Components: Provides exact structural or molecular information—such as SMILES strings, IUPAC names, purity, or CAS numbers. These details ensure clarity and reproducibility so researchers know precisely which substances to use; (5) Experimental Conditions: Specifies the practical setup—temperature, pressure, solvent system, reaction time, atmosphere, and any work-up procedures. This final layer describes how to carry out the reaction in a laboratory setting. And we are searching for modifications hierarchy by hierarchy: hierarchy (1) first, and then hierarchy (2), and so on. Hypothesis from a higher hierarchy is an expansion of the hypothesis from its previous hierarchy, with additional information described above."
        if assist_info[1] in [0, 1, 2, 3, 4]:
            hierarchy_info = hierarchy_info + f"\nHere we focus on hierarchy ({assist_info[1]+1}). "
        elif assist_info[1] == None:
            hierarchy_info = ""
        else:
            raise NotImplementedError
        prompts = [f"You are assisting chemists in refining a research hypothesis. Given a base research hypothesis (Research hypothesis candidate 1) and an updated version (Research hypothesis candidate 2), your task is to determine whether the updated hypothesis is an improvement over the base hypothesis. The aim of this iterative process is to gradually refine an initially coarse or partially novel hypothesis into one that is effective, scientifically valid, and sufficiently detailed for real {DISCIPLINE} experiments. \
                   A hypothesis is only improved if it demonstrably enhances its scientific validity, conceptual clarity, or experimental feasibility based on fundamental {DISCIPLINE} principles. More details do not automatically make a hypothesis better. A more specific hypothesis is generally preferred only when it corrects or addresses a shortcoming in the original proposal, or significantly improves the clarity and testability of the hypothesis. If an added element does not clearly resolve a known gap or make the hypothesis more feasible for experimental validation, then the base hypothesis should be preferred. \
                   While novelty can be valuable, validity and rigorous detail take precedence in this particular task. The original or coarse-grained hypothesis already establishes some novel direction or concept. Your role here is to decide if the updated hypothesis is more practically implementable or scientifically grounded, rather than simply more novel. Claims of better performance—such as a broader temperature range or superior output—should be evaluated in light of fundamental {DISCIPLINE} and adequacy of design details, not just based on self-reported figures. \
                   Justifications for added details are encouraged whenever possible. However, even if a new element is not fully explained, it may still be acceptable if it addresses a concrete gap or significantly strengthens the experimental basis of the hypothesis. The main criterion is whether the modification provides unique and necessary improvements over what was already in place. If the modification adds advanced-sounding techniques or claims to cover a broader range without showing how it is fundamentally achieved and supported by chemical reasoning, it should not be considered an improvement. \
                   Experimental feasibility should not be the sole deciding factor. At this stage, scientific soundness is more important than immediate ease of implementation. That said, a hypothesis that is rigorous and includes enough detail for chemists to conduct real experiments should be favored over one that is vague or leaves essential conditions undefined. Self-reported operational ranges or performance claims are not sufficient unless backed by credible design elements—like clearly justified material choices, mechanistic rationale, or realistic testing procedures—that convincingly cover the claimed range. \
                   If the modification involves new characterization tools, materials, or system designs, it should only be considered an improvement if it corrects a known deficiency, adds clearly required experimental coverage, or resolves an unaddressed mechanism in the original hypothesis. Simply stating that a technique or material could theoretically yield more data or handle wider conditions is not enough unless it fills a real gap in the existing design or provides critical data that was missing before. \
                   A strong hypothesis should become progressively more precise and experimentally grounded with each iteration, using meaningful refinements that improve scientific logic or fill gaps. It is not improved if it includes unnecessary, redundant, or overly restrictive details that do not serve a genuine scientific purpose. Nor is it improved if it removes essential components or relies on unsubstantiated claims of broader capability. The model should not favor a hypothesis purely because it sounds novel, claims bigger numerical ranges, or appears simpler. Instead, it should prefer hypotheses that are chemically coherent, realistically testable, and address any weaknesses in the prior version. \
                   Finally, more detail is generally useful so chemists do not have to guess key conditions. However, those details must be purposeful, chemically justified, and must enhance the hypothesis’s feasibility in real experiments. If the added detail or approach does not meaningfully strengthen the hypothesis’s ability to answer the research question, it is not an improvement. \
                   {hierarchy_info} \
                   {strict_to_hyp2_info} \
                   Decide whether the updated hypothesis is better than the base hypothesis, focusing on fundamental {DISCIPLINE} principles, meaningful details, and whether the modification addresses a concrete gap or enhances testability. If you accept the update, explain what shortcoming it corrects or how it strengthens the hypothesis. If you reject it, explain why the base hypothesis remains more valid or complete and why the modification does not add real value or fails to align with fundamental {DISCIPLINE}. \
                   The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nIMPORTANT: Please reason through your comparison first before making your selection. After giving your selection, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Selection of research hypothesis candidate starts:** [When specifying the selection, use the candidate's ID] **Selection of research hypothesis candidate ends**\n\n**Reasoning process starts:** [a concise summary of the key reasoning steps that led to this selection] **Reasoning process ends**"]
    elif module_name == "pairwise_compare_unify_response":
        # assist_info: [relation_hyp1_hyp2]
        #   relation_hyp1_hyp2: "strict_to_hyp2" or "same_hyp1_hyp2"
        assert len(assist_info) == 1
        assert assist_info[0] == "strict_to_hyp2" or assist_info[0] == "same_hyp1_hyp2" 
        # strict_to_hyp2_info
        if assist_info[0] == "strict_to_hyp2":
            strict_to_hyp2_info = "We should be careful when making modifications to research hypothesis candidate 1: since once we accept the modifications (Research hypothesis candidate 2), it might be hard to revert them back." 
        else:
            strict_to_hyp2_info = ""
        prompts = [f"You are assisting chemists in refining a research hypothesis. Given a base research hypothesis (Research hypothesis candidate 1) and an updated version (Research hypothesis candidate 2), your task is to determine whether the updated hypothesis is an improvement over the base hypothesis. The aim of this iterative process is to gradually refine an initially coarse or partially novel hypothesis into one that is effective, scientifically valid, and sufficiently detailed for real {DISCIPLINE} experiments. \
                   A hypothesis is only improved if it demonstrably enhances its scientific validity, conceptual clarity, or experimental feasibility based on fundamental {DISCIPLINE} principles. More details do not automatically make a hypothesis better. A more specific hypothesis is generally preferred only when it corrects or addresses a shortcoming in the original proposal, or significantly improves the clarity and testability of the hypothesis. If an added element does not clearly resolve a known gap or make the hypothesis more feasible for experimental validation, then the base hypothesis should be preferred. \
                   While novelty can be valuable, validity and rigorous detail take precedence in this particular task. The original or coarse-grained hypothesis already establishes some novel direction or concept. Your role here is to decide if the updated hypothesis is more practically implementable or scientifically grounded, rather than simply more novel. Claims of better performance—such as a broader temperature range or superior output—should be evaluated in light of fundamental {DISCIPLINE} and adequacy of design details, not just based on self-reported figures. \
                   Justifications for added details are encouraged whenever possible. However, even if a new element is not fully explained, it may still be acceptable if it addresses a concrete gap or significantly strengthens the experimental basis of the hypothesis. The main criterion is whether the modification provides unique and necessary improvements over what was already in place. If the modification adds advanced-sounding techniques or claims to cover a broader range without showing how it is fundamentally achieved and supported by chemical reasoning, it should not be considered an improvement. \
                   Experimental feasibility should not be the sole deciding factor. At this stage, scientific soundness is more important than immediate ease of implementation. That said, a hypothesis that is rigorous and includes enough detail for chemists to conduct real experiments should be favored over one that is vague or leaves essential conditions undefined. Self-reported operational ranges or performance claims are not sufficient unless backed by credible design elements—like clearly justified material choices, mechanistic rationale, or realistic testing procedures—that convincingly cover the claimed range. \
                   If the modification involves new characterization tools, materials, or system designs, it should only be considered an improvement if it corrects a known deficiency, adds clearly required experimental coverage, or resolves an unaddressed mechanism in the original hypothesis. Simply stating that a technique or material could theoretically yield more data or handle wider conditions is not enough unless it fills a real gap in the existing design or provides critical data that was missing before. \
                   A strong hypothesis should become progressively more precise and experimentally grounded with each iteration, using meaningful refinements that improve scientific logic or fill gaps. It is not improved if it includes unnecessary, redundant, or overly restrictive details that do not serve a genuine scientific purpose. Nor is it improved if it removes essential components or relies on unsubstantiated claims of broader capability. The model should not favor a hypothesis purely because it sounds novel, claims bigger numerical ranges, or appears simpler. Instead, it should prefer hypotheses that are chemically coherent, realistically testable, and address any weaknesses in the prior version. \
                   Finally, more detail is generally useful so chemists do not have to guess key conditions. However, those details must be purposeful, chemically justified, and must enhance the hypothesis’s feasibility in real experiments. If the added detail or approach does not meaningfully strengthen the hypothesis’s ability to answer the research question, it is not an improvement. \
                   {strict_to_hyp2_info} \
                   Now that three experts have provided their preference of the two hypotheses and their reason, please consider their advice and provide your final preference and reason. Note that it is not about voting, but about which preference is more reasonable. \
                   The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPreference and reasons from the three experts are: \n", f"\nNow, please decide whether the updated hypothesis is better than the base hypothesis, focusing on fundamental {DISCIPLINE} principles, meaningful details, and whether the modification addresses a concrete gap or enhances testability. If you accept the update, explain what shortcoming it corrects or how it strengthens the hypothesis. If you reject it, explain why the base hypothesis remains more valid or complete and why the modification does not add real value or fails to align with fundamental {DISCIPLINE}.\n\nIMPORTANT: Please reason through your comparison first before making your selection. After giving your selection, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Selection of research hypothesis candidate starts:** [When specifying the selection, use the candidate's ID] **Selection of research hypothesis candidate ends**\n\n**Reasoning process starts:** [a concise summary of the key reasoning steps considering the experts' opinions] **Reasoning process ends**"]
    # update pairwise_compare_prev_update's prompt with gpt-4o
    elif module_name == "pairwise_compare_between_final_hyp":
        # no need to care about assist_info
        prompts = [f"You are assisting chemists in refining a research hypothesis. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more effective in addressing the research question. The goal is to evaluate each hypothesis based on its scientific validity, conceptual clarity, experimental feasibility, and how well it addresses the inherent challenges of the research question. \
                   A hypothesis is considered better if it demonstrably enhances its scientific validity, experimental feasibility, and provides a clear pathway to answering the research question based on fundamental {DISCIPLINE} principles. More details do not automatically make a hypothesis better, but a more specific hypothesis is generally preferred when it addresses the research question more directly and improves the experimental approach. If an added detail or component does not resolve a genuine gap or make the hypothesis more feasible, it should not be considered an improvement. \
                   While novelty is valuable, scientific rigor and valid experimental design are paramount in this task. Both hypotheses are independent and should be evaluated on their own merits. Novelty and novelty claims should be evaluated based on their scientific validity and chemically grounded justification, not just their potential for new insights. Any addition that is chemically reasonable and improves testability of the hypothesis will be considered beneficial. \
                   Justifications for added details are encouraged whenever possible. However, lack of explicit justification should not automatically disqualify an improvement if the added detail is chemically reasonable and directly improves feasibility or address a critical gap in the design. The model should evaluate each modification or addition based on whether it addresses the primary scientific challenge or improves the experimental approach, making the hypothesis more suitable for testing the research question. \
                   Experimental feasibility should not dominate the comparison. Scientific soundness and experimentally grounded principles should be prioritized. A hypothesis that is scientifically robust but more challenging to implement is preferable to one that sacrifices scientific rigor for simplicity. The ultimate goal is to ensure that the hypothesis is testable and aligned with fundamental {DISCIPLINE} principles, regardless of the level of complexity involved. \
                   When comparing hypotheses, focus on whether the hypothesis addresses the core aspects of the research question and whether it provides a clear, experimentally feasible solution. A strong hypothesis should cover all aspects of the question, and more detail is encouraged if it serves a specific purpose in refining the experimental setup or in resolving scientific uncertainties. The model should not favor one hypothesis over the other based on simplicity alone. If a hypothesis is vague or lacks necessary details, it should be rejected for being underdeveloped, even if it sounds conceptually appealing. \
                   Now, compare the following two hypotheses. The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPlease use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID. Please evaluate **both hypotheses equally**, ensuring that the order in which they are presented does not influence your evaluation. Both hypotheses should be judged by the same criteria and standards, and no preference should be given to the first hypothesis simply because it is listed first."]
    # update pairwise_compare_prev_update's prompt with gpt-4o
    elif module_name == "pairwise_compare_between_final_hyp_unify_response":
        # no need to care about assist_info
        prompts = [f"You are assisting chemists in refining a research hypothesis. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more effective in addressing the research question. The goal is to evaluate each hypothesis based on its scientific validity, conceptual clarity, experimental feasibility, and how well it addresses the inherent challenges of the research question. \
                   A hypothesis is considered better if it demonstrably enhances its scientific validity, experimental feasibility, and provides a clear pathway to answering the research question based on fundamental {DISCIPLINE} principles. More details do not automatically make a hypothesis better, but a more specific hypothesis is generally preferred when it addresses the research question more directly and improves the experimental approach. If an added detail or component does not resolve a genuine gap or make the hypothesis more feasible, it should not be considered an improvement. \
                   While novelty is valuable, scientific rigor and valid experimental design are paramount in this task. Both hypotheses are independent and should be evaluated on their own merits. Novelty and novelty claims should be evaluated based on their scientific validity and chemically grounded justification, not just their potential for new insights. Any addition that is chemically reasonable and improves testability of the hypothesis will be considered beneficial. \
                   Justifications for added details are encouraged whenever possible. However, lack of explicit justification should not automatically disqualify an improvement if the added detail is chemically reasonable and directly improves feasibility or address a critical gap in the design. The model should evaluate each modification or addition based on whether it addresses the primary scientific challenge or improves the experimental approach, making the hypothesis more suitable for testing the research question. \
                   Experimental feasibility should not dominate the comparison. Scientific soundness and experimentally grounded principles should be prioritized. A hypothesis that is scientifically robust but more challenging to implement is preferable to one that sacrifices scientific rigor for simplicity. The ultimate goal is to ensure that the hypothesis is testable and aligned with fundamental {DISCIPLINE} principles, regardless of the level of complexity involved. \
                   When comparing hypotheses, focus on whether the hypothesis addresses the core aspects of the research question and whether it provides a clear, experimentally feasible solution. A strong hypothesis should cover all aspects of the question, and more detail is encouraged if it serves a specific purpose in refining the experimental setup or in resolving scientific uncertainties. The model should not favor one hypothesis over the other based on simplicity alone. If a hypothesis is vague or lacks necessary details, it should be rejected for being underdeveloped, even if it sounds conceptually appealing. \
                   Now that three experts have provided their preference of the two hypotheses and their reason, please consider their advice and provide your final preference and reason. Note that it is not about voting, but about which preference is more reasonable. \
                   The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPreference and reasons from the three experts are: \n", f"\nNow, please decide whether the updated hypothesis is better than the base hypothesis, focusing on fundamental {DISCIPLINE} principles, meaningful details, and whether the modification addresses a concrete gap or enhances testability. If you accept the update, explain what shortcoming it corrects or how it strengthens the hypothesis. If you reject it, explain why the base hypothesis remains more valid or complete and why the modification does not add real value or fails to align with fundamental {DISCIPLINE}. Use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID. Please evaluate **both hypotheses equally**, ensuring that the order in which they are presented does not influence your evaluation. Both hypotheses should be judged by the same criteria and standards, and no preference should be given to the first hypothesis simply because it is listed first."]
    elif module_name == "pairwise_compare_effectiveness":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more effective in addressing the research question. By effectiveness, we mean which hypothesis is more likely to result in a successful experimental outcome (e.g., better performance) when tested in a wet lab experiment. \
                   Your evaluation should focus solely on effectiveness, without considering aspects such as novelty, level of detail, or experimental complexity. The key objective is to assess the scientific validity and feasibility of the hypothesis in achieving the stated research goal. You should not assume that a hypothesis will perform well simply because it claims to do so; instead, base your judgment on your own understanding of the relevant {DISCIPLINE} principles.  \
                   When comparing hypotheses, specificity of parameters (e.g., a fixed 1:1 proportion) should not be considered a disadvantage, nor should it be framed as an inherent advantage. While more specific parameters may provide useful guidance for experimental design, their presence or absence should not influence the comparison unless they directly impact the effectiveness of the hypothesis in addressing the research question. \
                   Please evaluate the two hypotheses based on their effectiveness in addressing the research question. \
                   The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPlease use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    elif module_name == "pairwise_compare_effectiveness_unify_response":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more effective in addressing the research question. By effectiveness, we mean which hypothesis is more likely to result in a successful experimental outcome (e.g., better performance) when tested in a wet lab experiment. \
                   Your evaluation should focus solely on effectiveness, without considering aspects such as novelty, level of detail, or experimental complexity. The key objective is to assess the scientific validity and feasibility of the hypothesis in achieving the stated research goal. You should not assume that a hypothesis will perform well simply because it claims to do so; instead, base your judgment on your own understanding of the relevant {DISCIPLINE} principles.  \
                   When comparing hypotheses, specificity of parameters (e.g., a fixed 1:1 proportion) should not be considered a disadvantage, nor should it be framed as an inherent advantage. While more specific parameters may provide useful guidance for experimental design, their presence or absence should not influence the comparison unless they directly impact the effectiveness of the hypothesis in addressing the research question. \
                   Now that three experts have provided their preference of the two hypotheses and their reason, please consider their advice and provide your final preference and reason. Note that it is not about voting, but about which preference is more reasonable. \
                   The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPreference and reasons from the three experts are: \n", "\nNow, please decide which hypothesis is more effective in addressing the research question. Use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    elif module_name == "pairwise_compare_novelty":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more novel. By novelty, it means comparing to the existing literature, which hypothesis is more original, creative, or innovative in addressing the research question. \
                    You should focus solely on the novelty aspect of the hypothesis for comparison. Here are some example aspects that you don't need to consider for the comparison: detailedness, experimental feasibility, and effectiveness. \
                    Please evaluate the two hypotheses based on their novelty in addressing the research question. \
                    The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPlease use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    elif module_name == "pairwise_compare_novelty_unify_response":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more novel. By novelty, it means comparing to the existing literature, which hypothesis is more original, creative, or innovative in addressing the research question. \
                    You should focus solely on the novelty aspect of the hypothesis for comparison. Here are some example aspects that you don't need to consider for the comparison: detailedness, experimental feasibility, and effectiveness. \
                    Now that three experts have provided their preference of the two hypotheses and their reason, please consider their advice and provide your final preference and reason. Note that it is not about voting, but about which preference is more reasonable. \
                    The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPreference and reasons from the three experts are: \n", "\nNow, please decide which hypothesis is more novel in addressing the research question. Use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    elif module_name == "pairwise_compare_detailedness":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more detailed and complete. By detailedness, ideally it means (1) a hypothesis does not miss any methodological detail so that the full method is clear. As a bonus point, the reaction mechanism for each methodological detail is also clear so that scientist can understand why the method is designed in its current shape; (2) a hypothesis does not miss any experimental detail, so that scientists can directly test it in web lab experiment. \
                    You should focus solely on the detailedness aspect of the hypothesis for comparison. Here are some example aspects that you don't need to consider for the comparison: novelty, experimental feasibility, and effectiveness. \
                    Please evaluate the two hypotheses based on their detailedness in addressing the research question. \
                    The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPlease use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    elif module_name == "pairwise_compare_detailedness_unify_response":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more detailed and complete. By detailedness, ideally it means (1) a hypothesis does not miss any methodological detail so that the full method is clear. As a bonus point, the reaction mechanism for each methodological detail is also clear so that scientist can understand why the method is designed in its current shape; (2) a hypothesis does not miss any experimental detail, so that scientists can directly test it in web lab experiment. \
                    You should focus solely on the detailedness aspect of the hypothesis for comparison. Here are some example aspects that you don't need to consider for the comparison: novelty, experimental feasibility, and effectiveness. \
                    Now that three experts have provided their preference of the two hypotheses and their reason, please consider their advice and provide your final preference and reason. Note that it is not about voting, but about which preference is more reasonable. \
                    The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPreference and reasons from the three experts are: \n", "\nNow, please decide which hypothesis is more detailed and complete in addressing the research question. Use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    elif module_name == "pairwise_compare_feasibility":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more experimentally feasible. By feasibility, ideally it means a hypothesis does not miss important methodological or experimental details (so that it at least can be understood and therefore possible to be implemented in a real web lab experiment), and that the hypothesis is not overly or unnecessarily complex to implement. \
                    You should focus solely on the feasibility aspect of the hypothesis for comparison: whether a hypothesis is not overly or unnecessarily complex on the basis of providing enough details. Here are some example aspects that you don't need to consider for the comparison: effectiveness and novelty. \
                    Please evaluate the two hypotheses based on their feasibility in addressing the research question. \
                    The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPlease use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    elif module_name == "pairwise_compare_feasibility_unify_response":
        assert len(assist_info) == 1
        assert assist_info[0] == "same_hyp1_hyp2" 
        prompts = [f"You are assisting chemists in evaluating research hypotheses. Given two separate research hypotheses (Research hypothesis candidate 1 and Research hypothesis candidate 2), your task is to determine which hypothesis is more experimentally feasible. By feasibility, ideally it means a hypothesis does not miss important methodological or experimental details (so that it at least can be understood and therefore possible to be implemented in a real web lab experiment), and that the hypothesis is not overly or unnecessarily complex to implement. \
                    You should focus solely on the feasibility aspect of the hypothesis for comparison: whether a hypothesis is not overly or unnecessarily complex on the basis of providing enough details. Here are some example aspects that you don't need to consider for the comparison: effectiveness and novelty. \
                    Now that three experts have provided their preference of the two hypotheses and their reason, please consider their advice and provide your final preference and reason. Note that it is not about voting, but about which preference is more reasonable. \
                    The research question is: \n", "\nResearch hypothesis candidate 1 is: \n", "\nResearch hypothesis candidate 2 is: \n", "\nPreference and reasons from the three experts are: \n", "\nNow, please decide which hypothesis is more experimentally feasible in addressing the research question. Use the following response format: 'Reasoning process: \nSelection of research hypothesis candidate: \n'. When specifying the selection, use the candidate's ID."]
    else:
        raise NotImplementedError

    return prompts



def preprocessing_instruction_prompts(module_name, assist_info=None):
    if module_name == "preprocess_cg_hyp_to_research_direction":
        prompts = ["We want to replicate the interaction between professor and his phd student's interaction. Specifically, given a research question, the professor will give the student a general research direction, and then the student work on that direction to propose a specific research hypothesis.  \
                   Now we already know the research question, and the research hypothesis proposed by the student, and we want to infer what the research direction the professor has given. You will be also given the major functional components of the research hypothesis.  \
                   The professor's research direction in general should include the one-level more general chemical class of each of the major functional components (but not two-level more). For example, if one component is Barium Titanate (BaTiO₃), then the one-level more general level can be Perovskite Oxide, but not two-level more general level such as Ceramic. For another example, if one component is about Sulfate Ion, the one-level more general level can be Hofmeister series. One exception is that if a major component is already not specified enough (e.g., Perovskite Oxide, or Hofmeister series) and it is not directly mention in the research hypothesis, you should try to rephrase it with very similar concept (if there is any) but no need to make it one level more general, else if it is very hard to find a similar enough concept, you can keep it as it is.  For each major component however, there could be several one-level more general chemical class. For example, the Sulfate Ion can belong to both Hofmeister series and Oxyanion. But it is important that you should assign the one-level more general class that is more directly relevant to the research question and the research hypothesis.  However, if there is any component in the research hypothesis that is not in the summarized major components, you should keep it as it is in the research direction. For example, if there is Guanidine sulfate mentioned in the research hypothesis, and Hofmeister series is included in the major components (but Guanidine ion is not included in the major components), it is clear that Sulfate ions is related to Hofmeister series, but the Guanidine ion is not included in the major components, so in this case you should keep the Guanidine ion in the research direction given by the professor. \
                   The professor's research direction should also not specifiy any reaction mechanism, i.e., if there is any reaction mechanism in the research hypothesis, you should delete them but you should add enough descriptions that are indicative to the deleted reaction mechanism.  \
                   Although the research direction can only contain the one-level more general chemical class of each of the major components, and if there is any reaction mechanism involved, you should try not to include any exact same word of the major component or the reaction mechanism in the research direction, however the research direction should be enough inspiring to the students to come out of the research hypothesis: to do it you may consider to add some more descriptions in addition to give the one-level more general chemicals, or some other ways that you think is proper. You can mention the exact word if it is the case as we have discussed that the major component is already not specified enough (e.g., Hofmeister series, Perovskite Oxide). The research direction you are formulating should include information related to all of the listed major components (e.g., one-level more general chemical, or a similar/same concept if a major components if already not specified enough). \
                   Now let's have a try. The research question is: \n", "\nThe research hypothesis proposed by the student is: \n", "\nThe major components of the research hypothesis are: \n", "\nNow, please try to guess the research direction the professor give the phd student in the first place.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Research direction starts:** [the research direction] **Research direction ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this research direction] **Reasoning Process ends**"]
    elif module_name == "preprocess_cg_hyp_to_research_direction_refine":
        prompts = ["We want to replicate the interaction between professor and his phd student's interaction. Specifically, given a research question, the professor will give the student a general research direction, and then the student work on that direction to propose a specific research hypothesis.  \
                   Now we already know the research question, and the research hypothesis proposed by the student, and we want to infer what the research direction the professor has given. You will be also given the major functional components of the research hypothesis.  \
                   The professor's research direction in general should include the one-level more general chemical class of each of the major functional components (but not two-level more). For example, if one component is Barium Titanate (BaTiO₃), then the one-level more general level can be Perovskite Oxide, but not two-level more general level such as Ceramic. For another example, if one component is about Sulfate Ion, the one-level more general level can be Hofmeister series. One exception is that if a major component is already not specified enough (e.g., Perovskite Oxide, or Hofmeister series) and it is not directly mention in the research hypothesis, you should try to rephrase it with very similar concept (if there is any) but no need to make it one level more general, else if it is very hard to find a similar enough concept, you can keep it as it is.  For each major component however, there could be several one-level more general chemical class. For example, the Sulfate Ion can belong to both Hofmeister series and Oxyanion. But it is important that you should assign the one-level more general class that is more directly relevant to the research question and the research hypothesis.  However, if there is any component in the research hypothesis that is not in the summarized major components, you should keep it as it is in the research direction. For example, if there is Guanidine sulfate mentioned in the research hypothesis, and Hofmeister series is included in the major components (but Guanidine ion is not included in the major components), it is clear that Sulfate ions is related to Hofmeister series, but the Guanidine ion is not included in the major components, so in this case you should keep the Guanidine ion in the research direction given by the professor. \
                   The professor's research direction should also not specifiy any reaction mechanism, i.e., if there is any reaction mechanism in the research hypothesis, you should delete them but you should add enough descriptions that are indicative to the deleted reaction mechanism.  \
                   Although the research direction can only contain the one-level more general chemical class of each of the major components, and if there is any reaction mechanism involved, you should try not to include any exact same word of the major component or the reaction mechanism in the research direction, however the research direction should be enough inspiring to the students to come out of the research hypothesis: to do it you may consider to add some more descriptions in addition to give the one-level more general chemicals, or some other ways that you think is proper. You can mention the exact word if it is the case as we have discussed that the major component is already not specified enough (e.g., Hofmeister series, Perovskite Oxide). The research direction you are formulating should include information related to all of the listed major components (e.g., one-level more general chemical, or a similar/same concept if a major components if already not specified enough). \
                   Here is a trial to recover the professor's research direction. It might be not perfect, or violate some of the guildlines. Please help to refine the research direction. \
                   Now let's have a try. The research question is: \n", "\nThe research hypothesis proposed by the student is: \n", "\nThe major components of the research hypothesis are: \n", "\nThe trial to recover the professor's research direction is: \n", "\nNow, please try to guess the research direction the professor give the phd student in the first place.\n\nIMPORTANT: Please reason through this task first before providing your answer. After giving your answer, summarize your reasoning.\n\nYour response MUST follow this exact format:\n\n**Research direction starts:** [the research direction] **Research direction ends**\n\n**Reasoning Process starts:** [a concise summary of the key reasoning steps that led to this research direction] **Reasoning Process ends**"]
    else:
        raise NotImplementedError
    return prompts

