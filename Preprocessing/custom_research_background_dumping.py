import os, json, argparse


def research_background_to_json(custom_research_background_and_coarse_hyp_path):
    # YOUR RESEARCH QUESTION HERE
    research_question = '''
    YOUR RESEARCH QUESTION HERE
    '''

    # YOUR BACKGROUND SURVEY HERE
    background_survey = '''
    YOUR BACKGROUND SURVEY HERE
    '''

    # YOUR COARSE GRAINED HYPOTHESIS HERE
    coarse_grained_hypothesis = '''
    YOUR COARSE GRAINED HYPOTHESIS HERE
    '''


    # Save the research question and background survey to a JSON file
    with open(custom_research_background_and_coarse_hyp_path, "w") as f:
        json.dump([[research_question.strip(), background_survey.strip(), coarse_grained_hypothesis.strip()]], f, indent=4)
    print("Research background and coarse-grained hypothesis saved to", custom_research_background_and_coarse_hyp_path)



def moosechem_ranking_file_to_json(moosechem_ranking_file_path, custom_research_background_and_coarse_hyp_path):
    # Load the JSON file
    with open(moosechem_ranking_file_path, "r") as f:
        data = json.load(f)

    research_question = list(data[0].keys())[0]

    # YOUR BACKGROUND SURVEY HERE
    background_survey = '''
    YOUR BACKGROUND SURVEY HERE
    '''

    final_data_list = []
    for cur_id in range(len(data[0][research_question])):
        cur_hypothesis = data[0][research_question][cur_id][0]
        final_data_list.append([research_question, background_survey, cur_hypothesis])


    # Save the research question & background survey & coarse-grained hypothesis to a JSON file
    with open(custom_research_background_and_coarse_hyp_path, "w") as f:
        json.dump(final_data_list, f, indent=4)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moosechem_ranking_file_path", type=str, default="~/MOOSE-Chem/Geo/evaluation_GPT4o-mini.json", help="the path to the output file of evaluate.py")
    parser.add_argument("--custom_research_background_and_coarse_hyp_path", type=str, default="./custom_research_background_and_coarse_hyp.json", help="the path to the research background and coarse-grained hypothesis file. The format is [research question, background survey, coarse-grained hypothesis], and saved in a json file. ")
    parser.add_argument("--if_load_from_moosechem_ranking_file", type=int, default=1)
    args = parser.parse_args()

    args.moosechem_ranking_file_path = os.path.expanduser(args.moosechem_ranking_file_path)
    

    if args.if_load_from_moosechem_ranking_file == 0:
        research_background_to_json(args.custom_research_background_and_coarse_hyp_path)
    elif args.if_load_from_moosechem_ranking_file == 1:
        moosechem_ranking_file_to_json(args.moosechem_ranking_file_path, args.custom_research_background_and_coarse_hyp_path)
    else:
        raise ValueError("if_load_from_moosechem_ranking_file should be 0 or 1")

    