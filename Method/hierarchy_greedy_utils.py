import json, pickle, random



# Q: this ranking method might be unfair in terms of the index of the hypothesis in the list; but when args.num_init_for_EU is small (e.g., 3), it should be fine
# Function: rank the hypothesis based on pairwise_compare.compare 
# Input: 
#   hypothesis_reasons: [[hyp, reason], [hyp, reason], ...]
#   PairwiseCompare_object.compare(research_question, hypothesis1, hypothesis2, instruction_mode="same_hyp1_hyp2")
# Output: the index of the best hypothesis
def find_the_best_hypothesis_among_list(cur_q, cur_survey, hypothesis_reasons, PairwiseCompare_object):
    assert len(hypothesis_reasons) >= 1
    best_id = 0
    for cur_id in range(1, len(hypothesis_reasons)):
        # selection_reason: [selection (1/2), reason]
        selection_reason = PairwiseCompare_object.compare(cur_q, hypothesis_reasons[best_id][0], hypothesis_reasons[cur_id][0], instruction_mode="same_hyp1_hyp2")
        if selection_reason[0] == 2:
            best_id = cur_id
    return best_id



class HGNode:
    # base_hyp_reason: [hyp, reason]
    # full_generated_hyp: [search_results_init_0, search_results_init_1, ..., search_results_init_(num_init_for_EU), recombination_results_all_steps]
    #       search_results_init_0 / recombination_results_all_steps: [[hyp, reason], [hyp, reason], ...]
    # next_hierarchy_hyp: [[hyp, reason], [hyp, reason], ...]
    def __init__(self, hierarchy_id, base_hyp_reason, full_generated_hyp=None, next_hierarchy_hyp=None):
        assert isinstance(hierarchy_id, int), print("hierarchy_id: ", hierarchy_id)
        assert isinstance(base_hyp_reason[0], str), print("base_hyp_reason: ", base_hyp_reason)
        assert isinstance(full_generated_hyp, list) or full_generated_hyp == None, print("full_generated_hyp: ", full_generated_hyp)
        assert isinstance(next_hierarchy_hyp, list) or next_hierarchy_hyp == None, print("next_hierarchy_hyp: ", next_hierarchy_hyp)
        self.hierarchy_id = hierarchy_id
        # the input hyp from the previous hierarchy that this node starts its search from
        self.base_hyp_reason = base_hyp_reason
        # full_generated_hyp
        if full_generated_hyp != None:
            self.full_generated_hyp = full_generated_hyp
        else:
            self.full_generated_hyp = []
        # next_hierarchy_hyp
        if next_hierarchy_hyp != None:
            self.next_hierarchy_hyp = next_hierarchy_hyp
        else:
            self.next_hierarchy_hyp = []
        self.children = []
        self.parent = None

    def add_child(self, child):
        assert isinstance(child, HGNode)
        self.children.append(child)

    # Function: add new_next_hierarchy_hyp to self.next_hierarchy_hyp
    # new_next_hierarchy_hyp: [[hyp, reason]]
    # self.next_hierarchy_hyp: [[hyp, reason], [hyp, reason], ...]
    def update_next_hierarchy_hyp(self, new_next_hierarchy_hyp):
        assert isinstance(new_next_hierarchy_hyp, list)
        if len(self.next_hierarchy_hyp) > 0:
            print("Warning: self.next_hierarchy_hyp is not empty, next_hierarchy_hyp: ", self.next_hierarchy_hyp)
        print("Updating next_hierarchy_hyp. len(self.next_hierarchy_hyp): {}; len(new_next_hierarchy_hyp): {}".format(len(self.next_hierarchy_hyp), len(new_next_hierarchy_hyp)))
        self.next_hierarchy_hyp += new_next_hierarchy_hyp

    def replace_full_generated_hyp(self, full_generated_hyp):
        assert isinstance(full_generated_hyp, list)
        if len(self.full_generated_hyp) > 0:
            raise ValueError("self.full_generated_hyp is not empty, full_generated_hyp: ", self.full_generated_hyp)
        self.full_generated_hyp = full_generated_hyp

    def set_parent(self, parent):
        assert isinstance(parent, HGNode)
        assert self.parent is None, print("self.parent: ", self.parent)
        self.parent = parent

    # best_hyp_in_all_previous_hierarchies: [best_hyp_from_hierarchy_0, best_hyp_from_hierarchy_1, ...]; best_hyp_from_hierarchy_0/1: str
    def find_best_hyp_in_all_previous_hierarchies(self):
        # find the best hypothesis in all previous hierarchies
        best_hyp_in_all_previous_hierarchies = []
        cur_node = self
        while cur_node.parent is not None:
            cur_node = cur_node.parent
            # if a better hypothesis is found in this parant node (len(cur_node.full_generated_hyp) > 0), then update best_hyp_in_all_previous_hierarchies; else we skip it
            if len(cur_node.full_generated_hyp) > 0:
                best_hyp_cur_node = cur_node.full_generated_hyp[-1][-1][0]
                best_hyp_in_all_previous_hierarchies.append(best_hyp_cur_node)
        return best_hyp_in_all_previous_hierarchies


    def to_dict(self):
        """Convert the node to a dictionary for JSON serialization."""
        return {
            'hierarchy_id': self.hierarchy_id,
            'base_hyp_reason': self.base_hyp_reason,
            'full_generated_hyp': self.full_generated_hyp,
            'next_hierarchy_hyp': self.next_hierarchy_hyp,
            'children': [child.to_dict() for child in self.children],  # Recursively serialize children
            'parent': self.parent
        }



class HGTree:
    # base_hyp_reason: [hyp, reason]
    def __init__(self, hierarchy_id, base_hyp_reason, full_generated_hyp=None, next_hierarchy_hyp=None):
        assert isinstance(hierarchy_id, int), print("hierarchy_id: ", hierarchy_id)
        assert isinstance(base_hyp_reason[0], str), print("base_hyp_reason: ", base_hyp_reason)
        assert isinstance(full_generated_hyp, list) or full_generated_hyp == None, print("full_generated_hyp: ", full_generated_hyp)
        assert isinstance(next_hierarchy_hyp, list) or next_hierarchy_hyp == None, print("next_hierarchy_hyp: ", next_hierarchy_hyp)
        self.root = HGNode(hierarchy_id, base_hyp_reason, full_generated_hyp, next_hierarchy_hyp)

    # Function: 
    #   (1) find the top k hypothesis to enter a hierarchy; (2) update the next_hierarchy_hyp of each relevant node
    # Input
    #   hierarchy_id: int; hierarchy_id > 1
    #   PairwiseCompare_object.compare(research_question, hypothesis1, hypothesis2, instruction_mode="same_hyp1_hyp2")
    #   compare_mode: int; 0 or 1
    #       0: (a shortcut way) directly return the best hypothesis in each branch (in this case, the number of previous branches should be always equal to the beam size)
    #       1: (a more general way) compare all the hypothesis candidates in the previous hierarchy and select the top k hypothesis
    # Output
    #   if_success: bool
    #   topk_hypothesis: [[hyp, reason, node], [hyp, reason, node], ...]
    def find_the_top_k_hypothesis_to_enter_a_hierarchy_and_set_next_hierarchy_hyp_to_nodes(self, research_question, background_survey, k, hierarchy_id, PairwiseCompare_object, compare_mode=0):
        # hierarchy_id == 0 is the first hierarchy
        assert hierarchy_id > 0, print("hierarchy_id: ", hierarchy_id)

        ### all_all_local_minimum_and_recomb_hyp_candidates / all_recomb_hyp_candidates / all_best_local_minimum_hyp_candidates: used for select top k hypothesis in compare_mode 1
        # all_all_local_minimum_and_recomb_hyp_candidates: all of the hypothesis candidates in the previous hierarchy: [[hyp, reason, node], [hyp, reason, node], ...]
        all_all_local_minimum_and_recomb_hyp_candidates = []
        # all_recomb_hyp_candidates: [[hyp, reason, node], [hyp, reason, node], ...] (only the final recombination results for each node in the previous hierarchy)
        all_recomb_hyp_candidates = []
        # all_best_local_minimum_hyp_candidates: [[hyp, reason, node], [hyp, reason, node], ...] (only the best local minimum results for each node in the previous hierarchy)
        all_best_local_minimum_hyp_candidates = []
        ## iterate over all nodes in a breath-first way; if a node is in the previous hierarchy, then add its full_generated_hyp to all_all_local_minimum_and_recomb_hyp_candidates
        queue = [self.root]
        while len(queue) > 0:
            cur_node = queue.pop(0)
            # We consider two kinds of nodes: (1) the nodes in the previous hierarchy and have found better hypothesis in that hierarchy; (2) any other leaf nodes that have found no better hypothesis in whichever hierachy (but its base hypothesis can still be meaningful)
            if len(cur_node.full_generated_hyp) > 0:
                # the nodes in the previous hierarchy and have found better hypothesis in that hierarchy
                if cur_node.hierarchy_id == hierarchy_id - 1:
                    # should have at least one found better hyp + recombination hyp
                    assert len(cur_node.full_generated_hyp) > 1, print(f"len(cur_node.full_generated_hyp): {len(cur_node.full_generated_hyp)};\ncur_node.full_generated_hyp: {cur_node.full_generated_hyp}")
                    ## all_all_local_minimum_and_recomb_hyp_candidates
                    cur_node_all_local_minimum_and_recomb_hyp_candidates = [[cur_search_results_init[-1][0], cur_search_results_init[-1][1], cur_node] for cur_search_results_init in cur_node.full_generated_hyp]
                    all_all_local_minimum_and_recomb_hyp_candidates.append(cur_node_all_local_minimum_and_recomb_hyp_candidates)
                    ## all_recomb_hyp_candidates: the default best recombination is the last one in full_generated_hyp
                    cur_node_recomb_hyp_candidates = [cur_node.full_generated_hyp[-1][-1][0], cur_node.full_generated_hyp[-1][-1][1], cur_node]
                    all_recomb_hyp_candidates.append(cur_node_recomb_hyp_candidates)
                    ## all_best_local_minimum_hyp_candidates: the default best local minimum is the second last one in full_generated_hyp
                    cur_node_best_local_minimum_hyp_candidates = [cur_node.full_generated_hyp[-2][-1][0], cur_node.full_generated_hyp[-2][-1][1], cur_node]
                    all_best_local_minimum_hyp_candidates.append(cur_node_best_local_minimum_hyp_candidates)
            else:
                # leaf nodes that have found no better hypothesis in whichever hierachy (but its base hypothesis can still be meaningful)
                if len(cur_node.children) == 0:
                    cur_node_base_hyp_candidate = [cur_node.base_hyp_reason[0], cur_node.base_hyp_reason[1], cur_node]
                    all_all_local_minimum_and_recomb_hyp_candidates.append(cur_node_base_hyp_candidate)
                    all_recomb_hyp_candidates.append(cur_node_base_hyp_candidate)
                    all_best_local_minimum_hyp_candidates.append(cur_node_base_hyp_candidate)
            queue += cur_node.children

        ### select the top k hypothesis from all_all_local_minimum_and_recomb_hyp_candidates by using pairwise comparison; we should do it efficiently; we can use PairwiseCompare_object.compare(research_question, hypothesis1, hypothesis2, instruction_mode="same_hyp1_hyp2") to compare two hypotheses
        # topk_hypothesis: [[hyp, reason, node], [hyp, reason, node], ...]
        topk_hypothesis = []
        ## directly reply the recombination hypothesis of the nodes in the previous hierarchy
        if compare_mode == 0:
            print("compare_mode == 0")
            if len(all_recomb_hyp_candidates) == k:
                topk_hypothesis += all_recomb_hyp_candidates
            elif len(all_recomb_hyp_candidates) > k:
                # iteratively select the best hypothesis in all_recomb_hyp_candidates
                while len(topk_hypothesis) < k:
                    # find the best hypothesis in all_recomb_hyp_candidates
                    best_id = find_the_best_hypothesis_among_list(research_question, background_survey, all_recomb_hyp_candidates, PairwiseCompare_object)
                    topk_hypothesis.append(all_recomb_hyp_candidates[best_id])
                    # remove the best hypothesis from all_recomb_hyp_candidates
                    all_recomb_hyp_candidates.pop(best_id)
            else:
                # first collect the recombination hypothesis for topk_hypothesis
                topk_hypothesis += all_recomb_hyp_candidates
                num_hyp_to_collect = k - len(topk_hypothesis)
                assert num_hyp_to_collect > 0
                # as a replacement of the empty in all_recomb_hyp_candidates, collect the best from cur_node_best_local_minimum_hyp_candidates
                for id_hyp_replace in range(num_hyp_to_collect):
                    # find the best hypothesis in all_best_local_minimum_hyp_candidates
                    if len(all_best_local_minimum_hyp_candidates) > 0:
                        best_id = find_the_best_hypothesis_among_list(research_question, background_survey, all_best_local_minimum_hyp_candidates, PairwiseCompare_object)
                        topk_hypothesis.append(all_best_local_minimum_hyp_candidates[best_id])
                        # remove the best hypothesis from all_best_local_minimum_hyp_candidates
                        all_best_local_minimum_hyp_candidates.pop(best_id)
                if len(topk_hypothesis) < k:
                    print("Warning: len(all_recomb_hyp_candidates) is not equal to k. len(all_recomb_hyp_candidates): {}; len(all_best_local_minimum_hyp_candidates): {}; k: {}; len(topk_hypothesis): {}".format(len(all_recomb_hyp_candidates), len(all_best_local_minimum_hyp_candidates), k, len(topk_hypothesis)))
        ## select the top k hypothesis from all_recomb_hyp_candidates and all_best_local_minimum_hyp_candidates by using pairwise comparison
        elif compare_mode == 1:
            print("compare_mode == 1")
            # all_considered_hyp_candidates: [[hyp, reason, node], [hyp, reason, node], ...]
            all_considered_hyp_candidates = all_recomb_hyp_candidates + all_best_local_minimum_hyp_candidates
            # the order might be unfair, we shuffle it to avoid overly in favor of all_recomb_hyp_candidates or all_best_local_minimum_hyp_candidates
            random.shuffle(all_considered_hyp_candidates)
            if len(all_considered_hyp_candidates) == 0:
                assert len(topk_hypothesis) == 0, print("topk_hypothesis: ", topk_hypothesis)
                # topk_hypothesis: only an empty list []
                return False, topk_hypothesis
                # raise ValueError("len(all_considered_hyp_candidates) == 0")
            elif len(all_considered_hyp_candidates) <= k:
                print("Warning: len(all_considered_hyp_candidates) is less than k. len(all_considered_hyp_candidates): ", len(all_considered_hyp_candidates), "k: ", k)
                topk_hypothesis = all_considered_hyp_candidates
            else:
                print("Finding the top k hypothesis...")
                # find the top k hypothesis
                while len(topk_hypothesis) < k:
                    # find the best hypothesis in all_considered_hyp_candidates
                    best_id = find_the_best_hypothesis_among_list(research_question, background_survey, all_considered_hyp_candidates, PairwiseCompare_object)
                    topk_hypothesis.append(all_considered_hyp_candidates[best_id])
                    # remove the best hypothesis from all_considered_hyp_candidates
                    all_considered_hyp_candidates.pop(best_id)
                assert len(topk_hypothesis) == k, print("topk_hypothesis: ", topk_hypothesis)
        else:
            raise ValueError("Invalid compare_mode: ", compare_mode)
        assert len(topk_hypothesis) > 0, print("topk_hypothesis: ", topk_hypothesis)
        if len(topk_hypothesis) < k:
            print("Warning: len(topk_hypothesis) is less than k. len(topk_hypothesis): ", len(topk_hypothesis), "k: ", k)

        ### update the next_hierarchy_hyp of each relevant node of the top k hypothesis
        for cur_hyp_collection in topk_hypothesis:
            # cur_hyp_collection: [hyp, reason, node]
            cur_hyp_node = cur_hyp_collection[2]
            # print("updating next_hierarchy_hyp, cur_hyp_node: ", cur_hyp_node)
            cur_hyp_node.update_next_hierarchy_hyp([cur_hyp_collection[:2]])
            
        print("Found the top {} hypothesis. Expected number of hypothesis: {}".format(len(topk_hypothesis), k))
        return True, topk_hypothesis
                    

    # save children data in a depth-first manner
    def save(self, filename):
        """Save the tree to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.root.to_dict(), f)

    @staticmethod
    def load(filename):
        """Load a tree from a pickle file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            tree = HGTree(data['hierarchy_id'], data['base_hyp_reason'], data['full_generated_hyp'], data['next_hierarchy_hyp'])
            tree._load_children(tree.root, data['children'])
            return tree
        
    # load children data in a depth-first manner
    def _load_children(self, node, children_data):
        """Recursively load children into the node."""
        for child_data in children_data:
            child_node = HGNode(child_data['hierarchy_id'], child_data['base_hyp_reason'], child_data['full_generated_hyp'], child_data['next_hierarchy_hyp'])
            node.add_child(child_node)
            self._load_children(child_node, child_data['children'])
        # set parent
        for child_node in node.children:
            child_node.set_parent(node)





# Function: concatenate the background_survey with (1) input coarse-grained hypothesis and (2) the corresponding best hypothesis from all the previous hierarchies
# Input: 
#   corresponding_best_hyp_from_previous_hierarchies: [best_hyp_from_hierarchy_0, best_hyp_from_hierarchy_1, ...]; best_hyp_from_hierarchy_0/1: str
#   background_survey / input_cg_hypothesis_prompt: str
#   cur_hierarchy_id: int
# Output: 
#   cur_survey_with_additional_info: str
def get_all_previous_hierarchy_hypothesis_prompt(background_survey, input_cg_hyp, corresponding_best_hyp_from_previous_hierarchies, cur_hierarchy_id):
    # initialize input_cg_hypothesis_prompt
    input_cg_hypothesis_prompt = "\n\nThe coarse-grained hypothesis is: {}\nThe coarse-grained hypothesis is proposed by a student and has not been verified by experiments. ".format(input_cg_hyp)
    ## cur_survey_with_additional_info: (max) background_survey + input coarse-grained hypothesis + all previous hierarchy's best fine-grained hypothesis
    # merge input_cg_hypothesis_prompt and/or all_previous_hierarchy_hypothesis_prompt into background_survey
    if cur_hierarchy_id == 0:
        cur_survey_with_additional_info = background_survey + input_cg_hypothesis_prompt
    elif cur_hierarchy_id > 0:
        # initialize all_previous_hierarchy_hypothesis_prompt
        all_previous_hierarchy_hypothesis_prompt = "\n\nNext we introduce the best hypothesis we have found in each of the lower/previous hierarchies (which contain more general contents). When we make modifications to hypothesis in the current hierarchy, normally we should try to keep it consistent with the best hypotheses from the previous hierarchies (it is encouraged to maintain their general contents while adding details, instead of replacing the general contents with details), unless we have good reasons to change it. \n"
        # len(full_results_all_hierarchy) might be less than cur_hierarchy_id, since some hierarchies might not lead to better hypothesis than the previous hierarchy
        for cur_hierarchy_id_previous in range(len(corresponding_best_hyp_from_previous_hierarchies)):
            all_previous_hierarchy_hypothesis_prompt += "The best hypothesis found in hierarchy ({}) is: {}\n".format(cur_hierarchy_id_previous, corresponding_best_hyp_from_previous_hierarchies[cur_hierarchy_id_previous])
        cur_survey_with_additional_info = background_survey + input_cg_hypothesis_prompt + all_previous_hierarchy_hypothesis_prompt
    else:
        raise ValueError("Invalid cur_hierarchy_id: ", cur_hierarchy_id)
    return cur_survey_with_additional_info
        





# An old version of HierarchyGreedy.get_finegrained_hyp_for_one_research_question_Branching
#   not using the tree search data structure but using lists; only one branch in each hierarchy

    # # Function: get fine-grained hypothesis for one research question WITHOUT branching (one final hypothesis from one hierarchy)
    # # Output
    # # full_results_all_hierarchy: [search_results_all_init_hierarchy_0, search_results_all_init_hierarchy_1, ...]
    # #   search_results_all_init_hierarchy_0: [search_results_init_0, search_results_init_1, ..., search_results_init_(num_init_for_EU), recombination_results_all_steps]
    # #       search_results_init_x / recombination_results_all_steps: [[hyp, reason], [hyp, reason], ...]
    # def get_finegrained_hyp_for_one_research_question_noBranching(self, cur_bkg_id):
    #     # final results
    #     full_results_all_hierarchy = []
    #     ## we take a greedy approach between hierarchical levels
    #     prev_hierarchy_gene_fg_hyp = None
    #     for cur_hierarchy_id in range(self.args.num_hierarchy):
    #         print("Hierarchy ID: ", cur_hierarchy_id)
    #         # basic input information
    #         cur_q = self.bkg_q_list[cur_bkg_id]
    #         cur_survey = self.dict_bkg2survey[cur_q]
    #         cur_cg_hyp = self.dict_bkg2cg_hyp[cur_q]
    #         if cur_hierarchy_id == 0:
    #             print("Initial coarse-grained hypothesis: ", cur_cg_hyp)
    #         # initialize input_cg_hypothesis_prompt and all_previous_hierarchy_hypothesis_prompt
    #         # input_cg_hypothesis_prompt
    #         input_cg_hypothesis_prompt = "\n\nThe coarse-grained hypothesis is: {}\nThe coarse-grained hypothesis is proposed by a student and has not been verified by experiments. ".format(cur_cg_hyp) 
    #         # all_previous_hierarchy_hypothesis_prompt
    #         all_previous_hierarchy_hypothesis_prompt = "\n\nNext we introduce the best hypothesis we have found in each of the lower/previous hierarchies (which contain more general contents). When we make modifications to hypothesis in the current hierarchy, normally we should try to keep it consistent with the best hypotheses from the previous hierarchies (it is encouraged to maintain their general contents while adding details, instead of replacing the general contents with details), unless we have good reasons to change it. \n"
    #         # len(full_results_all_hierarchy) might be less than cur_hierarchy_id, since some hierarchies might not lead to better hypothesis than the previous hierarchy
    #         for cur_hierarchy_id_previous in range(len(full_results_all_hierarchy)):
    #             all_previous_hierarchy_hypothesis_prompt += "The best hypothesis found in hierarchy ({}) is: {}\n".format(cur_hierarchy_id_previous, full_results_all_hierarchy[cur_hierarchy_id_previous][-1][-1][0])
    #         # merge input_cg_hypothesis_prompt and/or all_previous_hierarchy_hypothesis_prompt into cur_survey
    #         if cur_hierarchy_id == 0:
    #             cur_survey += input_cg_hypothesis_prompt
    #         elif cur_hierarchy_id > 0:
    #             cur_survey += input_cg_hypothesis_prompt + all_previous_hierarchy_hypothesis_prompt
    #         else:
    #             raise ValueError("Invalid cur_hierarchy_id: ", cur_hierarchy_id)
    #         # search over one hierarchy
    #         search_results_all_init = self.search_over_one_hierarchy(cur_q, cur_survey, cur_cg_hyp, cur_hierarchy_id, prev_hierarchy_gene_fg_hyp)
    #         if len(search_results_all_init) > 0:
    #             full_results_all_hierarchy.append(search_results_all_init)
    #             # update prev_hierarchy_gene_fg_hyp
    #             prev_hierarchy_gene_fg_hyp = search_results_all_init[-1][-1][0]
    #             print("current hyp: ", prev_hierarchy_gene_fg_hyp)
    #         else:
    #             # print("INFO: No better searched hypothesis for this hierarchy.")
    #             # if the potential local minimum are all worse than the hypothesis from the previous hierarchy, then at least we have the hypothesis from the previous hierarchy, which can be used as the initial hypothesis for the next hierarchy
    #             assert prev_hierarchy_gene_fg_hyp != None

    #     # save the search results
    #     if self.args.if_save == 1:
    #         with open(self.args.output_dir, 'w') as f:
    #             json.dump(full_results_all_hierarchy, f, indent=4)