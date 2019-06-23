#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""This is the code for transducing the MG derivation trees into Xbar and MG derived phrase structure trees
Author: John torr (john.torr@cantab.net)"""
import sys
import nltk
import string
import pdb
import copy
import re
sys.setrecursionlimit(10000)

brackets='()'
open_b, close_b = brackets
open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
node_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
leaf_pattern = '[^%s%s]+' % (open_pattern, close_pattern)

right_merge = re.compile('\w+=')
right_merge_left_h_move = re.compile('>\w+=')
right_merge_right_h_move = re.compile('\w+<=')
right_merge_x_h_move = re.compile('\w+=\^')
left_merge = re.compile('=\w+')
left_merge_left_h_move = re.compile('=>\w+')
left_merge_right_h_move = re.compile('=\w+<')
left_merge_x_h_move = re.compile('=\w+\^')
left_adjoin = re.compile('\w+≈')
right_adjoin = re.compile('≈\w+')      
left_move = re.compile('\+\w+')
right_move = re.compile('(t)|(v)|(p)|(c)|(d)|(D)')
not_cat_feature = re.compile('=|(\+)|(\-)|~|≈|(\^)|&|\?|!|\>|\<')
Abar_features = ['-wh', '-foc', '-top']
A_features = ['-case', 'd']
R_features = ['t~', 'v~', 'p~', 'c~', 'd~']#do add a new category for rightward movement, change this and right_move and also in cky_mg.py extraposition_hosts and R_features and in autobankGUI extraposition_hosts
min_max_to_fix = {}
indices_to_fix = {}

derived_to_xbar_cat_mapping = {'lv':'v',
                               'v':'V',
                               'n':'N',
                               'p':'P',
                               'c':'C',
                               't':'T',
                               'd':'D',
                               'q':'Q',
                               'ln':'n',
                               'adv':'Adv',
                               'adj':'Adj'}

indices = []
    
class Node:
    def __init__(self, features = None, original_features = None, name = "", mother=None, daughters=[], movers = [], head = None, lexical_head = False, index = None, heads=[], sem_heads=[]):
        self.features = features
        self.original_features = original_features
        self.mother = mother
        self.daughters = daughters
        self.name = name
        self.head = head
        self.heads = heads
        self.sem_heads = sem_heads
        self.movers = movers
        #lexical_head tracks whether this node is a (complex or simplex) lexical item or a larger constituent
        self.lexical_head = lexical_head
        self.lowest_overt_trace = None
        self.index = index
        self.is_covert = False
        self.top_level_head_node = False
        self.terminal = False
        self.coordinator = False
        self.chain_pointer = None
        self.sem_node = False
        self.phon_node = False
        self.lexcoord = False
        self.top_level_lex_coord = False
        self.rightward_trace = False
        self.indices = []

    def get_terminal_heads(self, head_list=[], normalize_terminals = False, returnSynDeps=True):
        if not returnSynDeps:
            return []
        if self.daughters == []:
            self.terminal = True
        #returns the lexical heads for any constituent
        if self.terminal and self.name != 'λ' and self.name != 'μ' and self.name != 'ζ':
            head_list = []
            if self.name != "Λ":
                #if the head is a trace of head movement, we set
                #the semantic head as the antecedent, ie the overt copy..
                if normalize_terminals:
                    if len(self.name) > 2 and self.name[0] == self.name[-1] == '/':
                        self.name = self.name[1:-1]
                    self.name = self.name.lower()
                    try:
                        self.truncated_name = self.truncated_name.lower()
                    except Exception as e:
                        x=0
                head_list.append(self)
            else:
                if self.antecedent.lexcoord:
                    head_list=self.antecedent.get_terminal_heads(head_list, normalize_terminals=normalize_terminals)
                else:
                    head_list.append(self.antecedent)
            return head_list
        else:
            head_list = []
            if self.chain_pointer != None:
                NODE = self.chain_pointer
            else:
                NODE = self
            for head in NODE.heads:
                new_head_list=head.get_terminal_heads(head_list, normalize_terminals=normalize_terminals)
                head_list+=new_head_list
            return head_list

    def get_semantic_terminal_heads(self, head_list=[], normalize_terminals=False,returnSemDeps=True):
        if not returnSemDeps:
            return []
        #returns the semantic lexical heads for any constituent (works for xbar trees only)
        #currently not used as has a copy in gen_derived_tree.py
        if self.daughters == []:
            self.terminal = True
        if self.terminal and self.name != 'λ' and self.name != 'μ' and self.name != 'ζ':
            head_list = []
            if self.name != "Λ":
                #if the semantic lexical head is a trace of head movement, we set
                #the semantic head as the antecedent, ie the overt copy..
                if normalize_terminals:
                    if len(self.name) > 2 and self.name[0] == self.name[-1] == '/':
                        self.name = self.name[1:-1]
                    self.name = self.name.lower()
                    try:
                        self.truncated_name = self.truncated_name.lower()
                    except Exception as e:
                        x=0
                head_list.append(self)
            else:
                if self.antecedent.lexcoord:
                    head_list=self.antecedent.get_semantic_terminal_heads(head_list, normalize_terminals=normalize_terminals)
                else:
                    head_list.append(self.antecedent)
            return head_list
        else:
            head_list = []
            if self.chain_pointer != None:
                NODE = self.chain_pointer
            else:
                NODE = self
            for head_child in NODE.sem_heads:
                new_head_list=head_child.get_semantic_terminal_heads(head_list, normalize_terminals=normalize_terminals)
                head_list+=new_head_list
            return head_list

    def generate_bracketing(self):
        #I needed to insert dummy start and end brackets and also make sure there are spaces between any
        # )( sequences, so I created a wrapper around this core GENERATE_BRACKETING function (see below)
        bracketing = "(" + self.GENERATE_BRACKETING() + ")"
        index=0
        while True:
            if bracketing[index] == ")":
                if bracketing[index+1] == "(":
                    bracketing = bracketing[0:index+1] + " " + bracketing[index+1:]
                    index += 1
            index+=1
            if index == len(bracketing)-2:
                break
        #if we have ) ( inside a bracket it screws things up.. so we delete the space between the two brackets
        bracketing = re.sub("\) \(", ")(", bracketing, count=10000)
        return bracketing

    def GENERATE_BRACKETING(self, bracketing=""):
        bracketing+="("+self.name+" "
        for daughter in self.daughters:
            #if daughter is a terminal (word) then we don't generate another
            #open bracket - we check this by testing the type of daughter
            if daughter.terminal==True:
                bracketing+="("+daughter.name+")"
                if self.mother == None:
                    bracketing+=")"
                    return bracketing
                return bracketing
            else:
                bracketing=daughter.GENERATE_BRACKETING(bracketing)
                bracketing+=")"
        if self.mother==None:
            return bracketing+")"
        else:
            return bracketing

    def visualize_tree(self):
        self.bracketing=self.generate_bracketing()
        #we need an extra closing bracket if this is not a root tree
        if self.mother != None:
            self.bracketing+=")"
        #in the later version of nltk.Tree the relevant function for converting bracketings
        #to trees is called 'fromstring' so both are accommodated here
        try:
            self.tree=nltk.Tree.parse(self.bracketing, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
        except AttributeError:
            self.tree=nltk.Tree.fromstring(self.bracketing, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
        print(self.bracketing+"\n")
        print("\nClose the window containing the tree diagram to continue... (and then make sure the cursor is at the bottom of this screen)\n")
        self.tree.draw()

def get_nodes(node, nodes):
    nodes.append(node)
    for daughter in node.daughters:
        nodes = get_nodes(daughter, nodes)
    return nodes

def check_heads_are_in_tree(node, tree_nodes):
    #THIS FUNCTION IS USED FOR DEBUGGING
    for head in node.heads:
        if head not in tree_nodes:
            print("Found one that's not!")
        else:
            print("head node found in tree")
    else:
        for daughter in node.daughters:
            check_heads_are_in_tree(daughter, tree_nodes)

def main(derivation_bracketing, show_indices=True, return_xbar_tree=False, allowMoreGoals=False, allowOnlyGoals=True):
    global min_max_to_fix
    min_max_to_fix.clear()
    indices_to_fix.clear()
    #both checks that the derivation tree is good and generates a derived tree from it
    derivation_tree = gen_derivation_tree(derivation_bracketing)
    #doing deepcopy on derivation or derived trees was causing crashes, so we just generate
    #the copies from scratch, by applying gen_derivation_tree and gen_derived tree again..
    derived_tree = gen_derivation_tree(derivation_bracketing)
    set_lexcoord(derived_tree)
    set_top_level_lex_cooord_markers(derived_tree)
    if derived_tree.name != 'type_sat':
        delete_type_sat_nodes(derived_tree)
    else:
        #if the system is trying to delete a root type_sat node, then this is just
        #a tree fragment requested by autobank.. so for the phrase structure
        #trees we just make the root the daughter of the current root
        derived_tree = derived_tree.daughters[0]
    reset_indices()
    gen_derived_tree(derived_tree)
    for daughter in derived_tree.daughters:
        daughter.mother = derived_tree
    remove_fcide(derived_tree)
    set_missing_sem_heads_to_head(derived_tree)
    fix_missing_sem_node(derived_tree)
    fix_overt_to_rightward_traces(derived_tree)
    xbar_tree = gen_derivation_tree(derivation_bracketing)
    set_lexcoord(xbar_tree)
    set_top_level_lex_cooord_markers(xbar_tree)
    if xbar_tree.name != 'type_sat':
        delete_type_sat_nodes(xbar_tree)
    else:
        xbar_tree = xbar_tree.daughters[0]
    reset_indices()
    gen_derived_tree(xbar_tree)
    remove_fcide(xbar_tree)
    while "fcide" in xbar_tree.name:
        xbar_tree = xbar_tree.daughters[0]
        xbar_tree.mother = None
    fix_mother_relations(xbar_tree)
    fix_missing_sem_node(xbar_tree)
    fix_overt_to_rightward_traces(xbar_tree)
    gen_xbar_tree(xbar_tree)
    remove_hmove(xbar_tree)
    while xbar_tree.mother != None and xbar_tree in xbar_tree.mother.daughters and xbar_tree.mother.name != "type_sat":
        xbar_tree = xbar_tree.mother
    add_missing_xp_lex_coord(xbar_tree)
    while xbar_tree.mother != None and xbar_tree.mother.name != "type_sat":
        xbar_tree = xbar_tree.mother
    if xbar_tree.top_level_lex_coord:
        eliminate_vacuous_xp_xbar_nodes_inside_lex_coord(xbar_tree)
        if len(xbar_tree.name) > 1 and xbar_tree.name[-1] == 'P':
            xbar_tree.name = xbar_tree.name[:-1]
    add_top_head_nodes(xbar_tree)
    add_conj_features(xbar_tree)
    fix_heads(xbar_tree)
    add_sem_heads(xbar_tree)
    if show_indices == True:
        generate_xbar_indices(xbar_tree)
    eliminate_vacuous_trace_projections(xbar_tree)
    insert_additional_xbar_levels(xbar_tree)
    shift_leftover_indices_up(xbar_tree)
    #because xbar tree creation involves the insertion of lots of nodes, we want to make
    #sure that the xbar_tree variable still points to the top of the tree.
    while xbar_tree.mother != None and xbar_tree.mother.name != "type_sat":
        xbar_tree = xbar_tree.mother
    while "fcide" in derived_tree.name:
        derived_tree.daughters[0].features = derived_tree.features
        derived_tree = derived_tree.daughters[0]
        derived_tree.mother = None
    #The next function just ensures all mother relations in the xbar tree are correct
    fix_mother_relations(xbar_tree)
    fix_min_max_traces(xbar_tree)
    fix_min_max_traces_2(xbar_tree)
    unify_atb_indices(xbar_tree)
    index_counts = get_index_counts(xbar_tree, {})
    indices_to_delete = []
    for index in index_counts:
        if index_counts[index] == 1:
            indices_to_delete.append(int(index))
    delete_indices(xbar_tree, indices_to_delete)
    fix_pre_trace_heads(xbar_tree)
    set_missing_sem_heads_to_head(xbar_tree)
    percolate_sem_phon_indicators(xbar_tree)
    delete_indices_from_terminals(xbar_tree)
    fix_lex_coord_structure(xbar_tree)
    fix_lex_coord_multiple_bars(xbar_tree)
    insert_covert_traces(xbar_tree, 'xbar')
    node_index_mappings = get_overt_copy_indices(xbar_tree, {})
    add_heads_to_covert_traces(xbar_tree, node_index_mappings)
    insert_covert_traces(derived_tree, 'derived')
    #I'm attaching indices_to_fix to the root of the xbar tree so it is accessible
    #from autobank
    xbar_tree.indices_to_fix = indices_to_fix
    derived_bracketing = gen_bracketing(derived_tree)
    xbar_bracketing = gen_bracketing(xbar_tree)
    #now we need to check that the root node has only a c feature and no movers
    if len(derived_tree.features) == 0:
        raise Exception("Error! Root node has no c feature.. (no features at all)")
    if allowOnlyGoals:
        if len(derived_tree.features) > 1 and allowMoreGoals==False:
            raise Exception("Error! Root node has more than one feature.. (should just have c)")
    if return_xbar_tree == False:
        return (derived_bracketing, xbar_bracketing)
    else:
        return (derived_bracketing, xbar_bracketing, xbar_tree)

def fix_heads(node):
    #fixes a problem where in some cases the head is not found in the daughters but should be
    #and is a head from an old copy of a derived tree fragment
    for head in node.heads:
        if "<" in head.name:
            try:
                node.heads = [node.daughters[0]]
            except Exception as e:
                x=0
            break
        elif ">" in head.name:
            try:
                node.heads = [node.daughters[1]]
            except Exception as e:
                x=0
            break
    for daughter in node.daughters:
        fix_heads(daughter)

def fix_overt_to_rightward_traces(node):
    #there is a bug where if an item undergoes rightward movement and the node to which
    #it rightward moves does not move, but some other node dominating it does later move,
    #the trace is an overt not a covert one.. the trace's head also points to a terminal
    #not actually in the tree and this must be fixed for the xbar trees.. this function just fixes
    #the trace so that it is the zeta for rightward movement.. the function below fixes the
    #head part of the bug
    if "split" in node.name:
        if "<" in node.name:
            if node.daughters[1].name == 'λ':
                node.daughters[1].rightward_trace = True
                node.daughters[1].sem_node = True
                node.daughters[1].phon_node = False
        elif ">" in node.name:
            if node.daughters[0].name == 'λ':
                node.daughters[0].rightward_trace = True
                node.daughters[0].sem_node = True
                node.daughters[0].phon_node = False
    for daughter in node.daughters:
        fix_overt_to_rightward_traces(daughter)

def fix_missing_sem_node(node):
    #in parital trees generated using the derivation builder, sometimes a case of
    #successive cyclic covert movement has the wrong trace type until the full
    #tree is later generated.. this fixes that
    if "c_move" in node.name:
        if "<" in node.name:
            node.daughters[1].sem_node = True
        elif ">" in node.name:
            node.daughters[0].sem_node = True
    for daughter in node.daughters:
        fix_missing_sem_node(daughter)

def unify_atb_indices(node):
    parts = node.name.split("-")
    if len(parts) > 1:
        if parts[1] in indices_to_fix:
            node.name = parts[0]+"-"+indices_to_fix[parts[1]]
    for daughter in node.daughters:
        unify_atb_indices(daughter)

def insert_covert_traces(node, tree_type):
    #the covertly moved copies are cluttering up the xbar trees, so this
    #function changes them to μ 'traces'..
    if node.name == 'Λ':
        #owing to the possibility of min-max XP/X movement of a phrase/head, we can end up with
        #head trace nodes marked as sem nodes.. but we don't want to insert a covert trace here
        node.sem_node = False
    if (node.sem_node and not node.phon_node and node.name != ''):
        if node.rightward_trace:
            t = 'ζ'
        else:
            t = 'μ'
        if tree_type == 'xbar':
            trace_node = Node(features=[], original_features=[], name=t)
            trace_node.mother = node
            trace_node.daughters = []
            if t == 'μ':
                node.heads = [trace_node]
                node.sem_heads = [trace_node]
            else:
                trace_node.heads = node.heads
                node.heads = [trace_node]
                node.sem_heads = [trace_node]
            node.daughters = [trace_node]
        else:
            node.name = t
            node.truncated_name = t
            node.daughters = []
    else:
        for daughter in node.daughters:
            insert_covert_traces(daughter, tree_type)

def get_overt_copy_indices(node, node_index_mappings):
    if node.sem_node and not node.phon_node:
        return node_index_mappings
    if len(node.daughters) > 0 and len(node.name.split("-")) == 2:
        if node.daughters[0].name not in ['μ', 'Λ', 'ζ', 'λ', '']:
            index = node.name.split("-")[1]
            if index in node_index_mappings:
                #This happens sometimes if a node just dominating a moved head only
                #is overtly moved.. both are picked up as overt antecedents so we just ignore it here
                x=0
            else:
                node_index_mappings[index] = node
    for daughter in node.daughters:
        node_index_mappings = get_overt_copy_indices(daughter, node_index_mappings)
    return node_index_mappings

def add_heads_to_covert_traces(node, node_index_mappings):
    if len(node.heads) == 0 and (node.name == 'μ' or node.name == 'ζ'):
        try:
            index = node.mother.name.split("-")[1]
            node.heads = [node_index_mappings[index]]
            node.sem_heads = [node_index_mappings[index]]
        except Exception as e:
            x=0
    for daughter in node.daughters:
        add_heads_to_covert_traces(daughter, node_index_mappings)

def fix_lex_coord_multiple_bars(node):
    if node.top_level_lex_coord and node.mother != None:
        if node.mother.name.count("'") > 1:
            node.mother.name = node.mother.name.split("'")[0]+"'"
    for daughter in node.daughters:
        fix_lex_coord_multiple_bars(daughter)

def add_missing_xp_lex_coord(node):
    #after lexical coordination, there is a missing X' node, this corrects that
    if node.top_level_lex_coord and node.mother != None and node in node.mother.heads and node.mother.name[-1] == 'P':
        cat_feature = node.mother.name[:-1]
        new_xp_node = Node(features = node.mother.features, original_features = node.mother.original_features, name = node.mother.name, mother=node.mother.mother, daughters=[node.mother], movers = [], head = node.head, lexical_head = False, heads=[node.mother])
        insert_node(target_node=node.mother, above_below='above', new_node=new_xp_node)
        node.mother.name = node.mother.name[:-1]+"'"
        if node.name.count("'") == 1:
            new_xp_node.top_level_lex_coord = True
            node.top_level_lex_coord = False
        return
    for daughter in node.daughters:
        add_missing_xp_lex_coord(daughter)

def eliminate_vacuous_xp_xbar_nodes_inside_lex_coord(node):
    if node.name[-1] == 'P' and len(node.daughters) == 1 and node.daughters[0].name.count("'") == 1 and len(node.daughters[0].daughters) == 1:
        if node.mother != None:
            index = node.mother.daughters.index(node)
            node.mother.daughters.remove(node)
            node.mother.daughters.insert(index, node.daughters[0].daughters[0])
            if node in node.mother.sem_heads:
                node.mother.sem_heads.remove(node)
                node.mother.sem_heads.append(node.daughters[0].daughters[0])
            if node in node.mother.heads:
                node.mother.heads.remove(node)
                node.mother.heads.append(node.daughters[0].daughters[0])
                node.mother = None
        else:
            current_node = node
            while len(current_node.daughters) == 1:
                current_node = current_node.daughters[0]
            if len(current_node.daughters) == 2:
                node.daughters = current_node.daughters
                for daughter in node.daughters:
                    daughter.mother = node
                node.heads = current_node.heads
                node.sem_heads = current_node.sem_heads
                current_node.daughters = None
                node.name = node.name[:-1]
    for daughter in node.daughters:
        eliminate_vacuous_xp_xbar_nodes_inside_lex_coord(daughter)

def set_lexcoord(node):
    if "_lex" in node.name:
        node.lexcoord = True
    for daughter in node.daughters:
        set_lexcoord(daughter)

def set_top_level_lex_cooord_markers(node):
    if '_lex' in node.name:
        node.top_level_lex_coord = True
        #we only want to label the top one, hence we return here
        return
    for daughter in node.daughters:
        set_top_level_lex_cooord_markers(daughter)

def delete_type_sat_nodes(node):
    #we only want type sat nodes in the derivation tree.. they have no place in the derived/xbar trees
    if node.name == 'type_sat':
        #we will eliminate the intermediate type_sat node, but we need to change the terminal node name
        #to one which only has a single selectee feature, otherwise the system crashes when trying to generate the derived and xbar trees
        features = node.daughters[0].name.split(" ")[2:]
        for feature in features:
            FEATURE = re.sub('{.*?}', '', feature)
            if not not_cat_feature.search(FEATURE):
                cat_feature = feature
        saturated_node_name = " ".join(node.daughters[0].name.split(" ")[0:2])+" "+cat_feature
        type_sat_node_index = node.mother.daughters.index(node)
        node.mother.daughters.remove(node)
        node.mother.daughters.insert(type_sat_node_index, node.daughters[0])
        node.daughters[0].name = saturated_node_name
        return
    for daughter in node.daughters:
        delete_type_sat_nodes(daughter)

def fix_lex_coord_structure(node):
    #we don't want XP and X' nodes to appear in lexical coordination structures.. this function
    #therefore eliminates them, so that only X nodes are coordinated at the lexical level
    if node.mother != None and node.mother.lexcoord:
        node.mother.name = re.sub("'", "", node.mother.name)
        if node in node.mother.sem_heads and node.name[-1] == 'P' and len(node.name) > 1:
            #just checking to make sure this is definitely an XP node and not a prepositional P head
            xp_node = node
            x_node = xp_node.daughters[0].daughters[0]
            xp_node_index = xp_node.mother.daughters.index(xp_node)
            xp_node.mother.daughters.remove(xp_node)
            xp_node.mother.daughters.insert(xp_node_index, x_node)
            xp_node.mother.sem_heads.remove(xp_node)
            xp_node.mother.sem_heads.append(x_node)
    for daughter in node.daughters:
        fix_lex_coord_structure(daughter)

def remove_hmove(node):
    #this removes the hmove tags that we had to leave on while generating the xbar tree
    if len(node.daughters) > 0:
        node.name = re.sub('_hmove', '', node.name)
    for daughter in node.daughters:
        remove_hmove(daughter)

def get_index_counts(node, index_counts):
    try:
        parts = node.name.split("-")
        if len(parts) == 2:
            int(parts[-1])
            if parts[-1] in index_counts:
                index_counts[parts[-1]] += 1
            else:
                index_counts[parts[-1]] = 1
    except ValueError:
        x=0
    for daughter in node.daughters:
        index_counts = get_index_counts(daughter, index_counts)
    return index_counts

def delete_indices(node, indices_to_delete):
    #the fix_min_max_traces function incorrectly adds indices to the XP dominating an unergative X verb that has moved
    #this function corrects that by deleting all indices that only appear once in the tree
    try:
        parts = node.name.split("-")
        index = int(parts[-1])
        if index in indices_to_delete:
            node.name = "-".join(parts[:-1])
    except ValueError:
        x=0
    for daughter in node.daughters:
        delete_indices(daughter, indices_to_delete)

def delete_indices_from_terminals(node):
    #fixes a bug where indices can end up on a (null) terminal as well as its XP projection
    if len(node.daughters) == 0:
        parts = node.name.split("-")
        if len(parts) > 1:
            try:
                int(parts[-1])
                node.name = "-".join(parts[:-1])
            except ValueError:
                x=0
    for daughter in node.daughters:
        delete_indices_from_terminals(daughter)

def fix_min_max_traces(node):
    global min_max_to_fix
    global indices
    #in the case where the final head in the spine moves AND undergoes head movement
    #we end up with indices on the Ns rather than NPs.. we want them on the NP wherever there is
    #one (everywhere except on the moved head itself, which is only N) We then add new indices for
    #the X nodes themselves in a separate function.. however, we also 
    if node.name == 'Λ' and node.mother != None and len(node.mother.name.split("-")) == 2:
        index = node.mother.name.split("-")[1]
        if index not in min_max_to_fix:
            new_index = indices[0]
            del(indices[0])
            min_max_to_fix[index] = new_index
        else:
            new_index = min_max_to_fix[index]
            min_max_to_fix[index] = new_index
        current_node = node.mother
        while current_node.mother != None and len(current_node.mother.daughters) == 1:
            current_node = current_node.mother
        if "'" not in current_node.name and "-" not in current_node.name and node not in current_node.daughters:
            current_node.name+="-"+new_index
    for daughter in node.daughters:
        fix_min_max_traces(daughter)

def fix_min_max_traces_2(node):
    #the first function fixed the case for covert movement, this does the same for overt movement
    if node.name == 'λ':
        current_node = node.mother
        while current_node.mother != None and len(current_node.mother.daughters) == 1:
            current_node = current_node.mother
        parts = current_node.name.split("-")
        if len(parts) > 1:
            if parts[1] in min_max_to_fix:
                current_node.name = parts[0]+"-"+str(min_max_to_fix[parts[1]])
    for daughter in node.daughters:
        fix_min_max_traces_2(daughter)

def fix_pre_trace_heads(node):
    #this fixes a small issue whereby the mother of a trace node still points to X'
    #rather than to the trace directly.. the path is still good but it's not great to have
    #references to nodes no longer in the tree
    if node.name == 'λ':
        if node.heads not in [None, []]:
            node.mother.heads = [node]
    for daughter in node.daughters:
        fix_pre_trace_heads(daughter)

def fix_mother_relations(node):
    for daughter in node.daughters:
        daughter.mother = node
        fix_mother_relations(daughter)
                        
def percolate_sem_phon_indicators(xbar_tree):
    #we move up the indicators telling us if a node is semantic and/or phonetic
    #to the highest projection.. this function assumes that head movement cannot be
    #covert or phonetic in the sense that it leaves behind a /trace/ because
    #otherwise the whole phrase containing the head will be incorrectly tagged as
    #phoentic/semantic..
    if xbar_tree.sem_node or xbar_tree.phon_node:
        pointer = xbar_tree
        while pointer in pointer.mother.heads:
            pointer = pointer.mother
        pointer.sem_node = xbar_tree.sem_node
        pointer.phon_node = xbar_tree.phon_node
        pointer.rightward_trace = xbar_tree.rightward_trace
        pointer.maker = 1
        if pointer != xbar_tree:
            xbar_tree.sem_node = False
            xbar_tree.phon_node = False
    for daughter in xbar_tree.daughters:
        percolate_sem_phon_indicators(daughter)

def add_top_head_nodes(xbar_tree):
    #a function to annotate the top level node of complex heads with a label
    #indicating their status as the complex head node of the phrase.. needed
    #for add_sem_heads().. we clean up mother relations as we go too and also
    #set .terminal = True for all terminals
    for daughter in xbar_tree.daughters:
        daughter.mother = xbar_tree
        add_top_head_nodes(daughter)
    if len(xbar_tree.daughters) > 0:
        return
    else:
        xbar_tree.terminal = True
        current_top_node = xbar_tree.mother
        while current_top_node.mother != None and "'" not in current_top_node.mother.name and not (len(current_top_node.mother.name)>1 and current_top_node.mother.name[-1] == 'P'): 
            current_top_node = current_top_node.mother
        current_top_node.top_level_head_node = True

def add_conj_features(xbar_tree):
    #sets coordinator = True for all projections of a coordinator.. ie Coord, Coord' and CoordP
    #(needed to find semantic headedness in coordinator phrases given that Coord is not actually
    #a category in the grammar - so CoordP of category D is simply a D).. For semantic headedness
    #we will assume all conjuncts to be heads of the CoordP..
    for daughter in xbar_tree.daughters:
        add_conj_features(daughter)
    if len(xbar_tree.daughters) > 0:
        return
    else:
        if xbar_tree.coordinator == True:
            current_node = xbar_tree
            while current_node.mother != None and current_node in current_node.mother.heads and not current_node.top_level_lex_coord:
                current_node.mother.coordinator = True
                current_node = current_node.mother

def add_sem_heads(xbar_tree):
    #a function which adds in the head path to the head of the extended projection,
    #ie adds in all the semantic head daughters.. so whereas [decl] is the syntactic
    #head child of C' (and the lexical head of CP), TP is the semantic head child
    #because it leads ultimately to the lexical verb which is the semantic head of CP..
    #In addition, for relatives, we make the semantic head of the [nom] NP the moved DP..
    #this is so that any verb selecting the relative clause will still take the head noun as
    #its argument, which is important for the dep mappings when scoring MGbank trees
    top_level_head_found = False
    index = -1
    sem_heads_set = False
    if len(xbar_tree.daughters) == 1:
        xbar_tree.sem_heads = [xbar_tree.daughters[0]]
        sem_heads_set = True
    if sem_heads_set == False:
        try:
            if xbar_tree.get_terminal_heads()[0].name == '[nom]':
                if 'DP' in xbar_tree.daughters[0].name:
                    xbar_tree.sem_heads = [xbar_tree.daughters[0]]
                    sem_heads_set = True
        except Exception as e:
            x=0
    if sem_heads_set == False:
        for daughter in xbar_tree.daughters:
            index+=1
            #all intermediate Coord projections + all conjuncts must be labelled as semantic head children
            #to guarantee that all conjuncts will be treated as semantic heads of the CoordP,
            #just as they are in the Penn Treebank structures as we have defined them.. we need Coord' to be
            #a semantic head child so that we have a path down to the lower conjuncts..  top level CoordP should not by default
            #be a semantic head (though it will be unless its governor is a V or N)
            if daughter.coordinator and xbar_tree.coordinator and daughter in daughter.mother.heads and daughter.top_level_head_node == False:
                xbar_tree.sem_heads = [xbar_tree.daughters[0], xbar_tree.daughters[1]]
                sem_heads_set = True
            elif daughter.coordinator and xbar_tree.coordinator and daughter.top_level_head_node == True:
                #if we have reached the head of Coord, ie the Coord itself, we simply make its complement
                #the sole semantic head at this level..
                sem_heads_set = True
                if index == 0:
                    xbar_tree.sem_heads = [xbar_tree.daughters[1]]
                else:
                    xbar_tree.sem_heads = [xbar_tree.daughters[0]]
    index = -1
    if sem_heads_set == False:
        for daughter in xbar_tree.daughters:
            index+=1
            if daughter.top_level_head_node or daughter.top_level_lex_coord:
                top_level_head_found = True
                if len(daughter.name) >= 2 and daughter.name[:2] == 'AP':
                    adjunct = True
                else:
                    adjunct = False
                #if there is no complement, or the preterminal category is one of the lexical (ie non functional) categories, then
                #we set this item as the semantic head child of the mother.. AP is adjunct phrase and is never the head
                if not adjunct and (len(xbar_tree.daughters) == 1 or daughter.name == "V" or daughter.name == "N" or daughter.name == "Adj" or daughter.name == "Adv") and not (len(daughter.get_terminal_heads()[0].name) > 0 and daughter.get_terminal_heads()[0].name[0] == '[' and daughter.get_terminal_heads()[0].name[-1] == ']' and 'pro' not in daughter.get_terminal_heads()[0].name):
                    xbar_tree.sem_heads = [daughter]
                #in all other cases the sem_heads is the complement..
                else:
                    #choose the other daughter from the current one, the former being the complement node
                    if index == 0:
                        xbar_tree.sem_heads = [xbar_tree.daughters[1]]
                    else:
                        xbar_tree.sem_heads = [xbar_tree.daughters[0]]
                break
        if top_level_head_found == False:
            xbar_tree.sem_heads = xbar_tree.heads
    for daughter in xbar_tree.daughters:
        if len(daughter.daughters) > 0:
            add_sem_heads(daughter)

def set_missing_sem_heads_to_head(node):
    if node.sem_heads in [None, []]:
        node.sem_heads = node.heads
    for daughter in node.daughters:
        set_missing_sem_heads_to_head(daughter)
                    
def remove_fcide(node):
    for daughter in node.daughters:
        daughter.mother = node
        remove_fcide(daughter)
    if 'fcide' in node.name and node.mother != None:
        node.daughters[0].index = node.index
        node.daughters[0].sem_node = node.sem_node
        node.daughters[0].phon_node = node.phon_node
        node.daughters[0].rightward_trace = node.rightward_trace
        node.index = node.daughters[0]
        node_index = node.mother.daughters.index(node)
        node.mother.daughters.remove(node)
        node.mother.daughters.insert(node_index, node.daughters[0])
        node.daughters[0].mother = node.mother
        if node in node.mother.heads:
            node.mother.heads.remove(node)
            for head in node.heads:
                node.mother.heads.append(head)
        node.mother = None

def reset_indices():
    global indices
    del(indices[:])
    indices+=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110']

def generate_xbar_indices(node):
    for daughter in node.daughters:
        generate_xbar_indices(daughter)
    if node.index != None:
        #we need to track through the path of index pointers until we find an index which is an actual
        #index rather than a pointer to another node.
        while type(node.index) != type([]):
            node.index = node.index.index
        node.name += "-"+node.index[0]
        #we need to remove duplicate indices from daughters, ie we can't have an XP and its X head having the same index
        if len(node.daughters) != 0:
            for daughter in node.daughters:
                name_parts = daughter.name.split("-")
                if len(name_parts) == 2 and name_parts[1] == node.index[0]:
                    daughter.name = name_parts[0]

def shift_leftover_indices_up(node):
    #sometimes some stray indices get stuck on xbar levels.. so we remove them and move them up to the XP level
    for daughter in node.daughters:
        shift_leftover_indices_up(daughter)
    for daughter in node.daughters:
        if "'" in daughter.name and "-" in daughter.name:
            parts = daughter.name.split("-")
            daughter.name = parts[0]
            current_node = daughter.mother
            while "'" in current_node.name:
                current_node = current_node.mother
            if "-" not in current_node.name:
                current_node.name = current_node.name+"-"+parts[1]
            else:
                raise Exception("Error! Tried to move up a stray index from a bar level, but XP already has an index!")

def eliminate_vacuous_trace_projections(node):
    for daughter in node.daughters:
        eliminate_vacuous_trace_projections(daughter)
    if node.name == "λ" or node.name == "":
        if "'" in node.mother.mother.name:
            if len(node.mother.daughters) == 1 and len(node.mother.mother.daughters) == 1 and len(node.mother.mother.mother.daughters) == 1:
                node.mother.mother.mother.daughters = [node]
                node.mother = node.mother.mother.mother

def insert_additional_xbar_levels(node):
    #in classic xbar theory adjuncts are defined as daughters and sisters to the same bar level.. hence we must generate
    #additional xbar segments..
    for daughter in node.daughters:
        insert_additional_xbar_levels(daughter)
    bar_level = None
    AdjunctP = None
    reduce_bar_levels = False
    for daughter in node.daughters:
        if "'" in daughter.name:
            bar_level = daughter
        if "AdjunctP" in daughter.name:
            AdjunctP = daughter
        if "'" in node.name:
            #if the mother is also an xbar level, it will be a different xbar level (this is where we have
            #multiple adjuncts attached to one phrase).. we must therefore reduce all bar levels above this by one bar
            reduce_bar_levels = True
    if bar_level != None and AdjunctP != None:
        if reduce_bar_levels == False:
            new_xbar_node = Node(features = node.features, original_features = node.original_features, name = bar_level.name, mother=node, daughters=node.daughters, movers = [], head = node.head, lexical_head = False, heads = [node])
            insert_node(target_node=node, above_below="below", new_node=new_xbar_node)
        else:
            if "'" != node.name[-1]:
                raise Exception("Error! Xbar node exists whose name does not have ' as its last character..")
            else:
                node.name = node.name[:-1]
                while node.mother != None and "'" in node.mother.name:
                    current_node = node.mother
                    current_node.name = node.name
                    
def gen_xbar_tree(mother):
    #takes as input a derived tree and transforms it into an xbar tree
    for daughter in mother.daughters:
        gen_xbar_tree(daughter)
    if len(mother.daughters) > 0:
        return
    else:
        cat_feature = None
        #first we need to find the category feature
        if mother.name in ['[adjunctizer]', '[Adjunctizer]', '[adjunctiser]', '[Adjunctiser]'] or '\xe2\x89\x88lv' in mother.original_features:
            cat_feature = 'A'
        else:
            for feature in mother.original_features:
                not_cat = not_cat_feature.search(feature)
                if not not_cat:
                    cat_feature = feature
                    break
        if cat_feature == None:
            cat_feature = 'A'
        if cat_feature in derived_to_xbar_cat_mapping:
            #we use more tradition generative categories for the xbar trees, ie caps and v* instead of lvt etc..
            cat_feature = derived_to_xbar_cat_mapping[cat_feature]
        new_preterminal = Node(features = mother.features, original_features = mother.original_features, name = cat_feature, mother=mother.mother, daughters=[mother], movers = [], head = mother, lexical_head = False, heads = [mother])               
        insert_node(target_node=mother, above_below="above", new_node=new_preterminal)
        if new_preterminal.daughters[0].name == 'Λ':
            #need to correct some cases where a preterminal's mother is not reset to the
            #preterminal, and where the preterminal's head is currently the head trace rather than
            #the moved head.
            if new_preterminal.daughters[0].antecedent in new_preterminal.mother.heads:
                new_preterminal.mother.heads.remove(new_preterminal.daughters[0].antecedent)
                new_preterminal.mother.heads.append(new_preterminal)
            new_preterminal.heads = [new_preterminal.daughters[0].antecedent]
        current_node = mother.mother
        bar_level = 0
        while True:
            #the case where there is only one level of projection which is the root of the tree
            if bar_level == 1 and current_node.mother == None:
                new_xp_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"P", mother=None, daughters=[current_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node])
                insert_node(target_node=current_node, above_below='above', new_node=new_xp_node)
                break
            #the case where there is more than one level of projection and the highest is the root of the tree
            elif bar_level > 1 and current_node.mother == None:
                current_node.name = cat_feature+"P"
                break
            elif bar_level == 0 and current_node.mother == None:
                new_xbar_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"'", mother=current_node.mother, daughters=[current_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node])
                #this is the case for fragment words which are roots of single node trees
                new_xp_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"P", mother=None, daughters=[new_xbar_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node.mother])
                insert_node(target_node=new_xbar_node, above_below='above', new_node=new_xp_node)
                new_xbar_node.mother = new_xp_node
                break
            current_node_index = current_node.mother.daughters.index(current_node)
            if ("<" in current_node.mother.name and current_node_index == 0) or (">" in current_node.mother.name and current_node_index == 1) or 'fcide' in current_node.mother.name:
                if "hmove" not in current_node.mother.name:
                    bar_level += 1
                    if 'adjoin' in current_node.mother.name or 'r_move' in current_node.mother.name:
                        if current_node.name.count("'") == 1:
                            new_xp_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"P", mother=current_node.mother, daughters=[current_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node])
                            insert_node(target_node=current_node, above_below='above', new_node=new_xp_node)
                        elif current_node.name.count("'") > 1:
                            current_node.name = cat_feature+"P"
                            current_node.mother.name = cat_feature+"P"
                        elif current_node.name == cat_feature+"P":
                            current_node.mother.name = cat_feature+"P"
                        else:
                            #this is executed when we have adjuncts of adjuncts
                            new_xbar_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"'", mother=current_node.mother, daughters=[current_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node])
                            insert_node(target_node=current_node, above_below='above', new_node=new_xbar_node)
                    else:
                        current_node.mother.name = cat_feature+("'"*bar_level)
                else:
                    #we have to keep 'hmove' on all nodes for now and remove it with a separate
                    #function later.. this is because in the case of head movement to a right adjoined
                    #position, by the time we process the moved head node the system will think it's
                    #phrasal if no hmove tag is present..
                    current_node.mother.name = cat_feature+"_hmove"
            else:
                #we have to make sure that every phrase has at least three levels, X, X' and XP
                if bar_level == 0 and "hmove" not in current_node.mother.name:
                    new_xbar_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"'", mother=current_node.mother, daughters=[current_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node])
                    insert_node(target_node=current_node, above_below='above', new_node=new_xbar_node)
                    new_xp_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"P", mother=new_xbar_node.mother, daughters=[new_xbar_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node.mother])
                    insert_node(target_node=new_xbar_node, above_below='above', new_node=new_xp_node)
                    new_xbar_node.mother = new_xp_node
                elif bar_level == 1:
                    new_xp_node = Node(features = mother.features, original_features = mother.original_features, name = cat_feature+"P", mother=current_node.mother, daughters=[current_node], movers = [], head = current_node.head, lexical_head = False, heads=[current_node])
                    insert_node(target_node=current_node, above_below='above', new_node=new_xp_node)
                else:
                    if "hmove" not in current_node.mother.name:
                        current_node.name = cat_feature+'P'
                    else:
                        #this covers the case where we're looking at a simplex head-moved node..
                        current_node.name = cat_feature+'_hmove'
                break
            current_node = current_node.mother

def insert_node(target_node, above_below, new_node):
    #inserts a node either above or below a target node.. no need to specify the new node's daughters
    #since this will be done when the new node is created..(only above is currently implemented)
    if above_below == "above":
        if len(target_node.daughters) == 0 or (len(target_node.daughters) == 1 and target_node.daughters[0].name != "Λ"):
            new_node.index = target_node.index
        if target_node.mother != None:
            target_node_index = target_node.mother.daughters.index(target_node)
            target_node.mother.daughters.remove(target_node)
            #in case any other nodes in the tree point to here for their head node, we will redirect
            #them back to the main tree
            target_node.mother.daughters.insert(target_node_index, new_node)
            new_node.mother = target_node.mother
            if target_node in target_node.mother.heads:
                target_node.mother.heads.remove(target_node)
                target_node.mother.heads.append(new_node)
        target_node.mother = new_node

def gen_derived_tree(mother):
    #takes as input a derivation tree and transforms it into a derived tree
    for daughter in mother.daughters:
        gen_derived_tree(daughter)
    if "::" in mother.name or ':\\u0305:\\u0305' in mother.name or ":\xcc\x85:\xcc\x85" in mother.name:
        mother.head = mother
        mother.heads = []
        if ":\\u0305:\\u0305" in mother.name:
            parts = mother.name.split(":\\u0305:\\u0305")
            mother.coordinator = True
        elif ":\xcc\x85:\xcc\x85" in mother.name:
            parts = mother.name.split(":\xcc\x85:\xcc\x85")
            mother.coordinator = True
        else:
            parts = mother.name.split("::")
        mother.features = parts[1].split(" ")
        stripped_mother_features = []
        for feature in mother.features:
            feature = re.sub('{.*?}', '', feature, count=1)
            stripped_mother_features.append(feature)
        mother.features = stripped_mother_features 
        mother.original_features = copy.deepcopy(stripped_mother_features)
        #remove the unwanted '' first feature..
        mother.features.remove(mother.features[0])
        mother.original_features.remove(mother.original_features[0])
        mother.name = parts[0].split(" ")[0]
        mother.lexical_head = True
    #for each operation, we perform tests to make sure that the daughter nodes have the required
    #features.. this tests the correctness of the trees.
    for daughter in mother.daughters:
        if ("?" in daughter.features[0] or "!" in daughter.features[0]) and "fcide" in mother.name:
            del(daughter.features[0])
            mother.features = daughter.features
            mother.original_features = daughter.original_features
            mother.head = daughter.head
            mother.heads = [daughter]
            daughter.features = []
            mother.movers += daughter.movers
            daughter.movers = []
            break
        rmerge_xhm = right_merge_x_h_move.search(daughter.features[0])
        if rmerge_xhm:
            if mother.daughters[1].features[0].lower() == rmerge_xhm.group(0)[:-2].lower() and ('r_merge_xhm' in mother.name or 'r_merge_hatb' in mother.name):
                selector = mother.daughters[0]
                selectee = mother.daughters[1]
                merge(selector, selectee, mother=mother, head_pointer="<_", hmove=True)
                head_move(selectee, mother, hm_dir = 'excorp')
                break
            else:
                raise Exception("Error in rmerge_xhm! features and op in derivation tree don't match!")
        rmerge_lhm = right_merge_left_h_move.search(daughter.features[0])
        if rmerge_lhm:
            if mother.daughters[1].features[0].lower() == rmerge_lhm.group(0)[1:-1].lower() and 'r_merge_lhm' in mother.name:
                selector = mother.daughters[0]
                selectee = mother.daughters[1]
                merge(selector, selectee, mother=mother, head_pointer="<_", hmove=True)
                head_move(selectee, mother, hm_dir = 'left')
                break
            else:
                raise Exception("Error in rmerge_lhm! features and op in derivation tree don't match!")
        rmerge_rhm = right_merge_right_h_move.search(daughter.features[0])
        if rmerge_rhm:
            if mother.daughters[1].features[0].lower() == rmerge_rhm.group(0)[:-2].lower() and 'r_merge_rhm' in mother.name:
                selector = mother.daughters[0]
                selectee = mother.daughters[1]
                merge(selector, selectee, mother=mother, head_pointer="<_", hmove=True)
                head_move(selectee, mother, hm_dir = 'right')
                break
            else:
                raise Exception("Error in rmerge_rhm! features and op in derivation tree don't match!")
        rmerge = right_merge.search(daughter.features[0])
        if rmerge:
            if mother.daughters[1].features[0].lower() == rmerge.group(0)[:-1].lower() and 'r_merge' in mother.name:
                selector = mother.daughters[0]
                selectee = mother.daughters[1]
                merge(selector, selectee, mother=mother, head_pointer="<_")
                break
            else:
                raise Exception("Error in rmerge! features and op in derivation tree don't match!")
        lmerge_xhm = left_merge_x_h_move.search(daughter.features[0])
        if lmerge_xhm:
            if mother.daughters[0].features[0].lower() == lmerge_xhm.group(0)[1:-1].lower() and ('l_merge_xhm' in mother.name or 'l_merge_hatb' in mother.name):
                selector = mother.daughters[1]
                selectee = mother.daughters[0]
                merge(selector, selectee, mother=mother, head_pointer=">_", hmove=True)
                head_move(selectee, mother, hm_dir = 'excorp')
                break
            else:
                raise Exception("Error in lmerge_xhm! features and op in derivation tree don't match!")
        lmerge_lhm = left_merge_left_h_move.search(daughter.features[0])
        if lmerge_lhm:
            if mother.daughters[0].features[0].lower() == lmerge_lhm.group(0)[2:].lower() and 'l_merge_lhm' in mother.name:
                selector = mother.daughters[1]
                selectee = mother.daughters[0]
                merge(selector, selectee, mother=mother, head_pointer=">_", hmove=True)
                head_move(selectee, mother, hm_dir = 'left')
                break
            else:
                raise Exception("Error in lmerge_lhm! features and op in derivation tree don't match!")
        lmerge_rhm = left_merge_right_h_move.search(daughter.features[0])
        if lmerge_rhm:
            if mother.daughters[0].features[0].lower() == lmerge_rhm.group(0)[1:-1].lower() and 'l_merge_rhm' in mother.name:
                selector = mother.daughters[1]
                selectee = mother.daughters[0]
                merge(selector, selectee, mother=mother, head_pointer=">_", hmove=True)
                head_move(selectee, mother, hm_dir = 'right')
                break
            else:
                raise Exception("Error in lmerge_rhm! features and op in derivation tree don't match!")
        lmerge = left_merge.search(daughter.features[0])
        if lmerge and 'move' not in mother.name:
            #here we do a mother.name check here because l_merge features can also be checked by movement
            #(for MTC purposes)..
            if mother.daughters[0].features[0].lower() == lmerge.group(0)[1:].lower() and 'l_merge' in mother.name:
                selector = mother.daughters[1]
                selectee = mother.daughters[0]
                merge(selector, selectee, mother=mother, head_pointer=">_")
                break
            else:
                raise Exception("Error in lmerge! features and op in derivation tree don't match!")
        ladjoin = left_adjoin.search(daughter.features[0])
        if ladjoin:
            if mother.daughters[1].features[0].lower() == ladjoin.group(0)[:-3].lower() and 'l_adjoin' in mother.name:
                selector = mother.daughters[0]
                selectee = mother.daughters[1]
                merge(selector, selectee, mother=mother, head_pointer=">_", adjoin=True)
                break
            else:
                raise Exception("Error in ladjoin! features and op in derivation tree don't match!")
        radjoin = right_adjoin.search(daughter.features[0])
        if radjoin:
            if mother.daughters[0].features[0].lower() == radjoin.group(0)[3:].lower() and 'r_adjoin' in mother.name:
                selector = mother.daughters[1]
                selectee = mother.daughters[0]
                merge(selector, selectee, mother=mother, head_pointer="<_", adjoin=True)
                break
            else:
                raise Exception("Error in radjoin! features and op in derivation tree don't match!")
        lmove = left_move.search(daughter.features[0])
        if (lmove or lmerge) and ('l_move' in mother.name or 'c_move' in mother.name):
            #we allow lmerge because, owing to MTC, we allow selector's to be checked by licensees..
            #however, clearly the rightward movement cases and head-movement cases are all irrelevant here,
            #since control involves only leftward phrasal movement without subsequent head movement from the moved
            #element..
            if len(mother.daughters) == 1:
                if lmove:
                    if 'l_move' in mother.name:
                        move(mother, direction='left', move_feature=lmove, covert=False)
                    elif 'c_move' in mother.name:
                        move(mother, direction='left', move_feature=lmove, covert=True)
                else:
                    if 'c_move' in mother.name:
                        move(mother, direction='left', move_feature=lmerge, covert=True)
                    elif 'l_move' in mother.name:
                        move(mother, direction='left', move_feature=lmerge, covert=False)
                #this break is particularly important as we have just added a daughter which we don't want to
                #iterate over in the loop..
                break
            else:
                raise Exception("Error in lmove! Multiple daughters under move node!")
        rmove = right_move.search(daughter.features[0])
        not_rmove = not_cat_feature.search(daughter.features[0])
        if rmove and (not not_rmove) and 'r_move' in mother.name:
            if len(mother.daughters) == 1:
                move(mother, direction='right', move_feature=rmove, covert=False)
            else:
                raise Exception("Error in rmove! Multiple daughters under move node!")
            break
    #now a final check to make sure that both daughters have no features, which should always be the case
    #(given that any moving elements are kept separate anyway and the head child's features are passed to mother
    for daughter in mother.daughters:
        if len(daughter.features) > 0:
            raise Exception("Error! A non-moving daughter still has unchecked features!")

def move(mother, direction, move_feature, covert):
    daughter = mother.daughters[0]
    matching_mover = None
    num_matching_movers = 0
    mother.original_features = daughter.original_features
    mother.features = daughter.features
    mother.head = daughter.head
    mother.heads = [daughter]
    suicidal_checker = False
    node_split = False
    if '?' in mother.features[0] or '!' in mother.features[0]:#1234
        suicidal_feature = True
        if '!' in mother.features[0]:
            suicidal_checker = True
    else:
        suicidal_feature = False
    if direction != 'right':
        del(mother.features[0])
    #since the daughter's features have been passed up to mother, we can now erase all the daughter's features
    daughter.features = []
    for mover in daughter.movers:
        if direction == 'left':
            if mover.features[0].lower() == "-"+move_feature.group(0)[1:].lower() or mover.features[0].lower() == move_feature.group(0)[1:].lower():
                num_matching_movers += 1
                matching_mover = mover
                if num_matching_movers > 1:
                    raise Exception("Error! Shortest Move violation! (leftward)")
        elif direction == 'right':
            if re.search('\w+~', mover.features[0].lower()):
                num_matching_movers += 1
                matching_mover = mover
                if num_matching_movers > 1:
                    raise Exception("Error! Shortest Move violation! (rightward)")
    if matching_mover == None:
        raise Exception("Error! Couldn't find a mover!")
    else:
        matching_mover.mother = mother
        if direction == 'left':
            mother.name = ">_" + mother.name
            #if either this is a case of successive cyclic movement, or a case of A-movement feeding
            #another A-movement (control movement to spec v followed by movement to spec T) or A'-movement
            #(movement to spec T followed by wh-movement to spec C)..we create a trace and keep the moving element
            #separate..
            phon_node = None
            if matching_mover.is_covert == False:
                matching_mover_is_overt = True
            else:
                matching_mover_is_overt = False
            if '_phon' in mother.name:
                if '~' in matching_mover.features[-1]:
                    node_split = True
            if covert == True or node_split:
                if covert:
                    #after this movement, the mover will be covert, so we now change this..but we have a record that it was overt in its last position
                    matching_mover.is_covert = True
                #we now need two versions of this constituent - a (semantically rich) phonetic version which remains in situ
                #and whose terminals are surrounded by forward slashes: /SOUP/, and a moving semantic version
                #whose terminals are in caps: SOUP.
                sem_node = matching_mover
                #later have to move rightwards..
                #DO NOT point sem_node's heads towards this phon_node unless it actually gets inserted as daughter
                #(i.e. the try below succeeds) as this screw things up
                #for light verb constructions where you have covert movement within a covert mover
                #DO NOT MAKE THE BELOW COPY A DEEPCOPY OPERATION AS FOR CERTAIN PARSES (E.G. 0003 LINE 8) the trees contain too much covertly moved items for deepcopy to process and the program just freezes..
                phon_node = copy.copy(sem_node)
                phon_node.mother = sem_node.mother                    
                phon_node.index = sem_node
                if '<' in phon_node.name:
                    try:
                        phon_node.heads = [phon_node.daughters[0]]
                    except IndexError:
                        phon_node.heads = []
                elif '>' in phon_node.name:
                    try:
                        phon_node.heads = [phon_node.daughters[1]]
                    except IndexError:
                        phon_node.heads = []
                if not sem_node.sem_node:
                    #if sem_node is already .sem_node then this is not the first
                    #covert movement in the chain and we do not want to set
                    #the lower copy as a .phon_node
                    phon_node.phon_node = True
                    make_terminals_phonetic(phon_node, true_phonetic = False)
                if covert:
                    sem_node.sem_node = True
                    make_terminals_semantic(sem_node, remove_refs = True)
                #we delete all but any pf movement features (x~) from phon_node and delete the
                #pf movement feature from sem_node..
                new_phon_feature_list = []
                for feature in phon_node.features:
                    if "~" in feature:
                        node_split = True
                        #this is the case where we are splitting up the chain into a covertly moving one and a rightward moving one
                        new_phon_feature_list.append(feature)
                phon_node.features = new_phon_feature_list
                new_sem_feature_list = []
                for feature in sem_node.features:
                    if "~" not in feature:
                        new_sem_feature_list.append(feature)
                sem_node.features = new_sem_feature_list
                #if the phon_node still has any pf movement features, it should be kept to one side in mother's
                #movers (as will the semantic version in the code below).. otherwise it is inserted as the daughter
                #replacing the existing trace node inserted by merge()
                if len(phon_node.features) == 0:
                    try:
                        phon_index = matching_mover.lowest_overt_trace.mother.daughters.index(matching_mover.lowest_overt_trace)
                        phon_index = matching_mover.lowest_overt_trace.mother.daughters.index(matching_mover.lowest_overt_trace)
                        matching_mover.lowest_overt_trace.mother.daughters.remove(matching_mover.lowest_overt_trace)
                        matching_mover.lowest_overt_trace.mother.daughters.insert(phon_index, phon_node)
                        phon_node.mother = matching_mover.lowest_overt_trace.mother
                        matching_mover.lowest_overt_trace.mother = None
                        matching_mover.lowest_overt_trace = phon_node
                        sem_node.heads = [phon_node]
                        phon_node.features = []
                    except ValueError:
                        #if we get here it's because there were two covert movements, one after the other
                        #e.g. for pied-piping of 'to' where case and wh are both checked covertly on the preposition
                        #and we needn't insert anything in the base position..
                        x=0
                else:
                    mother.movers.append(phon_node)
                    phon_node.mother = None
                    phon_node.lowest_overt_trace = sem_node.lowest_overt_trace
                    if node_split:
                        phon_node.lowest_overt_trace.rightward_trace = True
                        phon_node.lowest_overt_trace.sem_node = True
                        phon_node.lowest_overt_trace.phon_node = False
                        phon_node.lowest_overt_trace.heads = [phon_node]
                        phon_node.lowest_overt_trace.sem_heads = [phon_node]
                        phon_node.lowest_overt_trace.lowest_overt_trace = phon_node
                        sem_node.lowest_overt_trace = phon_node
                        phon_node.lowest_overt_trace = None
                        phon_node.heads = []
                        phon_node.sem_heads = []
                #from this point on we just manipulate sem_node == matching_mover
            if "_sc" in mother.name or len(matching_mover.features) > 1 or (suicidal_feature and not suicidal_checker):
                if "_sc" not in mother.name and ((not suicidal_feature) or suicidal_checker):
                    del(matching_mover.features[0])
                mother.movers += daughter.movers
                daughter.movers = []
                #we leave a trace if subsequent movement is overt syntactico-semantic (leftward), a semantic copy
                #if it is phonetic (rightward), and a phonetic/semantic copy if it is the first covert leftward movement in the chain
                if len(matching_mover.features) == 1 and re.search('\w+~', matching_mover.features[0]):
                    phon_node = copy.deepcopy(matching_mover)
                    phon_node.mother = matching_mover.mother
                    if '<' in phon_node.name:
                        phon_node.heads = [phon_node.daughters[0]]
                    elif '>' in phon_node.name:
                        phon_node.heads = [phon_node.daughters[1]]
                    phon_node.index = matching_mover.index
                    matching_mover.index = phon_node
                    matching_mover.sem_node = True
                    make_terminals_semantic(matching_mover)
                    make_terminals_phonetic(phon_node, true_phonetic = True)
                    mother.movers.append(phon_node)
                    phon_node.mother = None
                    matching_mover.rightward_trace = True
                    mother.movers.remove(matching_mover)
                    matching_mover.features = []
                    mother.daughters.insert(0, matching_mover)
                    matching_mover.mother = mother
                    matching_mover.heads = [phon_node]
                else:
                    if '_phon' in mother.name:
                        if node_split:
                            matching_mover.sem_node = True
                            make_terminals_semantic(sem_node, remove_refs = True)
                        trace_node = copy.deepcopy(matching_mover)
                        if '<' in trace_node.name:
                            trace_node.heads = [trace_node.daughters[0]]
                        elif '>' in trace_node.name:
                            trace_node.heads = [trace_node.daughters[1]]
                        trace_node.features = []
                        trace_node.phon_node = True
                        make_terminals_phonetic(trace_node, true_phonetic=False)
                    else:
                        trace_node = Node(features=[], original_features=matching_mover.original_features, name="λ", mother=mother, movers=[], daughters=[], head=None, index=matching_mover)
                    if matching_mover_is_overt == True:
                        matching_mover.lowest_overt_trace = trace_node
                        trace_node.heads = [matching_mover]
                    if trace_node.name == "λ":
                        trace_node.heads = [matching_mover]
                    mother.daughters.insert(0, trace_node)
                    trace_node.mother = mother
            else:
                del(matching_mover.features[0])
                mother.daughters.insert(0, matching_mover)
                matching_mover.mother = mother
                mother.movers += daughter.movers
                #this time we remove the matching mover from mother's movers list since it is stopping here
                mother.movers.remove(matching_mover)
        elif direction == 'right':
            mother.movers += daughter.movers
            mother.movers.remove(matching_mover)
            #for now I assume that there is no successive cyclic rightward movement.  x~ features are always the
            #last in any constituent, hence we perform a check that there are no more features..
            mother.name = "<_" + mother.name
            del(matching_mover.features[0])
            mother.daughters.insert(1, matching_mover)
            matching_mover.mother = mother
            matching_mover.phon_node = True
            matching_mover.sem_node = False
            make_terminals_phonetic(matching_mover, true_phonetic = True)
            if len(matching_mover.features) < 0:
                raise Exception("Error! Rightward moved constituent still has unchecked features!")

def make_terminals_semantic(node, remove_refs=False):
    if remove_refs:
        node.daughters = []
        node.head = node
        node.heads = []
        node.sem_heads = []
        node.antecedent = None
        node.movers = []
        node.mother = None
        node.name = 'μ'
        node.terminal = True
        return
    if len(node.daughters) == 0:
        if node.name == "":
            return
        elif node.name[0] == "[" and node.name[-1] == "]":
            return
        elif node.name[0] != "/" and node.name[-1] != "/":
            #this function will never be called on truly phonetic-only constituents.. ie ones which have been right
            #moved.. hence any /X/ constituents are actually copies left by covert movement and actually
            #also have semantics..  
            node_name = node.name
        elif node.name[0] == "/" and node.name[-1] == "/":
            if node.name == node.name.lower():
                node.name = ""
                return
            else:
                node_name = node.name[1:-1]
        else:
            node_name = node.name[1:-1]
        sem_name = ""
        for char in node_name:
            sem_name+=char.upper()
        node.name = sem_name
    else:
        for daughter in node.daughters:
            make_terminals_semantic(daughter)

def make_terminals_phonetic(node, true_phonetic):
    #for cases where there is covert movement, we want to leave a phonetic AND semantically
    #rich copy in the base position.. we do this by using both // and capitals
    #so for "boy" the overt 'trace' would look like /BOY/, which is the same as just: boy,
    #but makes explicit its trace status..
    if len(node.daughters) == 0:
        if node.name == "":
            return
        elif node.name == "[extraposer]":
            return
        elif node.name[0] not in ['/', '['] and node.name[-1] not in ['/', ']'] and node.name != node.name.upper():
            if true_phonetic == True:
                node.name = "/"+node.name.lower()+"/"
            else:
                node.name = "/"+node.name.upper()+"/"
        elif true_phonetic == True and node.name[0] == "/" and node.name[-1] == "/" and node.name == node.name.upper():
            node.name = node.name.lower()
    else:
        for daughter in node.daughters:
            make_terminals_phonetic(daughter, true_phonetic = true_phonetic)

def head_move(selectee, mother, hm_dir):
    global indices
    head = selectee.head
    while head.mother.lexcoord:
        head = head.mother
    head_is_selectee = False
    if head.name == 'Λ' and selectee.name == 'Λ':
        #this happens if the phrase out of which head movement is taking place is also moving
        #and the moving phrase is at the bottom of the tree, because in that case in the
        #derived bare phrase structure tree the head == the moving phrase.. we don't want
        #the selectee and the head to be the same thing so we create a copy of head
        head = copy.deepcopy(head)
        head.name = head.old_name
        selectee.heads = [head]
        selectee.antecedent = head
        head_is_selectee = True
        index = indices[0]
        if len(head.daughters) == 0:
            head.terminal = True
    elif head.name == 'Λ':
        head.name = head.old_name
        #the case where the selectee is moving and its head is also moving
        if len(head.daughters) == 0:
            head.terminal = True
    if head.index == None:
        #the case where it wouldn't == None is the excorporation case where a head moves without rolling up
        index = indices[0]
        del(indices[0])
        head.index = [index]
    if not head_is_selectee:
        head_index = head.mother.daughters.index(head)
        head.mother.daughters.remove(head)
        head_trace = Node(features=[], original_features = selectee.head.original_features, name="Λ", mother=head.mother, movers=[], daughters=[], head=None, index=head)
    else:
        head_index = head.head_index
        #we need to use a different lambda symbol for head traces so that we can know when to percolate indices during xbar
        #tree generation (if a trace is a head trace, then its index does not percolate to any x' and xp nodes which are created..
        head_trace = Node(features=[], original_features = selectee.head.original_features, name="Λ", mother=head.mother, movers=[], daughters=[], head=None, index=head)
    head_trace.head = head_trace
    head_trace.antecedent = head
    head_trace.heads = [head]
    if not head_is_selectee:
        head.mother.daughters.insert(head_index, head_trace)
        head_trace.mother = head.mother
    if 'hatb' in mother.name and hm_dir == 'excorp':
        selectee.head.index = mother.head
        #if this is head atb-drop then we just finish here, having inserted the trace
        #we don't want to actually move the head anywhere
        return
    mother_head = mother.head
    mother_head_index = mother_head.mother.daughters.index(mother_head)
    mother_head.mother.daughters.remove(mother_head)
    mother_head_mother = mother_head.mother
    if hm_dir == 'left' or hm_dir == 'excorp':
        new_head = Node(features=[], original_features = mother_head.original_features, name=">_hmove_l", mother=mother_head.mother, movers=[], daughters=[head, mother_head], head=mother_head, lexical_head=True)
    else:
        new_head = Node(features=[], original_features = mother_head.original_features, name="<_hmove_r", mother=mother_head.mother, movers=[], daughters=[mother_head, head], head=mother_head, lexical_head=True)
    mother_head.mother = new_head
    new_head.heads = [mother_head]
    head.mother = new_head
    mother_head_mother.daughters.insert(mother_head_index, new_head)
    mother_head_mother.heads = [new_head]
    restore_terminals(new_head)
    if hm_dir != 'excorp':
        mother.head = new_head
    else:
        #if the moving head is excorporating, we set it as the head of mother so that it will
        #be the part that is moved next time round (when mother is the selectee).
        #THIS MEANS YOU CAN'T USE THE HEAD FEATURE TO CONSTRUCT XBAR TREES! USE > < INSTEAD
        mother.head = head

def restore_terminals(head):
    #removes the semanticised or phoneticized annotation.. sometimes heads are getting mad semantic or phonetic before they are head moved
    for daughter in head.daughters:
        restore_terminals(daughter)
    if len(head.daughters) == 0:
        if len(head.name) > 0:
            if '/' == head.name[0]:
                head.name = head.name[1:-1]
            head.name = head.name.lower()
                
def merge(selector, selectee, mother, head_pointer, adjoin=False, hmove=False):
    #takes as input a node and its two daughters and deletes relevant features, determines if the selectee
    #is a mover etc.. basically performs the merge operation..
    if 'ps' not in mother.name:
        del(selector.features[0])
    if adjoin == False:
        if "_sc" not in mother.name:
            del(selectee.features[0])
        mother.features = selector.features
        mother.original_features = selector.original_features
        mother.head = selector.head
        mother.heads = [selector]
        selector.features = []
        #since mother is a projection of selector (selectee for adjunction), we can now delete all of selector's features since the
        #relevant ones have been transferred to mother..
    else:
        mother.features = selectee.features
        mother.original_features = selectee.original_features
        mother.head = selectee.head
        mother.heads = [selectee]
        selectee.features = []
    mother.name = head_pointer + mother.name
    if (adjoin == False and len(selectee.features) > 0) or (adjoin == True and len(selector.features) > 0):
        #if the selectee (selector for adjunction) is moving, we remove it from mother.daughters and insert it into mother.movers,
        #leaving a trace as mother's new daughter if this is leftward movement or a semantic copy if
        #rightward (=pf) movement.
        selectee_index = mother.daughters.index(selectee)
        selector_index = mother.daughters.index(selector)
        if adjoin == False:
            insert_traces_copies(mother, selectee, selectee_index, hmove)
            #we also shift all of the selectee's movers (and the selectee itself) into mother.movers, as long as this is
            #a first merge, ie a head-comp not head-spec or adjunction situation (owing to CED)
            mother.movers += selector.movers
        else:
            insert_traces_copies(mother, selector, selector_index, hmove)
            mother.movers += selectee.movers
    else:
        if adjoin == False:
            mother.movers += selector.movers
        else:
            mother.movers += selectee.movers
    if selector.lexical_head == True and adjoin == False:
        mother.movers+=selectee.movers
        selectee.movers = []
    else:
        if adjoin == False and len(selectee.movers) > 0:
            #if there exists a mover inside the specifier or adjunct which does not exactly match
            #a mover inside the governing constituent, then we have a CED violation..otherwise we escape
            #CED by ATB, because the copy inside the dependent simply gets dropped.  In this code we can
            #only check that the features of the two constituents match since their spans are not available..
            #need to add this in at some point though!!!  Sometimes, with rightward movement, the derivational
            #history of the two movers may differ without this being picked up, e.g. if one of the movers
            #has something right adjoined to it which was moved to that position by rightward movement..we can't use
            #check_features to check this unfortunately because rightward movement checks no feature on the licensee..
            #May have to look at this at some point..
            while len(selectee.movers) > 0:
                mover = selectee.movers[0]
                mover_copy_found = False
                for MOVER in selector.movers:
                    if mover.features == MOVER.features:
                        mover_node_count = count_nodes(mover, 0)
                        MOVER_node_count = count_nodes(MOVER, 0)
                        if mover_node_count > MOVER_node_count:
                            #the movers can differ in that one or both of them may have had adjuncts
                            #adjoined to them at some stage in their derivation which moved away.. these
                            #do not affect the chains in the derivation tree since no features on the selectee
                            #are checked by adjunction.. we want to retain as much info in the tree as possible,
                            #(such as traces of rightward movement inside the adjunct clause) so we will drop the constituent with the least nodes..
                            #note that there is still a risk that we will lose SOME info from the phrase structure trees
                            #IF both movers have different but the same number of adjunctions.. but c'est la vie!
                            #NOTE THAT WHAT WE'RE DOING HERE ONLY AFFECTS THE XBAR AND DERIVED TREES, HENCE IS
                            #NOT ACTUALLY AFFECTING THE GRAMMAR AT ALL! THE DERIVATION TREE IS THE PRIMATIVE OBJECT
                            mother.movers.remove(MOVER)
                            mother.movers.append(mover)
                            selector.movers.remove(MOVER)
                            selector.movers.append(mover)
                            selectee.movers.remove(mover)
                            selectee.movers.append(MOVER)
                            old_mover = mover
                            mover = MOVER
                            MOVER = old_mover
                        mover_copy_found = True
                        break
                if re.search('\w+~', mover.features[0]):
                    rightward_mover = True
                else:
                    rightward_mover = False
                if not rightward_mover and mover_copy_found == False and 'edge' not in mother.name:
                    raise Exception("Error! Spec merged that contains movers, violating CED")
                elif not rightward_mover and 'atb' not in mother.name and 'edge' not in mother.name:
                    raise Exception("Error! ATB operation attempted but ATB not in mother's name")
                elif 'atb' in mother.name:
                    #we unify the indices of both movers.. but first we need to find all the other nodes in the tree
                    #that currently have the same index as the one we are about to change so we can change all of them too.
                    #actually, the safest way to do that is to add the indices that must be changed to a dictionary and
                    #then do the correction later
                    if mover.index != None:
                        mover_index = mover.index
                        while type(mover_index) != type([]):
                            mover_index = mover.index.index
                        new_mover_index = MOVER
                        while type(new_mover_index) != type([]):
                            new_mover_index = new_mover_index.index
                        indices_to_fix[mover_index[0]] = new_mover_index[0]
                    mover.index = MOVER
                    selectee.movers.remove(mover)
                    #unfortunately, we cannot perform the check to make sure that in the case of coordination
                    #both conjuncts have the same number of movers inside because the conj category is not currently visible
                    #in the trees.. So at the moment this code will let through "who did Jack punch Mary and kiss Sue" even though the
                    #parser itself will block these (because it has access to the special cat_feature which does have conj in it)
                elif 'edge' in mother.name or rightward_mover:
                    #this is only an approximate check.. the parser does the real checks and would
                    #only let through the right things..
                    mother.movers.append(mover)
                    selectee.movers.remove(mover)
        elif adjoin == True and len(selector.movers) > 0:
            while len(selector.movers) > 0:
                mover = selector.movers[0]
                mover_copy_found = False
                for MOVER in selectee.movers:
                    if mover.features == MOVER.features:
                        mover_copy_found = True
                        mover_node_count = count_nodes(mover, 0)
                        MOVER_node_count = count_nodes(MOVER, 0)
                        if mover_node_count > MOVER_node_count:
                            #the movers can differ in that one or both of them may have had adjuncts
                            #adjoined to them at some stage in their derivation which moved away.. these
                            #do not affect the chains in the derivation tree since no features on the selectee
                            #are checked by adjunction.. we want to retain as much info in the tree as possible,
                            #(such as traces of rightward movement inside the adjunct clause) so we will drop the constituent with the least nodes..
                            #note that there is still a risk that we will lose SOME info from the phrase structure trees
                            #IF both movers have different but the same number of adjunctions.. but c'est la vie!
                            mother.movers.remove(MOVER)
                            mother.movers.append(mover)
                            selectee.movers.remove(MOVER)
                            selectee.movers.append(mover)
                            selector.movers.remove(mover)
                            selector.movers.append(MOVER)
                            old_mover = mover
                            mover = MOVER
                            MOVER = old_mover
                        break
                if 'edge' not in mother.name and 'atb' not in mother.name:
                    raise Exception("Error! Movers inside adjunct violating CED!")
                elif 'atb' in mother.name:
                    if mover.index != None:
                        mover_index = mover.index
                        while type(mover_index) != type([]):
                            mover_index = mover.index.index
                        new_mover_index = MOVER
                        while type(new_mover_index) != type([]):
                            new_mover_index = new_mover_index.index
                        indices_to_fix[mover_index[0]] = new_mover_index[0]
                    mover.index = MOVER
                    try:
                        selector.movers.remove(mover)
                    except ValueError:
                        selector.movers.remove(MOVER)
                elif 'edge' in mother.name:
                    mother.movers+=selector.movers
                    selector.movers = []

def count_nodes(node, count):
    #returns the number of nodes in any tree passed to it
    count += 1
    for daughter in node.daughters:
        count = count_nodes(daughter, count)
    return count

def insert_traces_copies(mother, dependent, dependent_index, hmove=False):
    global variables
    global indices
    #an .index can either be an actual index, or it can point to another node.. the actual index must always be on the mover
    index = indices[0]
    del(indices[0])
    if len(dependent.features) == 1 and re.search('\w+~', dependent.features[0]):
        #if this element's only movement feature is a pf one, then we leave behind a semantic copy, not a trace
        #we don't want a deep copy of the index, we want an actual copy of it
        #we set the phon_node's index to be the dependent (ie the trace) could have been the other way around,
        #but doesn't really matter..
        phon_node = copy.deepcopy(dependent)
        if '<' in phon_node.name:
            phon_node.heads = [phon_node.daughters[0]]
        elif '>' in phon_node.name:
            phon_node.heads = [phon_node.daughters[1]]
        if hmove == True:
        #this is currently set up for the case where a moving complement/spec moves which contained an excorporating head
        #so we turn the moving head into a trace.. but I'm not sure if it will work for the case where the complement
        #is moving and its head also moving, but this time not an excorporating head.. all depends exactly what 'head' points to..
            phon_node.head.name = 'Λ'
            phon_node.head.antecedent = dependent.head
            phon_node.head.heads = [dependent.head]
            phon_node.head.daughters = []
        phon_node.index = [index]
        dependent.index = phon_node
        #we need a pointer from the trace to the antecedent for pf movement (which we get
        #automatically with normal movement) for dependency collection
        dependent.sem_node = True
        make_terminals_semantic(dependent)
        dependent.rightward_trace = True
        phon_node.phon_node = True
        make_terminals_phonetic(phon_node, true_phonetic = True)
        dependent.heads = [phon_node]
        mother.movers.append(phon_node)
        phon_node.mother = None
        dependent.features = []
    else:
        if hmove == True:
            dependent.head.old_name = dependent.head.name
            dependent.head.name = 'Λ'
            dependent.head.daughters = []
        dependent.index = [index]
        #the next line is needed in head_move() for the case where the selectee whose head is moving is also moving
        dependent.head_index = dependent.head.mother.daughters.index(dependent.head)
        mother.daughters.remove(dependent)
        #don't set dependent.mother = None at this point as it screws things up during dependency extraction
        #as a later trace will have its mother set to the dependent's mother
        if '_phon' in mother.name:
            trace_node = copy.deepcopy(dependent)
            trace_node.mother = dependent.mother
            if trace_node.name == 'Λ':
                #this is for the case where the selectee is at the bottom of the tree and is
                #undergoing phrasal movement and its head is also moving out..  since the dependent is
                #just a single node, that node has the name 'Λ' and we therefore must change this to
                #the phrasal trace lambda
                trace_node.name = 'λ'
            trace_node.features = []
            dependent.heads = [trace_node]
            trace_node.phon_node = True
            make_terminals_phonetic(trace_node, true_phonetic=False)
            if hmove:
                trace_node.head.heads = [dependent.head]
                trace_node.head.antecedent = dependent.head
            if '<' in trace_node.name:
                trace_node.heads = [trace_node.daughters[0]]
            elif '>' in trace_node.name:
                trace_node.heads = [trace_node.daughters[1]]
        else:
            trace_node = Node(features=[], original_features=dependent.original_features, name="λ", mother=mother, movers=[], daughters=[], head=None, index=dependent)
            trace_node.heads = [dependent]
        #if we later find that the moving element only moved covertly, we will need to replace the trace node
        #by a phonetic version of the original constituent.. hence we keep a pointer from mover to its lowest trace
        #CORRECTION, I CHANGED THE CODE SO THAT PHONETIC COPIES GET INSERTED RIGHT AWAY
        #SO THAT THIS WOULD SHOW UP IN THE DERIVATION BUILDER IN autobank.py.. HOWEVER
        #I HAVE LEFT ALL THE ORIGINAL CODE IN PLACE FOR NOW AS IT WOULD HAVE BEEN MESSY TO REMOVE/CHANGE IT ALL..
        dependent.lowest_overt_trace = trace_node
        mother.daughters.insert(dependent_index, trace_node)
        trace_node.mother = mother
        mother.movers.append(dependent)
        
def gen_derivation_tree(derivation_bracketing):
    #reads in the bracketing of a derivation tree and builds an object oriented
    #representation of the tree
    root_node = None
    mother = None
    for char in derivation_bracketing:
        if char == "(":
            if root_node != None:
                mother = current_node
            current_node = Node(features=[], original_features=[], name="", mother=mother, movers=[], daughters=[], head=None)
            if root_node == None:
                root_node = current_node
            if mother != None:
                current_node.mother.daughters.append(current_node)
        elif char not in [")"]:
            current_node.name += char
        elif char == ")":
            current_node = current_node.mother
    fix_mother_relations(root_node)
    remove_epsilons_from_terminals(root_node)
    return root_node

def remove_epsilons_from_terminals(node):
    if node.daughters == []:
        try:
            #we try both the unicode and utf8 encodings in case..
            node.name = re.sub(u'\u03b5; ', '', node.name)
            node.name = re.sub(u'; \u03b5', '', node.name)
            node.name = re.sub('\xce\xb5; ', '', node.name)
            node.name = re.sub('; \xce\xb5', '', node.name)
        except Exception as e:
            x=0
    for daughter in node.daughters:
        remove_epsilons_from_terminals(daughter)

def gen_bracketing(tree, bracketing = ""):
    #generates a bracketing given a tree (node) object
    bracketing += "(" + tree.name
    if len(tree.daughters) > 0:
        bracketing += " "
    for daughter in tree.daughters:
        bracketing = gen_bracketing(daughter, bracketing)
    bracketing += ")"
    return bracketing
