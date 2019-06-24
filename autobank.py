#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
from nltk import WordNetLemmatizer
from operator import itemgetter
import pdb
from Tkinter import *
import re
import nltk
import string
try:
    import cky_mg
    import autobankGUI
except Exception as e:
    pass
import os
from nltk.tree import Tree
from timeit import default_timer
import sys
import json
import ast
import copy
import time
import gc
import gen_derived_tree
import math
import argparse
from timeout import timeout
sys.setrecursionlimit(10000)
seeds = None
previous_bracketing0 = None
previous_bracketing1 = None
wordnet_lemmatizer = WordNetLemmatizer()
punctuation = string.punctuation
punctuation+="``"
punctuation+="..."
punctuation+=" "
punctuation = re.sub('%', '', punctuation)
punctuation = re.sub('&', '', punctuation)
punctuation = re.sub('\$', '', punctuation)
punctuation = re.sub('/', '', punctuation)
punctuation = re.sub(':', '', punctuation)
punctuation = re.sub(';', '', punctuation)
punctuation = re.sub('-', '', punctuation)
punctuation = re.sub("'", "", punctuation, count=1000)
punctuation = re.sub('@', '', punctuation)
cat_pattern = re.compile('\w+')
start_file = None
end_file = None
start_line = None
min_length = None
max_length = None
ccg_beam = None
ccg_beam_floor = None
auto_section_folders = None
time_out = None
parser_setting = None
stop_after = None
train_tagger = True
extract_dep_mappings = True
check_auto_mappings = False
max_mg_cats_per_word = None
useAllNull = False
skipRel = False
skipPro= False
super_rare_cutoff = None
super_tag_dict_min = None
super_category_cutoff = None
super_forward_beam_ratio = None
super_beam_width = None
super_beam_ratio = None
constrainMoveWithPTB = None
constrainConstWithPTBCCG = None
maxMoveDist = None
allowbesttag = False
allowMoreGoals = True
ptb_word_token_count = 0
sel_variables = ['x', 'y', 'z', 'w']

def Dict(**args): 
    """Return a dictionary with argument names as the keys, 
    and argument values as the key values"""
    return args

not_selectee_feature = re.compile('=|(\+)|(\-)|~|≈')

PTBcats = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS', 'IN', 'PDT', 'WDT', 'DT',
               'CC', 'CD', 'RB', 'RBR', 'RBS', 'WRB', 'RP', 'EX', 'FW', 'JJ', 'JJR', 'JJS',
               'LS', 'MD', 'POS', 'WP', 'WP$', 'PRP', 'SYM', 'TO', 'UH', '.', ':', "''", '``', '$', ',', '',
           'LRB', 'RRB', '#', 'NML', 'NNM', 'NNF', 'NNMF', 'NNPM', 'NNPF', 'NNPMF', 'PRP3SG', 'PRPM3SG', 'PRPF3SG',
           'PRP1SG', 'PRP2', 'PRP1PL', 'PRP3PL', 'PRP3SGSELF', 'PRPM3SGSELF', 'PRPF3SGSELF',
           'PRP1SGSELF', 'PRP2','PRP2SGSELF', 'PRP2PLSELF', 'PRP1PLSELF', 'PRP3PLSELF', 'PRP$3SG', 'PRP$M3SG', 'PRP$F3SG',
           'PRP$1SG', 'PRP$2', 'PRP$1PL', 'PRP$3PL', 'HYPH', 'NEG', 'DTSG', 'DTPL']

dispreferred_subcats = ['LV']

PTBcats.sort()

terminal_index = [-1]
find_syn_head_data = Dict(
    #list the possible heads in order of priority, most likely head first,
    #along with whether the daughters of the node should be search left to right
    #L or right to left R.. an A next to this letter = 'all' (see Julia Hockenmaier's thesis)
    #default tells the system which constituent to select as head in the event that
    #no items in the list are found... L=leftmost, R=rightmost.  For a head-initial language like English,
    #this should be set to L. This head-finder is adapted from both Collins and Hockenmaier.. Note that 
    #the rules have been modified so that there is always a direct head path from S, SBAR, SINV etc down to the
    #lexical verb, and from NP down to the lexical noun (rather than e.g. the possessive 's)..  This is a Jackendoffian (1977)
    #view of head-modifier relations and helps when determining the semantic heads of extended projections in MGs
    #I also do the same for PPs, so that the head of the PP in penn (but not in the MG trees) is the lexical noun
    default = "L",
    ADJP = [('JJ', 'L'), ('VBN', 'L'), ('NNS', 'L'), ('QP', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('$', 'L'), ('ADVP', 'L'), ('VBG', 'L'),
            (['ADJP', 'JJP'], 'L'), ('JJR', 'L'), ('NP', 'L'), ('JJS', 'L'), (['DT', 'DTSG', 'DTPL'], 'L'), ('FW', 'L'), ('RBR', 'L'), ('RBS', 'L'),
            ('*SBAR', 'L'), ('RB', 'L')],
    JJP = [('NNS', 'L'), ('QP', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('$', 'L'), ('ADVP', 'L'), ('JJ', 'L'), ('VBN', 'L'), ('VBG', 'L'),
            (['ADJP', 'JJP'], 'L'), ('JJR', 'L'), ('NP', 'L'), ('JJS', 'L'), (['DT', 'DTSG', 'DTPL'], 'L'), ('FW', 'L'), ('RBR', 'L'), ('RBS', 'L'),
            ('*SBAR', 'L'), ('RB', 'L')],
    ADVP = [('RB', 'R'), ('RBR', 'R'), ('RBS', 'R'), ('FW', 'R'), ('ADVP', 'R'), ('TO', 'R'), ('CD', 'R'), ('JJR', 'R'),
            ('JJ', 'R'), ('IN', 'R'), ('NP', 'R'), ('JJS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R')],
    CONJP = [('CC', 'R'), ('RB', 'R'), ('IN', 'R')],
    FRAG = 'R',
    INTJ = 'L',
    LST = [('LS', 'R'), (':', 'R')],
    NAC = [(['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('NNS', 'L'), (['NNP', 'NNPM', 'NNPF', 'NNPMF'], 'L'), ('NNPS', 'L'), ('NP', 'L'), ('NAC', 'L'), ('EX', 'L'), ('$', 'L'), ('CD', 'L'),
           ('QP', 'L'), (['PRP', 'PRP1SG', 'PRP2', 'PRP1PL', 'PRP3PL', 'PRPM3SG', 'PRPM3SG', 'PRPF3SG', 'PRPF3SG', 'PRP1SGSELF', 'PRP2SGSELF', 'PRP2PLELF', 'PRP1PLSELF', 'PRP3PLSELF', 'PRPM3SGSELF', 'PRPF3SGSELF', 'PRPM3SGSELF', 'PRPF3SGSELF'], 'L'), ('VBG', 'L'), ('JJ', 'L'), ('JJS', 'L'), ('JJR', 'L'), (['ADJP', 'JJP'], 'L'), ('FW', 'L')],
    NP = [('VP', 'R'), (['DT', 'DTSG', 'DTPL'], 'L'), ('QP', 'R'), ('NNS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R'), (['NNP', 'NNPM', 'NNPF', 'NNPMF'], 'L'), ('NNPS', 'L'), ('NX', 'R'), ('NP', 'L')],
    NX = [('VP', 'R'),(['DT', 'DTSG', 'DTPL'], 'L'), ('QP', 'R'), ('NNS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R'), (['NNP', 'NNPM', 'NNPF', 'NNPMF'], 'L'), ('NNPS', 'R'), ('NX', 'R'), ('NP', 'L'), ('QP', 'R')],
    NML = [('NNS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R'), (['NNP', 'NNPM', 'NNPF', 'NNPMF'], 'L'), ('NNPS', 'L'), ('NX', 'R'), ('NP', 'L'), ('QP', 'R')],
    PP = [('IN', 'R'), ('PP', 'R'), ('TO', 'R'), ('NP', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NNS'], 'R'), (['NNP', 'NNPM', 'NNPF', 'NNPMF', 'NML'], 'L'), ('S', 'R'), ('VBG', 'R'), ('VBN', 'R'), ('RP', 'R'), ('FW', 'R')],
    PRN = 'L',
    PRT =[('RP', 'R')],
    QP = [('CC', 'L'), ('$', 'L'), ('IN', 'L'), ('NNS', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('JJ', 'L'), ('RB', 'L'), (['DT', 'DTSG', 'DTPL'], 'L'), ('CD', 'L'),
          ('NCD', 'L'), ('QP', 'L'), ('JJR', 'L'), ('JJS', 'L')],#CC will only be set to head in PTB trees that lack the COORD tag.. otherwise the conjuncts are made the heads
    RRC = [('VP', 'R'), ('NP', 'R'), ('ADVP', 'R'), (['ADJP', 'JJP'], 'R'), ('PP', 'R')],
    S = [('IN', 'L'), ('S', 'L'), ('*SBAR', 'L'), ('TO', 'L'), ('VP', 'L'), (['ADJP', 'JJP'], 'L'), ('UCP', 'L'), ('NP', 'R')],
    SBAR = [('IN', 'L'), ('S', 'L'), ('*SQ', 'L'), ('*SINV', 'L'), ('*SBAR', 'L'), ('FRAG', 'L')],
    SBARQ = [('IN', 'L'), ('*SQ', 'L'), ('S', 'L'), ('*SINV', 'L'), ('*SBARQ', 'L'), ('FRAG', 'L')],
    SINV = [('VBZ', 'L'), ('VBD', 'L'), ('VBP', 'L'), ('VB', 'L'), ('MD', 'L'), ('VP', 'L'), ('S', 'L'), ('*SINV', 'L'),
            (['ADJP', 'JJP'], 'L'), ('NP', 'L')],
    SQ = [('VBZ', 'L'), ('VBD', 'L'), ('VBP', 'L'), ('VB', 'L'), ('MD', 'L'), ('VP', 'L'), ('*SQ', 'L')],
    UCP = 'R',
    VP = [('VBD', 'L'), ('TO', 'L'), ('VBN', 'L'), ('MD', 'L'), ('VBZ', 'L'), ('VB', 'L'), ('VBG', 'L'), ('VBP', 'L'), ('VP', 'L'),
          (['ADJP', 'JJP'], 'L'), (':', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('NNS', 'L'), ('NP', 'L')],
    WHADJP = [('CC', 'L'), ('WRB', 'L'), ('JJ', 'L'), (['ADJP', 'JJP'], 'L')],
    WHADVP = [('CC', 'R'), ('WRB', 'R')],
    WHNP = [('WDT', 'L'), ('WP', 'L'), ('WP$', 'L'), ('WHADJP', 'L'), ('WHPP', 'L'), ('WHNP', 'L')],
    WHPP = [('IN', 'R'), ('TO', 'R'), ('FW', 'R')],
    BP = [('LRB', 'L')],
    penn_argument_tags = ['-SBJ', '-CLR', '-DTV', '-TPC', '-PRD'],
    penn_adjunct_tags = ['-ADV', '-VOC', '-BNF', '-DIR', '-LOC', '-MNR', '-TMP'],
    propbank_argument_tags = ['-ARG0', '-ARG1', '-ARG2', '-ARG3', '-ARG4', '-ARG5'],
    propbank_adjunct_tags = ['-ARGM-DIR', '-ARGM-LOC', '-ARGM-MNR', '-ARGM-EXT', '-ARGM-REC', '-ARGM-PRD', '-ARGM-PNC', '-ARGM-CAU', '-ARGM-DIS', '-ARGM-ADV', '-ARGM-MOD', '-ARGM-NEG'],
    #function_tags holds all penn and pb argument and adjunct tags plus the -NOM tag.. it is used for disallowing daughters to be heads (if they have a function they
    #cannot be the head daughter by definition, since they are arguments or adjuncts..
    function_tags = ['-SBJ', '-CLR', '-DTV', '-TPC', '-PRD', '-ADV', '-VOC', '-BNF', '-DIR', '-LOC', '-MNR', '-TMP', '-NOM', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', '-ARGM-DIR', '-ARGM-LOC', '-ARGM-MNR', '-ARGM-EXT', '-ARGM-REC', '-ARGM-PRD', '-ARGM-PNC', '-ARGM-CAU', '-ARGM-DIS', '-ARGM-ADV', '-ARGM-MOD', '-ARGM-NEG'])

find_sem_head_data = Dict(
    default = "L",
    ADJP = [('JJ', 'L'), ('VBN', 'L'), ('NNS', 'L'), ('QP', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('$', 'L'), ('ADVP', 'L'), ('VBG', 'L'),
            (['ADJP', 'JJP'], 'L'), ('JJR', 'L'), ('NP', 'L'), ('JJS', 'L'), (['DT', 'DTSG', 'DTPL'], 'L'), ('FW', 'L'), ('RBR', 'L'), ('RBS', 'L'),
            ('*SBAR', 'L'), ('RB', 'L')],
    JJP = [('NNS', 'L'), ('QP', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('$', 'L'), ('ADVP', 'L'), ('JJ', 'L'), ('VBN', 'L'), ('VBG', 'L'),
            (['ADJP', 'JJP'], 'L'), ('JJR', 'L'), ('NP', 'L'), ('JJS', 'L'), (['DT', 'DTSG', 'DTPL'], 'L'), ('FW', 'L'), ('RBR', 'L'), ('RBS', 'L'),
            ('*SBAR', 'L'), ('RB', 'L')],
    ADVP = [('RB', 'R'), ('RBR', 'R'), ('RBS', 'R'), ('FW', 'R'), ('ADVP', 'R'), ('TO', 'R'), ('CD', 'R'), ('JJR', 'R'),
            ('JJ', 'R'), ('IN', 'R'), ('NP', 'R'), ('JJS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R')],
    CONJP = [('CC', 'R'), ('RB', 'R'), ('IN', 'R')],
    FRAG = 'R',
    INTJ = 'L',
    LST = [('LS', 'R'), (':', 'R')],
    NAC = [(['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('NNS', 'L'), (['NNP', 'NNPM', 'NNPF', 'NNPMF'], 'L'), ('NNPS', 'L'), ('NP', 'L'), ('NAC', 'L'), ('EX', 'L'), ('$', 'L'), ('CD', 'L'),
           ('QP', 'L'), (['PRP', 'PRP1SG', 'PRP2', 'PRP1PL', 'PRP3PL', 'PRPM3SG', 'PRPM3SG', 'PRPF3SG', 'PRPF3SG', 'PRP1SG', 'PRP2SGSELF', 'PRP2SLSELF', 'PRP1PLSELF', 'PRP3PLSELF', 'PRPM3SGSELF', 'PRPF3SGSELF', 'PRPM3SGSELF', 'PRPF3SGSELF'], 'L'), ('VBG', 'L'), ('JJ', 'L'), ('JJS', 'L'), ('JJR', 'L'), (['ADJP', 'JJP'], 'L'), ('FW', 'L')],
    NP = [('VP', 'R'), ('NNS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R'), (['JJ', 'NNP', 'NNPM', 'NNPF', 'NNPMF', 'NNPS'], 'L'), ('NNPS', 'L'), ('NX', 'R'), ('NP', 'L'), ('QP', 'R'), (['DT', 'DTSG', 'DTPL'], 'L')],
    NX = [('VP', 'R'), ('NNS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R'), (['JJ', 'NNP', 'NNPM', 'NNPF', 'NNPMF', 'NNPS'], 'L'), ('NNPS', 'L'), ('NX', 'R'), ('NP', 'L'), ('QP', 'R'), (['DT', 'DTSG', 'DTPL'], 'L')],
    NML = [('NNS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R'), (['NNP', 'NNPM', 'NNPF', 'NNPMF'], 'R'), ('NNPS', 'R'), ('NX', 'R'), ('NP', 'L'), ('QP', 'R')],
    PP = [('PP', 'R'), ('NP', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NNS', 'NNP', 'NNPM', 'NNPF', 'NNPMF', 'NML'], 'R'), ('S', 'R'), ('TO', 'R'), ('VBG', 'R'), ('VBN', 'R'), ('IN', 'R'), ('RP', 'R'), ('FW', 'R')],
    PRN = 'L',
    PRT =[('RP', 'R')],
    QP = [('$', 'L'), ('IN', 'L'), ('NNS', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('JJ', 'L'), ('RB', 'L'), (['DT', 'DTSG', 'DTPL'], 'L'), ('CD', 'L'),
          ('NCD', 'L'), ('QP', 'L'), ('JJR', 'L'), ('JJS', 'L')],
    RRC = [('VP', 'R'), ('NP', 'R'), ('ADVP', 'R'), (['ADJP', 'JJP'], 'R'), ('PP', 'R')],
    S = [('VP', 'L'), ('S', 'L'), ('*SBAR', 'L'), ('TO', 'L'), ('IN', 'L'), (['ADJP', 'JJP'], 'L'), ('UCP', 'L'), ('NP', 'R')],
    SBAR = [('S', 'L'), ('*SQ', 'L'), ('*SINV', 'L'), ('*SBAR', 'L'), ('FRAG', 'L')],
    SBARQ = [('*SQ', 'L'), ('S', 'L'), ('*SINV', 'L'), ('*SBARQ', 'L'), ('FRAG', 'L')],
    SINV = [('VP', 'L'), ('S', 'L'), ('*SINV', 'L'), ('VBZ', 'L'), ('VBD', 'L'), ('VBP', 'L'), ('VB', 'L'), ('MD', 'L'),
            (['ADJP', 'JJP'], 'L'), ('NP', 'L')],
    SQ = [('*SQ', 'L'), ('VP', 'L'), ('VBZ', 'L'), ('VBD', 'L'), ('VBP', 'L'), ('VB', 'L'), ('MD', 'L')],
    UCP = 'R',
    VP = [('VP', 'L'), ('VBD', 'L'), ('TO', 'L'), ('VBN', 'L'), ('MD', 'L'), ('VBZ', 'L'), ('VB', 'L'), ('VBG', 'L'), ('VBP', 'L'),
          (['ADJP', 'JJP'], 'L'), (':', 'L'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'L'), ('NNS', 'L'), ('NP', 'L')],
    WHADJP = [('CC', 'L'), ('WRB', 'L'), ('JJ', 'L'), (['ADJP', 'JJP'], 'L')],
    WHADVP = [('CC', 'R'), ('WRB', 'R')],
    WHNP = [('NNS', 'R'), (['NN', 'NNM', 'NNF', 'NNMF', 'NML'], 'R'), (['JJ', 'NNP', 'NNPM', 'NNPF', 'NNPMF', 'NNPS'], 'L'), ('NNPS', 'L'), ('NX', 'R'), ('NP', 'L'), ('WDT', 'L'), ('WP', 'L'), ('WP$', 'L'), ('WHADJP', 'L'), ('WHPP', 'L'), ('WHNP', 'L')],
    WHPP = [('IN', 'R'), ('TO', 'R'), ('FW', 'R')],
    BP = [('LRB', 'L')],
    penn_argument_tags = ['-SBJ', '-CLR', '-DTV', '-TPC', '-PRD'],
    penn_adjunct_tags = ['-ADV', '-VOC', '-BNF', '-DIR', '-LOC', '-MNR', '-TMP'],
    propbank_argument_tags = ['-ARG0', '-ARG1', '-ARG2', '-ARG3', '-ARG4', '-ARG5'],
    propbank_adjunct_tags = ['-ARGM-DIR', '-ARGM-LOC', '-ARGM-MNR', '-ARGM-EXT', '-ARGM-REC', '-ARGM-PRD', '-ARGM-PNC', '-ARGM-CAU', '-ARGM-DIS', '-ARGM-ADV', '-ARGM-MOD', '-ARGM-NEG'],
    #function_tags holds all penn and pb argument and adjunct tags plus the -NOM tag.. it is used for disallowing daughters to be heads (if they have a function they
    #cannot be the head daughter by definition, since they are arguments or adjuncts..
    function_tags = ['-SBJ', '-CLR', '-DTV', '-TPC', '-PRD', '-ADV', '-VOC', '-BNF', '-DIR', '-LOC', '-MNR', '-TMP', '-NOM', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', '-ARGM-DIR', '-ARGM-LOC', '-ARGM-MNR', '-ARGM-EXT', '-ARGM-REC', '-ARGM-PRD', '-ARGM-PNC', '-ARGM-CAU', '-ARGM-DIS', '-ARGM-ADV', '-ARGM-MOD', '-ARGM-NEG'])

country_adjectives = ['French', 'English', 'Dutch', 'German', 'Chinese', 'Japanese', 'American', 'Austrian', 'Swiss', 'Belgian', 'Itaian', 'Spanish', 'Australian', 'Canadian']
discourse_adverbs = ['though', 'however', 'moreover', 'nevertheless', 'anyway']
aux_verbs = ['do', 'does', 'did', 'done', 'doing', 'is', 'am', 'are', 'were', 'was', 'being', 'been', 'be', 'have', 'having', 'has', 'had', 'must', 'might', 'should', 'will', 'may', 'could', 'would', 'can', 'shall']

class Node:
    #the phrase parameter is for MG trees only, and identifies each node
    #be it a head, X' or XP with a given xbar schema, e.g. a specific vP or TP phrase etc
    def __init__(self, name='', mother = None, terminal = False, daughter_index=-1):
        self.daughters=[]
        self.terminal = terminal
        self.set_name(name)
        self.set_mother(mother, daughter_index)
        self.heads = []
        self.sem_heads = []
        self.chain_pointer = None
        self.indices = []
        if self.terminal == True:
            self.index = get_new_terminal_index()

    def __len__(self):
        return len(self.daughters)

    def __str__(self):
        self.generate_bracketing()

    def set_name(self, name):
        self.name=name
        #truncated name is the name with all tags, indices etc stripped off
        self.truncated_name = ""
        for char in self.name:
            if char != "-" or self.name[0] == char:
                self.truncated_name += char
            else:
                break

    def add_daughter(self, daughter, daughter_index=-1):
        #linearity is defined by the order of daughters in the daughters list
        #if no daughter index is supplied, the daughter is added to the right of
        #any other daughters
        if daughter not in self.daughters:
            if daughter_index==-1:
                self.daughters.append(daughter)
            else:
                self.daughters.insert(daughter_index, daughter)
        daughter.mother=self

    def set_mother(self, mother, daughter_index):
        self.mother=mother
        if self.mother != None and self not in self.mother.daughters:
            self.mother.add_daughter(self, daughter_index)

    def set_heads(self, head_type):
    #takes as input a Penn Treebank node and assigns a head to
    #the node. Note that the .heads attribute is the head children, eg VP for S.
    #to retrieve the ultimate terminal head for any phrase, eg 'said' for S
    #use the get_terminal_heads method.  Note that head assignment is done automatically
    #for MG trees as they are constructed..
        if head_type == 'syntactic':
            HEADS = self.heads
            FIND_HEAD_DATA = find_syn_head_data
        elif head_type == 'semantic':
            HEADS = self.sem_heads
            FIND_HEAD_DATA = find_sem_head_data
        if self.terminal == False:
            if not ('CC' in [daughter.name[:2] for daughter in self.daughters] and 'COORD' in ['COORD' for d in self.daughters if 'COORD' in d.name]):
                try:
                    if self.truncated_name in ['NP', 'NML']:
  	                #from Collins 1999.. we want the possessive s to be the head
                        if self.daughters[-1].truncated_name == 'POS':
                            HEADS.append(self.daughters[-1])
                            return
                        elif head_type == 'syntactic' and self.daughters[0].truncated_name == 'NP':
                            if self.daughters[0].daughters[-1].truncated_name == 'POS':
                                HEADS.append(self.daughters[0])
                                return
                    for entry in FIND_HEAD_DATA[self.truncated_name]:
                        if type(entry) == type(()):
                            if entry[1] == 'R':
                                self.daughters.reverse()
                            if type(entry[0]) == type("string"):
                                for daughter in self.daughters:
                                    if daughter.truncated_name == entry[0] or "*"+daughter.truncated_name == entry[0]:
                                    #before we accept a constituent as a head, we first
                                    #check to make sure it does not have a function tag..
                                    #if it has a ! tag we ignore the function tags as this indicates a split tree pointer
                                        modifier = False
                                        if "!" not in daughter.name:
                                            for tag in FIND_HEAD_DATA['function_tags']:
                                                if tag in daughter.name:
                                                    exists_non_modifier_daughter = False
                                                    for DAUGHTER in self.daughters:
                                                        MODIFIER = False
                                                        #unless there is a daughter without a function tag available, we will not block this
                                                        #daughter from being the head..  e.g. S with two NP daughters as in small clauses..both NPs have ARG tags
                                                        if DAUGHTER == daughter or DAUGHTER.truncated_name in [',', '.', '#', '$', '"', ':', '', 'HYPH', '``']:
                                                            continue
                                                        else:
                                                            for TAG in FIND_HEAD_DATA['function_tags']:
                                                                if TAG in DAUGHTER.name:
                                                                    MODIFIER = True
                                                                    break
                                                            if not MODIFIER:
                                                                exists_non_modifier_daughter = True
                                                                break
                                                    if exists_non_modifier_daughter:
                                                        modifier = self.check_head_exceptions(tag, daughter)
                                                        break
                                        if modifier == True:
                                            continue
                                        elif modifier == False:
                                            HEADS.append(daughter)
                                            if entry[1] == 'R':
                                                self.daughters.reverse()
                                            return
                            #because we introduced NML and JJP nodes, some head rules look for one of two categories..
                            elif type(entry[0]) == type([]):
                                for daughter in self.daughters:
                                    CONTINUE = False
                                    for ENTRY in entry[0]:
                                        if CONTINUE == True:
                                            continue
                                        if daughter.truncated_name == ENTRY or "*"+daughter.truncated_name == ENTRY:
                                        #before we accept a constituent as a head, we first
                                        #check to make sure it does not have a function tag
                                            modifier = False
                                            if "!" not in daughter.name:
                                                for tag in FIND_HEAD_DATA['function_tags']:
                                                    if tag in daughter.name:
                                                        exists_non_modifier_daughter = False
                                                        for DAUGHTER in self.daughters:
                                                            MODIFIER = False
                                                            #unless there is a daughter without a function tag available, we will not block this
                                                            #daughter from being the head..  e.g. S with two NP daughters as in small clauses..both NPs have ARG tags
                                                            if DAUGHTER == daughter:
                                                                continue
                                                            else:
                                                                for TAG in FIND_HEAD_DATA['function_tags']:
                                                                    if TAG in DAUGHTER.name:
                                                                        MODIFIER = True
                                                                        break
                                                                if not MODIFIER:
                                                                    exists_non_modifier_daughter = True
                                                                    break
                                                        if exists_non_modifier_daughter:
                                                            modifier = self.check_head_exceptions(tag, daughter)
                                                            break
                                            if modifier == True:
                                                CONTINUE=True
                                                continue
                                            elif modifier == False:
                                                HEADS.append(daughter)
                                                if entry[1] == 'R':
                                                    self.daughters.reverse()
                                                return
                                        
                            if entry[1] == 'R':
                                self.daughters.reverse()
                
                        else:
                    #if there's no list of prioritized heads in the data, then we just
                    #choose either the first or last daughter depending on L or R in entry
                            if entry == 'R':
                                self.daughters.reverse()
                            HEADS.append(self.daughters[0])
                            if entry == 'R':
                                self.daughters.reverse()
                            return
                except KeyError:
                    x=0
            else:
                #we will treat all conjuncts as heads for the trees that contain the COORD tags.. otherwise, we just
                #searched for the single head above as for any other phrase.
                for daughter in self.daughters:
                    if 'COORD' in daughter.name:
                        HEADS.append(daughter)
                        if "-UCP" not in self.name:
                            self.name = re.sub('UCP', HEADS[0].truncated_name+"-UCP", self.name, count=1)
                            self.truncated_name = re.sub('UCP', HEADS[0].truncated_name, self.truncated_name, count=1)
                if len(HEADS) != 0:
                    return
        elif self.terminal == True:
            HEADS.extend([self])
            return
        #if we make it here then no head has been found and we just pick either
        #the first (L) or last (R) daughter as head depending on the default parameter
        if len(HEADS) == 0:
            if FIND_HEAD_DATA['default'] == 'R':
                self.daughters.reverse()
            HEADS.append(self.daughters[0])
            if FIND_HEAD_DATA['default'] == 'R':
                self.daughters.reverse()
            return

    def check_head_exceptions(self, function_tag, daughter_node):
        #contains some exceptions to the rules which state that any node with a function tag
        #cannot be the head of its mother
        #there are a few examples in the Penn Treebank of where an SBAR dominates
        #another SBAR with a function tag.. in all but one cases, unless the tag is -ADV,
        #the daughter SBAR with tag(s) should in fact be the head daughter(s)
        if daughter_node.mother.truncated_name == 'NP':
            #we don't want to use the tags provided by nombank internal to NPs here as they often annotate the head as a
            #argument of one of the other daughters.. eg in 'Dutch publishing group', 'group' is the syntactic head but is marked as
            #the agent (ARG0) of publishing for semantic reasons.
            return False
        if daughter_node.name[:4] == 'SBAR':
            if function_tag != '-ADV':
                return False
            else:
                return True
        if daughter_node.name[:5] == 'S-NOM' and daughter_node.mother.truncated_name == 'PP':
            return False
        #in some cases, nombank annotates head nouns as their own ARG0..  We don't want to treat these as modifiers..
        elif len(daughter_node.daughters) == 1:
            if daughter_node.daughters[0].terminal == True:
                if '{'+daughter_node.daughters[0].name+'}' in daughter_node.name:
                    return False
        return True

    def get_head_children(self):
        #returns the head children (children, plural, because of coordination)
        return self.heads

    def get_terminal_heads(self,head_list=[],normalize_terminals=False,head_type='syntactic',returnSynDeps=True):
        if not returnSynDeps:
            return []
        #returns the lexical heads for any constituent
        if self.terminal == True and self.name != 'λ' and self.name != 'μ' and self.name != 'ζ':
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
                head_list.append(self.antecedent)
            return head_list
        else:
            head_list = []
            if self.chain_pointer != None:
                NODE = self.chain_pointer
            else:
                NODE = self
            if head_type == 'syntactic':
                #in order to get PTB deps with sem heads and syn dependents and vice versa,
                #I adapted this function to accommodate this.. the same thing is done for Xbar trees
                #but for these the get_semantic_terminal_heads() function (in gen_derived_tree) is used.. this is a bit messy
                #and at some point I need to eliminate the Node class from one of these programs
                #but they are not identical at present so I'm leaving it for the time being.
                HEADS = NODE.heads
            elif head_type == 'semantic':
                HEADS = NODE.sem_heads
            for head in HEADS:
                new_head_list=head.get_terminal_heads(head_list, normalize_terminals, head_type,returnSynDeps=returnSynDeps)
                head_list+=new_head_list
            return head_list

    def get_semantic_terminal_heads(self, head_list=[], normalize_terminals=False,returnSemDeps=True):
        if not returnSemDeps:
            return []
        #returns the semantic lexical heads for any constituent (works for xbar trees only)
        #currently not used as has a copy in gen_derived_tree.py
        if self.terminal == True and self.name != 'λ' and self.name != 'μ' and self.name != 'ζ':
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
                head_list.append(self.antecedent)
            return head_list
        else:
            head_list = []
            if self.chain_pointer != None:
                NODE = self.chain_pointer
            else:
                NODE = self
            for head_child in NODE.sem_heads:
                new_head_list=head_child.get_semantic_terminal_heads(head_list, normalize_terminals=normalize_terminals,returnSemDeps=returnSemDeps)
                head_list+=new_head_list
            return head_list

    def generate_bracketing(self, terminal_brackets=True):
        #I needed to insert dummy start and end brackets and also make sure there are spaces between any
        # )( sequences, so I created a wrapper around this core GENERATE_BRACKETING function (see below)
        bracketing = "(" + self.GENERATE_BRACKETING("", terminal_brackets) + ")"
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

    def GENERATE_BRACKETING(self, bracketing="", terminal_brackets=True):
        bracketing+="("+self.name+" "
        for daughter in self.daughters:
            #if daughter is a terminal (word) then we don't generate another
            #open bracket - we check this by testing the type of daughter
            if daughter.terminal==True:
                if terminal_brackets:
                    bracketing+="("+daughter.name+")"
                else:
                    bracketing+=daughter.name
                if self.mother == None:
                    bracketing+=")"
                    return bracketing
                return bracketing
            else:
                bracketing=daughter.GENERATE_BRACKETING(bracketing, terminal_brackets)
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
            reg_ex = re.compile(r'.*')
            self.tree=nltk.Tree.parse(self.bracketing, remove_empty_top_bracketing=True)
        except AttributeError:
            self.tree=nltk.Tree.fromstring(self.bracketing, remove_empty_top_bracketing=True)
        print self.bracketing+"\n"
        print "\nClose the window containing the tree diagram to continue... (and then make sure the cursor is at the bottom of this screen)\n"
        return self.tree.draw()

def time_taken(end_time):
    end_time = int(end_time)
    mins = int(end_time / 60)
    hours = int(mins / 60)
    if hours != 0:
        mins = mins - (hours * 60)
    secs = end_time % 60
    if hours == 1:
        HOUR = "hour"
    else:
        HOUR = "hours"
    if mins == 1:
        MIN = "minute"
    else:
        MIN = "minutes"
    if secs == 1:
        SEC = "second"
    else:
        SEC = "seconds"
    if hours != 0:
        return str(hours)+" "+HOUR+", "+str(mins)+" "+MIN+" and "+str(secs)+" "+SEC+".."
    elif mins != 0:
        return str(mins)+" "+MIN+" and "+str(secs)+" "+SEC+".."
    else:
        if secs == 0:
            return "less than a second.."
        return str(secs)+" "+SEC+".."

def orderSubcatFeaturesInLexicon(seedFolder):
    #use this to ensure all the items in the covertLexicons have their subcat features
    #sorted.. I had to do this as I imported then directly from cky_mg (copied and pasted).
    for f in sorted(os.listdir(seedFolder)):
        if 'Lexicon' in f:
            lexicon = json.load(open(seedFolder+'/'+f))
            for lex_entry in lexicon:
                sortedFeatures = autobankGUI.sortSubCat(lex_entry[1], return_list=True)
                lex_entry[1] = sortedFeatures
            with open(seedFolder+'/'+f, 'w') as lexFile:
                json.dump(lexicon, lexFile)

def gen_ccgbank_files(ccg_auto_folder, ptb_folder):
    print ""
    omitted_hyphenated = 0
    omitted = 0
    total_ccg_trees = 0
    for section_folder in sorted(os.listdir(ccg_auto_folder)):
        if section_folder == ".DS_Store":
            continue
        else:
            print "Processing CCGbank files for wsj section folder "+section_folder+"..."
            os.mkdir("CCGbank/"+section_folder)
            for FILE in sorted(os.listdir(ccg_auto_folder+"/"+section_folder)):
                if FILE != ".DS_Store":
                    ccg_auto_file = open(ccg_auto_folder+"/"+section_folder+"/"+FILE, 'r')
                    lines = []
                    for line in ccg_auto_file:
                        lines.append(line)
                    index = -1
                    id_parse_mappings = {}
                    for line in lines:
                        index += 1
                        if "ID=wsj_" in line:
                            total_ccg_trees += 1
                            ID = line.split(" ")[0][3:]
                            line_num = int(ID.split(".")[1])
                            ptb_file = open(ptb_folder+"/"+section_folder+"/"+FILE.split(".")[0]+".mrg")
                            ptb_index = 0
                            for line in ptb_file:
                                ptb_index += 1
                                if ptb_index == line_num:
                                    if line[-1] == '\n':
                                        line = line[:-1]
                                    ptb_tree = build_tree(line)
                                    ptb_terminals = ptb_tree[2]
                                    set_indices(ptb_terminals)
                                    break
                            if section_folder != ID[4:6]:
                                raise Exception("Oops, section folder is wrong!!")
                            new_line = ""
                            inside_node_name = False
                            char_index = -1
                            for char in lines[index+1]:
                                char_index += 1
                                if char == "<":
                                    inside_node_name = True
                                    if lines[index+1][char_index+1] == 'T':
                                        node_type = 'non_terminal'
                                    elif lines[index+1][char_index+1] == 'L':
                                        node_type = 'leaf'
                                    node_name = ""
                                    for CHAR in lines[index+1][char_index+1:]:
                                        if CHAR == ">":
                                            break
                                        else:
                                            node_name += CHAR
                                    node_name_fields = node_name.split(" ")
                                    new_node_name = node_name_fields[1]
                                    if node_type == 'leaf':
                                        new_node_name+="_"+node_name_fields[2]
                                    new_node_name = re.sub('\(', '{', new_node_name, count=1000)
                                    new_node_name = re.sub('\)', '}', new_node_name, count=1000)
                                    if node_type == 'leaf':
                                        new_node_name += " "+node_name_fields[4]
                                    new_line+=new_node_name
                                elif char == ">":
                                    inside_node_name = False
                                elif not inside_node_name:
                                    if char != '\n':
                                        new_line+= char
                            new_line = new_line.strip()
                            new_line = "("+new_line+")"
                            ccg_tree = build_tree(new_line)[0]
                            #because we added in the Ontonotes hyphenated compound structure and this is not contained in CCGbank,
                            #we will simply graft onto the ccg tree the appropriate node from the ptb tree.. the terminals will have ptb categories
                            #not ccg ones, but this is ok since these are mostly just nouns and adjectives anyway..
                            while len(build_tree(ccg_tree.generate_bracketing(terminal_brackets=False))[2]) != len(ptb_terminals):
                                ccg_tree = build_tree(ccg_tree.generate_bracketing(terminal_brackets=False))
                                ccg_terminals = ccg_tree[2]
                                ccg_tree = ccg_tree[0]
                                t_index = -1
                                for ptb_terminal in ptb_terminals:
                                    t_index += 1
                                    final_graft = False
                                    try:
                                        ccg_terminal = ccg_terminals[t_index]
                                    except IndexError:
                                        final_graft = True
                                        ccg_terminal = None
                                    if ccg_terminal == None or ccg_terminal.name.lower() != ptb_terminal.name.lower():
                                        hyphen = False
                                        if ptb_terminal.name == 'num' and "-" not in ccg_terminal.name and "/" not in ccg_terminal.name:
                                            continue
                                        elif ptb_terminal.name == 'num' and "-" in ccg_terminal.name:
                                            try:
                                                if ptb_terminals[t_index+1].name != "-":
                                                    continue
                                            except IndexError:
                                                hyphen = True
                                        elif ptb_terminal.name == 'num' and "/" in ccg_terminal.name:
                                            try:
                                                if ptb_terminals[t_index+1].name != "/":
                                                    continue
                                            except IndexError:
                                                hyphen = True
                                        if not hyphen and (final_graft or (("-" not in ccg_terminal.name and "/" not in ccg_terminal.name) or ccg_terminal.name in ['-LCB-', '-RCB-'])):
                                            if t_index != 0:
                                                ccg_terminals[t_index-1].mother.mother.add_daughter(copy.deepcopy(ptb_terminals[t_index].mother), daughter_index=-1)
                                                ccg_terminals[t_index-1].mother.mother.daughters[-1].name = ccg_terminals[t_index-1].mother.mother.daughters[-1].name+"_"+ccg_terminals[t_index-1].mother.mother.daughters[-1].name
                                            else:
                                                ccg_terminals[0].mother.mother.add_daughter(copy.deepcopy(ptb_terminals[t_index].mother), daughter_index=0)
                                                ccg_terminals[0].mother.mother.daughters[0].name+"_"+ccg_terminals[0].mother.mother.daughters[0].name
                                            break
                                        else:
                                            #num words = the words + hyhens
                                            if ccg_terminal.name != 'late-summer\\/early-FALL' and ccg_terminal.name != 'Republican-governor\\/Democratic-legislature':
                                                if "-" in ccg_terminal.name:
                                                    num_words_covered = (len(ccg_terminal.name.split("-"))*2)-1
                                                elif "/" in ccg_terminal.name:
                                                    num_words_covered = (len(ccg_terminal.name.split("/"))*2)-1
                                            else:
                                                num_words_covered = 3
                                            span_to_cover = [t_index, t_index+num_words_covered]
                                            lowest_common_dominator_found = False
                                            current_node = ptb_terminal.mother
                                            while current_node.indices == []:
                                                current_node = current_node.mother
                                            while span_to_cover != [current_node.indices[0], current_node.indices[-1]+1]:
                                                current_node = current_node.mother
                                            node_to_graft = copy.deepcopy(current_node)
                                            node_to_graft.name = "GRAFT"
                                            ccg_terminal.add_daughter(node_to_graft)
                                            ccg_terminal.terminal = False
                                            ccg_tree = build_tree(ccg_tree.generate_bracketing(terminal_brackets=False))
                                            ccg_terminals = ccg_tree[2]
                                            ccg_tree = ccg_tree[0]
                                            for terminal in ccg_terminals:
                                                if "_" not in terminal.mother.name:
                                                    terminal.mother.name = terminal.mother.name.split("-")[0]+"_"+terminal.mother.name.split("-")[0]
                                                    #we are simply doubling my tag as we did not have access to Julia's PTB tag
                                                    #as Julia was not using trees with the hyphenated structure.. we therefore need to shave off an person/number/gender info
                                                    #as even the backoff tagger won't work properly otherwise (unless we have an example of this exact tag in the set of hyphenated seeds which in some cases
                                                    #we won't)
                                                    parts = terminal.mother.name.split("_")
                                                    if 'PRP$' in parts[1]:
                                                        parts[1] = parts[1][:4]
                                                    elif 'PRP' in parts[1]:
                                                        parts[1] = parts[1][:3]
                                                    elif 'NNP' in parts[1]:
                                                        parts[1] = parts[1][:3]
                                                    elif len(parts[1]) > 1 and 'NN' == parts[1][:2]:
                                                        parts[1] = parts[1][:2]
                                                    terminal.mother.name = "_".join(parts)
                                            break
                            ccg_tree = build_tree(ccg_tree.generate_bracketing(terminal_brackets=False))
                            ccg_terminals = [terminal for terminal in ccg_tree[2]]
                            sentence = [terminal.name for terminal in ccg_tree[2]]
                            if len(sentence) == len(ptb_terminals):
                                terminal_index = -1
                                for word in ccg_terminals:
                                    tags_to_add = []
                                    terminal_index += 1
                                    #as well as Julia's modified PTB category, we also include my modified PTB category as this may have gender/person/number info
                                    word.mother.name = word.mother.name+"_"+ptb_terminals[terminal_index].mother.truncated_name
                                    word.mother.truncated_name = word.mother.name
                                    #for nouns, adverbs and prepositions/complementizers only, we will also add on any adverbial tags from the Penn set (DIR, LOC, MAN, TMP)
                                    current_node = ptb_terminals[terminal_index].mother
                                    if current_node.name.split("-")[0] in ['IN', 'TO'] or 'NN' in current_node.name.split("-")[0] or 'RB' in current_node.name.split("-")[0]:
                                        if not ('-TMP' in current_node.name or '-MNR' in current_node.name or '-LOC' in current_node.name or '-DIR' in current_node.name or '-DIS' in current_node.name):
                                            while current_node.mother != None and current_node in current_node.mother.heads:
                                                if '-TMP' in current_node.name or '-MNR' in current_node.name or '-LOC' in current_node.name or '-DIR' in current_node.name or '-DIS' in current_node.name:
                                                    break
                                                else:
                                                    current_node = current_node.mother
                                        #we'll use elif just so we don't end up with too much data sparsity.. usually only one tag on each non-terminal anyway, but not always, e.g. nombank adds in semantic roles for nouns too
                                        if '-TMP' in current_node.name:
                                            tags_to_add.append('TMP')
                                        elif '-LOC' in current_node.name:
                                            tags_to_add.append('LOC')
                                        elif '-DIR' in current_node.name:
                                            tags_to_add.append('DIR')
                                        elif '-MNR' in current_node.name:
                                            tags_to_add.append('MNR')
                                        elif '-DIS' in current_node.name:
                                            tags_to_add.append('DIS')
                                    terminal_mother_name_split = ptb_terminals[terminal_index].mother.name.split("-")
                                    if 'CC' in terminal_mother_name_split and 'MARK' not in terminal_mother_name_split and 'conj' in word.mother.name:
                                        CC_index = ptb_terminals[terminal_index].mother.mother.daughters.index(ptb_terminals[terminal_index].mother)
                                        CC_ind = -1
                                        LR = "L"
                                        left_coords = []
                                        right_coords = []
                                        for daughter in ptb_terminals[terminal_index].mother.mother.daughters:
                                            CC_ind += 1
                                            daughter_name_split = daughter.name.split("-")
                                            if CC_ind == CC_index:
                                                LR = "R"
                                            elif 'COORD' in daughter_name_split:
                                                if 'NX' == daughter_name_split[0]:
                                                    new_ptb_tag = daughter.heads[0].name.split("-")[0]
                                                    daughter_name_split[0] = new_ptb_tag
                                                    if len(daughter_name_split[0]) > 1:
                                                        tag = daughter_name_split[0][:2]
                                                    else:
                                                        tag = daughter_name_split[0][0]
                                                    if tag not in tags_to_add:
                                                        if LR == "L":
                                                            if tag not in left_coords:
                                                                left_coords.append(tag)
                                                        elif LR == "R":
                                                            if tag not in right_coords:
                                                                right_coords.append(tag)
                                                elif len(daughter.daughters) == 1 and daughter.daughters[0].terminal:
                                                    if len(daughter_name_split[0]) > 1:
                                                        tag = daughter_name_split[0][:2]
                                                    else:
                                                        tag = daughter_name_split[0][0]
                                                    if tag == 'PR':
                                                        tag = 'NP'
                                                    if tag not in tags_to_add:
                                                        if LR == "L":
                                                            if tag not in left_coords:
                                                                left_coords.append(tag)
                                                        elif LR == "R":
                                                            if tag not in right_coords:
                                                                right_coords.append(tag)
                                                else:
                                                    tag = daughter_name_split[0]
                                                    if tag == 'PRP':
                                                        tag = 'NP'
                                                    if LR == "L":
                                                        if tag not in left_coords:
                                                            left_coords.append(tag)
                                                    elif LR == "R":
                                                        if tag not in right_coords:
                                                            right_coords.append(tag)
                                        if len(right_coords+left_coords) > 0:
                                            current_node = ptb_terminals[terminal_index].mother.mother
                                            while 'NX' in current_node.name.split("-")[0]:
                                                current_node = current_node.mother
                                            CC_cat = current_node.name.split("-")[0]
                                            TAG = "CC"+CC_cat
                                            left_coords.reverse()
                                            NP_found = False
                                            for tag in left_coords+right_coords:
                                                if tag == 'NP':
                                                    NP_found = True
                                                    break
                                            if NP_found:
                                                #if one conjunct is an NP, then we change any NN conjuncts to NP also
                                                tag_ind = -1
                                                for tag in left_coords:
                                                    tag_ind += 1
                                                    if tag == 'NN':
                                                        left_coords[tag_ind] = 'NP'
                                                tag_ind = -1
                                                for tag in right_coords:
                                                    tag_ind += 1
                                                    if tag == 'NN':
                                                        right_coords[tag_ind] = 'NP'                                                        
                                            for tag in left_coords:
                                                TAG = "{"+TAG+"\\"+tag+"}"
                                            for tag in right_coords:
                                                TAG = "{"+TAG+"/"+tag+"}"
                                            word_parts = word.mother.name.split("_")
                                            word.mother.name = TAG[1:-1]+"_CC"+CC_cat+"_CC"+CC_cat
                                    elif len(ptb_terminals[terminal_index].mother.name) > 1:
                                        if ptb_terminals[terminal_index].mother.name[:2] == 'VB':
                                            current_node = ptb_terminals[terminal_index]
                                            ARG0_tag = 'ARG0{'+current_node.name+'<'+str(terminal_index)+','+str(terminal_index+1)+'>}'
                                            ARG0_tag_found = find_tag(ptb_tree[0], ARG0_tag)
                                            if ARG0_tag_found:
                                                tags_to_add.append('ARG0')
                                            ARG1_tag = 'ARG1{'+current_node.name+'<'+str(terminal_index)+','+str(terminal_index+1)+'>}'
                                            ARG1_tag_found = find_tag(ptb_tree[0], ARG1_tag)
                                            if ARG1_tag_found == 'S':
                                                tags_to_add.append('ARGS')
                                            elif ARG1_tag_found:
                                                tags_to_add.append('ARG1')
                                    for tag in tags_to_add:
                                        word.mother.name = word.mother.name+"_"+tag
                                        word.mother.truncated_name = word.mother.name
                            new_line = ccg_tree[0].generate_bracketing(terminal_brackets=False)
                            word_index = -1
                            indices_to_change_to_num = []
                            for word in sentence:
                                word_index += 1
                                try:
                                    float(word)
                                    indices_to_change_to_num.append(word_index)
                                except ValueError:
                                    x=0
                            for INDEX in indices_to_change_to_num:
                                sentence[INDEX] = 'num'
                            id_parse_mappings[ID] = new_line
                with open("CCGbank/"+section_folder+"/"+FILE.split(".")[0]+".ccg", 'w') as ccgbank_file:
                    json.dump(id_parse_mappings, ccgbank_file)
                ccg_auto_file.close()
    print "\nSuccessfully built CCGbank files."
    print "Total number of CCGbank trees: "+str(total_ccg_trees)

def find_tag(node, tag):
    if tag in node.name:
        if node.name[0] == 'S':
            return 'S'
        else:
            return True
    else:
        for daughter in node.daughters:
            tag_found = find_tag(daughter, tag)
            if tag_found:
                return tag_found
    return False

def gen_MGbank(ptb_folder, ccg_auto_folder, MGbankName, AUTO_SECTION_FOLDERS, overwrite_auto, START_LINE, START_FILE, END_FILE, MIN_LENGTH, MAX_LENGTH, CCG_BEAM, CCG_BEAM_FLOOR, TIMEOUT, PARSER_SETTING, STOP_AFTER, MAX_MG_CATS_PER_WORD, USEALLNULL, USEAUTOS, SUPERTAGGINGSTRATEGY, TRAIN_TAGGER, SUPER_RARE_CUTOFF, SUPER_TAG_DICT_MIN, SUPER_CATEGORY_CUTOFF, SUPER_FORWARD_BEAM_RATIO, SUPER_BEAM_RATIO, SUPER_BEAM_WIDTH, EXTRACT_DEP_MAPPINGS, SKIPREL, SKIPPRO, CONSTRAINMOVEWITHPTB, MAXMOVEDIST, CONSTRAINCONSTWITHPTBCCG, ALLOWBESTTAG, ALLOWMOREGOALS):
    global start_file
    global end_file
    global min_length
    global max_length
    global start_line
    global auto_section_folders
    global seeds
    global ccg_beam
    global ccg_beam_floor
    global parser_setting
    global time_out
    global stop_after
    global max_mg_cats_per_word
    global useAllNull
    global skipRel
    global skipPro
    global useAutos
    global allowbesttag
    global train_tagger
    global extract_dep_mappings
    global previous_bracketing0
    global previous_bracketing1
    global supertaggingStrategy
    global super_rare_cutoff
    global super_tag_dict_min
    global super_category_cutoff
    global super_forward_beam_ratio
    global super_beam_width
    global super_beam_ratio
    global ptb_word_token_count
    global constrainMoveWithPTB
    global constrainConstWithPTBCCG
    global maxMoveDist
    global allowMoreGoals
    allowMoreGoals = ALLOWMOREGOALS
    maxMoveDist = MAXMOVEDIST
    constrainMoveWithPTB = CONSTRAINMOVEWITHPTB
    constrainConstWithPTBCCG = CONSTRAINCONSTWITHPTBCCG
    useAutos = USEAUTOS
    allowbesttag = ALLOWBESTTAG
    train_tagger = TRAIN_TAGGER
    extract_dep_mappings = EXTRACT_DEP_MAPPINGS
    useAllNull = USEALLNULL
    skipRel = SKIPREL
    skipPro = SKIPPRO
    start_file = START_FILE
    end_file = END_FILE
    min_length = MIN_LENGTH
    max_length = MAX_LENGTH
    start_line = START_LINE
    auto_section_folders = AUTO_SECTION_FOLDERS
    ccg_beam = CCG_BEAM
    ccg_beam_floor = CCG_BEAM_FLOOR
    time_out = TIMEOUT
    parser_setting = PARSER_SETTING
    stop_after = STOP_AFTER
    max_mg_cats_per_word = MAX_MG_CATS_PER_WORD
    supertaggingStrategy = SUPERTAGGINGSTRATEGY
    super_rare_cutoff = SUPER_RARE_CUTOFF
    super_tag_dict_min = SUPER_TAG_DICT_MIN
    super_category_cutoff = SUPER_CATEGORY_CUTOFF
    super_forward_beam_ratio = SUPER_FORWARD_BEAM_RATIO
    super_beam_width = SUPER_BEAM_WIDTH
    super_beam_ratio = SUPER_BEAM_RATIO
    #the top level function of the MGbank generator
    seed_folder = ptb_folder+"_"+MGbankName+"Seed"
    auto_folder = ptb_folder+"_"+MGbankName+"Auto"
    ccg_folder = "CCGbank"
    if seed_folder not in os.listdir(os.getcwd()):
        os.mkdir(seed_folder)
    if auto_folder not in os.listdir(os.getcwd()):
        os.mkdir(auto_folder)
    loadData(ptb_folder, seed_folder, auto_folder)
    file_specified = False
    PTBbracketings = {}
    CCGbracketings = {}
    if ccg_folder not in os.listdir(os.getcwd()):
        os.mkdir(ccg_folder)
        gen_ccgbank_files(ccg_auto_folder, ptb_folder)
    if ptb_folder+"_strings" not in os.listdir(os.getcwd()):
        os.mkdir(os.getcwd()+'/'+ptb_folder+"_strings")
    for section_folder in sorted(os.listdir(ptb_folder)):
        if section_folder != ".DS_Store" and section_folder not in os.listdir(ptb_folder+"_strings"):
            os.mkdir(ptb_folder+"_strings/"+section_folder)
    ptb_strings_folder = ptb_folder+"_strings"
    ptb_word_token_count = 0
    for section_folder in sorted(os.listdir(ptb_folder)):
        if section_folder == '.DS_Store':
            continue
        print "Loading source trees from section folder: "+section_folder
        if section_folder not in os.listdir(seed_folder):
            os.mkdir(seed_folder+"/"+section_folder)
        for ptb_file in sorted(os.listdir(ptb_folder+"/"+section_folder)):
            if ptb_file == '.DS_Store':
                continue
            parses=open(ptb_folder+"/"+section_folder+"/"+ptb_file, "r")
            ccgParses = json.load(open("CCGbank/"+"/"+section_folder+"/"+ptb_file[:-4]+".ccg"))
            if ptb_file+"_strings" not in os.listdir(ptb_strings_folder+"/"+section_folder):
                string_file = open(ptb_strings_folder+"/"+section_folder+"/"+ptb_file+"_strings", 'w')
                for line in parses:
                    terminals = build_tree(line)[2]
                    sen = [t.name for t in terminals]
                    sentence = " ".join(sen)
                    string_file.write(sentence+'\n')
                    ptb_word_token_count += len(sen)
                string_file.close()
            else:
                string_file = open(ptb_strings_folder+"/"+section_folder+"/"+ptb_file+"_strings")
                for line in string_file:
                    ptb_word_token_count += len(line.split(" "))
            parses=open(ptb_folder+"/"+section_folder+"/"+ptb_file, "r")
            string_file = open(ptb_strings_folder+"/"+section_folder+'/'+ptb_file+"_strings")
            sentences = []
            for line in string_file:
                if line != '':
                    sentences.append(line)
            index = -1
            for line in parses:
                index += 1
                line = line.strip()
                try:
                    sentLength = len(sentences[index].split(" "))
                except IndexError:
                    string_file.close()
                    #if the program was interrupted while creating a string file we have to recreate the file
                    string_file = open(ptb_strings_folder+"/"+section_folder+"/"+ptb_file+"_strings", 'w')
                    PARSES=open(ptb_folder+"/"+section_folder+"/"+ptb_file, "r")
                    for line in PARSES:
                        terminals = build_tree(line)[2]
                        string_file.write(" ".join([t.name for t in terminals])+'\n')
                    string_file.close()
                    string_file = open(ptb_strings_folder+"/"+section_folder+'/'+ptb_file+"_strings")
                    sentences = []
                    for line in string_file:
                        if line != '':
                            sentences.append(line)
                    sentLength = len(sentences[index].split(" "))
                if sentLength in PTBbracketings:
                    PTBbracketings[sentLength].append((section_folder, ptb_file, index))
                else:
                    PTBbracketings[sentLength] = [(section_folder, ptb_file, index)]
                ccgIndex = ptb_file[:-4]+"."+str(index+1)
                if ccgIndex in ccgParses:
                    CCGbracketings[ccgIndex] = ccgParses[ccgIndex]
    print ""
    PTBbracketings['all'] = []
    stringLengths = ['all']
    maxStrLen = 0
    for sentLength in PTBbracketings:
        if sentLength != 'all':
            if sentLength not in stringLengths:
                if sentLength > maxStrLen:
                    maxStrLen = sentLength
                stringLengths.append(sentLength)
            PTBbracketings['all'] += PTBbracketings[sentLength]
    ptbBracketings = {}
    for entry in PTBbracketings:
        if entry != 'all':
            for ENTRY in PTBbracketings[entry]:
                if ENTRY[1] in ptbBracketings:
                    ptbBracketings[ENTRY[1]].append(ENTRY+(entry,))
                else:
                    ptbBracketings[ENTRY[1]] = [ENTRY+(entry,)]
    i = 9
    seedSentLen = None
    while True:
        i+=1
        match = False
        if i in stringLengths:
            (index, match) = search(0, PTBbracketings, CCGbracketings, 'forward', '', i, ptb_folder, seed_folder=seed_folder)
            if match == True:
                seedSentLen = i
                break
        if i > maxStrLen:
            break
    if seedSentLen == None:
        i = 10
        while True:
            i-=1
            match = False
            if i in stringLengths:
                seedSentLen = i
                (index, match) = search(0, PTBbracketings, CCGbracketings, 'forward', '', i, ptb_folder, seed_folder=seed_folder)
            if match == True:
                seedSentLen = i
                break
            if i == 0:
                break
    if seedSentLen == None:
        print "No parses detected in that folder!!!!!!"
        return
    if len(PTBbracketings['all']) == 0:
        #you need to handle properly the case where there are no bracketings of this length..
        print "no PTB parses detected in that folder!!!"
        return
    if 'parserSettings' in os.listdir(seed_folder) and 'timeAndDate' in os.listdir(auto_folder):
        parserSettings = json.load(open(seed_folder+"/"+"parserSettings"))
        lastAutoTimeAndDate = json.load(open(auto_folder+"/"+"timeAndDate"))
        if overwrite_auto == None and ('last_auto_date' not in parserSettings or (parserSettings['last_auto_date'] != lastAutoTimeAndDate['date'] or parserSettings['last_auto_time'] != lastAutoTimeAndDate['time'])):
            buildAutoMappings(auto_folder, seed_folder)
    nextPrev = 'next'
    #searchItem holds the reg exp or string that the user wishes to search for..
    #initially it is set to empty which means there is no search and all strings of the
    #given length are accessible..
    searchItem = ''
    match = True
    MATCH = True
    current_ptb_tree = get_current_ptb_tree(PTBbracketings, seedSentLen, index, ptb_folder)
    counts = {}
    print "Gathering corpus counts..."
    for entry in PTBbracketings:
        INDEX = 0
        (INDEX, match, count) = search(INDEX, PTBbracketings, CCGbracketings, 'forward', searchItem, entry, ptb_folder, searchAll=True, seed_folder=seed_folder)
        counts[entry] = count
    match = False
    for c in counts:
        if counts[c] > 0:
            match = True
            break
    #so we don't have to research for the empty string counts each time, we will take
    #a deepcopy here..
    emptyStringCounts = copy.deepcopy(counts)
    totalSourceTrees = len(PTBbracketings['all'])
    print "\nCorpus counts collected successfully."
    displayIndex = 1
    sameSearch = False
    newSentMode = False
    task = 'annotate'
    supertagger = None
    test_words = None
    untokenizedTestSentence = None
    while True:
        seeds = None
        previous_bracketing0 = None
        previous_bracketing1 = None
        (QUIT, start_auto_gen, overwriteAuto, nextPrev, searchItem, seedSentLen, match, displayIndex, sameSearch, goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI_MATCH, newSentMode, test_words, untokenizedTestSentence, lengthCountsOfRemovedTrees, task, supertagger) = gen_seed_MGbank(current_ptb_tree, ptb_folder, seed_folder, auto_folder, nextPrev, searchItem, match, seedSentLen, stringLengths, counts, emptyStringCounts, displayIndex, totalSourceTrees, newSentMode, test_words, untokenizedTestSentence, MGbankName, overwrite_auto, task, supertagger)
        if match != 'False' and GUI_MATCH != False and not (newSentMode and match == 'checked'):
            match = True
            MATCH = True
        else:
            MATCH = False
        if QUIT and not start_auto_gen:
            return
        elif QUIT and start_auto_gen:
            auto_gen_MGbank(ptb_folder, seed_folder, overwriteAuto)
            return
        elif nextPrev == 'test':
            #the user is just annotating/parsing a test sentence.. we therefore don't change any indices
            newSentMode = True
        elif nextPrev == 'search':
            displayIndex = 1
            if sameSearch == False:
                print ""
                if searchItem != "":
                    print "Searching all corpus files for: "+searchItem+"..."
                else:
                    print "Recalculating corpus counts..."
                counts={}
            index = 0
            if sameSearch == False:
                #if the user has entered a search term then we search both the string and the bracketing
                #using what they entered as a reg exp
                for entry in PTBbracketings:
                    INDEX = 0
                    (INDEX, match, count) = search(INDEX, PTBbracketings, CCGbracketings, 'forward', searchItem, entry, ptb_folder, searchAll=True, seed_folder=seed_folder)
                    counts[entry] = count
            (index, match) = search(index, PTBbracketings, CCGbracketings, 'forward', searchItem, seedSentLen, ptb_folder, seed_folder=seed_folder)
            print "Corpus counts collected successfully."
            if MATCH == False:
                match = 'False'
        elif nextPrev == 'ptb_search':
            (displayIndex, index) = do_ptb_search(fileManagerFolder, ptbLineNum, seedSentLen, PTBbracketings, seed_folder)
            counts = copy.deepcopy(emptyStringCounts)
        elif nextPrev == 'removed':
            #a seed tree was removed
            displayIndex = 1
            #we have to add a count to the PTB counts as one has been put back into our unannotated corpus,
            #unless the tree was removed from the autoset, in which case we don't need to do this..
            if lenRemovedTree != 'removedFromAutos' and lenRemovedTree != 'NewParseRemoved':
                #we need to reset the search to null because we don't know if the tree we just
                #added back in matched the search or not.. not worth the extra effort to determine this.
                emptyStringCounts[lenRemovedTree] += 1
                emptyStringCounts['all'] += 1
            index = 0
            counts = copy.deepcopy(emptyStringCounts)
            searchItem = ''
            (index, match) = search(index, PTBbracketings, CCGbracketings, 'forward', searchItem, seedSentLen, ptb_folder, seed_folder=seed_folder)
        elif nextPrev == 'removedSeveral':
            #the cases where a category was deleted, and as a result, one or more parses were removed from the seeds
            displayIndex = 1
            summation = 0
            for entry in lengthCountsOfRemovedTrees:
                summation += lengthCountsOfRemovedTrees[entry]
                emptyStringCounts[entry] += lengthCountsOfRemovedTrees[entry]
            emptyStringCounts['all'] += summation
            index = 0
            counts = copy.deepcopy(emptyStringCounts)
            searchItem = ''
            (index, match) = search(index, PTBbracketings, CCGbracketings, 'forward', searchItem, seedSentLen, ptb_folder, seed_folder=seed_folder)
        elif nextPrev == 'moved':
            #an auto tree was moved into the seed set
            displayIndex = 1
            #we have to subtract a count from the PTB counts as one has been put into the annotated seed corpus.
            emptyStringCounts[lenMovedTree] -= 1
            emptyStringCounts['all'] -= 1
            counts = copy.deepcopy(emptyStringCounts)
            searchItem=''
            index = 0
            (index, match) = search(index, PTBbracketings, CCGbracketings, 'forward', searchItem, seedSentLen, ptb_folder, seed_folder=seed_folder)
        elif nextPrev == 'next':
            oldIndex = index
            if index != len(PTBbracketings[seedSentLen])-1:
                index += 1
                (index, match) = search(index, PTBbracketings, CCGbracketings, 'forward', searchItem, seedSentLen, ptb_folder, seed_folder=seed_folder)
                if MATCH == False:
                    match = 'False'
                if match == False:
                    #we need to differentiate between the case where the system has found no matches at all
                    #and that where it finds no more when back and forward are pressed..
                    match = None
                #if our search turned up no new matches, we want to stay on the current tree..
                if match == True:
                    displayIndex += 1
            else:
                if MATCH == False:
                    match = 'False'
                else:
                    match = None
            if match == None:
                index = oldIndex
        elif nextPrev == 'previous':
            oldIndex = index
            if index != 0:
                index -= 1
            (index, match) = search(index, PTBbracketings, CCGbracketings, 'backward', searchItem, seedSentLen, ptb_folder, seed_folder=seed_folder)
            if MATCH == False:
                match = 'False'
            if match == False:
                #we need to differentiate between the case where the system has found no matches at all
                #and that where it finds no more when back and forward are pressed..
                match = None
            if match == True and index != 0:
                displayIndex -= 1
            if match == None:
                index = oldIndex
        elif nextPrev == 'same':
            #don't update index in this case
            nextPrev = 'next'
        elif nextPrev == 'goto':
            nextPrev = 'next'
            index=0
            (index, displayIndex) = search(index, PTBbracketings, CCGbracketings, 'forward', searchItem, seedSentLen, ptb_folder, searchAll=True, seed_folder=seed_folder, goto=goto)
        elif nextPrev == 'stop':
            nextPrev = 'next'
            if index == len(PTBbracketings[seedSentLen])-1:
                index = 0
                displayIndex = 1
            #we keep the search item variable the same so that the GUI will receive it,
            #but we allow the system here to just search for '' items so it doesn't keep getting sent back
            #here..
            (index, match) = search(index, PTBbracketings, CCGbracketings, 'forward', '', seedSentLen, ptb_folder, seed_folder=seed_folder)
            match = 'False'
        current_ptb_tree = get_current_ptb_tree(PTBbracketings, seedSentLen, index, ptb_folder)
        if nextPrev == 'search' or nextPrev == 'ptb_search':
            nextPrev = 'next'

def get_current_ptb_tree(PTBbracketings, seedSentLen, index, ptb_folder):
    current_ptb_tree = PTBbracketings[seedSentLen][index]
    bracketing = getPTBbracketing(current_ptb_tree, ptb_folder)
    current_ptb_tree+=(bracketing,)
    return current_ptb_tree

def getCCGbracketing(current_ptb_tree, CCGbracketings):
    index = current_ptb_tree[2]
    ccgIndex = current_ptb_tree[1][:-4]+"."+str(index+1)
    if ccgIndex in CCGbracketings:
        CCGbracketing = CCGbracketings[ccgIndex]
    else:
        CCGbracketing = ""
    return CCGbracketing

def getPTBbracketing(current_ptb_tree, ptb_folder):
    index = current_ptb_tree[2]
    f = open(ptb_folder+"/"+current_ptb_tree[0]+"/"+current_ptb_tree[1])
    i = -1
    for line in f:
        i += 1
        if i == index:
            if line[-1] == '\n':
                line = line[:-1]
            return line

def do_ptb_search(fileManagerFolder, ptbLineNum, seedSentLen, PTBbracketings, seed_folder):
    displayIndex = 0
    index = -1
    for entry in PTBbracketings[seedSentLen]:
        index+=1
        bracketing = get_current_ptb_tree(PTBbracketings, seedSentLen, index, ptb_folder)
        if bracketing[0] in os.listdir(seed_folder) and bracketing[1] in os.listdir(seed_folder+"/"+bracketing[0]):
            seeds = json.load(open(seed_folder+"/"+bracketing[0]+"/"+bracketing[1]))
        else:
            seeds = {}
        if str(bracketing[2]) not in seeds:
            displayIndex += 1
        if ptb_folder+"/"+bracketing[0]+"/"+bracketing[1] == fileManagerFolder and bracketing[2] == ptbLineNum:
            return (displayIndex, index)
    
def search(index, PTBbracketings, CCGbracketings, direction, searchItem, seedSentLen, ptb_folder, searchAll=False, seed_folder=None, goto=False):
    global seeds
    global previous_bracketing0
    global previous_bracketing1
    try:
        regexp = re.compile(searchItem)
    except Exception:
        regexp = False
    oldIndex = index
    if searchAll == True:
        count = 0
    #displayIndex is only used in here when the user is jumping to a particular tree
    #but to avoid lots of if statements we will just update it anyway all the time.
    displayIndex = 0
    bracketing = get_current_ptb_tree(PTBbracketings, seedSentLen, index, ptb_folder)
    CCGbracketing = getCCGbracketing(bracketing, CCGbracketings)
    strippedBracketing = re.sub('{.*?}', '', bracketing[3], count=10000)
    bracketing = (bracketing[0], bracketing[1], bracketing[2], strippedBracketing)
    sentence_file = open(ptb_folder+"_strings/"+bracketing[0]+"/"+bracketing[1]+"_strings")
    sentences = []
    for sentence in sentence_file:
        sentences.append(sentence)
    STRING = sentences[bracketing[2]]
    if STRING == '':
        regexp = False
    while True:
        if not (bracketing[0] == previous_bracketing0 and bracketing[1] == previous_bracketing1):
            previous_bracketing0 = bracketing[0]
            previous_bracketing1 = bracketing[1]
            if bracketing[0] in os.listdir(seed_folder) and bracketing[1] in os.listdir(seed_folder+"/"+bracketing[0]):
                seeds = json.load(open(seed_folder+"/"+bracketing[0]+"/"+bracketing[1]))
            else:
                seeds = {}
        if str(bracketing[2]) not in seeds and ((searchItem.lower() in STRING.lower() or searchItem in bracketing[3] or searchItem in CCGbracketing) or (regexp and (re.search(searchItem, STRING) or re.search(searchItem, strippedBracketing) or re.search(searchItem, CCGbracketing)))):
            if searchAll == True:
                count+=1
                if goto:
                    displayIndex += 1
                    if displayIndex == goto:
                        return (index, displayIndex)
                if index != len(PTBbracketings[seedSentLen])-1:
                    index += 1
                else:
                    break
            else:
                break
        else:
            if direction == 'forward':
                if index != len(PTBbracketings[seedSentLen])-1:
                    index += 1
                else:
                    if searchAll == True and count > 0:
                        break
                    elif searchAll == True:
                        return (index, True, 0)
                    else:
                        return (oldIndex, False)
            elif direction == 'backward':
                if index != 0:
                    index -= 1
                else:
                    if searchAll == True and count > 0:
                        break
                    elif searchAll == True:
                        return (index, True, 0)
                    else:
                        return (oldIndex, False)
        bracketing = get_current_ptb_tree(PTBbracketings, seedSentLen, index, ptb_folder)
        CCGbracketing = getCCGbracketing(bracketing, CCGbracketings)
        strippedBracketing = re.sub('{.*?}', '', bracketing[3], count=10000)
        sentence_file = open(ptb_folder+"_strings/"+bracketing[0]+"/"+bracketing[1]+"_strings")
        sentences = []
        for sentence in sentence_file:
            sentences.append(sentence)
        STRING = sentences[bracketing[2]]
    if searchAll == True:
        return (index, True, count)
    else:
        return (index, True)

def gen_seed_MGbank(current_ptb_tree, ptb_folder, seed_folder, auto_folder, nextPrev='next', searchItem='', match=True, seedSentLen=None, stringLengths=None, counts=None, emptyStringCounts=None, displayIndex=None, totalSourceTrees=None, newSentMode=None, test_words=None, untokenizedTestSentence=None, MGbankName=None, overwrite_auto=None, task=None, supertagger=None):
    global PosDepsMappings
    global TreeCatMappings
    global nullTreeCatMappings
    global CatTreeMappings
    global nullCatTreeMappings
    goto = False
    lenRemovedTree = None
    lenMovedTree = None
    fileManagerFolder = None
    section_folder = current_ptb_tree[0]
    ptb_file = current_ptb_tree[1]
    index = current_ptb_tree[2]
    ptb_bracketing = current_ptb_tree[3]
    dirs = sorted(os.listdir(os.getcwd()))
    ptbLineNum = None
    if ptb_file in os.listdir(seed_folder+"/"+section_folder):
        seeds = json.load(open(seed_folder+"/"+section_folder+"/"+ptb_file))
    else:
        seeds = {}
    #if str(index) in seeds:
        #print "does this ever get executed?  If not delete!!!!!"
        #we don't want to ask the user to annotate a sentence if it's already in the
        #seed set..
        #if match == False or match == None:
            #return (False, False, False, 'stop', searchItem, seedSentLen, match)
        #else:
            #return (False, False, False, nextPrev, searchItem, seedSentLen, match)
    (PTB_tree, sentence, terminals) = build_tree(ptb_bracketing)
    #we need an nltk Tree object for the GUI
    try:
        PTB_nltk_tree=nltk.Tree.parse(ptb_bracketing[1:-1], remove_empty_top_bracketing=True)
    except AttributeError:
        PTB_nltk_tree = Tree.fromstring(ptb_bracketing[1:-1], remove_empty_top_bracketing=True)
    GUI = autobankGUI.autobankGUI(PTB_nltk_tree, terminals, PosMappings, seeds, ptb_folder, seed_folder, auto_folder, section_folder, index, ptb_file, CovertLexicon, ExtraposerLexicon, TypeRaiserLexicon, ToughOperatorLexicon, NullExcorporatorLexicon, DepMappings, RevDepMappings, PosDepsMappings, covertCatComments, overtCatComments, searchItem, match, seedSentLen, stringLengths, counts, displayIndex, ptb_bracketing, totalSourceTrees, newSentMode, test_words, untokenizedTestSentence, MGbankName, overwrite_auto, useAutos, supertaggingStrategy, train_tagger, super_rare_cutoff, super_tag_dict_min, super_category_cutoff, super_forward_beam_ratio, super_beam_ratio, super_beam_width, extract_dep_mappings, ptb_word_token_count, task, supertagger)
    match = GUI.match
    ptbLineNum = GUI.ptbLineNum
    fileManagerFolder = GUI.fileManagerFolder
    searchItem = GUI.searchItem
    seedSentLen = GUI.seedSentLen
    lenRemovedTree = GUI.lenRemovedTree
    lenMovedTree = GUI.lenMovedTree
    #but we only save the trees and depedencies if the user clicked 'add parse' rather than 'quit'.
    if GUI.quit:
        return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, None, searchItem, seedSentLen, match, displayIndex, False, goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
    elif (not GUI.newSentMode or GUI.addedNewSentenceToSeeds) and not GUI.ptb_search and not GUI.search and not GUI.goto and not GUI.nextPTBtree and not GUI.previousPTBtree and not GUI.currentPTBtree:
        #The case where the user annotated the PTB tree and added it to seeds
        #save the updated seed set for this file and also save the PosMappings
        loadData(ptb_folder, seed_folder, auto_folder)
        realIndex = index
        if not GUI.addedNewSentenceToSeeds:
            counts[seedSentLen] -= 1
            counts['all'] -= 1
            emptyStringCounts[seedSentLen] -= 1
            emptyStringCounts['all'] -= 1
        else:
            section_folder = "new_parses"
            ptb_file = GUI.newSentFileName
            GUI.ptb_file_line_number = "0"
            seeds = GUI.seeds
            index = 0
        with open(seed_folder+"/"+section_folder+"/"+ptb_file, 'w') as seed_file:
            json.dump(seeds, seed_file)
        TreeCatMappings[str([seed_folder+"/"+section_folder+"/"+ptb_file, index])] = []
        nullTreeCatMappings[str([seed_folder+"/"+section_folder+"/"+ptb_file, index])] = []
        #we need to retrieve all the null categories that have been used in the tree so we can set up
        #the nullCatTreeMappings and nullTreeCatMappings
        subcat_derivation_bracketing = GUI.seeds[GUI.ptb_file_line_number][0]
        subcat_derivation_tree = gen_derived_tree.gen_derivation_tree(subcat_derivation_bracketing)
        MGcats = GUI.MGcatsCopy
        nullMGcats = get_null_MGcats(subcat_derivation_tree, [])
        (CatTreeMappings, TreeCatMappings) = updateMappings(MGcats, CatTreeMappings, TreeCatMappings, seed_folder, section_folder, ptb_file, index)
        nullMGcats = json.dumps(nullMGcats)
        nullMGcats = json.loads(nullMGcats)
        (nullCatTreeMappings, nullTreeCatMappings) = updateMappings(nullMGcats, nullCatTreeMappings, nullTreeCatMappings, seed_folder, section_folder, ptb_file, index)
        index = realIndex
        with open(seed_folder+"/"+'CatTreeMappings', 'w') as CatTreeMappingsFile:
            json.dump(CatTreeMappings, CatTreeMappingsFile)
        with open(seed_folder+"/"+'TreeCatMappings', 'w') as TreeCatMappingsFile:
            json.dump(TreeCatMappings, TreeCatMappingsFile)
        with open(seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
            json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
        with open(seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
            json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
        if not GUI.addedNewSentenceToSeeds:
            displayIndex -= 1
        if displayIndex != counts[seedSentLen]:
            return (False, False, False, 'next', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        else:
            if not GUI.addedNewSentenceToSeeds:
                displayIndex += 1
            return (False, False, False, 'previous', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        
    elif GUI.newSentMode == True:
        try:
            return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'test', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        except UnboundLocalError:
            return (False, False, False, 'test', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
    elif GUI.search == True:
        #GUI.search is only true immediately after the user presses search, not when they
        #are using the forward and back buttons to navigate through the search results..
        try:
            return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'search', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        except UnboundLocalError:
            return (False, False, False, 'search', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
    elif GUI.ptb_search == True:
        #This is where the user has selected a ptb tree to view using the fileManagerSystem
        try:
            return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'ptb_search', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        except UnboundLocalError:
            return (False, False, False, 'search', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
    elif GUI.previousPTBtree == True:
        try:
            return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'previous', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        except UnboundLocalError:
            return (False, False, False, 'previous', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
    elif GUI.goto:
        try:
            return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'goto', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        except UnboundLocalError:
            return (False, False, False, 'goto', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
    elif GUI.currentPTBtree == True:
        #this is the case where we don't move forward or back but just redisplay the same tree
        #unless we just removed a tree from the seed set or moved one into it from the
        #auto set.. in both of these cases we restart from the first tree that matches the current search
        try:
            if GUI.lengthCountsOfRemovedTrees != {}:
                #this is the case where we deleted categories, and as a result, one or more parses were removed from the seed bank
                return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'removedSeveral', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
            elif (GUI.treeRemoved == False or GUI.section_folder == 'new_parses') and GUI.treeMoved == False:
                return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'same', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
            elif GUI.treeMoved == True:
                return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'moved', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
            else:
                return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'removed', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        except UnboundLocalError:
            if GUI.treeRemoved == False:
                return (False, False, False, 'same', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
            else:
                return (False, False, False, 'removed', GUI.searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
    else:
        #covers the case where GUI.nextPTBtree == True...
        try:
            return (GUI.quit, GUI.start_auto_gen, GUI.overwriteAuto, 'next', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)
        except UnboundLocalError:
            return (False, False, False, 'next', searchItem, seedSentLen, match, displayIndex, GUI.sameSearch, GUI.goto, lenRemovedTree, lenMovedTree, fileManagerFolder, ptbLineNum, GUI.GUI_MATCH, GUI.newSentMode, GUI.test_words, GUI.untokenizedTestSentence, GUI.lengthCountsOfRemovedTrees, GUI.task, GUI.supertagger)

def updateMappings(cats, CatTreeMappings, TreeCatMappings, seed_folder, section_folder, ptb_file, index):
    #updates the tree to cat and cat to tree mappings
    for cat in cats:
        if cat in CatTreeMappings:
            if [seed_folder+"/"+section_folder+"/"+ptb_file, index] not in CatTreeMappings[cat]:
                CatTreeMappings[cat].append([seed_folder+"/"+section_folder+"/"+ptb_file, index])
        else:
            CatTreeMappings[cat] = [[seed_folder+"/"+section_folder+"/"+ptb_file, index]]
        TreeCatMappings[str([seed_folder+"/"+section_folder+"/"+ptb_file, index])].append(cat)
    return CatTreeMappings, TreeCatMappings

def buildAutoMappings(auto_folder, seed_folder):
    actm = {}
    anctm = {}
    atcm = {}
    antcm = {}
    print "\nBuilding auto mappings files...\n"
    for section_folder in sorted(os.listdir(auto_folder)):
        if section_folder not in ['timeAndDate', '.DS_Store', 'autoCatTreeMappings', 'autoNullCatTreeMappings', 'autoTreeCatMappings', 'autoNullTreeCatMappings', 'OvertLexicon']:
            if len(os.listdir(auto_folder+'/'+section_folder)) > 0 and not (len(os.listdir(auto_folder+'/'+section_folder)) == 1 and os.listdir(auto_folder+'/'+section_folder)[0] == '.DS_Store'):
                print "Processing auto trees from section folder: "+section_folder
            for FILE in sorted(os.listdir(auto_folder+'/'+section_folder)):
                if FILE != '.DS_Store':
                    autoParses = json.load(open(auto_folder+'/'+section_folder+'/'+FILE))
                    for parse in autoParses:
                        sdb_bracketing = autoParses[parse][0]
                        sd_tree = gen_derived_tree.gen_derivation_tree(sdb_bracketing)
                        MGcats = get_MGcats(sd_tree, [], returnWords=False)
                        nullMGcats = get_null_MGcats(sd_tree, [])
                        sfd_bracketing = autoParses[parse][3]
                        d_bracketing = autoParses[parse][4]
                        d_tree = gen_derived_tree.gen_derivation_tree(d_bracketing)
                        sfd_tree = gen_derived_tree.gen_derivation_tree(sfd_bracketing)
                        indices_mappings = get_overt_indices(d_tree, sfd_tree)
                        sortedMGcats = []
                        for MGcat in MGcats:
                            sortedMGcats.append(None)
                        for mapping in indices_mappings:
                            sortedMGcats[int(indices_mappings[mapping])] = MGcats[int(mapping)]
                        MGcats = sortedMGcats
                        while None in MGcats:
                            MGcats.remove(None)
                        PARSE = [auto_folder+'/'+section_folder+'/'+FILE, int(parse)]
                        atcm[str(PARSE)] = MGcats
                        antcm[str(PARSE)] = nullMGcats
                        for MGcat in MGcats:
                            if MGcat not in actm:
                                actm[MGcat] = [PARSE]
                            elif [auto_folder+'/'+section_folder+'/'+FILE, int(parse)] not in actm[MGcat]:
                                actm[MGcat].append([auto_folder+'/'+section_folder+'/'+FILE, int(parse)])
                        for nullMGcat in nullMGcats:
                            if nullMGcat not in anctm:
                                anctm[nullMGcat] = [[auto_folder+'/'+section_folder+'/'+FILE, int(parse)]]
                            elif [auto_folder+'/'+section_folder+'/'+FILE, int(parse)] not in anctm[nullMGcat]:
                                anctm[nullMGcat].append([auto_folder+'/'+section_folder+'/'+FILE, int(parse)])
    print ""
    with open(auto_folder+'/autoCatTreeMappings', 'w') as actmFile:
        json.dump(actm, actmFile)
    with open(auto_folder+'/autoNullCatTreeMappings', 'w') as anctmFile:
        json.dump(anctm, anctmFile)
    with open(auto_folder+'/autoTreeCatMappings', 'w') as atcmFile:
        json.dump(atcm, atcmFile)
    with open(auto_folder+'/autoNullTreeCatMappings', 'w') as antcmFile:
        json.dump(antcm, antcmFile)
    parserSettings = json.load(open(seed_folder+"/"+"parserSettings"))
    try:
        lastAutoTimeAndDate = json.load(open(auto_folder+"/"+"timeAndDate"))
        parserSettings['last_auto_date'] = lastAutoTimeAndDate['date']
        parserSettings['last_auto_time'] = lastAutoTimeAndDate['time']
        with open(seed_folder+"/"+"parserSettings", 'w') as parserSettingsFile:
            json.dump(parserSettings, parserSettingsFile)
    except Exception:
        x=0

def get_overt_indices(derivation_tree, sf_derivation_tree):
    tree_copy = copy.deepcopy(derivation_tree)
    assign_id_to_overt_terminals(tree_copy, 0)
    new_bracketing = tree_copy.generate_bracketing()
    new_bracketing = new_bracketing.encode('utf8')[1:-1]
    new_bracketing = re.sub(" \(", "(", new_bracketing, count=100000)
    new_bracketing = re.sub(" \)", ")", new_bracketing, count=100000)
    (X, Y, xbar_tree) = gen_derived_tree.main(new_bracketing, show_indices=True, return_xbar_tree=True, allowMoreGoals=True)
    mappings = {}
    get_id_indices_mappings(xbar_tree, mappings, 0)
    assign_indices_to_overt_terminals(sf_derivation_tree, 0, mappings, xbar_tree, new_bracketing)
    return mappings

def get_id_indices_mappings(node, mappings, index):
        if len(node.daughters) > 0:
            for daughter in node.daughters:
                index = get_id_indices_mappings(daughter, mappings, index)
        else:
            try:
                int(node.name[1:-1])
                mappings[node.name[1:-1]] = str(index)
                index += 1
            except ValueError:
                x=0
        return index

def assign_id_to_overt_terminals(node, ID):
        if len(node.daughters) != 0:
            for daughter in node.daughters:
                ID = assign_id_to_overt_terminals(daughter, ID)
        elif node.name[0] != '[':
            parts = node.name.split(" ")
            parts[0] = "*"+str(ID)+"*"
            node.name = " ".join(parts)
            ID += 1
        return ID

def assign_indices_to_overt_terminals(node, derivation_tree_index, id_indices_mappings, xbar_tree, new_bracketing):
        if len(node.daughters) != 0:
            for daughter in node.daughters:
                derivation_tree_index = assign_indices_to_overt_terminals(daughter, derivation_tree_index, id_indices_mappings, xbar_tree, new_bracketing)
        elif node.name[0] != '[':
            try:
                node.index = id_indices_mappings[str(derivation_tree_index)]
                derivation_tree_index += 1
            except KeyError:
                #owing to ATB movement, there can be overt terminals in the derivation tree that are missing
                #from the phrase structure tree..
                derivation_tree_index += 1
                node.index = -2
        else:
            node.index = -1
        return derivation_tree_index

def get_null_MGcats(node, nullMGcats):
    #takes as input a derivation tree object and returns a list of all the null
    #categories at the terminals of that tree in the format [decl] :: t= +WH
    if node.name[0] == '[':
        nullMGcats.append(node.name)
    else:
        for daughter in node.daughters:
            nullMGcats = get_null_MGcats(daughter, nullMGcats)
    return nullMGcats

def get_MGcats(node, MGcats, returnWords=False):
    #takes as input a derivation tree object and returns a list of all the overt
    #categories at the terminals of that tree in the format :: t= +WH
    if node.name[0] != '[' and len(node.daughters) == 0:
        if not returnWords:
            MGcat = " ".join(node.name.split(" ")[1:])
        else:
            MGcat = " ".join(node.name.split(" "))
        if MGcat[0] == " ":
            MGcat = MGcat.strip()
        MGcats.append(MGcat)
    else:
        for daughter in node.daughters:
            MGcats = get_MGcats(daughter, MGcats, returnWords)
    return MGcats
    
def get_MG_terminals(xbar_tree, terminals):
    if xbar_tree.sem_node and not xbar_tree.phon_node:
        return terminals
    #there's a weired bug in python that after exiting a program and re-entering it
    #some variables stick around.. this is sometimes the case with terminals so I have
    #to be sure to delete anything currently in it:
    #a function which returns a list of MG terminals.. we exclude any traces..
    for daughter in xbar_tree.daughters:
        if daughter.name != '' and daughter.terminal == True and "λ" not in daughter.name and "Λ" not in daughter.name and 'μ' not in daughter.name and 'ζ' not in daughter.name and (daughter.name[0] != "[" and daughter.name[-1] != "]"):
            terminals.append(daughter)
    for daughter in xbar_tree.daughters:
        terminals = get_MG_terminals(daughter, terminals)
    return terminals

def get_deps_terminals(ptb_bracketing, return_mg_deps, xbar_tree=None):
    (PTB_tree, sentence, terminals) = build_tree(ptb_bracketing)
    PTB_chains = get_chains(PTB_tree, chains={})
    #in some cases, multiple antecedents arise owing to splitTreePointers where the head was not part of the pointer
    #so we did not merge the sisters.. so we ignore these.. some times it's because both the head
    #of a relative clause and the relative pronoun are marked as antecedents in which case we'll just
    #make the relative the antecedent..We follow Penn in including 'that' as a relative pronoun here
    #even though in the MG tree it will not be..the system will learn this.
    for entry in PTB_chains:
        if len(PTB_chains[entry]['antecedents']) > 1:
            for antecedent in PTB_chains[entry]['antecedents']:
                if 'WDT' in antecedent.daughters[0].name or 'WP' in antecedent.daughters[0].name or 'WRB' in antecedent.daughters[0].name or ('IN' in antecedent.daughters[0].name and antecedent.daughters[0].daughters[0].name in ['that', 'That']):
                    PTB_chains[entry]['antecedents'] = [antecedent]
                    break
    entries_to_delete = []
    for entry in PTB_chains.keys():
        #if we still have more than one antecedent (ie more than one overt member
        #of the chain) then we simply remove this chain..we also remove if there are no
        #antecedents or no traces
        if len(PTB_chains[entry]['antecedents']) > 1 or len(PTB_chains[entry]['antecedents']) == 0 or len(PTB_chains[entry]['traces']) == 0:
            entries_to_delete.append(entry)
    while len(entries_to_delete) > 0:
        del(PTB_chains[entries_to_delete[0]])
        del(entries_to_delete[0])
    #now we go through and set all traces' chain_pointer attribute to point to the
    #overt antecedent
    for entry in PTB_chains:
        for trace in PTB_chains[entry]['traces']:
            trace.chain_pointer = PTB_chains[entry]['antecedents'][0]
    PTB_deps = get_PTBdeps(PTB_tree, terminals, deps=[], head_type='syntactic')
    PTB_sem_deps = get_PTBdeps(PTB_tree, terminals, deps=[], head_type='semantic')
    for dep in PTB_sem_deps:
        if dep not in PTB_deps:
            PTB_deps.append(dep)
    if return_mg_deps == False:
        return (PTB_deps, terminals, PTB_tree)
    #now we need to extract all dependencies from the xbar tree and create various mappings
    #between dependencies and categories..
    add_truncated_names(xbar_tree)
    MG_terminals = get_MG_terminals(xbar_tree, terminals = [])
    MG_chains = get_chains(xbar_tree, chains={})
    entries_to_delete = []
    for entry in MG_chains.keys():
        if len(MG_chains[entry]['antecedents']) > 1 or len(MG_chains[entry]['antecedents']) == 0 or len(MG_chains[entry]['traces']) == 0:
            entries_to_delete.append(entry)
    while len(entries_to_delete) > 0:
        del(MG_chains[entries_to_delete[0]])
        del(entries_to_delete[0])
    for entry in MG_chains:
        for trace in MG_chains[entry]['traces']:
            trace.chain_pointer = MG_chains[entry]['antecedents'][0]
    MG_deps = get_MGdeps(xbar_tree, MG_terminals, deps=[])
    return (PTB_deps, MG_deps, terminals, PTB_tree)

def get_null_c_lexicon(null_lexicon, null_c_lexicon, compose='supertags'):
    #we no longer just take c heads out of the supertags, but also the null heads with the
    #the licensee features for A'-movement..
    for entry in null_lexicon:
        if entry[0] in ['[det]', '[dat]', '[topicalizer]', '[focalizer]', '[wh]', '[relativizer]']:
            #because relative clauses have a nominal layer which is selected for by a determiner,
            #that determiner has no c= feature so we will not detect it.. we therefore have no choice but to allow
            #null [det] into the null_c_lexicon and also [dat] which governs [det] in coordinate constructions
            null_c_lexicon.append(entry)
            continue
        for feature in entry[1]:
            if '\xe2\x89\x88' not in feature.encode('utf8'):
                #we don't include stuff that adjoins to CP in the null_c_lexicon, only
                #stuff that selects it as complement (or spec)
                if cat_pattern.search(feature).group(0).lower() == 'c':
                    if 'NS' in compose:
                        entry = strip_features(entry)
                        if entry not in null_c_lexicon:
                            null_c_lexicon.append(entry)
                    else:
                        null_c_lexicon.append(entry)
                    break

def strip_features(cat):
    if type(cat) == type(u"") or type(cat) == type(""):
        parts = cat.split(" ")
        features = parts[1].split(" ")
    else:
        features = cat[1]
        new_cat = copy.deepcopy(cat)
    i = -1
    for feature in features:
        i+=1
        subcat_features = re.search('{.*?}', feature)
        if subcat_features:
            subcat_features = subcat_features.group(0)
        else:
            subcat_features = []
        sfs = []
        for sf in subcat_features:
            if sf in ['EDGE', '+NONE', 'MAIN', 'ANA', 'IT']+sel_variables:
                sfs.append(sf)
        if len(sfs) == 0:
            sfs = ''
        else:
            sfs = "{"+".".join(sfs)+"}"
        feature = re.sub('{.*}', sfs, feature)
        if type(cat) != type(u"") and type(cat) != type(""):
            new_cat[1][i] = feature
        else:
            features[i] = feature
    if type(cat) == type(u"") or type(cat) == type(""):
        new_cat = unicode(parts[0]+" "+" ".join(features))
    return new_cat

def strip_derivation_tree(node):
    chains = node.name.split(",")
    chain_index = -1
    for chain in chains:
        chain_index += 1
        char_ind = -1
        for char in chain:
            char_ind += 1
            if char == ":":
                break
        for char in chain[char_ind:]:
            char_ind+=1
            if char == " ":
                break
        features = chain[char_ind:].strip().split(" ")
        i=-1
        for feature in features:
            i+=1
            subcat_features = re.search('{.*?}', feature)
            if subcat_features:
                subcat_features = subcat_features.group(0)[1:-1].split(".")
            else:
                subcat_features = []
            sfs = []
            for sf in subcat_features:
                if sf in ['EDGE', '+NONE', 'MAIN', 'ANA', 'IT']+sel_variables:
                    sfs.append(sf)
            if len(sfs) == 0:
                sfs = ''
            else:
                sfs = "{"+".".join(sfs)+"}"
            feature = re.sub('{.*}', sfs, feature)
            features[i] = feature
            new_chain = chain[:char_ind]+" ".join(features)
            chains[chain_index] = new_chain
    node.name = ",".join(chains)
    for daughter in node.daughters:
        strip_derivation_tree(daughter)

def contains_vp_ellipsis(node):
    if node.name == "*?*" and node.mother != None and node.mother.name == '-NONE-' and node.mother.mother != None and (node.mother.mother.truncated_name == 'VP' or 'NP-PRD' in node.mother.mother.name):
        return True
    elif node.name.lower() in aux_verbs and node.mother != None and node.mother.mother != None and (len(node.mother.mother.daughters) == 1 or (len(node.mother.mother.daughters) == 2 and 'NEG' == node.mother.mother.daughters[1].truncated_name)):
        return True
    else:
        for daughter in node.daughters:
            if contains_vp_ellipsis(daughter):
                return True
    return False

def auto_gen_MGbank(ptb_folder, seed_folder, overwriteAuto):
    global PosDepsMappings
    global DepMappings
    global ccg_beam
    global ccg_beam_floor
    global supertaggingStrategy
    global constrainMoveWithPTB
    global constrainConstWithPTBCCG
    global use_deps
    MGbankName = seed_folder[4:-4]
    timeAndDate = {'time':time.strftime("%H:%M:%S"), 'date':time.strftime("%d/%m/%Y")}
    terminal_output_name = "terminal_output_"+MGbankName+"_"+time.strftime("%d_%m_%Y")+"@"+time.strftime("%H.%M.%S")
    if start_file != None:
        terminal_output_name += "_"+start_file
        if end_file != None:
            terminal_output_name += "_"+end_file
    elif auto_section_folders != None:
        terminal_output_name+="_"+"_".join(auto_section_folders)
    terminal_output = open(terminal_output_name, 'w')
    parserSettings = json.load(open(seed_folder+"/"+"parserSettings"))
    if parser_setting == 'basic':
        parser_strategy = 'basicOnly'
    elif parser_setting == 'full':
        parser_strategy = 'fullFirst'
    else:
        parser_strategy = 'basicOnly'
    if time_out == None:
        TIMEOUT = parserSettings['timeout_seconds']
    else:
        TIMEOUT = time_out
    if use_deps == None:
        use_deps = parserSettings['use_deps']
    if supertaggingStrategy == None:
        supertaggingStrategy = parserSettings['supertaggingStrategy']
    if all_verb_cats:
        avc = " with all_verb_cats set to True"
    else:
        avc = " with all_verb_cats set to False"
    if all_cats:
        avc = " with all_cats set to True"
    else:
        avc = " with all_cats set to False"
    print "\nAutomatically generating "+MGbankName+" using supertagging strategy: "+supertaggingStrategy+avc+", at "+timeAndDate['time']+" on "+timeAndDate['date']+'\n'
    terminal_output.write("Automatically generating "+MGbankName+" using supertagging strategy: "+supertaggingStrategy+avc+", at "+timeAndDate['time']+" on "+timeAndDate['date']+'\n')
    autocorpus_folder = ptb_folder+"_MGbankAuto"
    PosDepsMappings = json.load(open(seed_folder+'/PosDepsMappings'))
    DepMappings = json.load(open(seed_folder+'/DepMappings'))
    RevDepMappings = json.load(open(seed_folder+'/RevDepMappings'))
    overtLexicon = json.load(open(seed_folder+'/'+'OvertLexicon'))
    if supertaggingStrategy == 'CCG_OVERT_MG':
        stats_table1 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel1", 'r'))
        stats_table2 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel2", 'r'))
        stats_table3 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel3", 'r'))
        stats_table4 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel4", 'r'))
    elif supertaggingStrategy == 'CCG_MG_SUPERTAG':
        stats_table1 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel1", 'r'))
        stats_table2 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel2", 'r'))
        stats_table3 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel3", 'r'))
        stats_table4 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel4", 'r'))
    elif supertaggingStrategy == 'CCG_MG_SUPERTAG_MAXENT':
        uni_stats_table1 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel1", 'r'))
        uni_stats_table2 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel2", 'r'))
        uni_stats_table3 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel3", 'r'))
        uni_stats_table4 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_taggingModel4", 'r'))
        maxent_model1 = seed_folder+'/stagmaxent.model1'
        maxent_model2 = seed_folder+'/stagmaxent.model2'
        maxent_model3 = seed_folder+'/stagmaxent.model3'
        maxent_model4 = seed_folder+'/stagmaxent.model4'
        MGST_REF_table = json.load(open(seed_folder+'/MGST_REF_table'))
        REF_MGST_table = json.load(open(seed_folder+'/REF_MGST_table'))
        for entry in REF_MGST_table:
            REF_MGST_table[entry] = ast.literal_eval(REF_MGST_table[entry])
    elif supertaggingStrategy == 'CCG_MG_HYBRID':
        stats_table1 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel1", 'r'))
        stats_table2 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel2", 'r'))
        stats_table3 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel3", 'r'))
        stats_table4 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel4", 'r'))
    elif supertaggingStrategy == 'CCG_MG_HYBRID_MAXENT':
        uni_stats_table1 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel1", 'r'))
        uni_stats_table2 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel2", 'r'))
        uni_stats_table3 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel3", 'r'))
        uni_stats_table4 = json.load(open(seed_folder+"/CCG_MG_HYBRID_taggingModel4", 'r'))
        maxent_model1 = seed_folder+'/hybmaxent.model1'
        maxent_model2 = seed_folder+'/hybmaxent.model2'
        maxent_model3 = seed_folder+'/hybmaxent.model3'
        maxent_model4 = seed_folder+'/hybmaxent.model4'
        MGST_REF_table = json.load(open(seed_folder+'/MGHST_REF_table'))
        REF_MGST_table = json.load(open(seed_folder+'/REF_MGHST_table'))
        for entry in REF_MGST_table:
            REF_MGST_table[entry] = ast.literal_eval(REF_MGST_table[entry])
    elif supertaggingStrategy == 'CCG_MG_HYBRID_NS_MAXENT':
        uni_stats_table1 = json.load(open(seed_folder+"/CCG_MG_HYBRID_NS_taggingModel1", 'r'))
        uni_stats_table2 = json.load(open(seed_folder+"/CCG_MG_HYBRID_NS_taggingModel2", 'r'))
        uni_stats_table3 = json.load(open(seed_folder+"/CCG_MG_HYBRID_NS_taggingModel3", 'r'))
        uni_stats_table4 = json.load(open(seed_folder+"/CCG_MG_HYBRID_NS_taggingModel4", 'r'))
        maxent_model1 = seed_folder+'/hybnsmaxent.model1'
        maxent_model2 = seed_folder+'/hybnsmaxent.model2'
        maxent_model3 = seed_folder+'/hybnsmaxent.model3'
        maxent_model4 = seed_folder+'/hybnsmaxent.model4'
        MGST_REF_table = json.load(open(seed_folder+'/MGHSTNS_REF_table'))
        REF_MGST_table = json.load(open(seed_folder+'/REF_MGHSTNS_table'))
        for entry in REF_MGST_table:
            REF_MGST_table[entry] = ast.literal_eval(REF_MGST_table[entry])
    elif supertaggingStrategy == 'CCG_MG_SUPERTAG_NS':
        stats_table1 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel1", 'r'))
        stats_table2 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel2", 'r'))
        stats_table3 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel3", 'r'))
        stats_table4 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel4", 'r'))
    elif supertaggingStrategy == 'CCG_MG_SUPERTAG_NS_MAXENT':
        uni_stats_table1 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel1", 'r'))
        uni_stats_table2 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel2", 'r'))
        uni_stats_table3 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel3", 'r'))
        uni_stats_table4 = json.load(open(seed_folder+"/CCG_MG_SUPERTAG_NS_taggingModel4", 'r'))
        maxent_model1 = seed_folder+'/stnsagmaxent.model1'
        maxent_model2 = seed_folder+'/stnsagmaxent.model2'
        maxent_model3 = seed_folder+'/stnsagmaxent.model3'
        maxent_model4 = seed_folder+'/stnsagmaxent.model4'
        MGST_REF_table = json.load(open(seed_folder+'/MGSTNS_REF_table'))
        REF_MGST_table = json.load(open(seed_folder+'/REF_MGSTNS_table'))
        for entry in REF_MGST_table:
            REF_MGST_table[entry] = ast.literal_eval(REF_MGST_table[entry])
    elif supertaggingStrategy == 'CCG_OVERT_MG_MAXENT':
        uni_stats_table1 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel1", 'r'))
        uni_stats_table2 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel2", 'r'))
        uni_stats_table3 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel3", 'r'))
        uni_stats_table4 = json.load(open(seed_folder+"/CCG_MG_ATOMIC_taggingModel4", 'r'))
        maxent_model1 = seed_folder+'/tagmaxent.model1'
        maxent_model2 = seed_folder+'/tagmaxent.model2'
        maxent_model3 = seed_folder+'/tagmaxent.model3'
        maxent_model4 = seed_folder+'/tagmaxent.model4'
        MGT_REF_table = json.load(open(seed_folder+'/MGT_REF_table'))
        REF_MGT_table = json.load(open(seed_folder+'/REF_MGT_table'))
    if 'CCG' in supertaggingStrategy:
        if ccg_beam == None:
            ccg_beam = float(parserSettings['ccgBeam'])
        if ccg_beam_floor == None:
            ccg_beam_floor = float(parserSettings['ccgBeamFloor'])
        ccg_beam_text = "  CCG-MG starting beam: "+str(ccg_beam)+ "   Beam floor: "+str(ccg_beam_floor)+"   Maximum MG categories allowed per word: "+str(max_mg_cats_per_word)
    else:
        ccg_beam = None
        ccg_beam_floor = None
        ccg_beam_text = ""
    if constrainMoveWithPTB == None:
        constrainMoveWithPTB = parserSettings['constrainMoveWithPTB']
    if constrainConstWithPTBCCG == None:
        constrainConstWithPTBCCG = parserSettings['constrainConstWithPTBCCG']
    if constrainMoveWithPTB and not constrainConstWithPTBCCG:
        print "Using Penn Treebank constituencies to constrain phrasal movement in MG trees.."
        terminal_output.write("Using Penn Treebank constituencies to constrain phrasal movement in MG trees..")
    elif constrainConstWithPTBCCG and constrainMoveWithPTB:
        print "Using Penn Treebank constituencies to constrain phrasal movement and Penn Treebank and CCGbank constituencies to constrain phrase structure in MG trees.."
        terminal_output.write("Using Penn Treebank constituencies to constrain phrasal movement and Penn Treebank and CCGbank constituencies to constrain phrase structure in MG trees..")
    elif constrainConstWithPTBCCG and not constrainMoveWithPTB:
        print "Using Penn Treebank and CCGbank constituencies to constrain phrase structure in MG trees.."
        terminal_output.write("Using Penn Treebank constituencies to constrain phrase structure in MG trees..")
    instances_multi_perf = 0
    num_best_trees_added = 0
    instances_multi_best = 0
    PARSES_CONSIDERED = 0
    if auto_section_folders == None:
        section_folders = []
        for section_folder in parserSettings['section_folders']:
            section_folders.append(section_folder)
    else:
        section_folders = auto_section_folders
    section_folders.sort()
    total_start_time = default_timer()
    times = []
    parse_times = []
    if start_file != None:
        start_file_reached = False
    else:
        start_file_reached = True
    end_file_reached = False
    if start_line != None:
        start_line_reached = False
    else:
        start_line_reached = True
    for section_folder in section_folders:
        if auto_section_folders == None:
            if parserSettings['section_folders'][section_folder] == 0:
                continue
        elif section_folder not in auto_section_folders:
            continue
        if stop_after == None:
            if parserSettings['autoMaxTreesVar'] != 'no limit' and parserSettings['autoMaxTreesVar'] <= num_best_trees_added:
                break
        elif stop_after <= num_best_trees_added:
            break
        if section_folder == '.DS_Store':
            continue
        for ptb_file in sorted(os.listdir(ptb_folder+"/"+section_folder)):
            if end_file_reached:
                break
            if not start_file_reached:
                if ptb_file != start_file:
                    continue
                else:
                    start_file_reached = True
            if ptb_file == end_file:
                end_file_reached = True
            if stop_after == None:
                if parserSettings['autoMaxTreesVar'] != 'no limit' and parserSettings['autoMaxTreesVar'] <= num_best_trees_added:
                    break
            elif stop_after <= num_best_trees_added:
                break
            if ptb_file == '.DS_Store':
                continue
            parses=open(ptb_folder+"/"+section_folder+"/"+ptb_file, "r")
            PTBbracketings = []
            for line in parses:
                #all we need this time round is the bracketing so we pass None for the other values.
                PTBbracketings.append((None, None, None, line.strip()))
            #we need to check for each PTB tree whether or not it is already included in the
            #seed set or the auto set.. if it is we will skip it..
            if section_folder in os.listdir(seed_folder):
                if ptb_file in os.listdir(seed_folder+"/"+section_folder):
                    seed_set = json.load(open(seed_folder+"/"+section_folder+"/"+ptb_file))
                else:
                    seed_set = {}
            else:
                seed_set = {}
            if ptb_file in os.listdir(autocorpus_folder+"/"+section_folder):
                auto_set = json.load(open(autocorpus_folder+"/"+section_folder+"/"+ptb_file))
            else:
                auto_set = {}
            index = -1
            for bracketing in PTBbracketings:
                gc.collect()
                index+=1
                if stop_after == None:
                    if parserSettings['autoMaxTreesVar'] != 'no limit' and parserSettings['autoMaxTreesVar'] <= num_best_trees_added:
                        break
                elif stop_after <= num_best_trees_added:
                    break
                if not start_line_reached:
                    if index+1 != start_line:
                        continue
                    else:
                        start_line_reached = True
                if str(index) in seed_set or str(index) in auto_set:
                    continue
                else:
                    (PTB_deps, terminals, PTB_tree) = get_deps_terminals(bracketing[3], False)
                    vp_ellipsis = contains_vp_ellipsis(PTB_tree)
                    if constrainMoveWithPTB:
                        moveable_spans = []
                        get_moveable_spans(PTB_tree, terminals, moveable_spans)
                    else:
                        moveable_spans = None
                    if constrainConstWithPTBCCG:
                        source_spans = []
                        try:
                            parts = bracketing[3].split("/")
                            ccg_parses = json.load(open("CCGbank/"+section_folder+"/"+ptb_file.split(".")[0]+".ccg"))
                            ccg_bracketing = ccg_parses[ptb_file.split(".")[0]+"."+str(index+1)]
                            ccg_tree = build_tree(ccg_bracketing)
                            ccg_terminals = ccg_tree[2]
                            set_indices(ccg_terminals)
                            ccg_tree = ccg_tree[0]
                        except Exception as e:
                            ccg_tree = None
                        get_source_spans(PTB_tree, ccg_tree, source_spans, terminals)
                    else:
                        source_spans = None
                    if min_length != None:
                        if len(terminals) < min_length:
                            continue
                    if max_length != None:
                        if len(terminals) > max_length:
                            continue
                    elif parserSettings['autoMaxSentLenVar'] != 'all' and parserSettings['autoMaxSentLenVar'] < len(terminals):
                        continue
                    sen_len = len(terminals)
                    if PARSES_CONSIDERED != 0 and PARSES_CONSIDERED%1 == 0:
                        print ""
                        terminal_output.write("\n")
                        print "Total number of trees considered:", PARSES_CONSIDERED
                        terminal_output.write("Total number of trees considered: "+str(PARSES_CONSIDERED)+'\n')
                        print "Total number of trees added to the treebank:", num_best_trees_added
                        terminal_output.write("Total number of trees added to the treebank: "+str(num_best_trees_added)+'\n')
                        print "Total time taken so far to process the corpus:", time_taken(default_timer() - total_start_time)
                        terminal_output.write("Total time taken so far to process the corpus: "+str(time_taken(default_timer() - total_start_time)+'\n'))
                        print "Total time taken so far just for parsing: ", time_taken(sum(parse_times))
                        terminal_output.write("Total time taken so far just for parsing: "+str(time_taken(sum(parse_times)))+'\n')
                        try:
                            print "Quickest time taken so far to parse a tree: ", time_taken(min(parse_times))
                            terminal_output.write("Quickest time taken so far to parse a tree: "+str(time_taken(min(parse_times))+'\n'))
                            print "Slowest time taken so far to parse a tree: ", time_taken(max(parse_times))
                            terminal_output.write("Slowest time taken so far to parse a tree: "+str(time_taken(max(parse_times)))+'\n')
                            print "Average (mean) time taken so far to parse each tree: ", time_taken(sum(parse_times)/len(parse_times))
                            terminal_output.write("Average (mean) time taken so far to parse each tree: "+str(time_taken(sum(parse_times)/len(parse_times)))+'\n')
                        except Exception as e:
                            x=0
                        terminal_output.close()
                        terminal_output = open(terminal_output_name, 'a')
                    #now we need to determine a set of possible MG categories for each word
                    #based on the PTB categories and the dependencies they enter into
                    miniOvertLexicon = []
                    if True:
                        ID = ptb_file.split(".")[0]+"."+str(index+1)
                        ccg_parses = json.load(open("CCGbank/"+section_folder+"/"+ptb_file.split(".")[0]+".ccg", 'r'))
                        try:
                            ccg_tree = build_tree(ccg_parses[ID])
                            ccg_terminals = ccg_tree[2]
                            ccg_tree = ccg_tree[0]
                        except Exception as e:
                            print "\nNo CCG tree available for ID: "+str(ID)+".. Ignoring this sentence..\n"
                            #if there's no CCG tree, we ignore this tree and move to the next one
                            continue
                        if len(ccg_terminals) != len(terminals):
                            print "\nPTB and CCG trees have different number of terminals for tree with ID: "+str(ID)+".. Ignoring this sentence..\n"
                            #sometimes, because of the extra hyphenated structure imported from Ontonotes,
                            #the string length of the CCG and PTB trees does not match.. again we just move on to the
                            #next tree if so.. these trees can be generated once we have the pure MG supertagger
                            continue
                        ccg_tags = [terminal.mother.name for terminal in ccg_terminals]
                        prunedMGcats = []
                    else:
                        prunedMGcats = None
                    words = [t.name.lower() for t in terminals]
                    timeAndDate = {'time':time.strftime("%H:%M:%S"), 'date':time.strftime("%d/%m/%Y")}
                    print "\nParsing sentence: '"+" ".join(words)+", at fringe of PTB tree:", autocorpus_folder+"/"+section_folder+"/"+ptb_file, "ln:", index+1, "on", timeAndDate['date'], "at", timeAndDate['time'], ccg_beam_text, "   Timeout: ", str(TIMEOUT)+"\n"
                    terminal_output.write("\nParsing sentence: '"+" ".join(words)+"', at fringe of PTB tree: "+autocorpus_folder+"/"+section_folder+"/"+ptb_file+" ln: "+str(index+1)+" on "+timeAndDate['date']+" at "+timeAndDate['time']+" "+ccg_beam_text+"\n\n")
                    PARSES_CONSIDERED += 1
                    terminal_output.close()
                    terminal_output = open(terminal_output_name, 'a')
                    abort = False
                    supertags = []
                    if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                        supertags = []
                    else:
                        miniOvertLexicon = []
                    if 'MAXENT' in supertaggingStrategy:
                        ccg_tagged_sentence_model1 = ""
                        ccg_tagged_sentence_model2 = ""
                        ccg_tagged_sentence_model3 = ""
                        ccg_tagged_sentence_model4 = ""
                        ts_index = -1
                        for word in words:
                            ts_index += 1
                            ccg_tagged_sentence_model1+=words[ts_index]+"|"+ccg_tags[ts_index]+" "
                            ccg_tagged_sentence_model2+=words[ts_index]+"|"+"_".join(ccg_tags[ts_index].split("_")[:3])+" "
                            ccg_tagged_sentence_model3+=words[ts_index]+"|"+ccg_tags[ts_index].split("_")[0]+" "
                            ccg_tagged_sentence_model4+=words[ts_index]+"|"+"_".join(ccg_tags[ts_index].split("_")[1:])+" "
                        ccg_tagged_sentence_model1 = ccg_tagged_sentence_model1.strip()
                        ccg_tagged_sentence_model1_file = open('ccg_tagged_sentence_model1_'+ptb_file, 'w')
                        ccg_tagged_sentence_model1_file.write(ccg_tagged_sentence_model1)
                        ccg_tagged_sentence_model1_file.close()
                        ccg_tagged_sentence_model2 = ccg_tagged_sentence_model2.strip()
                        ccg_tagged_sentence_model2_file = open('ccg_tagged_sentence_model2_'+ptb_file, 'w')
                        ccg_tagged_sentence_model2_file.write(ccg_tagged_sentence_model2)
                        ccg_tagged_sentence_model2_file.close()
                        ccg_tagged_sentence_model3 = ccg_tagged_sentence_model3.strip()
                        ccg_tagged_sentence_model3_file = open('ccg_tagged_sentence_model3_'+ptb_file, 'w')
                        ccg_tagged_sentence_model3_file.write(ccg_tagged_sentence_model3)
                        ccg_tagged_sentence_model3_file.close()
                        ccg_tagged_sentence_model4 = ccg_tagged_sentence_model4.strip()
                        ccg_tagged_sentence_model4_file = open('ccg_tagged_sentence_model4_'+ptb_file, 'w')
                        ccg_tagged_sentence_model4_file.write(ccg_tagged_sentence_model4)
                        ccg_tagged_sentence_model4_file.close()
                        #now tag the sentence and save the result to file 'mg_tagged_sentence'
                        error = True
                        while error:
                            #sometimes if this is running on a network and the connection is bad then saving files doesn't work momentarily, so we keep trying until it does
                            try:
                                os.system("./candc-1.00/bin/msuper --model "+maxent_model1+" --input ccg_tagged_sentence_model1_"+ptb_file+" --output mg_tagged_sentence_model1_"+ptb_file+" --super-forward_beam_ratio "+str(super_forward_beam_ratio)+" --super-category_cutoff "+str(super_category_cutoff)+" --super-rare_cutoff "+str(super_rare_cutoff)+" --super-tagdict_min "+str(super_tag_dict_min)+" --super-beam_ratio "+str(super_beam_ratio)+" --super-beam_width "+str(super_beam_width))
                                mg_tagged_sentence_model1 = open('mg_tagged_sentence_model1_'+ptb_file, 'r')
                                error = False
                            except Exception as e:
                                x=0
                        error = True
                        while error:
                            try:
                                os.system("./candc-1.00/bin/msuper --model "+maxent_model2+" --input ccg_tagged_sentence_model2_"+ptb_file+" --output mg_tagged_sentence_model2_"+ptb_file+" --super-forward_beam_ratio "+str(super_forward_beam_ratio)+" --super-category_cutoff "+str(super_category_cutoff)+" --super-rare_cutoff "+str(super_rare_cutoff)+" --super-tagdict_min "+str(super_tag_dict_min)+" --super-beam_ratio "+str(super_beam_ratio)+" --super-beam_width "+str(super_beam_width))
                                mg_tagged_sentence_model2 = open('mg_tagged_sentence_model2_'+ptb_file, 'r')
                                error = False
                            except Exception as e:
                                x=0
                        error = True
                        while error:
                            try:
                                os.system("./candc-1.00/bin/msuper --model "+maxent_model3+" --input ccg_tagged_sentence_model3_"+ptb_file+" --output mg_tagged_sentence_model3_"+ptb_file+" --super-forward_beam_ratio "+str(super_forward_beam_ratio)+" --super-category_cutoff "+str(super_category_cutoff)+" --super-rare_cutoff "+str(super_rare_cutoff)+" --super-tagdict_min "+str(super_tag_dict_min)+" --super-beam_ratio "+str(super_beam_ratio)+" --super-beam_width "+str(super_beam_width))
                                mg_tagged_sentence_model3 = open('mg_tagged_sentence_model3_'+ptb_file, 'r')
                                error = False
                            except Exception as e:
                                x=0
                        error = True
                        while error:
                            try:
                                os.system("./candc-1.00/bin/msuper --model "+maxent_model4+" --input ccg_tagged_sentence_model4_"+ptb_file+" --output mg_tagged_sentence_model4_"+ptb_file+" --super-forward_beam_ratio "+str(super_forward_beam_ratio)+" --super-category_cutoff "+str(super_category_cutoff)+" --super-rare_cutoff "+str(super_rare_cutoff)+" --super-tagdict_min "+str(super_tag_dict_min)+" --super-beam_ratio "+str(super_beam_ratio)+" --super-beam_width "+str(super_beam_width))
                                mg_tagged_sentence_model4 = open('mg_tagged_sentence_model4_'+ptb_file, 'r')
                                error = False
                            except Exception as e:
                                x=0
                        os.system('rm mg_tagged_sentence_model1_'+ptb_file)
                        os.system('rm ccg_tagged_sentence_model1_'+ptb_file)
                        os.system('rm mg_tagged_sentence_model2_'+ptb_file)
                        os.system('rm ccg_tagged_sentence_model2_'+ptb_file)
                        os.system('rm mg_tagged_sentence_model3_'+ptb_file)
                        os.system('rm ccg_tagged_sentence_model3_'+ptb_file)
                        os.system('rm mg_tagged_sentence_model4_'+ptb_file)
                        os.system('rm ccg_tagged_sentence_model4_'+ptb_file)
                        #now we will construct the maxent stats_table using the statistics provided by the MaxEnt supertagger
                        line_index = -1
                        line_model2_index = -1
                        line_model3_index = -1
                        line_model4_index = -1
                        stats_table = {}
                        abort = False
                        print ""
                        for line in mg_tagged_sentence_model1:
                            line_index += 1
                            if line == '\n':
                                break
                            backoff_cat1 = "_".join(ccg_tags[line_index].split("_")[:3])
                            backoff_cat2 = ccg_tags[line_index].split("_")[0]
                            backoff_cat3 = "_".join(ccg_tags[line_index].split("_")[1:])
                            stats_table[line_index] = {}
                            fields = line.split('\t')
                            field_index = -1
                            FIELDS = fields[3:]
                            for field in FIELDS:
                                if ccg_tags[line_index] in uni_stats_table1: 
                                    field_index += 1
                                    if field_index % 2 == 0:
                                        if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                            MGtag = str(REF_MGST_table[field])
                                        else:
                                            MGtag = REF_MGT_table[field]
                                        #We allow in only those MG tags that were paired with this CCG tag in the seedset at this stage..unless allowbesttag == True
                                        if (allowbesttag and field_index == 0) or MGtag in uni_stats_table1[ccg_tags[line_index]]:
                                            prob = FIELDS[field_index+1]
                                            if prob[-1] == '\n':
                                                prob = prob[:-1]
                                            prob = float(prob)
                                            stats_table[line_index][MGtag] = prob
                                else:
                                    #we try backing off first to model2 then to model3 then to model4
                                    if backoff_cat1 in uni_stats_table2:
                                        print "CCG category: "+ccg_tags[line_index]+" not seen in seed set..  backing off to another model with category: "+backoff_cat1+"..."
                                        terminal_output.write("CCG category: "+ccg_tags[line_index]+" not seen in seed set..  backing off to another model with category: "+backoff_cat1+"...\n")
                                        terminal_output.close()
                                        terminal_output = open(terminal_output_name, 'a')
                                        for line in mg_tagged_sentence_model2:
                                            line_model2_index += 1
                                            if line_model2_index == line_index:
                                                fields = line.split('\t')
                                                field_index = -1
                                                FIELDS = fields[3:]
                                                for field in FIELDS:
                                                    field_index += 1
                                                    if field_index % 2 == 0:
                                                        if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                                            MGtag = str(REF_MGST_table[field])
                                                        else:
                                                            MGtag = REF_MGT_table[field]
                                                        if MGtag in uni_stats_table2[backoff_cat1]:
                                                            prob = FIELDS[field_index+1]
                                                            if prob[-1] == '\n':
                                                                prob = prob[:-1]
                                                            prob = float(prob)
                                                            stats_table[line_index][MGtag] = prob
                                                break
                                    elif backoff_cat2 in uni_stats_table3:
                                        print "CCG category: "+ccg_tags[line_index]+" not seen in seed set..  backing off to another model with category: "+backoff_cat2+"..."
                                        terminal_output.write("CCG category: "+ccg_tags[line_index]+" not seen in seed set..  backing off to another model with category: "+backoff_cat2+"...\n")
                                        terminal_output.close()
                                        terminal_output = open(terminal_output_name, 'a')
                                        for line in mg_tagged_sentence_model3:
                                            line_model3_index += 1
                                            if line_model3_index == line_index:
                                                fields = line.split('\t')
                                                field_index = -1
                                                FIELDS = fields[3:]
                                                for field in FIELDS:
                                                    field_index += 1
                                                    if field_index % 2 == 0:
                                                        if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                                            MGtag = str(REF_MGST_table[field])
                                                        else:
                                                            MGtag = REF_MGT_table[field]
                                                        if MGtag in uni_stats_table3[backoff_cat2]:
                                                            prob = FIELDS[field_index+1]
                                                            if prob[-1] == '\n':
                                                                prob = prob[:-1]
                                                            prob = float(prob)
                                                            stats_table[line_index][MGtag] = prob
                                                break
                                    elif backoff_cat3 in uni_stats_table4:
                                        print "CCG category: "+ccg_tags[line_index]+" not seen in seed set..  backing off to another model with category: "+backoff_cat3+"..."
                                        terminal_output.write("CCG category: "+ccg_tags[line_index]+" not seen in seed set..  backing off to another model with category: "+backoff_cat3+"...\n")
                                        terminal_output.close()
                                        terminal_output = open(terminal_output_name, 'a')
                                        for line in mg_tagged_sentence_model4:
                                            line_model4_index += 1
                                            if line_model4_index == line_index:
                                                fields = line.split('\t')
                                                field_index = -1
                                                FIELDS = fields[3:]
                                                for field in FIELDS:
                                                    field_index += 1
                                                    if field_index % 2 == 0:
                                                        if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                                            MGtag = str(REF_MGST_table[field])
                                                        else:
                                                            MGtag = REF_MGT_table[field]
                                                        if MGtag in uni_stats_table4[backoff_cat3]:
                                                            prob = FIELDS[field_index+1]
                                                            if prob[-1] == '\n':
                                                                prob = prob[:-1]
                                                            prob = float(prob)
                                                            stats_table[line_index][MGtag] = prob
                                                break
                                    else:
                                        print "CCG category: "+ccg_tags[line_index]+" not seen in seed set (backing off also failed)..  Aborting parse..."
                                        terminal_output.write("CCG category: "+ccg_tags[line_index]+" not seen in seed set (backing off also failed)..  Aborting parse...\n")
                                        terminal_output.close()
                                        terminal_output = open(terminal_output_name, 'a')
                                        abort = True
                                        break
                                    break
                            if len(stats_table[line_index]) == 0 and ccg_tags[line_index] in uni_stats_table1:
                                for MGcat in uni_stats_table1[ccg_tags[line_index]]:
                                    #if a word has been paired with a given pair of CCG and MG categories in the seed set, and we
                                    #currently have no MG categories in the stats_table for this word, we just add all such MGcats with equal probability
                                    if MGcat not in stats_table[line_index]:
                                        if words[line_index].lower() in uni_stats_table1[ccg_tags[line_index]][MGcat][1]:
                                            stats_table[line_index][MGcat] = 1
                                if len(stats_table[line_index]) > 0:
                                    print "Failed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt and backoff models.. Using all MG categories from model 1 seen with that word and CCG category in the seeds with equal probability.."
                                    terminal_output.write("Failed to find satisfactory MG categories for for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt.. Using all MG categories from model 1 seen with that word and CCG category in the seeds with equal probability..\n")
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                                for MGcat in stats_table[line_index]:
                                    stats_table[line_index][MGcat] = 1/len(stats_table[line_index])
                            if len(stats_table[line_index]) == 0 and backoff_cat1 in uni_stats_table2:
                                #if we still have no MG cats then we try the same as above but using the first backoff category
                                for MGcat in uni_stats_table2[backoff_cat1]:
                                    if MGcat not in stats_table[line_index]:
                                        if words[line_index].lower() in uni_stats_table2[backoff_cat1][MGcat][1]:
                                            stats_table[line_index][MGcat] = 1
                                if len(stats_table[line_index]) > 0:
                                    print "Failed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt and backoff models.. Using all MG categories from model 2 seen with that word and CCG category in the seeds with equal probability.."
                                    terminal_output.write("Failed to find satisfactory MG categories for for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt.. Using all MG categories from model 2 seen with that word and CCG category in the seeds with equal probability..\n")
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                                for MGcat in stats_table[line_index]:
                                    stats_table[line_index][MGcat] = 1/len(stats_table[line_index])
                            if len(stats_table[line_index]) == 0 and backoff_cat2 in uni_stats_table3:
                                #if still no luck then try the same with the second backoff cat
                                for MGcat in uni_stats_table3[backoff_cat2]:
                                    if MGcat not in stats_table[line_index]:
                                        if words[line_index].lower() in uni_stats_table3[backoff_cat2][MGcat][1]:
                                            stats_table[line_index][MGcat] = 1
                                if len(stats_table[line_index]) > 0:
                                    print "Failed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt and backoff models.. Using all MG categories from model 3 seen with that word and CCG category in the seeds with equal probability.."
                                    terminal_output.write("Failed to find satisfactory MG categories for for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt.. Using all MG categories from model 3 seen with that word and CCG category in the seeds with equal probability..\n")
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                                for MGcat in stats_table[line_index]:
                                    stats_table[line_index][MGcat] = 1/len(stats_table[line_index])
                            if len(stats_table[line_index]) == 0:
                                if ccg_tags[line_index] in uni_stats_table1:
                                    print "\nFailed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt... backing off to unigram model 1..\n"
                                    terminal_output.write("\nFailed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using maxEnt... backing off to unigram model 1..\n")
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                                    #if we still have no MG categories for this word, but we did see its CCG tag in the seed set, we fall back to the unigram model 1
                                    for MGcat in uni_stats_table1[ccg_tags[line_index]]:
                                        stats_table[line_index][MGcat] = uni_stats_table1[ccg_tags[line_index]][MGcat][0]
                            if len(stats_table[line_index]) == 0:
                                if backoff_cat1 in uni_stats_table2:
                                    print "\nFailed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt... backing off to unigram model 2..\n"
                                    terminal_output.write("\nFailed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using maxEnt... backing off to unigram model 2..\n")
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                                    #if we still have no MG categories for this word, but we did see its CCG tag in the seed set, we fall back to the unigram model 2
                                    for MGcat in uni_stats_table2[backoff_cat1]:
                                        stats_table[line_index][MGcat] = uni_stats_table2[backoff_cat1][MGcat][0]
                            if len(stats_table[line_index]) == 0:
                                if backoff_cat2 in uni_stats_table3:
                                    print "\nFailed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using MaxEnt... backing off to unigram model 3..\n"
                                    terminal_output.write("\nFailed to find satisfactory MG categories for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" using maxEnt... backing off to unigram model 3..\n")
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                                    #if we still have no MG categories for this word, but we did see its CCG tag in the seed set, we fall back to the unigram model 3
                                    for MGcat in uni_stats_table3[backoff_cat2]:
                                        stats_table[line_index][MGcat] = uni_stats_table3[backoff_cat2][MGcat][0]
                            #we don't backoff to unigram model 4 as it will be too unreliable
                            added_cat = False
                            if ccg_tags[line_index] in uni_stats_table1:
                                cats_to_add = []
                                max_prob = 0
                                for MGCAT in stats_table[line_index]:
                                    if stats_table[line_index][MGCAT] > max_prob:
                                        max_prob = stats_table[line_index][MGCAT]
                                for MGcat in uni_stats_table1[ccg_tags[line_index]]:
                                    #if either 1. a word has been paired with a given pair of CCG and MG categories in the seed set, and the
                                    #CCG category and word are seen together in the test set or 2. if all_verb_cats is True and this is supertagging strategy
                                    #then we add all MG cats that were seen with the full CCG_PTB_PROBANK tag during training.. this
                                    #helps to combat the fact that the supertagger is not good at choosing verbs with relativization
                                    #and other A' movement, owing to the unbounded nature of such movement.. in eithet case, if the MG cat is not already
                                    #in the table, we add it with the lowest probability required for it to be considered..
                                    if MGcat not in stats_table[line_index]:
                                        if (words[line_index].lower() in uni_stats_table1[ccg_tags[line_index]][MGcat][1]) or ('SUPERTAG' in supertaggingStrategy and (all_cats or (all_verb_cats and '_VB' in ccg_tags[line_index] or '_MD' in ccg_tags[line_index]))):
                                            cats_to_add.append([MGcat, uni_stats_table1[ccg_tags[line_index]][MGcat][0]])
                                            added_cat = True
                                cats_to_add = sorted(cats_to_add, key=lambda x: x[1])
                                new_prob = round((max_prob*ccg_beam_floor) + 0.00001, 5)
                                if len(cats_to_add) > 0:
                                    last_uni_prob = cats_to_add[0][1]
                                #having collected additional categories from the unigram model, we rank them according to their unigram probability,
                                #then add the first one at the lowest value at which it will get into the beam, then increment that probability for each one by the same 0.00001
                                #amount that the beam is incremented in cky_mg.py.. in this way, all new categories will not suddenly just be added at once..
                                cat_ind = -1
                                for MGcat in cats_to_add:
                                    cat_ind+=1
                                    stats_table[line_index][MGcat[0]] = new_prob
                                    if len(cats_to_add)-1 > cat_ind and cats_to_add[cat_ind][1] > last_uni_prob:
                                        new_prob += 0.00001
                                        last_uni_prob = cats_to_add[cat_ind][1]
                            if not added_cat and backoff_cat1 in uni_stats_table2:
                                max_prob = 0
                                for MGCAT in stats_table[line_index]:
                                    if stats_table[line_index][MGCAT] > max_prob:
                                        max_prob = stats_table[line_index][MGCAT]
                                for MGcat in uni_stats_table2[backoff_cat1]:
                                    if MGcat not in stats_table[line_index]:
                                        if words[line_index].lower() in uni_stats_table2[backoff_cat1][MGcat][1]:
                                            stats_table[line_index][MGcat] = round((max_prob*ccg_beam_floor) + 0.00001, 5)
                            if not added_cat and backoff_cat2 in uni_stats_table3:
                                max_prob = 0
                                for MGCAT in stats_table[line_index]:
                                    if stats_table[line_index][MGCAT] > max_prob:
                                        max_prob = stats_table[line_index][MGCAT]
                                for MGcat in uni_stats_table3[backoff_cat2]:
                                    if MGcat not in stats_table[line_index]:
                                        if words[line_index].lower() in uni_stats_table3[backoff_cat2][MGcat][1]:
                                            stats_table[line_index][MGcat] = round((max_prob*ccg_beam_floor) + 0.0005, 4)
                            if len(stats_table[line_index]) == 0:
                                print "Failed to find any MG category for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" aborting parse..."
                                terminal_output.write("Failed to find any MG category for '"+words[line_index].lower()+"'... "+ccg_tags[line_index]+" aborting parse...\n")
                                terminal_output.close()
                                terminal_output = open(terminal_output_name, 'a')
                                abort = True
                                break
                        if abort:
                            continue
                    INDEX = -1
                    for terminal in terminals:
                        ccg_tag_text = ''
                        INDEX += 1
                        tagged_terminal = False
                        if 'CCG' in supertaggingStrategy:
                            ccg_tag_text ="     (CCG tag: "+ccg_tags[INDEX]+")"
                            MGcats = []
                            OmitOtherCats = False
                            tagged_terminal = True
                            if 'SUPERTAG' in supertaggingStrategy:
                                print "Assigning MG supertags to: '"+terminal.name.lower()+"'"+ccg_tag_text
                                terminal_output.write("Assigning MG supertags to: '"+terminal.name.lower()+"'"+ccg_tag_text+'\n')
                            elif 'HYBRID' in supertaggingStrategy:
                                print "Assigning MG hybrid supertags to: '"+terminal.name.lower()+"'"+ccg_tag_text
                                terminal_output.write("Assigning MG hybrid supertags to: '"+terminal.name.lower()+"'"+ccg_tag_text+'\n')
                            else:
                                print "Assigning MG tags to: '"+terminal.name.lower()+"'"+ccg_tag_text
                                terminal_output.write("Assigning MG tags to: '"+terminal.name.lower()+"'"+ccg_tag_text+'\n')
                            terminal_output.close()
                            terminal_output = open(terminal_output_name, 'a')
                            max_prob = 0
                            if "MAXENT" not in supertaggingStrategy:
                                found_source_cat = False
                                backoff_cat1 = "_".join(ccg_tags[INDEX].split("_")[:3])
                                backoff_cat2 = ccg_tags[INDEX].split("_")[0]
                                backoff_cat3 = "_".join(ccg_tags[INDEX].split("_")[1:])
                                if ccg_tags[INDEX] in stats_table1:
                                    stats_table = stats_table1
                                    source_cat = ccg_tags[INDEX]
                                    found_source_cat = True
                                elif backoff_cat1 in stats_table2:
                                    print "CCG category: "+ccg_tags[INDEX]+" not seen in seeds.. backing off to another model with category: "+backoff_cat1
                                    terminal_output.write("CCG category: "+ccg_tags[INDEX]+" not seen in seeds.. backing off to another model with category: "+backoff_cat1+'\n')
                                    stats_table = stats_table2
                                    source_cat = backoff_cat1
                                    found_source_cat = True
                                elif backoff_cat2 in stats_table3:
                                    print "CCG category: "+ccg_tags[INDEX]+" not seen in seeds.. backing off to another model with category: "+backoff_cat2
                                    terminal_output.write("CCG category: "+ccg_tags[INDEX]+" not seen in seeds.. backing off to another model with category: "+backoff_cat2+'\n')
                                    stats_table = stats_table3
                                    source_cat = backoff_cat2
                                    found_source_cat = True
                                elif backoff_cat3 in stats_table4:
                                    print "CCG category: "+ccg_tags[INDEX]+" not seen in seeds.. backing off to another model with category: "+backoff_cat3
                                    terminal_output.write("CCG category: "+ccg_tags[INDEX]+" not seen in seeds.. backing off to another model with category: "+backoff_cat3+'\n')
                                    stats_table = stats_table4
                                    source_cat = backoff_cat3
                                    found_source_cat = True
                                if found_source_cat:
                                    for MGcat in stats_table[source_cat]:
                                        if stats_table[source_cat][MGcat][0] > max_prob:
                                            max_prob = stats_table[source_cat][MGcat][0]
                                    for MGcat in sorted(stats_table[source_cat], key=itemgetter(0), reverse=True):
                                        if "SUPERTAG" not in supertaggingStrategy and 'HYBRID' not in supertaggingStrategy:
                                            MGCATS = miniOvertLexicon+MGcats
                                        else:
                                            MGCATS = supertags
                                        if len(MGCATS) > 0:
                                            if (INDEX+1) * max_mg_cats_per_word <= len(MGCATS):
                                                break
                                        #if we have seen a word linked with an MG tag/supertag and CCG category in the seeds (or optionally autos for supertags),
                                        #we include that MG cat with that word regardless of the MG|CCG category unigram probility
                                        #and we disallow any other categories.. this increases efficiency and should have very little
                                        #cost.. it prevents e.g. copula-adj, copula-pp, prog be and pass be from being added as soon as the beam is dropped as I originally had it
                                        #when copula-NP is required
                                        if terminal.name.lower() in stats_table[source_cat][MGcat][1]:
                                            if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                                #if we are dealing with supertags, we need to insert the string index into the over cats
                                                MGcatLit = ast.literal_eval(MGcat)
                                                if type(MGcatLit[0]) == type(()):
                                                    if MGcatLit[0][0] == 'OVERT_WORD':
                                                        MGcatLit[0] = (words[INDEX], MGcatLit[0][1], MGcatLit[0][2])
                                                    MGcatLit[2] = INDEX
                                                else:
                                                    for link in MGcatLit:
                                                        if link[0][0][0] == 'OVERT_WORD':
                                                            link[0][0] = (words[INDEX], link[0][0][1], link[0][0][2])
                                                        if link[0][2] == None:
                                                            link[0][2] = INDEX
                                                        if link[2][0][0] == 'OVERT_WORD':
                                                            link[2][0] = (words[INDEX], link[2][0][1], link[2][0][2])
                                                        if link[2][2] == None:
                                                            link[2][2] = INDEX
                                                supertags.append((MGcatLit, stats_table[source_cat][MGcat][0]))
                                                MGcats.append(MGcat)
                                            else:
                                                MGcats.append((MGcat, stats_table[source_cat][MGcat][0]))
                                            OmitOtherCats = True
                                    if len(MGcats) == 0:
                                        #we only include other MG categories at this stage that are within the beam if we didn't already find an exact match for that word..
                                        for MGcat in sorted(stats_table[source_cat], key=itemgetter(0), reverse=True):
                                            if "SUPERTAG" not in supertaggingStrategy and 'HYBRID' not in supertaggingStrategy:
                                                MGCATS = miniOvertLexicon+MGcats
                                            else:
                                                MGCATS = supertags
                                            if len(MGCATS) > 0:
                                                if (INDEX+1) * max_mg_cats_per_word <= len(MGCATS):
                                                    break
                                            if MGcat not in MGcats and stats_table[source_cat][MGcat][0] >= max_prob*ccg_beam:
                                                if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                                    MGcatLit = ast.literal_eval(MGcat)
                                                    if type(MGcatLit[0]) == type(()):
                                                        if MGcatLit[0][0] == 'OVERT_WORD':
                                                            MGcatLit[0] = (words[INDEX], MGcatLit[0][1], MGcatLit[0][2])
                                                        MGcatLit[2] = INDEX
                                                    else:
                                                        for link in MGcatLit:
                                                            if link[0][0][0] == 'OVERT_WORD':
                                                                link[0][0] = (words[INDEX], link[0][0][1], link[0][0][2])
                                                            if link[0][2] == None:
                                                                link[0][2] = INDEX
                                                            if link[2][0][0] == 'OVERT_WORD':
                                                                link[2][0] = (words[INDEX], link[2][0][1], link[2][0][2])
                                                            if link[2][2] == None:
                                                                link[2][2] = INDEX
                                                    supertags.append((MGcatLit, stats_table[source_cat][MGcat][0]))
                                                    MGcats.append(MGcat)
                                                else:
                                                    MGcats.append((MGcat, stats_table[source_cat][MGcat][0]))
                                    if not OmitOtherCats:
                                        #if we found an extact word and cat match then we dont include pruned categories, otherwise we do so they can be intruduced as the beam lowers
                                        for MGcat in stats_table[source_cat]:
                                            if MGcat not in MGcats:
                                                if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                                    MGcatLit = ast.literal_eval(MGcat)
                                                    if type(MGcatLit[0]) == type(()):
                                                        if MGcatLit[0][0] == 'OVERT_WORD':
                                                            MGcatLit[0] = (words[INDEX], MGcatLit[0][1], MGcatLit[0][2])
                                                        MGcatLit[2] = INDEX
                                                    else:
                                                        for link in MGcatLit:
                                                            if link[0][0][0] == 'OVERT_WORD':
                                                                link[0][0] = (words[INDEX], link[0][0][1], link[0][0][2])
                                                            if link[0][2] == None:
                                                                link[0][2] = INDEX
                                                            if link[2][0][0] == 'OVERT_WORD':
                                                                link[2][0] = (words[INDEX], link[2][0][1], link[2][0][2])
                                                            if link[2][2] == None:
                                                                link[2][2] = INDEX
                                                    prunedMGcats.append([max_prob, stats_table[source_cat][MGcat][0], MGcatLit])
                                                else:
                                                    lex = constructMGlexEntry(terminal.name.lower(), MGcat.split(" "))
                                                    prunedMGcats.append([[max_prob, stats_table[source_cat][MGcat][0], lex], INDEX])
                            else:
                                max_prob = 0
                                for MGcat in stats_table[INDEX]:
                                    if stats_table[INDEX][MGcat] > max_prob:
                                        max_prob = stats_table[INDEX][MGcat]
                                mgcats = []
                                for MGcat in stats_table[INDEX]:
                                    mgcats.append([MGcat, stats_table[INDEX][MGcat]])
                                for MGcat in sorted(mgcats, key = itemgetter(1), reverse=True):
                                    MGcat = MGcat[0]
                                    if 'SUPERTAG' in supertaggingStrategy or 'HYBRID' in supertaggingStrategy:
                                        MGcatLit = ast.literal_eval(MGcat)
                                        if type(MGcatLit[0]) == type(()):
                                            if MGcatLit[0][0] == 'OVERT_WORD':
                                                MGcatLit[0] = (words[INDEX], MGcatLit[0][1], MGcatLit[0][2])
                                            MGcatLit[2] = INDEX
                                        else:
                                            for link in MGcatLit:
                                                if link[0][0][0] == 'OVERT_WORD':
                                                    link[0][0] = (words[INDEX], link[0][0][1], link[0][0][2])
                                                if link[0][2] == None:
                                                    link[0][2] = INDEX
                                                if link[2][0][0] == 'OVERT_WORD':
                                                    link[2][0] = (words[INDEX], link[2][0][1], link[2][0][2])
                                                if link[2][2] == None:
                                                    link[2][2] = INDEX
                                        catLimitReached = False
                                        if len(supertags) > 0:
                                            if (INDEX+1) * max_mg_cats_per_word <= len(supertags):
                                                catLimitReached = True
                                        if stats_table[INDEX][MGcat] >= max_prob*ccg_beam and not catLimitReached:
                                            supertags.append((MGcatLit, stats_table[INDEX][MGcat]))
                                        else:
                                            prunedMGcats.append([max_prob, stats_table[INDEX][MGcat], MGcatLit])
                                    else:
                                        catLimitReached = False
                                        if len(miniOvertLexicon+MGcats) > 0:
                                            if (INDEX+1) * max_mg_cats_per_word <= len(miniOvertLexicon+MGcats):
                                                catLimitReached = True
                                        if stats_table[INDEX][MGcat] >= max_prob*ccg_beam and not catLimitReached:
                                            MGcats.append((MGcat, stats_table[INDEX][MGcat]))
                                        else:
                                            lex = constructMGlexEntry(terminal.name.lower(), MGcat.split(" "))
                                            prunedMGcats.append([[max_prob, stats_table[INDEX][MGcat], lex], INDEX])
                        else:
                            ccg_tag_text = ""
                        if not tagged_terminal:
                            print "Failed to find any MG category for '"+terminal.name.lower()+"'... "+ccg_tag_text+" aborting parse..."
                            terminal_output.write("Failed to find any MG category for '"+terminal.name.lower()+"'... "+ccg_tag_text+" aborting parse...\n")
                            terminal_output.close()
                            terminal_output = open(terminal_output_name, 'a')
                            abort = True
                            break
                        if 'SUPERTAG' not in supertaggingStrategy and 'HYBRID' not in supertaggingStrategy:
                            for MGcat in MGcats:
                                features = MGcat[0].split(" ")
                                #ignoring subcat frames for now and just adding everything to the lexicon..
                                lex = constructMGlexEntry(terminal.name, features)
                                if lex != None and lex not in miniOvertLexicon:
                                    miniOvertLexicon.append(([lex, INDEX], MGcat[1]))
                    if abort:
                        continue
                    if True:
                        try:
                            derivation_bracketings = []
                            if 'OVERT_MG' in supertaggingStrategy:
                                with timeout(TIMEOUT):
                                    start_time = default_timer()
                                    if parser_strategy == 'basicOnly':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=CovertLexicon, ExtraposerLexicon=ExtraposerLexicon, TypeRaiserLexicon=TypeRaiserLexicon, ToughOperatorLexicon=ToughOperatorLexicon, NullExcorporatorLexicon=NullExcorporatorLexicon, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, useAllNull=useAllNull, lexical_scoring=True, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                    elif parser_strategy == 'fullFirst':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=CovertLexicon, ExtraposerLexicon=ExtraposerLexicon, TypeRaiserLexicon=TypeRaiserLexicon, ToughOperatorLexicon=ToughOperatorLexicon, NullExcorporatorLexicon=NullExcorporatorLexicon, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, useAllNull=useAllNull, lexical_scoring=True, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                    elif parser_strategy == 'basicAndRight':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=CovertLexicon, ExtraposerLexicon=ExtraposerLexicon, TypeRaiserLexicon=TypeRaiserLexicon, ToughOperatorLexicon=ToughOperatorLexicon, NullExcorporatorLexicon=NullExcorporatorLexicon, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, useAllNull=useAllNull, lexical_scoring=True, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                    elif parser_strategy == 'basicAndExcorp':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=CovertLexicon, ExtraposerLexicon=ExtraposerLexicon, TypeRaiserLexicon=TypeRaiserLexicon, ToughOperatorLexicon=ToughOperatorLexicon, NullExcorporatorLexicon=NullExcorporatorLexicon, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, useAllNull=useAllNull, lexical_scoring=True, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                    elif parser_strategy == 'basicAndTough':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=CovertLexicon, ExtraposerLexicon=ExtraposerLexicon, TypeRaiserLexicon=TypeRaiserLexicon, ToughOperatorLexicon=ToughOperatorLexicon, NullExcorporatorLexicon=NullExcorporatorLexicon, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, useAllNull=useAllNull, lexical_scoring=True, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                    end_time = default_timer() - start_time
                                    times.append(int(end_time))
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                                    parse_times.append(parse_time)
                            elif 'SUPERTAG' in supertaggingStrategy:
                                with timeout(TIMEOUT):
                                    start_time = default_timer()
                                    if parser_strategy == 'basicOnly':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'fullFirst':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'basicAndRight':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'basicAndExcorp':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'basicAndTough':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=allowMoreGoals, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, SOURCE_SPANS=source_spans)
                                    end_time = default_timer() - start_time
                                    times.append(int(end_time))
                                    parse_times.append(parse_time)
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                            elif 'HYBRID' in supertaggingStrategy:
                                null_c_lexicon = []
                                for lexicon in [CovertLexicon, ExtraposerLexicon, TypeRaiserLexicon, ToughOperatorLexicon, NullExcorporatorLexicon]:
                                    get_null_c_lexicon(lexicon, null_c_lexicon)
                                with timeout(TIMEOUT):
                                    start_time = default_timer()
                                    if parser_strategy == 'basicOnly':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'fullFirst':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=True, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'basicAndRight':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=True, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'basicAndExcorp':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, allowMoreGoals=True, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                                    elif parser_strategy == 'basicAndTough':
                                        (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, xbar_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings, lex_scores) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True,allowMoreGoals=True, printPartialAnalyses=parserSettings['printPartialAnalyses'], limitRightwardMove=False, prunedMGcats=prunedMGcats, ccg_beam=ccg_beam, ccg_beam_floor=ccg_beam_floor, terminal_output=terminal_output, terminal_output_name=terminal_output_name, max_mg_cats_per_word=max_mg_cats_per_word, supertags=supertags, lexical_scoring=True, start_time=start_time, MOVEABLE_SPANS=moveable_spans, MAXMOVEDIST=maxMoveDist, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                                    end_time = default_timer() - start_time
                                    times.append(int(end_time))
                                    parse_times.append(parse_time)
                                    terminal_output.close()
                                    terminal_output = open(terminal_output_name, 'a')
                        except Exception as e:
                            print "Parser timeout or error!!!"
                            #for some reason, at this point sometimes the terminal_output file is closed..
                            #could be that the timeout kicks in between closing and reopening it above..
                            #we'll just open it again here..
                            succeeded = False
                            while not succeeded:
                                try:
                                    terminal_output = open(terminal_output_name, 'a')
                                    succeeded = True
                                except Exception as e:
                                    x=0
                            terminal_output.write("\nError!!!\n\n")
                            terminal_output.close()
                            terminal_output = open(terminal_output_name, 'a')
                            derivation_bracketings = []
                    if len(derivation_bracketings) > 0:
                        if len(derivation_bracketings) == 1:
                            tree_text = 'tree'
                        else:
                            tree_text = 'trees'
                        print "\nFound "+str(len(xbar_trees))+" MG "+tree_text+" for sentence:'"+" ".join(words)+", at fringe of PTB tree: "+autocorpus_folder+"/"+section_folder+"/"+ptb_file+" ln: "+str(index+1)+",  Total time taken for processing was: ", time_taken(end_time)
                        terminal_output.write("\nFound "+str(len(xbar_trees))+" MG "+tree_text+" for sentence:'"+" ".join(words)+", at fringe of PTB tree: "+autocorpus_folder+"/"+section_folder+"/"+ptb_file+" ln: "+str(index+1)+",  Total time taken for processing was:  "+str(time_taken(end_time))+'\n')
                        terminal_output.close()
                        terminal_output = open(terminal_output_name, 'a')
                        (best_trees, max_matched_mg_chains) = evaluate_trees(xbar_trees, PTB_deps, DepMappings, RevDepMappings, PTB_tree, terminals, False, derivation_bracketings, subcat_derivation_bracketings, terminal_output, lex_scores, xbar_bracketings, ccg_tree=ccg_tree)
                        if len(best_trees) > 1:
                            if len(best_trees) > 1:
                                eliminate_trees_with_certain_subcats(best_trees, subcat_derivation_bracketings)
                            if len(best_trees) > 1:
                                retain_trees_with_least_x(best_trees, xbar_bracketings, '[pro-')
                            if len(best_trees) > 1:
                                #to stop logophors being preferred over reflexives
                                retain_trees_with_least_x(best_trees, xbar_bracketings, '[self]')
                            if len(best_trees) > 1:
                                #make sure this comes after the least pro check as otherwise adjunct control will tend
                                #to be analyzed as involving pro as atb has an extra overt lexical node in the derivation tree
                                #meaning its probability is lower..
                                eliminate_lowest_lex_scoring_trees(best_trees, lex_scores, reduce_to=1)
                            if len(best_trees) > 1:
                                retain_trees_with_least_traces(best_trees, derived_bracketings)
                            if len(best_trees) > 1:
                                prefer_right_branching(best_trees, terminal_output)
                            if len(best_trees) > 1:
                                retain_smallest_trees(best_trees)
                        print "\nAdding best MG parse for PTB tree", autocorpus_folder+"/"+section_folder+"/"+ptb_file, "ln: ", index+1, "to auto corpus.."
                        terminal_output.write("\nAdding best MG parse for PTB tree "+autocorpus_folder+"/"+section_folder+"/"+ptb_file+" ln: "+str(index+1)+" to auto corpus..\n")
                        terminal_output.close()
                        terminal_output = open(terminal_output_name, 'a')
                        entries_to_delete = []
                        for entry in best_trees:
                            if entry[2] != max_matched_mg_chains:
                                entries_to_delete.append(entry)
                        for entry in entries_to_delete:
                            best_trees.remove(entry)
                        num_best_trees_added+=1
                        auto_set[index] = (subcat_derivation_bracketings[best_trees[0][1]], xbar_bracketings[best_trees[0][1]], derived_bracketings[best_trees[0][1]], subcat_full_derivation_bracketings[best_trees[0][1]], derivation_bracketings[best_trees[0][1]], full_derivation_bracketings[best_trees[0][1]], parserSettings['parserSetting'])
                        subcat_derivation_tree = gen_derived_tree.gen_derivation_tree(subcat_derivation_bracketings[best_trees[0][1]])
                        derivation_terminals = get_overt_derivation_terminals(subcat_derivation_tree, [])
                        if len(best_trees) > 1:
                            instances_multi_best += 1
                        best_trees = []
                    else:
                        print "\nNo MG parses found for PTB tree:", autocorpus_folder+"/"+section_folder+"/"+ptb_file, "ln:", index+1
                        terminal_output.write("\nNo MG parses found for PTB tree: "+autocorpus_folder+"/"+section_folder+"/"+ptb_file+" ln: "+str(index+1)+'\n')
                        terminal_output.close()
                        terminal_output = open(terminal_output_name, 'a')
                success = False
                while not success:
                    #making sure these json objects are saved properly and can be opened without crashing
                    try:
                        with open(autocorpus_folder+"/"+section_folder+"/"+ptb_file, 'w') as auto_file:
                            json.dump(auto_set, auto_file)
                        autoset = json.load(open(autocorpus_folder+"/"+section_folder+"/"+ptb_file))
                        with open(autocorpus_folder+"/"+"timeAndDate", 'w') as timeAndDateFile:
                            json.dump(timeAndDate, timeAndDateFile)
                        timeAndDate = json.load(open(autocorpus_folder+"/"+"timeAndDate"))
                        success = True
                    except Exception as e:
                        x=0
    print ""
    terminal_output.write("\n")
    if not start_file_reached:
        print "Start file not found! Make sure start file is in specified section folder list!"
        terminal_output.write("Start File not found! Make sure start file is in specified section folders!")
    else:
        try:
            timeAndDate = {'time':time.strftime("%H:%M:%S"), 'date':time.strftime("%d/%m/%Y")}
            print "Finished automatic generation at "+timeAndDate['time']+" on "+timeAndDate['date']
            terminal_output.write("Finished automatic generation at "+timeAndDate['time']+" on "+timeAndDate['date']+'\n')
            print "Total PTB trees considered: "+str(PARSES_CONSIDERED)
            terminal_output.write("Total PTB trees considered:  "+str(PARSES_CONSIDERED)+'\n')
            print "Total number of MG trees added to MGbank:  "+str(num_best_trees_added)
            terminal_output.write("Total number of MG trees added to MGbank:  "+str(num_best_trees_added)+'\n')
            print "Number of times there were multiple best trees:  "+str(instances_multi_best)
            terminal_output.write("Number of times there were multiple best trees:  "+str(instances_multi_best)+'\n')
            print "Total time taken to process corpus (including scoring): ", time_taken(default_timer() - total_start_time)
            terminal_output.write("Total time taken to process corpus (including scoring):  "+str(time_taken(default_timer() - total_start_time))+'\n')
            print "Total time taken just for parsing: ", time_taken(sum(parse_times))
            terminal_output.write("Total time taken just for parsing:  "+str(time_taken(sum(parse_times)))+'\n')
            print "Quickest time taken to parse a tree: ", time_taken(min(parse_times))
            terminal_output.write("Quickest time taken to parse a tree:  "+str(time_taken(min(parse_times)))+'\n')
            print "Slowest time taken to parse a tree: ", time_taken(max(parse_times))
            terminal_output.write("Slowest time taken to parse a tree:  "+str(time_taken(max(parse_times)))+'\n')
            print "Average (mean) time taken to parse each tree: ", time_taken(sum(parse_times)/len(parse_times))
            terminal_output.write("Average (mean) time taken to parse each tree:  "+str(time_taken(sum(parse_times)/len(parse_times)))+'\n\n')
            terminal_output.close()
        except ValueError:
            print "\nFinished..  No sentences were successfully parsed!!"
            terminal_output.write("\nFinished.. No sentences were successfully parsed!!")
            terminal_output.close()

def get_moveable_spans(PTB_tree, terminals, moveable_spans):
    set_indices(terminals)
    get_spans(PTB_tree, moveable_spans, legit_movers=True)
    for i in range(len(moveable_spans)):
        moveable_spans[i][1]+=1
    legit_terminal_movers = get_legit_terminal_mover_spans(terminals)
    spans_to_remove = []
    for span in moveable_spans:
        if span[0] == span[1]-1:
            if span not in legit_terminal_movers:
                spans_to_remove.append(span)
    for span in spans_to_remove:
        moveable_spans.remove(span)
    add_colon_spans(moveable_spans, terminals)
    add_quot_inv_extrap_spans(PTB_tree, moveable_spans)

def get_source_spans(PTB_tree, ccg_tree, source_spans, terminals):
    set_indices(terminals)
    #gets a set of spans from the PTB tree and (if available) the ccg tree which
    #are then used to constrain the MG parser..
    get_spans(PTB_tree, source_spans, include_all_words=True)
    if ccg_tree != None:
        get_spans(ccg_tree, source_spans)
    i=-1
    for span in source_spans:
        i+=1
        source_spans[i][1] += 1
    add_n_pp_spans(PTB_tree, source_spans)
    add_vp_minus_tmp_spans(PTB_tree, source_spans)
    add_sbar_minus_dis_spans(PTB_tree, source_spans)
    add_qp_spans(PTB_tree, source_spans)
    add_pre_det_spans(PTB_tree, source_spans)
    add_quot_inv_extrap_spans(PTB_tree, source_spans)
    add_vp_minus_apposition_spans(PTB_tree, source_spans)
    add_prt_spans(PTB_tree, source_spans)
    add_top_level_frag_spans(PTB_tree, source_spans)
    add_than_spans(PTB_tree, source_spans)
    add_coord_final_conj_spans(PTB_tree, source_spans)
    add_hyph_spans(PTB_tree, source_spans)
    get_final_det_spans(PTB_tree, source_spans)
    add_punc_spans(PTB_tree, source_spans)
    add_det_nnp_nnp_spans(PTB_tree, source_spans)

def add_qp_spans(node, source_spans):
    #QPs in PTB are completely different to those in MGbank, hence we must create
    #lots of additional source spans for them..including internally to the QP and also
    #linked with the nominal it modifies in a right branching config..
    if node.truncated_name == 'QP' and node.mother != None:
        if len(node.mother.daughters) == 1 and node.mother.truncated_name == 'NP' and node.mother.mother != None and node.mother.mother.truncated_name == 'NP':
            #there was a case where a QP was dominated by a unary NP, which was dominated by a binary NP it doesn't head..(sentence 0807 line 10)
            NODE = node.mother
        else:
            NODE = node
        qp_daughter_index = NODE.mother.daughters.index(NODE)
        qp_indices = set(NODE.indices)
        if qp_indices != set([]):
            qp_span = [min(qp_indices), max(qp_indices)+1]
            if len(NODE.mother.daughters) > qp_daughter_index+1 and ((NODE.mother.daughters[qp_daughter_index+1].truncated_name in ['NP', 'PP'] or 'NN' in NODE.mother.daughters[qp_daughter_index+1].truncated_name) and NODE.mother.daughters[qp_daughter_index+1].indices != []):
                np_indices = set(NODE.mother.daughters[qp_daughter_index+1].indices)
                if np_indices != set([]):
                    np_span = [min(np_indices), max(np_indices)+1]
                    for i in range(qp_span[1]-qp_span[0]):
                        new_span = [qp_span[0]+i, np_span[1]]
                        if new_span not in source_spans:
                            source_spans.append(new_span)
            for i in range(qp_span[1]-qp_span[0]):
                for j in range(qp_span[1]-qp_span[0]):
                    if j > i:
                        new_span = [qp_span[0]+i, qp_span[0]+j]
                        if new_span not in source_spans:
                            source_spans.append(new_span)
    for daughter in node.daughters:
        add_qp_spans(daughter, source_spans)

def add_than_spans(node, source_spans):
    #built for cases where in PTB 'than' is dominated by a ADVP that it doesn't head and there is material to the right that it should form a constituent with
    if node.mother != None and node.mother.mother != None and node.truncated_name == 'IN' and node.mother.truncated_name not in ['PP', 'SBAR', 'S'] and len(node.mother.daughters) > 1 and node == node.mother.daughters[-1]:
        p_indices = set(node.indices)
        if p_indices != set([]):
            p_span = [min(p_indices), max(p_indices)+1]
            node_mother_mother_indices = set(node.mother.mother.indices)
            if node_mother_mother_indices != set([]):
                node_mother_mother_span = [min(node_mother_mother_indices), max(node_mother_mother_indices)+1]
                new_span = [p_span[0], node_mother_mother_span[1]]
                if new_span not in source_spans:
                    source_spans.append(new_span)
    for daughter in node.daughters:
        add_than_spans(daughter, source_spans)

def add_coord_final_conj_spans(node, source_spans):
    found_coord = False
    for daughter in node.daughters:
        if daughter.truncated_name == 'CC':
            coord_indices = set(daughter.indices)
            if coord_indices != set([]):
                found_coord = True
                coord_span = [min(coord_indices), max(coord_indices)+1]
        elif found_coord and '-COORD' in daughter.name:
            #set found_coord back to False so that we can keep iterating in case of list coordination structure with multiple coordinators: Tom and Dick and Harry
            conj_indices = set(daughter.indices)
            if conj_indices != set([]):
                conj_span = [min(conj_indices), max(conj_indices)+1]
                found_coord = False
                new_span = [coord_span[0], conj_span[1]]
                if new_span not in source_spans:
                    source_spans.append(new_span)
    for daughter in node.daughters:
        add_coord_final_conj_spans(daughter, source_spans)

def add_hyph_spans(node, source_spans):
    #sometimes we need a hyphen to form a single constituent with the following constituent
    for daughter in node.daughters:
        if daughter.truncated_name == 'HYPH' and len(node.daughters) > 2 and daughter == node.daughters[-2]:
            hyph_indices = set(daughter.indices)
            if hyph_indices != set([]):
                hyph_span = [min(hyph_indices), max(hyph_indices)+1]
                hyph_comp_indices = set(node.daughters[-1].indices)
                if hyph_comp_indices != set([]):
                    hyph_comp_span = [min(hyph_comp_indices), max(hyph_comp_indices)+1]
                    new_span = [hyph_span[0], hyph_comp_span[1]]
                    if new_span not in source_spans:
                        source_spans.append(new_span)
    for daughter in node.daughters:
        add_hyph_spans(daughter, source_spans)

def add_punc_spans(node, source_spans):
    for daughter in node.daughters:
        if daughter.truncated_name in [":", "LRB", "RRB"]:
            punc_indices = set(daughter.indices)
            if punc_indices != set([]):
                punc_span = [min(punc_indices), max(punc_indices)+1]
                for span in source_spans:
                    if span[1] == punc_span[0]:
                        new_span = [span[0], punc_span[1]]
                        if new_span not in source_spans:
                            source_spans.append(new_span)
                    elif span[0] == punc_span[1]:
                        new_span = [punc_span[0], span[1]]
                        if new_span not in source_spans:
                            source_spans.append(new_span)
                    elif span[1] == punc_span[1]:
                        new_span = [span[0], punc_span[0]]
                        if new_span not in source_spans:
                            source_spans.append(new_span)
                    elif span[0] == punc_span[0]:
                        new_span = [punc_span[1], span[1]]
                        if new_span not in source_spans:
                            source_spans.append(new_span)
    for daughter in node.daughters:
        add_punc_spans(daughter, source_spans)

def add_pre_det_spans(node, source_spans):
    #sometimes in PTB you have DT DT flat sequenecs which are not corrected in CCGbank..
    index = -1
    det_indices = []
    for daughter in node.daughters:
        index += 1
        if daughter.truncated_name in ['DT', 'DTSG', 'DTPL', 'PDT', 'WP']:
            det_indices.append(index)
            while len(node.daughters) > index+1 and node.daughters[index+1].truncated_name in ['DT', 'DTSG', 'DTPL', 'PDT', 'WP']:
                index+=1
                det_indices.append(index)
            break
    if len(det_indices) > 1:
        np_indices = set(node.indices)
        np_span = [min(np_indices), max(np_indices)+1]
        for det_index in det_indices[1:]:
            new_span = [np_span[0]+det_index, np_span[-1]]
            if new_span not in source_spans:
                source_spans.append(new_span)
    for daughter in node.daughters:
        add_pre_det_spans(daughter, source_spans)

def add_top_level_frag_spans(node, source_spans):
    #do not traverse the whole tree! This should only apply to root frags..
    if node.truncated_name == 'FRAG':
        daughter_spans = []
        for daughter in node.daughters:
            daughter_indices = set(daughter.indices)
            if daughter_indices != set([]):
                daughter_span = [min(daughter_indices), max(daughter_indices)+1]
                daughter_spans.append(daughter_span)
        if len(daughter_spans) > 1:
            sen_len = daughter_spans[-1][1]
            for ds in daughter_spans[1:]:
                new_span_end = ds[1]
                for new_span_start in range(sen_len):
                    if new_span_start < new_span_end-1 and new_span_start < daughter_spans[-1][0]:
                        #-1 because word spans are already included
                        new_span = [new_span_start, new_span_end]
                        if new_span not in source_spans:
                            source_spans.append(new_span)
       
def add_n_pp_spans(node, source_spans):
    #originally just for NP with following PP adjunct in PTB but now extended for post nominal ADJP too, and also where there is a nominal adjective as the head 'noun'
    if node.truncated_name in ['PP', 'ADJP'] and node.mother != None and node.mother.truncated_name == 'NP' and len(node.mother.daughters) > 1:
        node_daughter_index = node.mother.daughters.index(node)
        pp_indices = set(node.indices)
        if pp_indices != set([]) and node_daughter_index != 0 and node.mother.daughters[node_daughter_index-1].truncated_name == 'NP' and ('NN' in node.mother.daughters[node_daughter_index-1].daughters[-1].truncated_name or node.mother.daughters[node_daughter_index-1].daughters[-1].truncated_name in ['JJ', 'NML']) and node.mother.daughters[node_daughter_index-1].daughters[-1].indices != []:
            pp_span = [min(pp_indices), max(pp_indices)+1]
            #as well as getting the n+pp or n+adjP span, we also need to take account of any modifers
            #such as compound nouns, adjectives, quantifiers etc..
            np_indices = set(node.mother.daughters[node_daughter_index-1].indices)
            np_span = [min(np_indices), max(np_indices)+1]
            for i in range(np_span[1]-np_span[0]):
                new_span = [np_span[0]+i, pp_span[1]]
                if new_span not in source_spans:
                    source_spans.append(new_span)
    for daughter in node.daughters:
        add_n_pp_spans(daughter, source_spans)

def add_det_nnp_nnp_spans(node, source_spans):
    if node.truncated_name == 'NP' and len(node.daughters) >= 3 and node.daughters[0].truncated_name == 'DT' and node.daughters[-1].truncated_name == 'NNP' and node.daughters[-2].truncated_name == 'NNP':
        det_index = set(node.daughters[0].indices)
        if det_index != set([]):
            node_indices = set(node.indices)
            end_pos = max(node_indices)+1
            for i in range(len(node.daughters)-2):
                daught_indices = set(node.daughters[-(i+2)].indices)
                if daught_indices != set([]):
                    start_pos = min(daught_indices)
                    new_span = [start_pos, end_pos]
                    source_spans.append(new_span)
    for daughter in node.daughters:
        add_det_nnp_nnp_spans(daughter, source_spans)
            
def add_quot_inv_extrap_spans(node, source_spans):
    #PTB has insane constituency for senetenecs like "[after all], [he says], [even to make love] [you] [need experience]", with very flat structures in which
    #all five of the bracketed items are sisters.. this corrects this somewhat by making everything after the PRN 'he says' node a constituent
    if node.truncated_name == 'S':
        prn_index = -1
        for daughter in node.daughters:
            prn_index += 1
            if 'PRN' in daughter.name and 'S' in [dau.truncated_name for dau in daughter.daughters]:
                spans_to_unify = []
                for DAUGHTER in node.daughters[prn_index:]:
                    if '-SBJ' in DAUGHTER.name or DAUGHTER.name[0] == 'S' or DAUGHTER.truncated_name == 'VP':
                        dau_indices = set(DAUGHTER.indices)
                        if dau_indices != set([]):
                            dau_span = [min(dau_indices), max(dau_indices)+1]
                            spans_to_unify.append(dau_span)
                if len(spans_to_unify) > 1:
                    new_span = [spans_to_unify[0][0], spans_to_unify[-1][1]]
                    if new_span not in source_spans:
                        source_spans.append(new_span)
    for daughter in node.daughters:
        add_quot_inv_extrap_spans(daughter, source_spans)

def add_prt_spans(node, source_spans):
    if node.truncated_name == 'VP':
        prt_index = -1
        for daughter in node.daughters:
            prt_index+=1
            if daughter.truncated_name == 'PRT' and daughter != node.daughters[-1]:
                prt_indices = set(daughter.indices)
                if prt_indices == set([]):
                    continue
                prt_span = [min(prt_indices), max(prt_indices)+1]
                spans_to_right_of_prt = []
                for DAUGHTER in node.daughters[prt_index+1:]:
                    d_indices = set(DAUGHTER.indices)
                    if d_indices != set([]):
                        d_span = [min(d_indices), max(d_indices)+1]
                        spans_to_right_of_prt.append(d_span)
                if len(spans_to_right_of_prt) > 0:
                    new_span = [prt_span[0], spans_to_right_of_prt[-1][-1]]
                    if new_span not in source_spans:
                        source_spans.append(new_span)
                    new_span = [prt_span[-1], spans_to_right_of_prt[-1][-1]]
                    if new_span not in source_spans:
                        source_spans.append(new_span)
    for daughter in node.daughters:
        add_prt_spans(daughter, source_spans)

def add_vp_minus_tmp_spans(node, source_spans):
    tmp_indices = set(node.indices)
    if tmp_indices != set([]) and 'TMP' in node.name and node.mother != None and node.mother.truncated_name == 'VP' and node == node.mother.daughters[-1]:
        tmp_span = [min(tmp_indices), max(tmp_indices)+1]
        current_node = node
        while current_node.mother.truncated_name == 'VP':
            vp_indices = set(current_node.mother.indices)
            if vp_indices != set([]):
                vp_span = [min(vp_indices), max(vp_indices)+1]
                new_span = [vp_span[0], tmp_span[0]]
                if new_span not in source_spans:
                    source_spans.append(new_span)
                current_node = current_node.mother
                if current_node.mother == None:
                    break
    for daughter in node.daughters:
        add_vp_minus_tmp_spans(daughter, source_spans)

def get_final_det_spans(node, source_spans):
    #sometimes an adverb modifies a determiner in PTB so you get a RB+DT constituent that will not
    #appear in the MG tree because heads can't be modified in MGbank at present
    if len(node.daughters) > 1 and node.mother != None and node.mother.truncated_name == 'NP' and node.daughters[-1].truncated_name == 'DT':
        top_np_node = node.mother
        while top_np_node.mother != None and top_np_node.mother.truncated_name == 'NP' and top_np_node in top_np_node.mother.heads:
            top_np_node = top_np_node.mother
        dt_indices = set(node.daughters[-1].indices)
        if dt_indices != set([]):
            dt_span = [min(dt_indices), max(dt_indices)+1]
            np_indices = set(top_np_node.daughters[-1].indices)
            if np_indices != set([]):
                np_span = [min(np_indices), max(np_indices)+1]
                new_span = [dt_span[0], np_span[1]]
                if new_span not in source_spans:
                    source_spans.append(new_span)
    for daughter in node.daughters:
        get_final_det_spans(daughter, source_spans)
    
def add_vp_minus_apposition_spans(node, source_spans):
    #sometimes in PTB, a constitiuent in apposition to the clause in a daughter of VP, rather than S/SBAR and is offset by a comma..
    #this function adds in spans as if it were attached to the S/SBAR node and all nodes in between
    app_indices = set(node.indices)
    if app_indices != set([]) and node.mother != None and node.mother.truncated_name == 'VP' and len(node.mother.daughters) > 1 and node == node.mother.daughters[-1] and node.mother.daughters[-2].name == ",":
        app_span = [min(app_indices), max(app_indices)+1]
        if node.mother != None:
            current_node = node.mother
        while current_node.truncated_name == 'VP' or current_node.name[0] == 'S':
            non_app_indices = set(current_node.indices)
            if non_app_indices != set([]):
                non_app_span = [min(non_app_indices), max(non_app_indices)+1]
                new_span = [non_app_span[0], app_span[0]]
                if new_span not in source_spans:
                    source_spans.append(new_span)
                if current_node.mother != None:
                    current_node = current_node.mother
                else:
                    break
    for daughter in node.daughters:
        add_vp_minus_apposition_spans(daughter, source_spans)

def add_sbar_minus_dis_spans(node, source_spans):
    dis_indices = set(node.indices)
    if dis_indices != set([]) and ('DIS' in node.name or node.get_terminal_heads()[0].truncated_name.lower() in discourse_adverbs) and node.mother != None and node.mother.truncated_name == 'VP' and node == node.mother.daughters[-1]:
        dis_span = [min(dis_indices), max(dis_indices)+1]
        if node.mother != None:
            current_node = node.mother
        while current_node.truncated_name == 'VP' or current_node.name[0] == 'S':
            non_dis_indices = set(current_node.indices)
            if non_dis_indices != set([]):
                non_dis_span = [min(non_dis_indices), max(non_dis_indices)+1]
                new_span = [non_dis_span[0], dis_span[0]]
                if new_span not in source_spans:
                    source_spans.append(new_span)
                if current_node.mother != None:
                    current_node = current_node.mother
                else:
                    break
    for daughter in node.daughters:
        add_sbar_minus_dis_spans(daughter, source_spans)

def get_spans(node, spans, legit_movers=False, include_all_words=False):
    #if legit_movers is True then this just returns XP and S spans, as these are the only ones that can move
    if node.indices != []:
        indices = set(node.indices)
        span = [min(indices), max(indices)]
        if include_all_words:
            if span not in spans:
                spans.append(span)
        elif not legit_movers:
            if span not in spans and span[0] != span[1]:
                #when scoring trees using PTB constituencies vs MG constituencies, there's no point in
                #using the words as constituents, as these all match for all trees
                spans.append(span)
        elif span not in spans and ((node.name == 'VB' and node.daughters[0].name.lower() == 'try') or node.truncated_name == 'LS' or node.truncated_name == 'WP$' or (node.mother != None and 'NN' in node.truncated_name and len(node.mother.daughters) > 1 and ":" in [da.name for da in node.mother.daughters]) or (node.mother != None and 'NN' in node.truncated_name and len(node.mother.daughters) == 2 and node.mother.daughters[-1].truncated_name == 'POS') or (node.truncated_name not in ['VBP', 'NNP', 'RP', 'WP', 'PRP'] and ('-MARK' in node.name or node.truncated_name == 'NML' or 'PRP' in node.truncated_name or node.truncated_name[-1] == 'P' or node.truncated_name[0] == 'S')) or (node.truncated_name in ['RB', 'DT', 'CC'] and node.daughters[0].name.lower() in ['both', 'all', 'either', 'neither', 'each', 'nor'])):
            #this is where we are looking at which constituents are allowed to undergo phrasal movement, which includes some words
            #so here we DO retain some word spans unlike above for scoring.
            spans.append(span)
    for daughter in node.daughters:
        get_spans(daughter, spans, legit_movers=legit_movers, include_all_words=include_all_words)

def add_colon_spans(moveable_spans, terminals):
    #adds colon spans onto the spans that appear to the left and right of the colon (creates new spans for these)
    index = -1
    new_spans = []
    for terminal in terminals:
        index += 1
        if terminal.mother.name == ":":
            for span in moveable_spans:
                if span[0] == index + 1:
                    new_span = [span[0]-1, span[1]]
                    new_spans.append(new_span)
                elif span[1] == index:
                    new_span = [span[0], span[1]+1]
                    new_spans.append(new_span)
    for span in new_spans:
        moveable_spans.append(span)

def get_legit_terminal_mover_spans(terminals):
    #This function identifies heads of phrases with overt dependents, since these should be unable to undergo
    #phrasal movement..
    legit_movers = []
    index = -1
    for terminal in terminals:
        index += 1
        legit = False
        current_node = terminal
        while current_node.mother != None:
            if len(current_node.mother.daughters) == 1:
                current_node = current_node.mother
            elif len(current_node.mother.daughters) > 1:
                if current_node.truncated_name == 'NP' and current_node.mother.truncated_name == 'NP' and ":" in [da.name for da in current_node.mother.daughters]:
                    legit = True
                    break
                if current_node.truncated_name in 'WP$' or (current_node.truncated_name == 'NP' and 'PRP' in current_node.daughters[0].truncated_name):
                    legit = True
                    break
                if current_node not in current_node.mother.heads and current_node not in current_node.mother.sem_heads:
                    legit = True
                    break
                if current_node.name == 'VB' and current_node.daughters[0].name.lower() == 'try':
                    legit = True
                    break
                dominates_other_terminals = False
                for daughter in current_node.mother.daughters:
                    if daughter == current_node:
                        continue
                    elif len(daughter.indices) > 0:
                        dominates_other_terminals = True
                        break
                if dominates_other_terminals:
                    break
                else:
                    current_node = current_node.mother
        if legit:
            legit_movers.append([index, index+1])
    return legit_movers

def set_indices(terminals):
    if terminals[0].indices != []:
        return
    #decorates each non-unary, non-root, non-terminal with all the index positions of the
    #terminals it dominates
    index = -1
    for terminal in terminals:
        index += 1
        current_node = terminal
        while current_node.mother != None:
            current_node = current_node.mother
            current_node.indices.append(index)

def eliminate_lowest_lex_scoring_trees(trees, lex_scores, reduce_to=1):
    if reduce_to >= len(trees):
        return
    len_trees = len(trees)
    tree_score_pairings = []
    for tree in trees:
        index = tree[1]
        tree_score_pairings.append([tree, lex_scores[index]])
    tree_score_pairings = sorted(tree_score_pairings, key=itemgetter(1), reverse=True)
    trees_retained = tree_score_pairings[:reduce_to]
    #we don't want to prune any trees that have the same score as the top one
    for tree in tree_score_pairings[reduce_to:]:
        if tree[1] == tree_score_pairings[0][1]:
            trees_retained.append(tree)
        else:
            break
    del(trees[:])
    for tree in trees_retained:
        trees.append(tree[0])
    if len(trees) < len_trees:
        print "\nRemoved", len_trees - len(trees), "trees with lower lexical scores..\n"

def eliminate_trees_with_certain_subcats(trees, subcat_derivation_bracketings):
    #sometimes we need a way to resolve draws and one way to do this is to disprefer
    #trees with particular subcat features..  For example, we can disprefer a lightverb construction
    #for an ordinary transitive sentence by retaining trees with the least number of LV features.
    for tree in trees:
        score = 0
        index = tree[1]
        derivation_bracketing = subcat_derivation_bracketings[index]
        subcatGroups = re.findall('{.*?}', derivation_bracketing)
        for subcatGroup in subcatGroups:
            subcat_group = subcatGroup[1:-1]
            subcat_group = subcat_group.split(".")
            for subcatFeature in subcat_group:
                if subcatFeature in dispreferred_subcats:
                    score += 1
        tree[0].dispreferred_subcats = score
    low_score = 99999999999
    for tree in trees:
        if tree[0].dispreferred_subcats < low_score:
            low_score = tree[0].dispreferred_subcats
    trees_to_remove = []
    for tree in trees:
        if tree[0].dispreferred_subcats > low_score:
            trees_to_remove.append(tree)
    for tree in trees_to_remove:
        trees.remove(tree)

def retain_trees_with_least_x(best_trees, xbar_bracketings, STRING):
    least_pro_score = 99999
    for best_tree in best_trees:
        bracketing = xbar_bracketings[best_tree[1]]
        pro_count = bracketing.count(STRING)
        if pro_count < least_pro_score:
            least_pro_score = pro_count
    trees_to_remove = []
    for best_tree in best_trees:
        bracketing = xbar_bracketings[best_tree[1]]
        if bracketing.count(STRING) > least_pro_score:
            trees_to_remove.append(best_tree)
    if trees_to_remove != []:
        print "\nRemoving", str(len(trees_to_remove)), "trees that have the greatest number of "+STRING+" heads..."
    for tree in trees_to_remove:
        best_trees.remove(tree)

def retain_smallest_trees(best_trees):
    lowest_node_count = 99999999999999
    for tree in best_trees:
        tree[0].node_count = get_node_count(tree[0], 0)
        if tree[0].node_count < lowest_node_count:
            lowest_node_count = tree[0].node_count
    trees_to_remove = []
    for tree in best_trees:
        if tree[0].node_count > lowest_node_count:
            trees_to_remove.append(tree)
    if trees_to_remove != []:
        print "\nRemoving", str(len(trees_to_remove)), "trees with a greater number of nodes..."
    for tree in trees_to_remove:
        best_trees.remove(tree)

def get_node_count(node, node_count):
    node_count += 1
    for daughter in node.daughters:
        node_count = get_node_count(daughter, node_count)
    return node_count

def retain_trees_with_least_traces(best_trees, derived_bracketings):
    least_trace_score = 99999
    for best_tree in best_trees:
        bracketing = derived_bracketings[best_tree[1]]
        trace_count = 0
        trace_count += bracketing.count("λ")
        trace_count += bracketing.count("ζ")
        trace_count += bracketing.count("Λ")
        trace_count += bracketing.count("μ")
        if trace_count < least_trace_score:
            least_trace_score = trace_count
    trees_to_remove = []
    for best_tree in best_trees:
        bracketing = derived_bracketings[best_tree[1]]
        trace_count = 0
        trace_count += bracketing.count("λ")
        trace_count += bracketing.count("ζ")
        trace_count += bracketing.count("Λ")
        trace_count += bracketing.count("μ")
        if trace_count > least_trace_score:
            trees_to_remove.append(best_tree)
    if trees_to_remove != []:
        print "\nRemoving", str(len(trees_to_remove)), "trees that have the greatest number of traces..."
    for tree in trees_to_remove:
        best_trees.remove(tree)

def evaluate_trees(xbar_trees, PTB_deps, DepMappings, RevDepMappings, PTB_tree, terminals, returnReifiedDepMappings, derivation_bracketings, subcat_derivation_bracketings, terminal_output, lex_scores, xbar_bracketings, ccg_tree=None):
    best_trees = []
    max_matched_mg_chains = 0
    checked_constituency = False
    checked_reverse_constituency = False
    pre_prune = False
    global use_deps
    original_len_xbar_trees = len(xbar_trees)
    if len(terminals) * len(xbar_trees) > 1000:
        #if we have too many candidate trees, then calculating all the dependencies takes up too much memory, so we first prune some of the trees away using other means
        pre_prune = True
    if pre_prune and returnReifiedDepMappings == False:
        print "\nAttempting to reduce number of trees using constituencies and other heuristics before scoring with dependencies..\n"
        if terminal_output != None:
            terminal_output.write("\nAttempting to reduce number of trees using constituencies and other heuristics before scoring with dependencies..\n")
        INDEX = -1
        BEST_TREES = []
        for xbar_tree in xbar_trees:
            INDEX+=1
            xbar_trees[INDEX] = (xbar_tree, INDEX)
            BEST_TREES.append((xbar_tree, INDEX)) 
        constituency_evaluation(BEST_TREES, terminals, PTB_tree, reverse=False, ccg_tree=ccg_tree)
        checked_constituency = True
        #if len(terminals) * len(BEST_TREES) > 1001:
            #checked_reverse_constituency = True
            #constituency_evaluation(BEST_TREES, terminals, PTB_tree, reverse=True, ccg_tree=ccg_tree)
        if len(terminals) * len(BEST_TREES) > 1001:
            retain_trees_with_least_x(BEST_TREES, xbar_bracketings, '[pro-')
        if len(terminals) * len(BEST_TREES) > 1001:
            #to stop logophors being preferred over reflexives, we do the following
            retain_trees_with_least_x(BEST_TREES, xbar_bracketings, '[self]')
        if lex_scores != None:
            #make sure this comes after least pro check as for adjunct control there is an extra overt
            #terminal in the derivation tree for the correct atb analysis meaning that pro analysis will be preferred
            if len(terminals) * len(BEST_TREES) > 1000:
                eliminate_lowest_lex_scoring_trees(BEST_TREES, lex_scores, reduce_to=int(math.ceil(1000/len(terminals))))
        if len(terminals) * len(BEST_TREES) > 1001:
            retain_smallest_trees(BEST_TREES)
        if original_len_xbar_trees > len(BEST_TREES):
            print "\nReduced candidates to "+str(len(BEST_TREES))+" trees..\n"
            if terminal_output != None:
                terminal_output.write("\nReduced candidates to "+str(len(BEST_TREES))+" trees.. now starting dependency scoring..\n")
        else:
            print "\nFailed to reduced the number of trees with other methods..\n"
            if terminal_output != None:
                terminal_output.write("Failed to reduced the number of trees with other methods..\n")
    try:
        if use_deps and ((pre_prune and len(BEST_TREES) > 1) or (not pre_prune and len(xbar_trees) > 1)):
            print "\nScoring candidates using PTB-MG dependency mappings...\n"
            if terminal_output != None:
                terminal_output.write("\nScoring candidates using PTB-MG dependency mappings...\n")
    except NameError:
        use_deps = True
        if use_deps and ((pre_prune and len(BEST_TREES) > 1) or (not pre_prune and len(xbar_trees) > 1)):
            print "\nScoring candidates using PTB-MG dependency mappings...\n"
            if terminal_output != None:
                terminal_output.write("\nScoring candidates using PTB-MG dependency mappings...\n")
    if not pre_prune:
        INDEX = -1
    else:
        index = -1
    if use_deps:
        for xbar_tree in xbar_trees:
            if not pre_prune:
                INDEX += 1
                index = INDEX
                len_cand = len(xbar_trees)
            else:
                INDEX = xbar_tree[1]
                if xbar_tree not in BEST_TREES:
                    continue
                len_cand = len(BEST_TREES)
                index += 1
                xbar_tree = xbar_tree[0]
            if (pre_prune and len(BEST_TREES) > 1) or (not pre_prune and len(xbar_trees) > 1):
                print "Calculating dependency score for tree "+str(index+1)+" of "+str(len_cand)+"...\n"
            matched_deps = 0
            add_truncated_names(xbar_tree)
            MG_terminals = get_MG_terminals(xbar_tree, terminals=[])
            MG_chains = get_chains(xbar_tree, chains={})
            for entry in MG_chains.keys():
                if len(MG_chains[entry]['antecedents']) > 1 or len(MG_chains[entry]['antecedents']) == 0 or len(MG_chains[entry]['traces']) == 0:
                    del(MG_chains[entry])
            for entry in MG_chains:
                for trace in MG_chains[entry]['traces']:
                    trace.chain_pointer = MG_chains[entry]['antecedents'][0]
            MG_deps = get_MGdeps(xbar_tree, MG_terminals, deps=[])
            (sent_dep_mappings, sent_reverse_dep_mappings) = get_dep_mappings(PTB_deps, MG_deps)
            sent_dep_mappings_head_head_word = copy.deepcopy(sent_dep_mappings)
            sent_dep_mappings_both_head_words = copy.deepcopy(sent_dep_mappings)
            sent_dep_mappings = remove_word_info_from_mapping(sent_dep_mappings, True)
            sent_dep_mappings_head_head_word = remove_word_info_from_mapping(sent_dep_mappings_head_head_word, True, True)
            sent_dep_mappings_both_head_words = remove_word_info_from_mapping(sent_dep_mappings_both_head_words, True, True, True)
            sent_reverse_dep_mappings_head_head_word = copy.deepcopy(sent_reverse_dep_mappings)
            sent_reverse_dep_mappings_both_head_words = copy.deepcopy(sent_reverse_dep_mappings)
            sent_reverse_dep_mappings = remove_word_info_from_mapping(sent_reverse_dep_mappings, True)
            sent_reverse_dep_mappings_head_head_word = remove_word_info_from_mapping(sent_reverse_dep_mappings_head_head_word, True, True)
            sent_reverse_dep_mappings_both_head_words = remove_word_info_from_mapping(sent_reverse_dep_mappings_both_head_words, True, True, True)
            sent_dep_mappings = sent_dep_mappings+sent_dep_mappings_head_head_word+sent_dep_mappings_both_head_words
            sent_reverse_dep_mappings = sent_reverse_dep_mappings+sent_reverse_dep_mappings_head_head_word+sent_reverse_dep_mappings_both_head_words
            sent_dep_mappings = json.dumps(sent_dep_mappings)
            sent_dep_mappings = json.loads(sent_dep_mappings)
            sent_reverse_dep_mappings = json.dumps(sent_reverse_dep_mappings)
            sent_reverse_dep_mappings = json.loads(sent_reverse_dep_mappings)
            matched_mg_chains = []
            matched_mappings = score_trees(sent_dep_mappings=sent_dep_mappings, DepMappings=DepMappings, matched_mg_chains=matched_mg_chains, matched_mappings=[])
            matched_mappings += score_trees(sent_dep_mappings=sent_reverse_dep_mappings, DepMappings=RevDepMappings, matched_mg_chains=matched_mg_chains, matched_mappings=[])
            #the score is all the number of matched mg chains for all levels of abstraction of dependency.. I also add in the
            #number of mappings between the source and target since this captures simple unlabelled
            #dependencies between words.
            sent_score = len(matched_mg_chains)+len(sent_dep_mappings)+len(sent_reverse_dep_mappings)
            if sent_score > max_matched_mg_chains:
                max_matched_mg_chains = sent_score
                best_trees = []
                best_trees.append((xbar_tree,INDEX,max_matched_mg_chains))
            elif sent_score == max_matched_mg_chains:
                best_trees.append((xbar_tree,INDEX,max_matched_mg_chains))
    else:
        INDEX = -1
        max_matched_mg_chains = 0
        best_trees = []
        len_cand = len(xbar_trees)
        for xbar_tree in xbar_trees:
            INDEX+=1
            best_trees.append((xbar_tree,INDEX,max_matched_mg_chains))
    if len_cand != 1 and len(best_trees) < len_cand:
        print "\nRemoved", len_cand - len(best_trees), " trees with lower dependency scores.. trees remaining:", len(best_trees), '\n'
        if terminal_output != None:
            terminal_output.write("\nRemoved "+str(len_cand - len(best_trees))+" trees with lower dependency scores.. trees remaining: "+str(len(best_trees))+'\n')
    elif len_cand > 1 and use_deps:
        print "\nAll trees have the same dependency scores!  Cannot remove any on this basis..\n"
        if terminal_output != None:
            terminal_output.write("\nAll trees have the same dependency scores!  Cannot remove any on this basis..\n")
    if not returnReifiedDepMappings and len(best_trees) > 1 and not checked_constituency:
        constituency_evaluation(best_trees, terminals, PTB_tree, ccg_tree=ccg_tree)
    #if not returnReifiedDepMappings and len(best_trees) > 1:
        #prefer_right_branching(best_trees, terminal_output)
    if not returnReifiedDepMappings and len(best_trees) > 1 and not checked_reverse_constituency:
        constituency_evaluation(best_trees, terminals, PTB_tree, reverse=True, ccg_tree=ccg_tree)
    if returnReifiedDepMappings:
        return (best_trees, max_matched_mg_chains, sent_dep_mappings_both_head_words+sent_reverse_dep_mappings_both_head_words, matched_mappings)
    return (best_trees, max_matched_mg_chains)

def prefer_right_branching(xbar_trees, terminal_output=None):
    #removes trees with less right branching
    rb_high_score = 0
    for xbar_tree in xbar_trees:
        #we don't want to favour trees with more null heads or traces which of course often have more right branching nodes in them.. so we subtract the count of these from the right branch count
        xbar_tree[0].rb_score = get_rb_score(xbar_tree[0], 0)
        if xbar_tree[0].rb_score > rb_high_score:
            rb_high_score = xbar_tree[0].rb_score
    trees_to_remove = []
    for xbar_tree in xbar_trees:
        if xbar_tree[0].rb_score < rb_high_score:
            trees_to_remove.append(xbar_tree)
    if len(trees_to_remove) > 0:
        print "\nRemoving "+str(len(trees_to_remove))+" trees with less right branching..\n"
        if terminal_output != None:
            terminal_output.write("\nRemoving "+str(len(trees_to_remove))+" trees with less right branching..\n")
    while len(trees_to_remove) > 0:
        xbar_trees.remove(trees_to_remove[0])
        del(trees_to_remove[0])

def get_rb_score(node, rb_score):
    if len(node.daughters) == 2 and not (len(node.daughters[0].daughters)==1 and (node.daughters[0].daughters[0].name in ['μ', 'ζ', 'λ', 'Λ'] or (node.daughters[0].daughters[0].name[0] == '[' and node.daughters[0].daughters[0].name[-1] == ']'))):
        current_node = node.daughters[1]
        if len(current_node.daughters) == 2:
            rb_score += 1
        else:
            while not len(current_node.daughters) == 0:
                current_node = current_node.daughters[0]
                if len(current_node.daughters) == 2:
                    rb_score += 1
                    break
    for daughter in node.daughters:
        rb_score = get_rb_score(daughter, rb_score)
    return rb_score

def constituency_evaluation(xbar_trees, terminals, PTB_tree, reverse=False, ccg_tree=None):
    #if the dependencies were not enough to reduce the candidates to a single tree
    #we use the constituency of the Penn Tree to filter out additional xbar trees
    set_indices(terminals)
    PTB_spans = []
    get_source_spans(PTB_tree, ccg_tree, PTB_spans, terminals)
    best_const_score = -99999999
    for xbar_tree in xbar_trees:
        MG_terminals = get_MG_terminals(xbar_tree[0], terminals=[])
        set_indices(MG_terminals)
        MG_spans = []
        get_spans(xbar_tree[0], MG_spans)
        for span in MG_spans:
            span[1] = span[1]+1
        xbar_tree[0].const_score = 0
        for span in MG_spans:
            if not reverse:
                if span in PTB_spans:
                    xbar_tree[0].const_score += 1
            else:
                if span not in PTB_spans:
                    xbar_tree[0].const_score -= 1
        if xbar_tree[0].const_score > best_const_score:
            best_const_score = xbar_tree[0].const_score
    trees_to_remove = []
    for xbar_tree in xbar_trees:
        if xbar_tree[0].const_score < best_const_score:
            trees_to_remove.append(xbar_tree)
    if trees_to_remove != []:
        print "\nFiltering out "+str(len(trees_to_remove))+" trees based on constituency.."
        while len(trees_to_remove) > 0:
            xbar_trees.remove(trees_to_remove[0])
            del(trees_to_remove[0])

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
    
def score_trees(sent_dep_mappings, DepMappings, matched_mg_chains, matched_mappings):
    seed_chain_keys = []
    for m in sent_dep_mappings:
        for sent_chain in m:
            #there's only ever 1 sent_chain in m
            sent_chain_lit = ast.literal_eval(sent_chain)
            #first we need to identify the correct ptb_chain_frame in DepMappings
            #which we can't do by just using == as this doesn't work owing to ordering
            #issues in python dictionaries
            matched_chain = None
            ind = 0
            for seed_chain in DepMappings:
                ind +=1
                if matched_chain != None:
                    if matched_chain not in seed_chain_keys:
                        seed_chain_keys.append(matched_chain)
                    break
                matched_chain = seed_chain
                seed_chain_lit = ast.literal_eval(seed_chain)
                if len(seed_chain_lit) != len(sent_chain_lit):
                    matched_chain = None
                    continue
                for sent_dep in sent_chain_lit:
                    if sent_dep not in seed_chain_lit:
                        matched_chain = None
                        break
                if ind == len(DepMappings):
                    #if we are at the end of DepMappings then we have to add the matched chain
                    #to seed_chain_keys now
                    if matched_chain != None:
                        if matched_chain not in seed_chain_keys:
                            seed_chain_keys.append(matched_chain)
                        break
    for seed_ptb_chain in seed_chain_keys:
        for m in sent_dep_mappings:
            for sent_chain in m:
                if seed_ptb_chain == sent_chain:
                    if [] in m[sent_chain] and [] in DepMappings[seed_ptb_chain]:
                        matched_mg_chains.append(seed_ptb_chain)
                        matched_mappings.append(m)
                        continue
                    else:
                        matched_chain = None
                    for seed_mg_chain in DepMappings[seed_ptb_chain]:
                        if matched_chain != None:
                            break
                        ind = 0
                        for sent_mg_chain in m[seed_ptb_chain]:
                            ind+=1
                            if matched_chain != None:
                                matched_mg_chains.append(seed_ptb_chain)
                                matched_mappings.append(m)
                                break
                            matched_chain = seed_mg_chain
                            if len(sent_mg_chain) != len(seed_mg_chain):
                                matched_chain = None
                            for sent_dep in sent_mg_chain:
                                if sent_dep not in seed_mg_chain:
                                    matched_chain = None
                                    break
                            if ind == len(m[seed_ptb_chain]):
                                if matched_chain != None:
                                    matched_mg_chains.append(seed_ptb_chain)
                                    matched_mappings.append(m)
                                    break
    return matched_mappings

def del_word_info(dep):
    dep_copy = copy.deepcopy(dep)
    del(dep_copy['head_word'])
    del(dep_copy['head_word_span'])
    del(dep_copy['non_head_word'])
    del(dep_copy['non_head_word_span'])
    return dep_copy

def get_overt_derivation_terminals(derivation_node, derivation_terminals):
    if len(derivation_node.daughters) == 0:
        if derivation_node.name[0] != '[':
            derivation_terminals.append(" ".join(derivation_node.name.split(" ")[1:]))
    else:
        for daughter in derivation_node.daughters:
            derivation_terminals = get_overt_derivation_terminals(daughter, derivation_terminals)
    return derivation_terminals

def remove_word_info_from_mapping(mapping, return_as_list=False, retain_head_head_word=False, retain_both_head_words=False):
    if return_as_list:
        #this is the case where the system is trying to score a particular sentence,
        #so if the same ptb dep appears twice, we want to record that so the parse can get 2 points
        list_of_mappings=[]
    new_mapping = {}
    for ptb_chain in mapping:
        ptb_chain_lit = ast.literal_eval(ptb_chain)
        for ptb_dep in ptb_chain_lit:
            if retain_both_head_words:
                if ptb_dep['head_word'] != wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='v'):
                    ptb_dep['head_word'] = wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='v')
                elif ptb_dep['head_word'] != wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='n'):
                    ptb_dep['head_word'] = wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='n')
                if ptb_dep['non_head_word'] != wordnet_lemmatizer.lemmatize(ptb_dep['non_head_word'], pos='v'):
                    ptb_dep['non_head_word'] = wordnet_lemmatizer.lemmatize(ptb_dep['non_head_word'], pos='v')
                elif ptb_dep['non_head_word'] != wordnet_lemmatizer.lemmatize(ptb_dep['non_head_word'], pos='n'):
                    ptb_dep['non_head_word'] = wordnet_lemmatizer.lemmatize(ptb_dep['non_head_word'], pos='n')
                ptb_dep['reification'] = 'both'
            elif retain_head_head_word:
                if ptb_dep['head_word'] != wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='v'):
                    ptb_dep['head_word'] = wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='v')
                elif ptb_dep['head_word'] != wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='n'):
                    ptb_dep['head_word'] = wordnet_lemmatizer.lemmatize(ptb_dep['head_word'], pos='n')
                ptb_dep['reification'] = 'head'
            else:
                ptb_dep['reification'] = None
            if not retain_head_head_word and not retain_both_head_words:
                del(ptb_dep['head_word'])
            del(ptb_dep['head_word_span'])
            if not retain_both_head_words:
                del(ptb_dep['non_head_word'])
            del(ptb_dep['non_head_word_span'])
        if not return_as_list:
            if str(ptb_chain_lit) not in new_mapping:
                new_mapping[str(ptb_chain_lit)] = mapping[ptb_chain]
            else:
                for mg_chain in mapping[ptb_chain]:
                    if mg_chain not in new_mapping[str(ptb_chain_lit)]:
                        new_mapping[str(ptb_chain_lit)].append(mg_chain)
        else:
            list_of_mappings.append({str(ptb_chain_lit):mapping[ptb_chain]})
            MG_MAPPINGS = list_of_mappings[-1][str(ptb_chain_lit)]
            for MG_MAPPING in MG_MAPPINGS:
                for mg_dep in MG_MAPPING:
                    if retain_both_head_words:
                        if mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v'):
                            mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v')
                        elif mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n'):
                            mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n')
                        if mg_dep['non_head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='v'):
                            mg_dep['non_head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='v')
                        elif mg_dep['non_head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='n'):
                            mg_dep['non_head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='n')
                        mg_dep['reification'] = 'both'
                    elif retain_head_head_word:
                        if mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v'):
                            mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v')
                        elif mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n'):
                            mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n')
                        mg_dep['reification'] = 'head'
                    else:
                        mg_dep['reification'] = None
                    if not retain_head_head_word and not retain_both_head_words:
                        del(mg_dep['head_word'])
                    del(mg_dep['head_word_span'])
                    if not retain_both_head_words:
                        del(mg_dep['non_head_word'])
                    del(mg_dep['non_head_word_span'])
    if not return_as_list:
        for ptb_chain in new_mapping:
            for mg_chain in new_mapping[ptb_chain]:
                for mg_dep in mg_chain:
                    try:
                        if retain_both_head_words:
                            if mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v'):
                                mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v')
                            elif mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n'):
                                mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n')
                            if mg_dep['non_head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='v'):
                                mg_dep['non_head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='v')
                            elif mg_dep['non_head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='n'):
                                mg_dep['non_head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['non_head_word'], pos='n')                                
                            mg_dep['reification'] = 'both'
                        elif retain_head_head_word:
                            if mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v'):
                                mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='v')
                            elif mg_dep['head_word'] != wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n'):
                                mg_dep['head_word'] = wordnet_lemmatizer.lemmatize(mg_dep['head_word'], pos='n')
                            mg_dep['reification'] = 'head'
                        else:
                            mg_dep['reification'] = None
                        if not retain_head_head_word and not retain_both_head_words:
                            del(mg_dep['head_word'])
                        del(mg_dep['head_word_span'])
                        if not retain_both_head_words:
                            del(mg_dep['non_head_word'])
                        del(mg_dep['non_head_word_span'])
                    except KeyError:
                        #a key error here means that we have the same dep listed in this chain
                        #twice.. in fact it seems to be exactly the same dep so when we removed the
                        #head info first time around it already got removed this time..
                        raise Exception("Oops, for some reason head info has already been deleted and its not owing to copies of the same deps inside a chain")
        chains = []
        duplicates = []
        for ptb_chain in new_mapping:
            for mg_chain in new_mapping[ptb_chain]:
                if mg_chain not in chains:
                    chains.append(mg_chain)
                else:
                    duplicates.append(mg_chain)
            for dup in duplicates:
                new_mapping[ptb_chain].remove(dup)
            chains = []
            duplicates = []
    if return_as_list:
        return list_of_mappings
    else:
        return new_mapping

def constructMGlexEntry(word, features):
    #get the features
    while '' in features:
        #in case the user entered in an extra space between features
        features.remove('')
    if features == []:
    #if no features were added for a word, we don't create an entry
    #because it cannot possibly partake in syntactic operations..
        return None
    #identify the selectee feature, if any..
    selectee_feature = ''
    for feature in features[1:]:
        #we need to strip off the subcat stuff as this contains + and - not relevant to the
        #determining if this is a selectee feature and also not included on the PoS feature..
        FEATURE = re.sub('{.*}', '', feature)
        #need to get rid of the pesky unicode introduced by the coord separator,
        #as the parser doesn't like it
        try:
            selectee_feature = FEATURE.encode('ascii')
        except UnicodeEncodeError:
            #doesn't work for adjuncts 
            selectee_feature = FEATURE
        not_cat = not_selectee_feature.search(selectee_feature)
        if not_cat or u'd\u2248' in selectee_feature:
            selectee_feature = ''
        else:
            break
    #determine whether this is a coordinator
    if features[0] == u':\u0305:\u0305' or features[0] == ':\u0305:\u0305':
        conj = "-conj"
    else:
        conj = ""
    features = features[1:]
    findex=-1
    for feature in features:
        findex+=1
        try:
            features[findex] = feature.encode('ascii')
        except UnicodeEncodeError:
            features[findex] = feature
    #now put it all together into a lexical entry the parser will recognize..
    entry = (word, features, selectee_feature.lower()+conj)
    return entry

def get_chains(tree, chains):
    #returns a set of chains, where each chain has an antecedent(s) node and a number of trace nodes
    if tree.terminal:
        return chains
    else:
        tags = tree.name.split("-")
        #setting indices as a list just so we can double check there' only one per node..
        #so we'll only ever use indices[0]..
        indices = []
        for tag in tags:
            try:
                int(tag)
                indices.append(tag)
            except ValueError:
                x=0
        if len(indices) > 1:
            print "Watch out!!!!!!!! Some nodes still have multipe indices!!!"
        if len(indices) > 0:
            if indices[0] not in chains:
                chains[indices[0]] = {'antecedents':[], 'traces':[]}
            if is_antecedent(tree):#"-NONE-" not in [d.name for d in tree.daughters] and 'Λ' not in [d.name for d in tree.daughters] and 'λ' not in [d.name for d in tree.daughters] and 'μ' not in [d.name for d in tree.daughters] and 'ζ' not in [d.name for d in tree.daughters]:
                chains[indices[0]]['antecedents'].append(tree)
            else:
                chains[indices[0]]['traces'].append(tree)
        for daughter in tree.daughters:
            chains = get_chains(daughter, chains)
        return chains

def is_antecedent(node):
    if len(node.daughters) == 1:
        if node.daughters[0].name in ['Λ', 'λ', 'μ', 'ζ', '-NONE-']:
            return False
    elif len(node.daughters) == 2:
        #moved sentences in the treebank for some reason still have the null complementizer at the extraction site.. so we
        #have to be careful to check for this as it, like a trace, has -NONE- tag
        if node.daughters[0].name == '-NONE-' and len(node.daughters[1].daughters) == 1 and node.daughters[1].daughters[0].name == '-NONE-':
            return False
    return True

def get_dep_mappings(PTB_deps, MG_deps):
    #looks for matching dependencies in the MG and PTB trees.. works by matching the head and
    #non head word spans.. also looks for where the head and dependent in the MG tree are reversed
    #relative to the PTB tree, since during seed set creation we can record this fact and know that this
    #particular dependency is/can be reversed in future..
    dep_mappings = {}
    reverse_dep_mappings = {}
    #both of these matching_deps dictionaries have the PTB deps as a key and a list of
    #matching MG_deps as values..
    PTB_chains = {}
    for PTB_dep in PTB_deps:
        #we don't want to keep the dummy deps around.. we only needed them for PosDepsMappings
        #so we could ensure we always match MG preterminals to PTB preterminals correctly. Keeping
        #the dummy deps around here mucks up the scoring for the reverse deps.. long story!
        if PTB_dep['non_head_word_span'] == [-1, -1]:
            continue
        #we start by gathering together all the PTB_deps with the same head and dependency word..
        #once I have dealt with indexed traces in the PTB, it could often be the case that two PTB
        #deps map to two or more MG deps..
        spans = str((PTB_dep['head_word_span'], PTB_dep['non_head_word_span']))
        if spans not in PTB_chains:
            PTB_chains[spans] = [copy.deepcopy(PTB_dep)]
        else:
            PTB_chains[spans].append(copy.deepcopy(PTB_dep))
    #now we do the same for MG_deps except here we also construct MG reverse chains..
    #to catch the reverse deps, all we need to do is simply reverse every MG dep and then
    #try matching these with the forward PTB deps.. any which matched as forward deps will
    #now not match, but any which previously did not, will now match..
    MG_chains = {}
    MG_reverse_chains = {}
    for MG_dep in MG_deps:
        if MG_dep['non_head_word_span'] == [-1, -1]:
            continue
        spans = str((MG_dep['head_word_span'], MG_dep['non_head_word_span']))
        reverse_spans = str((MG_dep['non_head_word_span'], MG_dep['head_word_span']))
        if spans not in MG_chains:
            MG_chains[spans] = [copy.deepcopy(MG_dep)]
        else:
            MG_chains[spans].append(copy.deepcopy(MG_dep))
        if reverse_spans not in MG_reverse_chains:
            MG_reverse_chains[reverse_spans] = [copy.deepcopy(MG_dep)]
        else:
            MG_reverse_chains[reverse_spans].append(copy.deepcopy(MG_dep))
    #now, to get our PTB to MG dependency mappings, we iterate over the PTB chains
    #and find any matching MG chains.. if there are none, then we record the fact that
    #the PTB chain does not map to any MG chain.. this is important, since it gives
    #the system a way to choose the correct MG tree even if some PTB deps are missing in
    #it..(I'm not sure yet if this will ever be the case, but we want the option of allowing it).
    for PTB_chain in PTB_chains:
        if PTB_chain in MG_chains:
            #remember that here 'PTB_chain' and 'MG_chain' refer to spans, ie the keys
            dep_mappings[str(PTB_chains[PTB_chain])] = [MG_chains[PTB_chain]]
        #we also need to check that the reverse dep doesn't match before we enter a
        #null mapping into dep_mappings..
        elif PTB_chain in MG_reverse_chains:
            reverse_dep_mappings[str(PTB_chains[PTB_chain])] = [MG_reverse_chains[PTB_chain]]
    return (dep_mappings, reverse_dep_mappings)

def get_PTBdeps(PTB_node, terminals, deps, head_type, returnSynDeps=True,returnSemDeps=True):
    if PTB_node.terminal == True or (len(PTB_node.daughters) == 1 and PTB_node.daughters[0].terminal == True):
        return deps
    if head_type == 'syntactic':
        HEADS = PTB_node.heads
    elif head_type == 'semantic':
        HEADS = PTB_node.sem_heads
    #traverses a PTB tree top down and returns a list of all dependencies
    #in the tree.  Each depedency is a python dictionary with the following keys:
    #parent_cat, head_child_cat, head_word, head_word_span, non_head_child_cat,
    #non_head_word, non_head_word_span, direction, relations.
    parent_cat = PTB_node.truncated_name
    for head_child in HEADS:
        #there could be multiple head children, as in coordinate structures where each conjunct
        #will be treated as a head (in 'Jack likes honey and lemon', it is more useful
        #to regard 'honey' and 'lemon' as dependents of 'likes' than to regard 'and' as the dependent).
        head_child_cat = head_child.truncated_name
        if returnSynDeps:
            head_child_head_words = head_child.get_terminal_heads(normalize_terminals=True, head_type='syntactic')
        else:
            head_child_head_words = []
        if returnSemDeps:
            sem_head_child_head_words = head_child.get_terminal_heads(normalize_terminals=True, head_type='semantic')
        else:
            sem_head_child_head_words = []
        for shchw in sem_head_child_head_words:
            if shchw not in head_child_head_words:
                head_child_head_words.append(shchw)
        for head_child_head_word in head_child_head_words:
            #in turn, each head child could have multiple lexical heads owing again to coordination..
            head_word = head_child_head_word.name
            try:
                head_word_span = [terminals.index(head_child_head_word), terminals.index(head_child_head_word)+1]#1
            except ValueError:
                continue#we have to ignore dependencies with null items as head or dependent.. this is because there could be more than one null head/dependent in position [2,2] or [3,3] etc which screws things up when collecting dependency mappings
            for daughter in PTB_node.daughters:
                if daughter in HEADS:
                    continue
                else:
                    #direction records the direction of the non-head child relative to the head child
                    if PTB_node.daughters.index(daughter) < PTB_node.daughters.index(head_child):
                        direction = 'left'
                    else:
                        direction = 'right'
                    non_head_child_cat = daughter.truncated_name
                    if returnSynDeps:
                        non_head_child_head_words = daughter.get_terminal_heads(normalize_terminals=True, head_type='syntactic')
                    else:
                        non_head_child_head_words = []
                    if returnSemDeps:
                        sem_non_head_child_head_words = daughter.get_terminal_heads(normalize_terminals=True, head_type='semantic')
                    else:
                        sem_non_head_child_head_words = []
                    for snhchw in sem_non_head_child_head_words:
                        if snhchw not in non_head_child_head_words:
                            non_head_child_head_words.append(snhchw)
                    for non_head_child_head_word in non_head_child_head_words:
                        #again, there could be multiple head words inside the dependent
                        #owing to coordination, so we create a separate dependency relation
                        #for each dependent head.
                        non_head_word = non_head_child_head_word.name
                        try:
                            non_head_word_span = [terminals.index(non_head_child_head_word), terminals.index(non_head_child_head_word)+1]
                        except ValueError:
                            continue
                        #we now need to extract all the thematic and modificational relations..
                        relations = extract_propbank_relations(daughter.name, head_word_span)
                        ignore_dep = False
                        dep = Dict(
                            parent_cat = parent_cat,
                            head_child_cat = head_child_cat,
                            head_word = head_word,
                            head_word_span = head_word_span,
                            non_head_child_cat = non_head_child_cat,
                            non_head_word = non_head_word,
                            non_head_word_span = non_head_word_span,
                            direction = direction,
                            relations = relations
                            )
                        if dep['non_head_word'] != None:
                            #we'll ignore punctuation because it's messy..
                            deps.append(dep)
    for daughter in PTB_node.daughters:
        deps = get_PTBdeps(daughter,terminals,deps,head_type,returnSynDeps=returnSynDeps,returnSemDeps=returnSemDeps)
    return deps

def get_MGdeps(MG_node, terminals, deps, returnSynDeps=True, returnSemDeps=True):
    if len(MG_node.daughters) == 1:
        if MG_node.daughters[0].name in ['Λ', 'λ', 'μ', 'ζ']:
            return deps
    if MG_node.terminal == True or (len(MG_node.daughters) == 1 and MG_node.daughters[0].terminal == True):
        #if this is a terminal or preterminal, we stop
        return deps
    if MG_node.sem_node and not MG_node.phon_node:
        #when we hit a sem node, we don't to loop back into the antecedent or
        #we duplicate deps
        return deps
    #traverses a MG tree top down and returns a list of all dependencies
    #in the tree.  Each depedency is a python dictionary with the following keys:
    #parent_cat, head_child_cat, head_word, head_word_span, non_head_child_cat,
    #non_head_word, non_head_word_span, direction.  For MG trees, both syntactic and
    #semantic dependencies are returned.. (e.g. the syntactic head of CP is C,
    #but the semantic head of CP is V).
    parent_cat = MG_node.truncated_name
    #first we deal with the syntactic dependencies..
    for head_child in MG_node.heads:
        if not returnSynDeps:
            continue
        #there could be multiple head children, as in coordinate structures where each conjunct
        #will be treated as a head (in 'Jack likes honey and lemon', it is more useful
        #to regard 'honey' and 'lemon' as dependents of 'likes' than to regard 'and' as the dependent).
        #NOTE: I think its only for the semantic heads that we treat the conjuncts as heads
        head_child_cat = head_child.truncated_name
        head_child_head_words = head_child.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
        while '' in [hw.name for hw in head_child_head_words]:
            #we want to ignore the deleted semantic node remnants that occur after pf movement
            for hw in head_child_head_words:
                if hw.name == '':
                    head_child_head_words.remove(hw)
                    break               
        head_found = False
        new_daughter = None
        if len(MG_node.daughters) == 1:
            new_daughter = MG_node.daughters[0]
        else:
            new_daughter_set = False
            for d in MG_node.daughters:
                if d in MG_node.heads and not d.top_level_head_node:
                    new_daughter = d
                    new_daughter_set = True
            if not new_daughter_set:
                if (len(d.name) >= 2 and d.name[:2] == 'AP') or 'punc' in d.name or 'Punc' in d.name:
                    adjunct = True
                else:
                    adjunct = False
                for d in MG_node.daughters:
                    if d not in MG_node.heads and not d.top_level_head_node and not adjunct:
                        new_daughter = d
                        new_daughter_set = True
                if not new_daughter_set:
                    new_daughter = None
        for head_child_head_word in head_child_head_words:
            original_hws = head_child_head_words
            original_hw = head_child_head_word
            while not head_found and len(head_child_head_words) > 0 and new_daughter != None:
                for hw in head_child_head_words:
                    if hw.name != '' and not (hw.name[0] == '[' and 'pro' not in hw.name) and hw.name != 'Λ' and hw.name != 'λ' and hw.name != 'μ' and hw.name != 'ζ':
                        head_found = True
                        head_child_head_word = hw
                        break                
                if not head_found and new_daughter != None:
                    if len(new_daughter.daughters) == 1:
                        new_daughter = new_daughter.daughters[0]
                        #this will not change the top level head_child_head_words that we are iterating over, which is
                        #important because for conjunction we may have two heads (though I think this is just for semantic heads anyway
                        #and we don't bother with all this there because they go down to the bottom of the extended
                        #projection anyway, which is very rarely a null [head], and even if it were there'd be nowhere
                        #to go.
                        head_child_head_words = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                        continue
                    elif new_daughter.name == 'Λ':
                        new_daughter = new_daughter.antecedent
                    elif new_daughter.name == 'λ' or new_daughter.name == 'μ':
                        new_daughter = new_daughter.heads[0]
                    else:
                        new_daughter_set = False
                        for d in new_daughter.daughters:
                            if d in new_daughter.heads and not d.top_level_head_node:
                                #.top_level_head_node tests to see if this is a complex or simple head node, in which case we do not want to continue
                                #down the head path as we already know it leads to a [null]
                                new_daughter = d
                                new_daughter_set = True
                                head_child_head_words = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                while '' in [hw.name for hw in head_child_head_words]:
                                    #we want to ignore the deleted semantic node remnants that occur after pf movement
                                    for hw in head_child_head_words:
                                        if hw.name == '':
                                            head_child_head_words.remove(hw)
                                            break
                                break
                        if not new_daughter_set:
                            for da in new_daughter.daughters:
                                if (len(da.name) >= 2 and da.name[:2] == 'AP') or 'punc' in da.name or 'Punc' in da.name:
                                    adjunct = True
                                else:
                                    adjunct = False
                                if da not in new_daughter.heads and not adjunct:
                                    #.top_level_head_node tests to see if this is a complex or simple head node, in which case we do not want to continue
                                    #down the head path as we already know it leads to a [null]
                                    new_daughter = da
                                    new_daughter_set = True
                                    head_child_head_words = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                    while '' in [hw.name for hw in head_child_head_words]:
                                        #we want to ignore the deleted semantic node remnants that occur after pf movement
                                        for hw in head_child_head_words:
                                            if hw.name == '':
                                                head_child_head_words.remove(hw)
                                                break
                                    break
                        if not new_daughter_set:
                            #end of the line if we get here, so setting back to original
                            head_child_head_words = original_hws
                            head_child_head_word = original_hw
                            head_found = True
                else:
                    #if we get here, then we couldn't find a path down from the top node for some reason..
                    #must mean there's no head and bizarrely two adjuncts, or that the head is the top node of a complex
                    #or simplex head word.
                    head_found = True
                    head_child_head_word = hw
                    break
            #in turn, each head child could have multiple lexical heads owing again to coordination..
            head_word = head_child_head_word.name
            if head_child_head_word.name[0] == '[' and 'pro' in head_child_head_word.name:
                #for null heads in the spine we carry on searching down for the head.. but where
                #we have a null head due to ellipsis, we do not want to carry on searching down as then we'd end up
                #selecting the trace of an argument of the ellipsed verb.. e.g. in 'yours isn't working; mine is [pro-v] t' we'd select t, the trace of 'mine', as the head of the CP. 
                continue
            try:
                head_word_span = [terminals.index(head_child_head_word), terminals.index(head_child_head_word)+1]#2
            except ValueError:
                continue
            for daughter in MG_node.daughters:
                if daughter == head_child:
                    continue
                else:
                    direction = None
                    #direction records the direction of the non-head child relative to the head child
                    try:
                        if MG_node.daughters.index(daughter) < MG_node.daughters.index(head_child):
                            direction = 'left'
                        else:
                            direction = 'right'
                    except ValueError:
                        #if we get here it is again the issue that the head is the moved head, not one of the daughters
                        #so we'll establish the index by looking for the head trace
                        index = -1
                        for daughter in MG_node.daughters:
                            index+=1
                            if len(daughter.daughters) == 1 and daughter.daughters[0].name == 'Λ':
                                if index == 0:
                                    direction = 'right'
                                else:
                                    direction = 'left'
                        if direction == None:
                            #if we get here it probably means that the head 'child' is not actually a child but an overt
                            #trace elsewhere in the tree, hence we couldn't get an index to determine directionality..
                            #we can easily get this though simply by looking at the category of the head 'child'
                            #then selecting the daughter with the same category (and index, hence we just use name)
                            ultimate_head = MG_node.heads[0]
                            while ultimate_head.name == MG_node.name:
                                ultimate_head = ultimate_head.heads[0]
                            ultimate_head = ultimate_head.mother
                            while len(ultimate_head.daughters) == 1:
                                ultimate_head = ultimate_head.daughters[0]
                            if len(ultimate_head.daughters) != 2 or ultimate_head.heads[0] not in ultimate_head.daughters:
                                return deps
                            for daughter in ultimate_head.daughters:
                                if daughter == ultimate_head.heads[0]:
                                    continue
                                elif ultimate_head.daughters.index(daughter) < ultimate_head.daughters.index(ultimate_head.heads[0]):
                                    direction = 'left'
                                else:
                                    direction = 'right'
                    non_head_child_cat = daughter.truncated_name
                    non_head_child_head_words = daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                    non_head_child_head_words_sem = daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                    for nhchws in [non_head_child_head_words, non_head_child_head_words_sem]:
                        while '' in [hw.name for hw in nhchws]:
                            #we want to ignore the deleted semantic node remnants that occur after pf movement
                            for hw in nhchws:
                                if hw.name == '':
                                    nhchws.remove(hw)
                                    break
                    if len(non_head_child_head_words+non_head_child_head_words_sem) == 0:
                        continue
                    #now, if the dependent head turned out to be a null morpheme e.g. [decl]
                    #we want to continue down the tree until we find the first overt head in the spine..
                    #this is because otherwise we would not match the dependency between 'want' and 'to' which
                    #is captured in the PTB owing to the intervening [decl] in the MG tree
                    nhchws_ind = 0
                    for nhchws in [non_head_child_head_words, non_head_child_head_words_sem]:
                        nhchws_ind += 1
                        head_found = False
                        if len(daughter.daughters) > 0:
                            new_daughter = daughter
                            while not head_found and len(nhchws) > 0:
                                for hw in nhchws:
                                    if hw.name != '' and not (hw.name[0] == '[' and 'pro' not in hw.name) and hw.name not in ['Λ', 'λ', 'μ', 'ζ']:
                                        head_found = True
                                        break
                                if not head_found:
                                    if len(new_daughter.daughters) == 1:
                                        new_daughter = new_daughter.daughters[0]
                                        if nhchws_ind == 1:
                                            nhchws = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                        else:
                                            nhchws = new_daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                                        continue
                                    elif new_daughter.name == 'Λ':
                                        new_daughter = new_daughter.antecedent
                                    elif new_daughter.name == 'λ' or new_daughter.name == 'μ' or new_daughter.name == 'ζ':
                                        new_daughter = new_daughter.heads[0]
                                    else:
                                        new_daughter_set = False
                                        for d in new_daughter.daughters:
                                            if d in new_daughter.heads and not d.top_level_head_node:
                                                new_daughter = d
                                                new_daughter_set = True
                                                if nhchws_ind == 1:
                                                    nhchws = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                                else:
                                                    nhchws = new_daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                                                while '' in [hw.name for hw in nhchws]:
                                                    #we want to ignore the deleted semantic node remnants that occur after pf movement
                                                    for hw in nhchws:
                                                        if hw.name == '':
                                                            nhchws.remove(hw)
                                                            break
                                                break
                                        if not new_daughter_set:
                                            for da in new_daughter.daughters:
                                                if (len(da.name) >= 2 and da.name[:2] == 'AP') or 'punc' in da.name or 'Punc' in da.name:
                                                    adjunct = True
                                                else:
                                                    adjunct = False
                                                if da not in new_daughter.heads and not adjunct:
                                                    #.top_level_head_node tests to see if this is a complex or simple head node, in which case we do not want to continue
                                                    #down the head path as we already know it leads to a [null]
                                                    new_daughter = da
                                                    new_daughter_set = True
                                                    if nhchws_ind == 1:
                                                        nhchws = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                                    else:
                                                        nhchws = new_daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                                                    while '' in [hw.name for hw in nhchws]:
                                                        #we want to ignore the deleted semantic node remnants that occur after pf movement
                                                        for hw in nhchws:
                                                            if hw.name == '':
                                                                nhchws.remove(hw)
                                                                break
                                                    break
                                        if not new_daughter_set:
                                            #end of the line
                                            nhchws = []
                        if nhchws_ind == 1:
                            non_head_child_head_words = nhchws
                        else:
                            non_head_child_head_words_sem = nhchws
                    for non_head_child_head_word in non_head_child_head_words+non_head_child_head_words_sem:
                        if non_head_child_head_word.name == '':
                            continue
                        #again, there could be multiple head words inside the dependent
                        #owing to coordination, so we create a separate dependency relation
                        #for each dependent head.
                        if non_head_child_head_word.name[0] == '[' and 'pro' in non_head_child_head_word.name:
                            continue
                        non_head_word = non_head_child_head_word.name
                        try:
                            non_head_word_span = [terminals.index(non_head_child_head_word), terminals.index(non_head_child_head_word)+1]#3
                        except ValueError:
                            continue
                        dep = Dict(
                            parent_cat = parent_cat,
                            head_child_cat = head_child_cat,
                            head_word = head_word,
                            head_word_span = head_word_span,
                            non_head_child_cat = non_head_child_cat,
                            non_head_word = non_head_word,
                            non_head_word_span = non_head_word_span,
                            direction = direction
                            )
                        if dep not in deps:
                            if dep['non_head_word'] != None:
                                if dep['head_word_span'] != dep['non_head_word_span']:
                                    deps.append(dep)
    #now for dependencies for the semantic head..
    for head_child in MG_node.sem_heads:
        if not returnSemDeps:
            continue
        #there could be multiple head children, as in coordinate structures where each conjunct
        #will be treated as a head (in 'Jack likes honey and lemon', it is more useful
        #to regard 'honey' and 'lemon' as dependents of 'likes' than to regard 'and' as the dependent).
        head_child_cat = head_child.truncated_name
        head_child_head_words = head_child.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
        while '' in [hw.name for hw in head_child_head_words]:
            #we want to ignore the deleted semantic node remnants that occur after pf movement
            for hw in head_child_head_words:
                if hw.name == '':
                    head_child_head_words.remove(hw)
                    break
        for head_child_head_word in head_child_head_words:
            #in turn, each head child could have multiple lexical heads owing again to coordination..
            head_word = head_child_head_word.name
            try:
                head_word_span = [terminals.index(head_child_head_word), terminals.index(head_child_head_word)+1]#3
            except ValueError:
                continue
            for daughter in MG_node.daughters:
                if daughter == head_child or (len(daughter.daughters) > 0 and daughter.daughters[0].name == 'Λ'):
                    continue
                else:
                    direction = None
                    try:
                        #direction records the direction of the non-head child relative to the head child
                        if MG_node.daughters.index(daughter) < MG_node.daughters.index(head_child):#2
                            direction = 'left'
                        else:
                            direction = 'right'
                    except ValueError:
                        index = -1
                        for daughter in MG_node.daughters:
                            index+=1
                            if len(daughter.daughters) == 1 and daughter.daughters[0].name == 'Λ':
                                if index == 0:
                                    direction = 'right'
                                else:
                                    direction = 'left'
                        if direction == None:
                            #if we get here it probably means that the head 'child' is not actually a child but an overt
                            #trace elsewhere in the tree, hence we couldn't get an index to determine directionality..
                            #we can easily get this though simply by looking at the category of the head 'child'
                            #then selecting the daughter with the same category (and index, hence we just use name)
                            ultimate_head = MG_node.heads[0]
                            while ultimate_head.name == MG_node.name:
                                ultimate_head = ultimate_head.heads[0]
                            ultimate_head = ultimate_head.mother
                            while len(ultimate_head.daughters) == 1:
                                ultimate_head = ultimate_head.daughters[0]
                            if len(ultimate_head.daughters) != 2 or ultimate_head.heads[0] not in ultimate_head.daughters:
                                return deps
                            for daughter in ultimate_head.daughters:
                                if daughter == ultimate_head.heads[0]:
                                    continue
                                elif ultimate_head.daughters.index(daughter) < ultimate_head.daughters.index(ultimate_head.heads[0]):
                                    direction = 'left'
                                else:
                                    direction = 'right'
                    non_head_child_cat = daughter.truncated_name
                    non_head_child_head_words = daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                    non_head_child_head_words_syn = daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                    for nhchws in [non_head_child_head_words, non_head_child_head_words_syn]:
                        while '' in [hw.name for hw in nhchws]:
                            #we want to ignore the deleted semantic node remnants that occur after pf movement
                            for hw in nhchws:
                                if hw.name == '':
                                    nhchws.remove(hw)
                                    break
                    if len(non_head_child_head_words+non_head_child_head_words_syn) == 0:
                        continue
                    #again, we will weed out [null] heads
                    nhchws_ind = 0
                    for nhchws in [non_head_child_head_words, non_head_child_head_words_syn]:
                        nhchws_ind += 1
                        head_found = False
                        if len(daughter.daughters) > 0:
                            new_daughter = daughter
                            while not head_found and len(nhchws) > 0:
                                for hw in nhchws:
                                    if hw.name != '' and not (hw.name[0] == '[' and 'pro' not in hw.name) and hw.name not in ['Λ', 'λ', 'μ', 'ζ']:
                                        head_found = True
                                        break
                                if not head_found:
                                    if len(new_daughter.daughters) == 1:
                                        new_daughter = new_daughter.daughters[0]
                                        if nhchws_ind == 1:
                                            nhchws = new_daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                                        else:
                                            nhchws = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                        continue
                                    elif new_daughter.name == 'Λ':
                                        new_daughter = new_daughter.antecedent
                                    elif new_daughter.name == 'λ' or new_daughter.name == 'μ' or new_daughter.name == 'ζ':
                                        new_daughter = new_daughter.heads[0]
                                    else:
                                        new_daughter_set = False
                                        for d in new_daughter.daughters:
                                            if d in new_daughter.heads and not d.top_level_head_node:
                                                new_daughter = d
                                                new_daughter_set = True
                                                if nhchws_ind == 1:
                                                    nhchws = new_daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                                                else:
                                                    nhchws = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                                while '' in [hw.name for hw in nhchws]:
                                                    #we want to ignore the deleted semantic node remnants that occur after pf movement
                                                    for hw in nhchws:
                                                        if hw.name == '':
                                                            nhchws.remove(hw)
                                                            break
                                                break
                                        if not new_daughter_set:
                                            for da in new_daughter.daughters:
                                                if (len(da.name) >= 2 and da.name[:2] == 'AP') or 'punc' in da.name or 'Punc' in da.name:
                                                    adjunct = True
                                                else:
                                                    adjunct = False
                                                if da not in new_daughter.heads and not adjunct:
                                                    #.top_level_head_node tests to see if this is a complex or simple head node, in which case we do not want to continue
                                                    #down the head path as we already know it leads to a [null]
                                                    new_daughter = da
                                                    new_daughter_set = True
                                                    if nhchws_ind == 1:
                                                        nhchws = new_daughter.get_semantic_terminal_heads(normalize_terminals=True,returnSemDeps=returnSemDeps)
                                                    else:
                                                        nhchws = new_daughter.get_terminal_heads(normalize_terminals=True,returnSynDeps=returnSynDeps)
                                                    while '' in [hw.name for hw in nhchws]:
                                                        #we want to ignore the deleted semantic node remnants that occur after pf movement
                                                        for hw in nhchws:
                                                            if hw.name == '':
                                                                nhchws.remove(hw)
                                                                break
                                                    break
                                        if not new_daughter_set:
                                            #end of the line
                                            nhchws = []
                        if nhchws_ind == 1:
                            non_head_child_head_words = nhchws
                        else:
                            non_head_child_head_words_syn = nhchws
                    for non_head_child_head_word in non_head_child_head_words+non_head_child_head_words_syn:
                        if non_head_child_head_word.name == '':
                            continue
                        #again, there could be multiple head words inside the dependent
                        #owing to coordination, so we create a separate dependency relation
                        #for each dependent head.
                        non_head_word = non_head_child_head_word.name
                        try:
                            non_head_word_span = [terminals.index(non_head_child_head_word), terminals.index(non_head_child_head_word)+1]#1
                        except ValueError:
                            continue
                        dep = Dict(
                            parent_cat = parent_cat,
                            head_child_cat = head_child_cat,
                            head_word = head_word,
                            head_word_span = head_word_span,
                            non_head_child_cat = non_head_child_cat,
                            non_head_word = non_head_word,
                            non_head_word_span = non_head_word_span,
                            direction = direction,
                            )
                        if dep not in deps:
                            if dep['head_word_span'] != dep['non_head_word_span']:
                                deps.append(dep)
    for daughter in MG_node.daughters:
        deps = get_MGdeps(daughter, terminals, deps, returnSynDeps=returnSynDeps, returnSemDeps=returnSemDeps)
    return deps

def add_truncated_names(xbar_tree):
    #we need xbar_tree nodes to have a .truncated_name attribute with indices stripped off
    #so these are no included in the dependency tuples..
    if xbar_tree.terminal == True:
        return
    else:
        xbar_tree.truncated_name = xbar_tree.name.split("-")[0]
    for daughter in xbar_tree.daughters:
        add_truncated_names(daughter)

def extract_propbank_relations(dependent_node_name, head_word_span):
    #we need to remove any - symbols which appear internal to each relation label,
    #because these also appear as separators between labels and we will split the
    #string based on these.. so ARG1-to{give} becomes ARG1to{give}..
    dependent_node_name = re.sub('ARGM-', 'ARGM', dependent_node_name, count = 100)
    dependent_node_name = re.sub('ARG0-', 'ARG0', dependent_node_name, count = 100)
    dependent_node_name = re.sub('ARG1-', 'ARG1', dependent_node_name, count = 100)
    dependent_node_name = re.sub('ARG2-', 'ARG2', dependent_node_name, count = 100)
    dependent_node_name = re.sub('ARG3-', 'ARG3', dependent_node_name, count = 100)
    dependent_node_name = re.sub('ARG4-', 'ARG4', dependent_node_name, count = 100)
    dependent_node_name = re.sub('ARG5-', 'ARG5', dependent_node_name, count = 100)
    relations = dependent_node_name.split("-")[1:]
    #we will remove any relations whose head word does not match the specific head
    #entering into the current dependency..and also any indices or ! tags..
    RELATIONS = []
    for relation in relations:
        HEAD_WORD_SPAN = re.search("<.*?>", relation)
        if HEAD_WORD_SPAN != None:
            HEAD_WORD_SPAN = HEAD_WORD_SPAN.group(0)[1:-1]
            span_members = HEAD_WORD_SPAN.split(",")
            HEAD_WORD_SPAN = [int(span_members[0]), int(span_members[1])]
            if HEAD_WORD_SPAN == head_word_span:
                #we can remove the {head<x,y>} markup now..
                relation = re.sub("{.*?}", "", relation)
                RELATIONS.append(relation)
            #if the spans didn't match, this relation is simply not included in this
            #dependency.. the reason for this is that one word can enter into multiple
            #dependencies with different heads in propbank.. e.g. in 'it has no bearing on
            #this case' propbank/nombank marks 'it' as an argument of both 'has' and 'bearing'
            #whereas we are only representing the dependency on 'has' in the syntax..
        else:
            try:
                #if its an index we ignore it..
                int(relation)
            except ValueError:
                if "!" not in relation:
                    RELATIONS.append(relation)
    return RELATIONS

def build_tree(bracketing):
    #having taken the space out earlier, it appears we now need it again!
    bracketing = re.sub('\)\(', ') (', bracketing, count=10000)
    reset_terminal_index_counter()
    #returns a penn tree Node() object and the sentence at its terminals given a penn bracketing
    sentence=""
    current_mother = None
    #terminals is a list of all terminal nodes.  This is useful for adding the propbank annotation to our trees,
    #since propbank provides tree locations by terminal and height index.
    terminals = []
    tokenized_bracketing = bracketing.split(" ")
    #we need to remove the vacuous outer brackets
    tokenized_bracketing[0] = tokenized_bracketing[0][1:]
    tokenized_bracketing[-1] = tokenized_bracketing[-1].strip()[:-1]
    #we now need to remove all the empty entries in tokenized bracketing
    #that were caused by unnecessary spaces
    new_tokenized_bracketing = [entry for entry in tokenized_bracketing if entry != '']
    for item in new_tokenized_bracketing:
        if item[0] == "(":
            if current_mother == None:
                root_tree = Node(item[1:])
                current_mother = root_tree
            else:
                current_mother.add_daughter(Node(item[1:]))
                current_mother = current_mother.daughters[-1]
        else:
            closures = item.count(")")
            if item[0] != ")":
                current_mother.add_daughter(Node(item[0:-closures], terminal=True))
                #since we need to match spans between the MG tree and the Penn tree, and since
                #the number of traces in the two trees is usually different, we ignore all traces
                #when constructing our terminal list here..
                if "-NONE-" not in current_mother.name and current_mother.daughters[0].name not in punctuation and current_mother.daughters[0].name != "''":
                    terminals.append(current_mother.daughters[0])
                if item[0] != '*':
                    sentence+=item[0:-closures]+" "
            #now we need to backtrack up the tree
            for i in range(closures):
                current_mother = current_mother.mother
    #first set the heads of the root node
    root_tree.set_heads('syntactic')
    root_tree.set_heads('semantic')
    eliminate_bracket_hyphens(root_tree)
    #now recursively set the heads for all daughters
    set_all_tree_heads(root_tree)
    return (root_tree, sentence, terminals)

def eliminate_bracket_hyphens(node):
    if not node.terminal:
        if node.name in ['-LRB-', '-RRB-']:
            node.name = node.name[1:-1]
            node.truncated_name = node.name
    for daughter in node.daughters:
        eliminate_bracket_hyphens(daughter)

def get_new_terminal_index():
    terminal_index[0] += 1
    return terminal_index[0]

def reset_terminal_index_counter():
    terminal_index[0] = -1

def set_all_tree_heads(tree):
    for daughter in tree.daughters:
        #sometimes the ARG tags appear on the S rather than its SBAR mother.. the following corrects this..
        if daughter.truncated_name == 'SBAR':
            for DAUGHTER in daughter.daughters:
                if DAUGHTER.truncated_name == 'S' or DAUGHTER.truncated_name == 'SQ' or DAUGHTER.truncated_name == 'SINV':
                    if 'ARG' in DAUGHTER.name:
                        tags = DAUGHTER.name.split("-")
                        daughter.name += ("-"+"-".join(tags[1:]))
                        DAUGHTER.name = tags[0]
                    break
        daughter.set_heads('syntactic')
        daughter.set_heads('semantic')
        if daughter.terminal == False:
            set_all_tree_heads(daughter)

def loadData(ptb_folder, seed_folder, auto_folder):
    dirs = sorted(os.listdir(seed_folder))
    dirsAuto = os.listdir(auto_folder)
    global PosMappings
    global PosDepsMappings
    global DepMappings
    global RevDepMappings
    global CovertLexicon
    global ExtraposerLexicon
    global TypeRaiserLexicon
    global ToughOperatorLexicon
    global NullExcorporatorLexicon
    global CatTreeMappings
    global TreeCatMappings
    global nullCatTreeMappings
    global nullTreeCatMappings
    global overtCatComments
    global covertCatComments
    if 'covertCatComments' not in dirs:
        covertCatComments = {}
        with open(seed_folder+'/'+'covertCatComments', 'w') as covertCatCommentsFile:
            json.dump(covertCatComments, covertCatCommentsFile)
    else:
        covertCatComments = json.load(open(seed_folder+"/"+"covertCatComments"))
    if 'overtCatComments' not in dirs:
        overtCatComments = {}
        with open(seed_folder+'/'+'overtCatComments', 'w') as overtCatCommentsFile:
            json.dump(overtCatComments, overtCatCommentsFile)
    else:
        overtCatComments = json.load(open(seed_folder+"/"+"overtCatComments"))
    if 'PosMappings' not in dirs:
    #PosMappings is a mapping from PTB preterminal catgories (VB, NNP etc)
    #to MG preterminal categories, which are features sequences (=d d= v, n, etc)
    #It is used to populate the drop down menus to aid the user in annotating a new
    #tree with MG categories..
        PosMappings = {}
        for cat in PTBcats:
            PosMappings[cat] = ['No categories available', '']
        with open(seed_folder+'/'+'PosMappings', 'w') as PosMappingsFile:
            json.dump(PosMappings, PosMappingsFile)
    else:
        PosMappings = json.load(open(seed_folder+"/"+"PosMappings"))
        for cat in PTBcats:
            if cat not in PosMappings:
                PosMappings[cat] = ['No categories available', '']
        with open(seed_folder+'/'+'PosMappings', 'w') as PosMappingsFile:
            json.dump(PosMappings, PosMappingsFile)
    overtMGcatList = []
    for ptbCat in PosMappings:
        for MGcat in PosMappings[ptbCat]:
            if MGcat not in overtMGcatList:
                overtMGcatList.append(MGcat)
    #PosDepsMappings contains mappings from PTB preterminal categories to MG preterminal
    #categories+a set of PTB dependencies.. this will be used by the automatic corpus
    #generator to decide which MG categories to consider for each sentence based on the category
    #AND subcat environment of each word in the PTB tree..
    PosDepsMappings = {}
    for cat in PTBcats:
        PosDepsMappings[cat] = {}
    #the forward mappings from PTB dependency chains to MG dependency chains..
    DepMappings = {}
    #the reverse mappings from PTB dependency chains to MG dependency chains..
    RevDepMappings = {}
    covertMGcatsList = []
    if 'parserSettings' not in dirs:
        use_section_folder = {}
        parserSettings = {"printPartialAnalyses": 0, "timeout_seconds": 60, "parserSetting": "basicOnly", "autoMaxTreesVar": "no limit", "autoMaxSentLenVar": 6, "supertaggingStrategy": "CCG_OVERT_MG", "ccgBeam": "0.5000", "ccgBeamFloor": "0.2000", "useAllNull": 0, "skipRel": 0, "skipPro": 0, "constrainMoveWithPTB": 0, "constrainConstWithPTBCCG": 0, "use_deps": 1}
        for section_folder in os.listdir(ptb_folder):
            if section_folder != '.DS_Store':
                use_section_folder[section_folder] = 1
        parserSettings['section_folders'] = use_section_folder
        with open(seed_folder+"/parserSettings", 'w') as parserSettingsFile:
            json.dump(parserSettings, parserSettingsFile)
    if 'CovertLexicon' not in dirs:
        #CovertLexicon is a lexicon of null heads, e.g. [decl]
        CovertLexicon = []
        with open(seed_folder+'/'+'CovertLexicon', 'w') as CovertLexiconFile:
            json.dump(CovertLexicon, CovertLexiconFile)
    else:
        CovertLexicon = json.load(open(seed_folder+"/"+"CovertLexicon"))
        for entry in CovertLexicon:
            if 'conj' in entry[2]:
                separator = ":\u0305:\u0305"
            else:
                separator = "::"
            covertMGcatsList.append(entry[0]+" "+separator+" "+" ".join(entry[1]))
    if 'ExtraposerLexicon' not in dirs:
        ExtraposerLexicon = []
        with open(seed_folder+'/'+'ExtraposerLexicon', 'w') as ExtraposerLexiconFile:
            json.dump(ExtraposerLexicon, ExtraposerLexiconFile)
    else:
        ExtraposerLexicon = json.load(open(seed_folder+"/"+"ExtraposerLexicon"))
        for entry in ExtraposerLexicon:
            if 'conj' in entry[2]:
                separator = ":\u0305:\u0305"
            else:
                separator = "::"
            covertMGcatsList.append(entry[0]+" "+separator+" "+" ".join(entry[1]))

    if 'TypeRaiserLexicon' not in dirs:
        TypeRaiserLexicon = []
        with open(seed_folder+'/'+'TypeRaiserLexicon', 'w') as TypeRaiserLexiconFile:
            json.dump(TypeRaiserLexicon, TypeRaiserLexiconFile)
    else:
        TypeRaiserLexicon = json.load(open(seed_folder+"/"+"TypeRaiserLexicon"))
        for entry in TypeRaiserLexicon:
            if 'conj' in entry[2]:
                separator = ":\u0305:\u0305"
            else:
                separator = "::"
            covertMGcatsList.append(entry[0]+" "+separator+" "+" ".join(entry[1]))

    if 'ToughOperatorLexicon' not in dirs:
        ToughOperatorLexicon = []
        with open(seed_folder+'/'+'ToughOperatorLexicon', 'w') as ToughOperatorLexiconFile:
            json.dump(ToughOperatorLexicon, ToughOperatorLexiconFile)
    else:
        ToughOperatorLexicon = json.load(open(seed_folder+"/"+"ToughOperatorLexicon"))
        for entry in ToughOperatorLexicon:
            if 'conj' in entry[2]:
                separator = ":\u0305:\u0305"
            else:
                separator = "::"
            covertMGcatsList.append(entry[0]+" "+separator+" "+" ".join(entry[1]))

    if 'NullExcorporatorLexicon' not in dirs:
        NullExcorporatorLexicon = []
        with open(seed_folder+'/'+'NullExcorporatorLexicon', 'w') as NullExcorporatorLexiconFile:
            json.dump(NullExcorporatorLexicon, NullExcorporatorLexiconFile)
    else:
        NullExcorporatorLexicon = json.load(open(seed_folder+"/"+"NullExcorporatorLexicon"))
        for entry in NullExcorporatorLexicon:
            if 'conj' in entry[2]:
                separator = ":\u0305:\u0305"
            else:
                separator = "::"
            covertMGcatsList.append(entry[0]+" "+separator+" "+" ".join(entry[1]))
    if "CCG_MG_ATOMIC_taggingModel1" not in dirs:
        with open(seed_folder+'/CCG_MG_ATOMIC_taggingModel1', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if "CCG_MG_ATOMIC_taggingModel2" not in dirs:
        with open(seed_folder+'/CCG_MG_ATOMIC_taggingModel2', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if "CCG_MG_ATOMIC_taggingModel3" not in dirs:
        with open(seed_folder+'/CCG_MG_ATOMIC_taggingModel3', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if "CCG_MG_ATOMIC_taggingModel4" not in dirs:
        with open(seed_folder+'/CCG_MG_ATOMIC_taggingModel4', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if "CCG_MG_SUPERTAG_taggingModel1" not in dirs:
        with open(seed_folder+'/CCG_MG_SUPERTAG_taggingModel1', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if "CCG_MG_SUPERTAG_taggingModel2" not in dirs:
        with open(seed_folder+'/CCG_MG_SUPERTAG_taggingModel2', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if "CCG_MG_SUPERTAG_taggingModel3" not in dirs:
        with open(seed_folder+'/CCG_MG_SUPERTAG_taggingModel3', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if "CCG_MG_SUPERTAG_taggingModel4" not in dirs:
        with open(seed_folder+'/CCG_MG_SUPERTAG_taggingModel4', 'w') as stats_table_file:
            json.dump({}, stats_table_file)
    if 'CatTreeMappings' not in dirs:
    #CatTreeMappings is a mapping from MG categories (feature sequences) to
    #the set of derivation trees in the seed set which contain that category..
    #we need this because whenever the user wishes to go back and modify or
    #delete an MG category it is important to perform regression tests to make
    #sure that the sentences are still parsable and that they still have the same
    #structure..
        CatTreeMappings = {}
        with open(seed_folder+'/'+'CatTreeMappings', 'w') as CatTreeMappingsFile:
            json.dump(CatTreeMappings, CatTreeMappingsFile)
    else:
        CatTreeMappings = json.load(open(seed_folder+"/"+"CatTreeMappings"))
    if 'TreeCatMappings' not in dirs:
    #we also need a mapping in the other direction so we can keep track of which categories
    #each parse used.. this is so that right before we begin auto generating the full corpus
    #we can create the relevant PosDepsMappings..
        TreeCatMappings = {}
        with open(seed_folder+'/'+'TreeCatMappings', 'w') as TreeCatMappingsFile:
            json.dump(TreeCatMappings, TreeCatMappingsFile)
    else:
        TreeCatMappings = json.load(open(seed_folder+"/"+"TreeCatMappings"))
    if 'nullCatTreeMappings' not in dirs:
        nullCatTreeMappings = {}
        with open(seed_folder+'/'+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
            json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
    else:
        nullCatTreeMappings = json.load(open(seed_folder+"/"+"nullCatTreeMappings"))
    if 'nullTreeCatMappings' not in dirs:
        nullTreeCatMappings = {}
        with open(seed_folder+'/'+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
            json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
    else:
        nullTreeCatMappings = json.load(open(seed_folder+"/"+"nullTreeCatMappings"))
    if 'autoCatTreeMappings' not in dirsAuto:
        autoCatTreeMappings = {}
        with open(auto_folder+'/'+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
            json.dump(autoCatTreeMappings, autoCatTreeMappingsFile)
    else:
        autoCatTreeMappings = json.load(open(auto_folder+"/"+"autoCatTreeMappings"))
    if 'autoTreeCatMappings' not in dirsAuto:
        autoTreeCatMappings = {}
        with open(auto_folder+'/'+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
            json.dump(autoTreeCatMappings, autoTreeCatMappingsFile)
    else:
        autoTreeCatMappings = json.load(open(auto_folder+"/"+"autoTreeCatMappings"))
    if 'autoNullCatTreeMappings' not in dirsAuto:
        autoNullCatTreeMappings = {}
        with open(auto_folder+'/'+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
            json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
    else:
        autoNullCatTreeMappings = json.load(open(auto_folder+"/"+"autoNullCatTreeMappings"))
    if 'autoNullTreeCatMappings' not in dirsAuto:
        autoNullTreeCatMappings = {}
        with open(auto_folder+'/'+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
            json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)
    else:
        autoNullTreeCatMappings = json.load(open(auto_folder+"/"+"autoNullTreeCatMappings"))
    #the following fixes a bug that if the program is stopped while it is trying to modify a category
    #a space can end up being inserted at the start of the category
    #DELETE ALL THE FOLLOWING AFTER A FEW WEEKS WHEN WE KNOW THIS BUG IS DEFINITELY FIXED!!!!
    cat_pairs = []
    for mappings in [(CatTreeMappings, TreeCatMappings, nullCatTreeMappings, nullTreeCatMappings), (autoCatTreeMappings, autoTreeCatMappings, autoNullCatTreeMappings, autoNullTreeCatMappings)]:
        CTM = mappings[0]
        TCM = mappings[1]
        NCTM = mappings[2]
        NTCM = mappings[3]
        if CTM == CatTreeMappings:
            autos = False
        else:
            autos = True
            if not check_auto_mappings:
                continue
        for cat in CTM:
            if cat not in overtMGcatList:
                if not autos:
                    print "Error! Found MG category:", cat, "in CatTreeMappings that is not in PosMappings..."
                else:
                    print "Error! Found MG category:", cat, "in autoCatTreeMappings that is not in PosMappings..."
                pdb.set_trace()
                raise Exception
            if cat[0] == ' ':
                if not autos:
                    print "Warning! Fixing space bug in category from CatTreeMappings.."
                else:
                    print "Warning! Fixing space bug in category from autoCatTreeMappings.."
                print "category: "+cat
                new_cat = cat.strip()
                cat_pairs.append((cat, new_cat))
        for (cat, new_cat) in cat_pairs:
            CTM[new_cat] = CTM[cat]
            del(CTM[cat])
        tree_cat_indices = []
        for tree in TCM:
            index = -1
            for cat in TCM[tree]:
                index += 1
                if cat[0] == ' ':
                    if not autos:
                        print "Warning! Fixing space bug in TreeCatMappings!!!"
                    else:
                        print "Warning! Fixing space bug in autoTreeCatMappings!!!"
                    print "category: "+cat
                    new_cat = cat.strip()
                    tree_cat_indices.append((tree, index, new_cat))
        for (tree, index, new_cat) in tree_cat_indices:
            del(TCM[tree][index])
            TCM[tree].insert(index, new_cat)
        #now for the null cat mappings
        cat_pairs = []
        for cat in NCTM:
            if cat not in covertMGcatsList:
                if not autos:
                    print "Error! Found covert MG category:", cat, "in nullCatTreeMappings that is not in any covert lexicon... press c to continue"
                else:
                    print "Error! Found covert MG category:", cat, "in autoNullCatTreeMappings that is not in any covert lexicon... press c to continue"
                pdb.set_trace()
                raise Exception
            if cat[0] == ' ':
                if not autos:
                    print "Warning! Fixing space bug in nullCatTreeMappings!!!!"
                else:
                    print "Warning! Fixing space bug in autoNullCatTreeMappings!!!!"
                print "category: "+cat
                new_cat = cat.strip()
                cat_pairs.append((cat, new_cat))
        for (cat, new_cat) in cat_pairs:
            NCTM[new_cat] = NCTM[cat]
            del(NCTM[cat])
        tree_cat_indices = []
        for tree in NTCM:
            index = -1
            for cat in NTCM[tree]:
                index += 1
                if cat[0] == ' ':
                    if not autos:
                        print "Warning!!!  Fixing space bug in nullTreeCatMappings!!!!"
                    else:
                        print "Warning!!!  Fixing space bug in autoNullTreeCatMappings!!!!"
                    print "category: "+cat
                    new_cat = cat.strip()
                    tree_cat_indices.append((tree, index, new_cat))
        for (tree, index, new_cat) in tree_cat_indices:
            del(NTCM[tree][index])
            NTCM[tree].insert(index, new_cat)
        for tree in TCM:
            for cat in TCM[tree]:
                if cat not in CTM:
                    trees_to_add = []
                    for TREE in TCM:
                        if cat in TCM[TREE]:
                            trees_to_add.append(ast.literal_eval(TREE))
                    CTM[cat] = trees_to_add
                    if not autos:
                        print "Error!!! having to add trees and cat to CatTreeMappings!!!!"
                    else:
                        print "Error!!! having to add trees and cat to autoCatTreeMappings!!!!"
                    pdb.set_trace()
                    raise Exception
                elif ast.literal_eval(tree) not in CTM[cat]:
                    CTM[cat].append(ast.literal_eval(tree))
                    if not autos:
                        print "Error!!!! having to add a tree to CatTreeMappings!!!!"
                    else:
                        print "Error!!!! having to add a tree to autoCatTreeMappings!!!!"
                    pdb.set_trace()
                    raise Exception
        for tree in NTCM:
            for cat in NTCM[tree]:
                if cat not in NCTM:
                    print "Oops, it's happening here too :("
                    pdb.set_trace()
                    raise Exception("")
                if ast.literal_eval(tree) not in NCTM[cat]:
                    print "Oops, it's happening here too :("
                    pdb.set_trace()
                    raise Exception("")
        #now we do some checks the other way
        if True:
            for cat in CTM:
                CTM[cat].sort()
                for tree in CTM[cat]:
                    if str([tree[0].encode('utf8'), tree[1]]) not in TCM or cat not in TCM[str([tree[0].encode('utf8'), tree[1]])]:
                        if not autos:
                            print "Oops, there are trees in CatTreeMappings for a cat that does not appear with that tree in TreeCatMappings"
                        else:
                            print "Oops, there are trees in autoCatTreeMappings for a cat that does not appear with that tree in autoTreeCatMappings"
                        pdb.set_trace()
                        raise Exception
            for cat in NCTM:
                NCTM[cat].sort()
                for tree in NCTM[cat]:
                    if str([tree[0].encode('utf8'), tree[1]]) not in NTCM or cat not in NTCM[str([tree[0].encode('utf8'), tree[1]])]:
                        if not autos:
                            print "Oops, there are trees in nullCatTreeMappings for a cat that does not appear with that tree in TreeCatMappings"
                        else:
                            print "Oops, there are trees in autoNullCatTreeMappings for a cat that does not appear with that tree in autoTreeCatMappings"
                        pdb.set_trace()
                        raise Exception
    with open(seed_folder+"/"+'CatTreeMappings', 'w') as CatTreeMappingsFile:
        json.dump(CatTreeMappings, CatTreeMappingsFile)
    with open(seed_folder+"/"+'TreeCatMappings', 'w') as TreeCatMappingsFile:
        json.dump(TreeCatMappings, TreeCatMappingsFile)
    with open(seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
        json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
    with open(seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
        json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
    with open(auto_folder+"/"+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
        json.dump(autoCatTreeMappings, autoCatTreeMappingsFile)
    with open(auto_folder+"/"+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
        json.dump(autoTreeCatMappings, autoTreeCatMappingsFile)
    with open(auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
        json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
    with open(auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
        json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)

if __name__ == '__main__':
    #full list of parameters: -f, -o, -a, -avc, -ac, -l, -n, -bc, -bf, -t, -p, -s, -mmd,
    #-src, -stdm, -scc, sfbr, sbr, sbw, -ua, -ptbmv, -ptbct, -tt, -edm, -st, -m, -c, -sr, -sp, -ud, -abt
    global all_verb_cats
    global all_cats
    global use_deps
    cmd_parser = argparse.ArgumentParser(description='Autobank command line arguments.')
    cmd_parser.add_argument('MGbankName', metavar='MGbankName', type=str, nargs=1, help='Specifies the name of the MG treebank project.')
    cmd_parser.add_argument('--auto-gen', dest='auto_gen', default=False, action='store_true', help='Start automatic annotator.')
    cmd_parser.add_argument('--files', dest='files_folders', metavar='F', type=str, nargs=1, default=["00-24"], help='Specifies a ptb file, folder or range of files/folders containing the sentences to be annotated (e.g. 02 or wsj_1204.mrg or 00-05 or wsj_1204.mrg-wsj_1505.mrg.)')
    cmd_parser.add_argument('--overwrite-autos', dest='overwrite_auto', default=False, action='store_true', help='Specifies that any existing automatically generated corpus should be overwritten.')
    cmd_parser.add_argument('--check-auto-mappings', dest='check_auto_mappings', default=False, action='store_true', help='Causes the system to check that the auto mappings files are consistent (used during the development of Autobank).')
    cmd_parser.add_argument('--all-verbal-cats', dest='all_verb_cats', default=False, action='store_true', help='In maxent modes, all MG categories seen with a verbal CCGbank during training supertag will be sent to the parser (with minimal probability if they were not predicted by the supertagger).')
    cmd_parser.add_argument('--all-cats', dest='all_cats', default=False, action='store_true', help='In maxent modes, all MG categories seen with a CCGbank supertag during training will be sent to the parser (with minimal probability if they were not predicted by the supertagger).')
    cmd_parser.add_argument('--start-line', dest='start_line', metavar='L', type=int, nargs=1, default=[1], help='Specifies which line to start at in the first file to be processed.')
    cmd_parser.add_argument('--sent-len', dest='sent_len', metavar='N', type=str, nargs=1, default=['all'], help='Specifies an exact string length or range of string lengths for the sentences to be processed (e.g. 13, or 10-20).')
    cmd_parser.add_argument('--beam-ceil', dest='beam_ceil', metavar='BC', type=float, nargs=1, default=[0.1], help='Specifies the beam ceiling for the supertagger (default is 0.1).')
    cmd_parser.add_argument('--beam-floor', dest='beam_floor', metavar='BF', type=float, nargs=1, default=[0.0001], help='Specifies the beam floor for the supertagger (default is 0.0001).')
    cmd_parser.add_argument('--timeout', dest='timeout', metavar='T', type=int, nargs=1, default=[21600], help='Specifies a timeout in seconds for the parser (default = 21600 = 6 hours).')
    cmd_parser.add_argument('--parser', dest='parser', metavar='P', type=str, nargs=1, default=['basic'], help="Specifies the version of the parser to be used ('basic' vs. 'full') (default is basic, full adds in [extraposers] and tough movement operators [op]).")
    cmd_parser.add_argument('--stop-after', dest='stop', metavar='S', type=int, nargs=1, default=[99999999], help='Tells the automatic generator how many trees to annotate (if unspecified it will just keep going until it reaches the end of the files to be processed).')
    cmd_parser.add_argument('--max-move-dist', dest='max_move', metavar='MMD', type=int, nargs=1, default=[15], help='An integer specifying a string distance limit for phrasal movement (improves efficiency, default is 15).')
    cmd_parser.add_argument('--use-autos', dest='use_autos', default=False, action='store_true', help='Including this flag will tell the system to include any pre-existing auto corpus when it constructs the ccg-mg supertagging model and also for the purposes of extracting the dependency mappings for scoring candidate trees.')
    cmd_parser.add_argument('--allow-best-tag', dest='allow_best_tag', default=False, action='store_true', help='Including this flag ensures that the parser will always have access to the highest scoring tag, even if this was never seen with a given word during training (and may therefore otherwise have been blocked).')
    cmd_parser.add_argument('--ptb-constrain-move', dest='ptbmv', default=False, action='store_true', help='Including this flag will cause the system to use the PTB constituencies to constrain the allowable movement operations (improves efficiency and precision at limited cost to robustness).')
    cmd_parser.add_argument('--ptb-constrain-const', dest='ptbct', default=False, action='store_true', help='Including this flag will cause the system to use the PTB (and CCGbank) constituencies to constrain the allowable constituencies in the MG trees (improves efficiency and precision at limited cost to robustness).')
    cmd_parser.add_argument('--allow-frag', dest='frag', default=False, action='store_true', help='Including this flag means that the system can return fragment parses (although it will only do this if it cannot find a main clause root CP analysis).')
    cmd_parser.add_argument('--train-tagger', dest='train', default=False, action='store_true', help='Specifies that a new set of supertagging models should be trained before annotation commences.')
    cmd_parser.add_argument('--extract-deps', dest='extract_deps', default=False, action='store_true', help='Specifies that a new set of dependency mappings should be extracted before annotation commences.')
    cmd_parser.add_argument('--supertagger', dest='supertagger', metavar='ST', type=str, nargs=1, default=['tmax'], help="Specifies the type of supertagger to use (parameter can take one of the following values (default is tmax): 'tuni', 'stuni', 'stmax', 'tmax', 'hybuni', 'hybmax', 'hybabmax', 'stabuni', 'stabmax'; t = atomic tags, st = supertags, hyb (hybrid) = supertags delexicalised for A'-movement, ab = abstract supertags with most subcategorization features removed.)")
    cmd_parser.add_argument('--max-mg-cats', dest='max_cats', metavar='M', type=float, nargs=1, default=[2.5], help='Floating point number specifying a maximum number of MG categories that the parser can try per word (defaults to 2.5).')
    cmd_parser.add_argument('--use-all-null', dest='use_all_null', default=False, action='store_true', help='Tells the parser to introduce all null heads immediately into the chart rather than incrementally (prevents correct analyses being bled by incorrect ones but is very inefficient for many sentences - not recommended).')
    cmd_parser.add_argument('--skip-rel', dest='skip_rel', default=False, action='store_true', help='Tells the parser not to use [relativizer] heads.')
    cmd_parser.add_argument('--skip-pro', dest='skip_pro', default=False, action='store_true', help='Tells the parser not to use [pro] heads.')
    args = cmd_parser.parse_args()
    MAX_MG_CATS_PER_WORD = None
    USEALLNULL = False
    SKIPREL = False
    SKIPPRO = False
    USEAUTOS = True
    TRAIN_TAGGER = True
    EXTRACT_DEP_MAPPINGS = True
    SUPERTAGGINGSTRATEGY = None
    SUPER_RARE_CUTOFF = 0
    SUPER_TAG_DICT_MIN = 1
    SUPER_CATEGORY_CUTOFF = 1
    SUPER_FORWARD_BEAM_RATIO = 0.0001
    SUPER_BEAM_RATIO = 0.0001
    SUPER_BEAM_WIDTH = 10000
    ALLOWBESTTAG = False
    ALLOWMOREGOALS = True
    ptb_folder = 'wsj'
    ccg_auto_folder = 'ccg'
    if "-" in args.files_folders[0]:
        if ".mrg" in args.files_folders[0]:
            START_FILE = args.files_folders[0].split("-")[0]
            END_FILE = args.files_folders[0].split("-")[1]
            if START_FILE == END_FILE:
                raise Exception("Error! start file and end file must not be the same!")
            if not (int(START_FILE.split("_")[1][:4]) < int(END_FILE.split("_")[1][:4])):
                raise Exception("Error! Start file must precede end file!")
            start_folder = START_FILE.split("_")[1][:2]
            end_folder = END_FILE.split("_")[1][:2]
        else:
            start_folder = args.files_folders[0].split("-")[0]
            end_folder = args.files_folders[0].split("-")[1]
            START_FILE = None
            END_FILE = None
        AUTO_SECTION_FOLDERS = []
        found_start_folder = False
        found_end_folder = False
        for folder in os.listdir(ptb_folder):
            if folder == start_folder:
                found_start_folder = True
            if found_start_folder:
                AUTO_SECTION_FOLDERS.append(folder)
            if folder == end_folder:
                found_end_folder = True
                break
        if not (found_start_folder and found_end_folder):
            raise Exception("Error! Make sure end file follows start file!")
    elif ".mrg" in args.files_folders[0]:
        START_FILE = args.files_folders[0]
        END_FILE = args.files_folders[0]
        AUTO_SECTION_FOLDERS = [START_FILE.split("_")[1][:2]]
    else:
        AUTO_SECTION_FOLDERS = args.files_folders
        START_FILE = None
        END_FILE = None
    check_auto_mappings = args.check_auto_mappings
    all_verb_cats = args.all_verb_cats
    all_cats = args.all_cats
    try:
        START_LINE = args.start_line[0]
        if START_LINE < 1:
            raise Exception("Error! Start line must be a positive integer")
    except ValueError:
        raise Exception("Error! Start line must be a positive integer")
    sent_length = args.sent_len[0]
    try:
        if sent_length not in ['all', 'All', 'ALL']:
            if "-" in sent_length:
                MIN_LENGTH = int(sent_length.split("-")[0])
                MAX_LENGTH = int(sent_length.split("-")[1])
            else:
                MIN_LENGTH = int(sent_length)
                MAX_LENGTH = int(sent_length)
        else:
            MIN_LENGTH = 0
            MAX_LENGTH = 10000
    except Exception as e:
        raise Exception("Error! Sentence length must be entered as integer or range (e.g. 10-12)..")
    CCG_BEAM = float("{0:.4f}".format(float(args.beam_ceil[0])))
    if CCG_BEAM < 0 or CCG_BEAM > 1:
        raise Exception("Error! Beam ceiling must be a floating number between 0 and 1!")
    CCG_BEAM_FLOOR = float("{0:.4f}".format(float(args.beam_floor[0])))
    if CCG_BEAM_FLOOR > CCG_BEAM:
        raise Exception("Error! Beam floor must be lower than beam ceiling!")
    if CCG_BEAM_FLOOR < 0 or CCG_BEAM_FLOOR > 1:
        raise Exception("Error! Beam ceiling must be a floating number between 0 and 1!")
    TIMEOUT = args.timeout[0]
    if TIMEOUT < 1:
        raise Exception("Error! Timeout value must be a positive integer")
    PARSER_SETTING = args.parser[0].lower()
    if PARSER_SETTING not in ['basic', 'full']:
        raise Exception("Error! Parser setting must be either 'basic' or 'full'..")
    STOP_AFTER = args.stop[0]
    if STOP_AFTER < 1:
        raise Exception("Error! Number of trees to generate must be a positive integer..")
    MAXMOVEDIST = args.max_move[0]
    if MAXMOVEDIST < 1:
        raise Exception("Error! Max Move Distance must be a non-negative integer...")
    TRAIN_TAGGER = args.train
    use_deps = True
    EXTRACT_DEP_MAPPINGS = args.extract_deps
    SUPERTAGGINGSTRATEGY = args.supertagger[0]
    if SUPERTAGGINGSTRATEGY not in ['tuni', 'stuni', 'stmax', 'tmax', 'hybuni', 'hybmax', 'hybabmax', 'stabuni', 'stabmax']:
        raise Exception("Error! Illicit parameter value for supertag strategy..")
    if SUPERTAGGINGSTRATEGY == 'tuni':
        SUPERTAGGINGSTRATEGY = 'CCG_OVERT_MG'
    elif SUPERTAGGINGSTRATEGY == 'stuni':
        SUPERTAGGINGSTRATEGY = 'CCG_MG_SUPERTAG'
    elif SUPERTAGGINGSTRATEGY == 'stmax':
        SUPERTAGGINGSTRATEGY = 'CCG_MG_SUPERTAG_MAXENT'
    elif SUPERTAGGINGSTRATEGY == 'tmax':
        SUPERTAGGINGSTRATEGY = 'CCG_OVERT_MG_MAXENT'
    elif SUPERTAGGINGSTRATEGY == 'hybmax':
        SUPERTAGGINGSTRATEGY = 'CCG_MG_HYBRID_MAXENT'
    elif SUPERTAGGINGSTRATEGY == 'hybabmax':
        SUPERTAGGINGSTRATEGY = 'CCG_MG_HYBRID_NS_MAXENT'
    elif SUPERTAGGINGSTRATEGY == 'hybuni':
        SUPERTAGGINGSTRATEGY = 'CCG_MG_HYBRID'
    elif SUPERTAGGINGSTRATEGY == 'stabmax':
        SUPERTAGGINGSTRATEGY = 'CCG_MG_SUPERTAG_NS_MAXENT'
    elif SUPERTAGGINGSTRATEGY == 'stabuni':
        SUPERTAGGINGSTRATEGY = 'CCG_MG_SUPERTAG_NS'
    MAX_MG_CATS_PER_WORD = float("{0:.2f}".format(args.max_cats[0]))
    if MAX_MG_CATS_PER_WORD < 0:
        raise Exception("Error! Max MG categories must be a positive floating point number..")
    if AUTO_SECTION_FOLDERS not in [None, 'all', 'ALL', 'All']:
        for item in AUTO_SECTION_FOLDERS:
            if item not in os.listdir(ptb_folder):
                raise Exception("Error! The section folder or file '"+item+"' does not exist!")
    if ptb_folder[-1] == '/':
        ptb_folder=ptb_folder[:-1]
    seed_folder = ptb_folder+"_"+args.MGbankName[0]+"Seed"
    try:
        if ('SUPERTAG' not in SUPERTAGGINGSTRATEGY or 'MAXENT' not in SUPERTAGGINGSTRATEGY) and (all_verb_cats or all_cats):
            raise Exception("Error! all_cats/all_verb_cats options only works with maxent supertagging strategy (either without or without subcats)!")
        elif all_verb_cats and all_cats:
            all_verb_cats = False
    except Exception as e:
        x=0
    if seed_folder not in os.listdir(os.getcwd()):
        os.mkdir(seed_folder)
    if args.auto_gen:
        overwrite_autos = args.overwrite_auto
    else:
        overwrite_autos = None
    gen_MGbank(ptb_folder, ccg_auto_folder, args.MGbankName[0], AUTO_SECTION_FOLDERS, overwrite_autos, START_LINE, START_FILE, END_FILE, MIN_LENGTH, MAX_LENGTH, CCG_BEAM, CCG_BEAM_FLOOR, TIMEOUT, PARSER_SETTING, STOP_AFTER, MAX_MG_CATS_PER_WORD, args.use_all_null, args.use_autos, SUPERTAGGINGSTRATEGY, TRAIN_TAGGER, SUPER_RARE_CUTOFF, SUPER_TAG_DICT_MIN, SUPER_CATEGORY_CUTOFF, SUPER_FORWARD_BEAM_RATIO, SUPER_BEAM_RATIO, SUPER_BEAM_WIDTH, args.extract_deps, args.skip_rel, args.skip_pro, args.ptbmv, MAXMOVEDIST, args.ptbct, args.allow_best_tag, args.frag)
    
