#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Note, you must install simplediff for this code to work.. on mac do: pip install simplediff

from __future__ import division
from functools import partial
from Tkinter import *
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from nltk import tokenize
from nltk import Tree
import nltk
import pdb
import re
from functools import partial
from timeit import default_timer
import cky_mg
import json
import gen_derived_tree
import autobank
import ast
import os
import shutil
import copy
import string
import astar_mg
sys.path.append('./SuperSuperTagger/scripts/supertagging')
try:
    import Supertagger
except ImportError:
    x=0
from timeout import timeout
import time
try:
    import simplediff
    simpleDiffEnabled = True
except ImportError:
    #to get simple diff, type into the terminal: pip install simplediff
    simpleDiffEnabled = False
from shutil import copytree
from shutil import rmtree
sys.setrecursionlimit(10000)
brackets='()'
open_b, close_b = brackets
open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
node_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
leaf_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
names_to_remove_from_male = ['Claire', 'Georgia']
names_to_remove_from_female = []
non_section_folders = ['mg_lstm_corpus', 'mg_lstm_corpus_ns', 'nameCats', 'new_parses', 'new_parse_strings', 'CCG_MG_SUPERTAG_taggingModel1', 'CCG_MG_SUPERTAG_taggingModel2', 'CCG_MG_SUPERTAG_taggingModel3', 'CCG_MG_SUPERTAG_taggingModel4', 'CCG_MG_SUPERTAG_NS_taggingModel1', 'CCG_MG_SUPERTAG_NS_taggingModel2', 'CCG_MG_SUPERTAG_NS_taggingModel3', 'CCG_MG_SUPERTAG_NS_taggingModel4', 'CCG_MG_ATOMIC_taggingModel1', 'CCG_MG_ATOMIC_taggingModel2', 'CCG_MG_ATOMIC_taggingModel3', 'CCG_MG_ATOMIC_taggingModel4', 'CCG_MG_HYBRID_taggingModel1', 'CCG_MG_HYBRID_taggingModel2', 'CCG_MG_HYBRID_taggingModel3', 'CCG_MG_HYBRID_taggingModel4', 'OvertLexicon', '.DS_Store', 'parserSettings', 'timeAndDate', 'CatTreeMappings', 'covertCatComments', 'CovertLexicon', 'ExtraposerLexicon', 'nullCatTreeMappings', 'NullExcorporatorLexicon', 'nullTreeCatMappings', 'overtCatComments', 'PosMappings', 'ToughOperatorLexicon', 'TreeCatMappings', 'TypeRaiserLexicon', 'DepMappings', 'PosDepsMappings', 'RevDepMappings', 'ccg_mg_supertagged_model1_corpus', 'ccg_mg_supertagged_model2_corpus', 'ccg_mg_supertagged_model3_corpus', 'ccg_mg_supertagged_model4_corpus', 'ccg_mg_supertagged_ns_model1_corpus', 'ccg_mg_supertagged_ns_model2_corpus', 'ccg_mg_supertagged_ns_model3_corpus', 'ccg_mg_supertagged_ns_model4_corpus', 'ccg_mg_hybrid_model1_corpus', 'ccg_mg_hybrid_ns_model1_corpus', 'ccg_mg_hybrid_ns_model2_corpus', 'ccg_mg_hybrid_ns_model3_corpus', 'ccg_mg_hybrid_ns_model4_corpus', 'ccg_mg_hybrid_model2_corpus', 'ccg_mg_hybrid_model3_corpus', 'ccg_mg_hybrid_model4_corpus', 'REF_MGST_table', 'MGST_REF_table', 'REF_MGHSTNS_table', 'MGHSTNS_REF_table', 'REF_MGHST_table', 'MGHST_REF_table', 'ccg_mg_tagged_corpus_model1', 'ccg_mg_tagged_corpus_model2', 'ccg_mg_tagged_corpus_model3', 'ccg_mg_tagged_corpus_model4', 'REF_MGT_table', 'MGT_REF_table', 'REF_MGSTNS_table', 'MGSTNS_REF_table', 'stagmaxent.model1', 'stagmaxent.model2', 'stagmaxent.model3', 'stagmaxent.model4', 'stnsagmaxent.model1', 'stnsagmaxent.model2', 'stnsagmaxent.model3', 'stnsagmaxent.model4', 'tagmaxent.model1', 'tagmaxent.model2', 'tagmaxent.model3', 'tagmaxent.model4', 'hybmaxent.model1', 'hybmaxent.model2', 'hybmaxent.model3', 'hybmaxent.model4', 'autoCatTreeMappings', 'autoNullCatTreeMappings', 'autoTreeCatMappings', 'autoNullTreeCatMappings']

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

names = nltk.corpus.names
male_names = names.words('male.txt')+['mr.', 'mr', 'mister']
for name in names_to_remove_from_male:
    male_names.remove(name)
index = -1
for name in male_names:
    index+=1
    male_names[index] = male_names[index].lower()
female_names = names.words('female.txt')+['mrs.', 'mrs', 'miss.', 'miss', 'ms', 'ms.']
for name in names_to_remove_from_female:
    female_names.remove(name)
index = -1
for name in female_names:
    index+=1
    female_names[index] = female_names[index].lower()

cat_feature_pattern = re.compile("[^(=|(\+)|(\-)|~|≈|(\^)|&|\?|\!|\>|\<)]+")

extraposition_hosts = ['t', 'v', 'p', 'c', 'd', 'D']

autoFilesToIgnore = ['timeAndDate', '.DS_Store', 'autoCatTreeMappings', 'autoNullCatTreeMappings', 'autoTreeCatMappings', 'autoNullTreeCatMappings', 'OvertLexicon']

PTBcats = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS', 'IN', 'PDT', 'WDT', 'DT',
               'CC', 'CD', 'RB', 'RBR', 'RBS', 'WRB', 'RP', 'EX', 'FW', 'JJ', 'JJR', 'JJS',
               'LS', 'MD', 'POS', 'WP', 'WP$', 'PRP', 'SYM', 'TO', 'UH', '.', ':', "''", '``', '$', ',', '',
           'LRB', 'RRB', '#', 'NML', 'NNM', 'NNF', 'NNMF', 'NNPM', 'NNPF', 'NNPMF', 'PRP3SG', 'PRPM3SG', 'PRPF3SG',
           'PRP1SG', 'PRP1PL', 'PRP3PL', 'PRP3SGSELF', 'PRPM3SGSELF', 'PRPF3SGSELF',
           'PRP1SGSELF', 'PRP2','PRP2SGSELF', 'PRP2PLSELF', 'PRP1PLSELF', 'PRP3PLSELF', 'PRP$3SG', 'PRP$M3SG', 'PRP$F3SG',
           'PRP$1SG', 'PRP$2', 'PRP$1PL', 'PRP$3PL', 'HYPH', 'NEG', 'DTSG', 'DTPL']
covertLexiconFiles = ['CovertLexicon', 'ExtraposerLexicon', 'NullExcorporatorLexicon', 'ToughOperatorLexicon',
                      'TypeRaiserLexicon']
PTBcats.sort()
not_selectee_feature = re.compile('=|(\+)|(\-)|~')

globVar = None

class TerminalsFrame(Frame):
    def __init__(self, root):
        Frame.__init__(self, root)
        self.canvas = Canvas(root, borderwidth=0, background="#ffffff", height=620, width=425)
        self.frame = Frame(self.canvas, background="#ffffff")
        self.vsb = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.hsb = Scrollbar(root, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.hsb.set)
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=False)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", 
                                  tags="self.frame")

    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

class autobankGUI():
    def __init__(self, PTBtree, terminals, PosMappings, seeds, ptb_folder, seed_folder, auto_folder, section_folder, ptb_file_line_number, ptb_file, CovertLexicon, ExtraposerLexicon, TypeRaiserLexicon, ToughOperatorLexicon, NullExcorporatorLexicon, DepMappings, RevDepMappings, PosDepsMappings, covertCatComments, overtCatComments, searchItem, match, seedSentLen, stringLengths, counts, displayIndex, PTBbracketing, totalSourceTrees, newSentMode, test_words, untokenizedTestSentence, MGbankName, overwrite_auto, useAutos, supertaggingStrategy, train_tagger, super_rare_cutoff, super_tag_dict_min, super_category_cutoff, super_forward_beam_ratio, super_beam_ratio, super_beam_width, extract_dep_mappings, ptb_word_token_count, task, supertagger):
        self.DepMappings = DepMappings
        self.overwrite_auto = overwrite_auto
        self.RevDepMappings = RevDepMappings
        self.PosDepsMappings = PosDepsMappings
        self.ptb_folder = ptb_folder
        self.seed_folder = seed_folder
        self.auto_folder = auto_folder
        self.section_folder = section_folder
        self.ptb_file = ptb_file
        self.CovertLexicon = CovertLexicon
        self.ExtraposerLexicon = ExtraposerLexicon
        self.TypeRaiserLexicon = TypeRaiserLexicon
        self.ToughOperatorLexicon = ToughOperatorLexicon
        self.NullExcorporatorLexicon = NullExcorporatorLexicon
        self.covertLexicons = [(self.CovertLexicon, 'CovertLexicon'), (self.ExtraposerLexicon, 'ExtraposerLexicon'), (self.TypeRaiserLexicon, 'TypeRaiserLexicon'), (self.ToughOperatorLexicon, 'ToughOperatorLexicon'), (self.NullExcorporatorLexicon, 'NullExcorporatorLexicon')]
        self.covertCatComments = covertCatComments
        self.overtCatComments = overtCatComments
        self.seeds = seeds
        self.previousPTBtree = False
        self.ptb_file_line_number = ptb_file_line_number
        self.miniOvertLexicon = []
        self.PosMappings = PosMappings
        self.words = [ terminal.name.lower() for terminal in terminals ]
        self.useAutos = useAutos
        self.train_tagger = train_tagger
        self.extract_dep_mappings = extract_dep_mappings
        preterminals = [ terminal.mother for terminal in terminals ]
        i = -1
        for preterminal in preterminals:
            if preterminal.name == '-LRB-':
                preterminal.name = 'LRB'
            elif preterminal.name == '-RRB-':
                preterminal.name = 'RRB'
        self.PTBPosTags = [ preterminal.name.split('-')[0] for preterminal in preterminals ]
        self.PTBtree = PTBtree
        self.quit = False
        self.overwriteAuto = None
        self.start_auto_gen = False
        self.nextPTBtree = False
        self.currentPTBtree = False
        self.checkedForDuplicates = False
        self.morphemeName = None
        self.MGfeatures = None
        self.covertLexUpdated = False
        self.oldCatIsNewCat = False
        self.search = False
        self.length = None
        self.PTBbracketingWindow = None
        self.confirmTrainTaggerWindow = None
        self.confirmLoadTaggerWindow = None
        self.sameSearch = False
        self.goto = False
        self.removingFromFilebox = False
        self.movingFromFilebox = False
        self.backButton = None
        self.removeParseButton = None
        self.lenRemovedTree = None
        self.lenMovedTree = None
        self.treeMoved = False
        self.ptb_search = False
        self.inLoop = False
        self.ptb_search_len = None
        self.ptbLineNum = None
        self.fileManagerFolder = None
        self.newSeedSentLen = None
        self.oldAndNewPosMappingsMatch = False
        self.oldAndNewPosMappingsSubset = False
        self.innerInnerInnerLowerTreeFrame = None
        self.terminals_frame = None
        self.arg1 = None
        self.arg2 = None
        self.arg1CatsBox = None
        self.arg2CatsBox = None
        self.arg1CatsBoxIndex = None
        self.arg2CatsBoxIndex = None
        self.lowerTreeTypeDerivationSpin = None
        self.derivationSpin = None
        self.lowerTreeLabel = None
        self.lowerTreeSpinFrame = None
        self.arg1TreeTypeSpin = None
        self.arg2TreeTypeSpin = None
        self.GUI_MATCH = True
        self.override_GUI_MATCH = False
        self.parserSettingsLastParse = None
        self.catsBox = None
        self.save_button = None
        self.treeRemoved = False
        self.supertagger = supertagger
        self.task = 'annotate'
        self.super_rare_cutoff = super_rare_cutoff
        self.super_tag_dict_min = super_tag_dict_min
        self.super_category_cutoff = super_category_cutoff
        self.super_forward_beam_ratio = super_forward_beam_ratio
        self.super_beam_ratio = super_beam_ratio
        self.super_beam_width = super_beam_width
        self.supertagStrategy = supertaggingStrategy
        self.lengthCountsOfRemovedTrees = {}
        self.newSentMode = newSentMode
        self.TASK = task
        self.test_words = test_words
        self.untokenizedTestSentence = untokenizedTestSentence
        self.totalSourceTrees = totalSourceTrees
        self.xbar_bracketings = []
        self.PTBbracketing = PTBbracketing
        self.parserSettings = json.load(open(self.seed_folder + '/' + 'parserSettings'))
        self.match = match
        self.seedSentLen = seedSentLen
        self.searchItem = searchItem
        self.stringLengths = stringLengths
        self.counts = counts
        self.displayIndex = displayIndex
        self.xbar_trees = []
        self.derivation_trees = []
        self.derived_trees = []
        self.MGbankName = MGbankName
        self.ptb_word_token_count = ptb_word_token_count
        if overwrite_auto == None:
            self.genMainWindow()
            if not self.newSentMode:
                self.showtrees(treeType='PTB', trees=[self.PTBtree])
        if 'nameCats' in os.listdir(self.seed_folder):
            self.nameCats = json.load(open(self.seed_folder + '/nameCats'))
        else:
            self.nameCats = {'male': [], 'female': []}
        with open(self.seed_folder + '/parserSettings', 'w') as (parserSettingsFile):
            json.dump(self.parserSettings, parserSettingsFile)
        if overwrite_auto != None:
            self.autoGenerateCorpus()
        else:
            mainloop()
        return

    def delete_empty_files_folders(self):
        #if a file is empty, it is deleted from the seed and auto file system
        for root_folder in [self.auto_folder, self.seed_folder]:
            for section_folder in sorted(os.listdir(root_folder)):
                if section_folder != 'new_parses' and section_folder in non_section_folders:
                    continue
                for FILE in sorted(os.listdir(root_folder+'/'+section_folder)):
                    if FILE == '.DS_Store':
                        continue
                    parses = json.load(open(root_folder+'/'+section_folder+'/'+FILE))
                    if parses == {}:
                        os.remove(root_folder+'/'+section_folder+'/'+FILE)

    def genMainWindow(self, mode='annotation'):
        self.mode = mode
        self.covertLexUpdated = False
        self.oldCatIsNewCat = False
        self.oldAndNewPosMappingsMatch = False
        self.oldAndNewPosMappingsSubset = False
        self.checkedForDuplicates = False
        self.eliminateButton = None
        self.eliminateEntry = None
        self.compareButton = None
        #self.autoSelectButton = None
        self.scoreButton = None
        self.add2SeedsButton = None
        self.overwriteWindow = None
        self.PTBtreeWidget = None
        self.TERMINALS = None
        self.globaltframe = None
        self.addOvertCatWindow = None
        self.addCovertCatWindow = None
        self.alreadyExistsWindow = None
        self.confirmReparseWindow = None
        self.nothingEntered = None
        self.noParsesFound = None
        self.PTBtreeWidget = None
        self.MGtreeWidget = None
        self.PTBwindow = None
        self.XBARwindow = None
        self.xbar_tframe = None
        self.treeTypeSpin = None
        self.spin = None
        self.noParse2accept = None
        self.viewDelCatsWindow = None
        self.confirmWindow = None
        self.modifyWindow = None
        self.statsWindow = None
        self.depMappingsWindow = None
        self.treeCompareWindow = None
        self.parsing = None
        self.viewParseQuestionWindow = None
        self.upperBracketingWindow = None
        self.lowerBracketingWindow = None
        self.diffWindow = None
        self.mainBracketingWindow = None
        self.fileManagerWindow = None
        self.parserSettingsWindow = None
        self.derivationWindow = None
        self.testSentenceWindow = None
        self.untaggedWordsWindow = None
        self.resultsTrees = None
        self.resultsExpressionList = None
        self.constructLexiconWindow = None
        self.aboutWindow = None
        self.confirmDeleteTreeWindow = None
        self.saveTreeWindow = None
        self.useSectionFoldersWindow = None
        self.useAutosWindow = None
        self.addedNewSentenceToSeeds = False
        self.confirmAddDerivationToSeedsWindow = None
        self.ptb_ccg_toggle = "ptb"
        self.mainWindow = Tk()
        self.mainWindow.protocol('WM_DELETE_WINDOW', lambda: self.mainWindowQuit())
        if self.mode == 'annotation':
            if self.newSentMode == True:
                self.mainWindow.title("New Sentence Annotator")
            else:
                self.mainWindow.title("Seed Set Annotator: "+self.seed_folder[4:-4])
            self.MGcats = []
            #we have an array housing the categories that the user selects for each word,
            #these are initially all set to None..
            #the widget housing the self.words and their drop-down menus
            #we arrange them into a grid
            self.TERMINALS_FRAME = Frame(self.mainWindow)
            self.TERMINALS_FRAME.pack(side='left', expand=True)
            self.refreshTerminals()
        elif self.mode == 'viewer':
            if self.corpusType == 'seed':
                self.mainWindow.title("Seed Tree Viewer: "+self.seed_folder[4:-4])
            elif self.corpusType == 'auto':
                self.mainWindow.title("Auto Tree Viewer: "+self.seed_folder[4:-4])
        self.mainWindow.geometry("10000x10000")
        menubar = Menu(self.mainWindow)
        self.mainWindow.config(menu=menubar)
        aboutMenu = Menu(menubar)
        aboutMenu.add_command(label="about", command = self.about)
        aboutMenu.add_separator()
        aboutMenu.add_command(label="quit", command = self.mainWindowQuit)
        menubar.add_cascade(label="Autobank", underline=0, menu=aboutMenu)
        categoriesMenu = Menu(menubar)
        overtSubmenu = Menu(categoriesMenu)
        overtSubmenu.add_command(label="Add Overt Category", command = self.addOvertCat, accelerator="Cmd+O")
        self.mainWindow.bind_all("<Command-o>", self.addOvertCatShortcut)
        overtSubmenu.add_command(label="View/Delete/Modify Overt Categories", command=lambda: self.viewDelCats('overt'), accelerator="Ctrl+O")
        self.mainWindow.bind_all("<Control-o>", self.viewDelCatsOvertShortcut)
        covertSubmenu = Menu(categoriesMenu)
        covertSubmenu.add_command(label="Add Covert Category", command=self.addCovertCat, accelerator="Cmd+N")
        self.mainWindow.bind_all("<Command-n>", self.addCovertCatShortcut)
        covertSubmenu.add_command(label="View/Delete/Modify Covert Categories", command=lambda: self.viewDelCats('covert'), accelerator="Ctrl+N")
        self.mainWindow.bind_all("<Control-n>", self.viewDelCatsCovertShortcut)
        categoriesMenu.add_cascade(label='Overt Categories', menu=overtSubmenu, underline=0)
        categoriesMenu.add_cascade(label='Covert Categories', menu=covertSubmenu, underline=0)
        categoriesMenu.add_separator()
        #categoriesMenu.add_command(label="Exit", underline=0, command=self.printExit)
        menubar.add_cascade(label="Categories", underline=0, menu=categoriesMenu)
        if self.newSentMode == False:
            #I had to hide the corpus menu when in new sentence mode.. can't remember why!  But don't mess with this.. there was a good reason!
            corpusMenu = Menu(menubar)
            corpusSubmenu = Menu(corpusMenu)
            corpusSubmenu.add_command(label="View Corpora Stats", command = self.viewCorpusStats, accelerator="Ctrl+S")
            self.mainWindow.bind_all("<Control-s>", self.viewCorpusStatsShortcut)
            corpusSubmenu.add_command(label="View (non-seed) PTB Trees", command=lambda: self.fileManager(self.ptb_folder), accelerator="Cmd+U")
            self.mainWindow.bind_all("<Command-u>", self.fileManagerUnannotatedShortcut)
            corpusSubmenu.add_command(label="View Seed MGbank Trees", command=lambda: self.fileManager(self.seed_folder), accelerator="Cmd+S")
            self.mainWindow.bind_all("<Command-s>", self.fileManagerSeedShortcut)
            corpusSubmenu.add_command(label="View Auto MGbank Trees", command=lambda: self.fileManager(self.auto_folder), accelerator="Cmd+A")
            self.mainWindow.bind_all("<Command-a>", self.fileManagerAutoShortcut)
            corpusSubmenu.add_command(label="Update Dependency Mappings",  command=lambda: self.autoGenerateCorpus(updateDepMappingsOnly=True), accelerator="Cmd+Shift+U")
            self.mainWindow.bind_all("<Command-Shift-u>", self.updateDepMappingsShortcut)
            corpusSubmenu.add_command(label="Reparse all seeds",  command=lambda: self.confirm_reparse_all_trees(corpus='Seeds'), accelerator="Cmd+Shift+S")
            self.mainWindow.bind_all("<Command-Shift-s>", self.reparse_all_seedsShortcut)
            corpusSubmenu.add_command(label="Reparse all autos",  command=lambda: self.confirm_reparse_all_trees(corpus='Autos'), accelerator="Cmd+Shift+A")
            self.mainWindow.bind_all("<Command-Shift-a>", self.reparse_all_autosShortcut)
            corpusSubmenu.add_command(label="Extract Overt Lexicon",  command=lambda: self.confirmConstructLexicon(), accelerator="Cmd+L")
            self.mainWindow.bind_all("<Command-l>", self.confirmConstructLexiconShortcut)
            corpusSubmenu.add_command(label="Backup seeds and autos", command = self.createBackup, accelerator="Cmd+Shift+B")
            self.mainWindow.bind_all("<Command-Shift-b>", self.createBackupShortcut)
            corpusMenu.add_separator()
            menubar.add_cascade(label="Corpora", underline=0, menu=corpusSubmenu)
        settingsMenu = Menu(menubar)
        settingsMenu.add_command(label="View/edit Parser Settings", command = self.displayParserSettings, accelerator="Cmd+P")
        self.mainWindow.bind_all("<Command-p>", self.displayParserSettingsShortcut)
        menubar.add_cascade(label='Settings', underline=0, menu=settingsMenu)
        testSentenceMenu = Menu(menubar)
        testSentenceMenu.add_command(label='Annotate New Sentence', command=lambda : self.newTestSent('annotate'), accelerator='Cmd+Shift+N')
        self.mainWindow.bind_all('<Command-Shift-n>', self.annotateTestSentShortcut)
        testSentenceMenu.add_command(label='Parse New Sentence', command=lambda : self.newTestSent('parse'), accelerator='Cmd+Shift+P')
        self.mainWindow.bind_all('<Command-Shift-p>', self.parseTestSentShortcut)
        testSentenceMenu.add_command(label='Train LSTM supertagger', command=self.trainLstmSupertagger, accelerator='Cmd+T')
        self.mainWindow.bind_all('<Command-Shift-t>', self.trainLstmSupertaggerShortcut)
        testSentenceMenu.add_command(label='load LSTM supertagger', command=self.loadLstmSupertagger, accelerator='Cmd+L')
        self.mainWindow.bind_all('<Command-Shift-l>', self.loadLstmSupertaggerShortcut)
        testSentenceMenu.add_separator()
        if self.newSentMode == True:
            testSentenceMenu.add_command(label="Exit New Sentence Mode", command = self.exitnewSentMode, accelerator="Cmd+Shift+E")
            self.mainWindow.bind_all("<Command-Shift-e>", self.exitnewSentModeShortcut)
        menubar.add_cascade(label="New Sentence", underline=0, menu=testSentenceMenu)
        testSentenceMenu.add_separator()
        self.globaltframe = Frame(self.mainWindow)
        self.globaltframe.pack(side='left', fill=BOTH, expand=True)
        if not self.newSentMode and not self.section_folder == 'new_parses':
            self.uppertframe = Frame(self.globaltframe)
            self.uppertframe.pack(fill=BOTH, expand=True)
        else:
            self.ptb_label = Label(self.globaltframe, text=self.untokenizedTestSentence)
            self.ptb_label.pack()
        if self.newSentMode or self.section_folder == 'new_parses':
            #if we're in new sentence mode then we want the buttons below lowertframe so we
            #create the frame earlier than normal
            self.lowertframe = Frame(self.globaltframe)
            self.lowertframe.pack(fill=BOTH, expand=True)
        self.mainButtonFrame = Frame(self.globaltframe)
        self.mainButtonFrame.pack()
        if mode == 'annotation' and not self.TASK == 'parse':
            derivation_button = Button(self.mainButtonFrame, text='derive', command=self.derivationBuilder)
            derivation_button.pack(side='left')
            parse_button = Button(self.mainButtonFrame, text='parse', command=self.Parse)
            parse_button.pack(side='left')
            if self.newSentMode == False and (self.match == False or self.match == 'False' or self.match == 'checked'):
                derivation_button.config(state=DISABLED)
                parse_button.config(state=DISABLED)
        if mode == 'viewer':
            cycleBackButton = Button(self.mainButtonFrame, text='<', command=lambda: self.cycleParses(direction='back'))
            cycleBackButton.pack(side='left')
            cycleForwardButton = Button(self.mainButtonFrame, text='>', command=lambda: self.cycleParses(direction='forward'))
            cycleForwardButton.pack(side='left')
            emptyLabel = Label(self.mainButtonFrame, text="   ")
            emptyLabel.pack(side='left')
        quit_button = Button(self.mainButtonFrame, text='quit', command=self.mainWindowQuit)
        quit_button.pack(side='left')
        if self.newSentMode:
            if not self.TASK == 'parse':
                exit_button = Button(self.mainButtonFrame, text='exit new sentence mode', command=self.exitnewSentMode)
                exit_button.pack(side='left')
                self.save_button = Button(self.mainButtonFrame, text='save to seeds', command=lambda: self.saveTree1())
                self.save_button.pack(side='left')
                self.save_button.config(state=DISABLED)
            else:
                exit_button = Button(self.mainButtonFrame, text='exit new parse mode', command=self.exitnewSentMode)
                exit_button.pack(side='left')
        if mode == 'viewer':
            if self.section_folder != 'new_parses':
                i=-1
                for line in open(self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file):
                    i+=1
                    if i == self.ptb_file_line_number:
                        ptb_bracketing = line
                        ptb_bracketing = ptb_bracketing.encode('ascii')
                if self.corpusType == 'seed':
                    seeds = json.load(open(self.seed_folder+"/"+self.section_folder+"/"+self.ptb_file))
                    subcat_derivation_bracketing = seeds[str(self.ptb_file_line_number)][0]
                elif self.corpusType == 'auto':
                    autos = json.load(open(self.auto_folder+"/"+self.section_folder+"/"+self.ptb_file))
                    subcat_derivation_bracketing = autos[str(self.ptb_file_line_number)][0]
                try:
                    subcat_derivation_bracketing = subcat_derivation_bracketing.encode('utf8')#.decode('unicode_escape')
                except UnicodeDecodeError:
                    x=0
                (X, Y, xbar_tree) = gen_derived_tree.main(subcat_derivation_bracketing, show_indices=True, return_xbar_tree=True, allowMoreGoals=True)
                (PTB_deps, terminals, PTB_tree) = autobank.get_deps_terminals(ptb_bracketing, False)
                self.scoreButton = Button(self.mainButtonFrame, text='score', command=lambda: self.displayTreeScore([xbar_tree], PTB_deps, PTB_tree, terminals))
                self.scoreButton.pack(side='left')
            if self.corpusType == 'seed':
                removeButton = Button(self.mainButtonFrame, text='remove from seeds', command=lambda: self.confirm(confirmWhat='removeParse', message = "Are you sure you wish to remove\nthis parse from the seed set?"))
                reparseButton = Button(self.mainButtonFrame, text='reparse', command=lambda: self.confirm_reparse_all_trees(singleTree=self.seed, corpus='Seeds'))
                reparseButton.pack(side='left')
            elif self.corpusType == 'auto':
                removeButton = Button(self.mainButtonFrame, text='remove from autos', command=lambda: self.confirm(confirmWhat='removeParse', message = "Are you sure you wish to remove\nthis parse from the auto set?"))
                openInAnnotatorButton = Button(self.mainButtonFrame, text='open in annotator', command=self.startNewPtbSearch)
                openInAnnotatorButton.pack(side='left')
                reparseButton = Button(self.mainButtonFrame, text='reparse', command=lambda: self.confirm_reparse_all_trees(singleTree=self.seed, corpus='Autos'))
                reparseButton.pack(side='left')
                self.moveParseButton = Button(self.mainButtonFrame, text='move to seeds', command=lambda: self.moveParse('view'))
                self.moveParseButton.pack(side='left')
            returnButton = Button(self.mainButtonFrame, text='return to annotation', command=self.viewCurrentPTBtree)
            removeButton.pack(side='left')
            returnButton.pack(side='left')
        if not self.newSentMode and not self.section_folder == "new_parses":
            self.lowertframe = Frame(self.globaltframe)
            self.lowertframe.pack(fill=BOTH, expand=True)
            self.outer_PTB_tframe = Frame(self.uppertframe, relief=SUNKEN)
            self.outer_PTB_tframe.pack(side='left', fill=BOTH, expand=True)
            self.PTB_tframe = Frame(self.outer_PTB_tframe, relief=SUNKEN)
            self.PTB_tframe.pack(side='left', fill=BOTH, expand=True)
            self.PTBwindow = CanvasFrame(self.PTB_tframe, highlightthickness=2, highlightbackground='black', bg='white',height=10)
            self.PTBwindow.pack(fill=BOTH, expand=True)
        if self.match != False and self.match != 'False' and self.newSentMode == False and self.section_folder != "new_parses":
            self.ptb_label = Label(self.PTB_tframe, text='Source   (File:  '+self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"      Line Number: "+str(self.ptb_file_line_number)+")")
            self.ptb_label.pack()
        self.outer_xbar_tframe = Frame(self.lowertframe, relief=SUNKEN)
        self.outer_xbar_tframe.pack(side='left', fill=BOTH, expand=True)
        self.freshXbarWindow()
        if self.newSentMode == False and self.section_folder != "new_parses" and (self.match == False or self.match == 'False'):
            self.match = 'checked'
            self.nothing(self.mainWindow, "No matches found at that string length!\nHint: try selecting another string length.", height = 70)
        self.autoSelectEliminateQuestionWindow = None
        if self.TASK == 'parse':
            parse_result = self.astar_parse()
            if not parse_result:
                self.nothing(self.mainWindow, 'The parser failed to parse this sentence!')
                self.task = 'annotate'
                self.TASK = 'annotate'
        return

    def trainLstmSupertagger(self):
        if self.confirmTrainTaggerWindow != None:
            self.destroyWindow(self.confirmTrainTaggerWindow, 'confirmTrainTaggerWindow')
        self.confirmTrainTaggerWindow = Toplevel(self.mainWindow)
        self.confirmTrainTaggerWindow.title("Confirm Train Supertagger")
        self.confirmTrainTaggerWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.confirmTrainTaggerWindow, 'confirmTrainTaggerWindow'))
        w=500
        h=55
        (x, y) = self.getCentrePosition(w, h)
        self.confirmTrainTaggerWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.confirmTrainTaggerWindow, text="What type of supertagger would you like to train?")
        label.pack()
        buttonFrame = Frame(self.confirmTrainTaggerWindow)
        buttonFrame.pack()
        reifiedButton = Button(buttonFrame, text="Reified Supertagger", command=lambda: self.trainLstmSupertagger2(False))
        reifiedButton.pack(side='left')
        abstractButton = Button(buttonFrame, text="Abstract Supertagger", command=lambda: self.trainLstmSupertagger2(True))
        abstractButton.pack(side='left')
        cancelButton = Button(buttonFrame, text="Cancel", command=lambda: self.destroyWindow(self.confirmTrainTaggerWindow, 'confirmTrainTaggerWindow'))
        cancelButton.pack(side='left')

    def trainLstmSupertagger2(self, ns):
        if self.confirmTrainTaggerWindow != None:
            self.destroyWindow(self.confirmTrainTaggerWindow, 'confirmTrainTaggerWindow')
        if 'SuperSuperTagger' not in os.listdir('./'):
            self.nothing(self.mainWindow, "Could not find the folder "+os.getcwd()+'/SuperSuperTagger/')
            return
        self.extractTaggingModels('supertag_lstm', gen_new_ccg_trees=False, ns=ns, buildCorpusOnly=True, train=90, dev=5, test=5)
        os.chdir('./SuperSuperTagger')
        if ns:
            model_dir = 'abstract_model'
            data_dir = 'abstract_data'
        else:
            model_dir = 'reified_model'
            data_dir = 'reified_data'
        os.system("./scripts/run.sh edin.supertagger.MainTrain --model_dir ./"+model_dir+" --hyper_params_file ./configs/MG_tagger_ELMo.yaml --embeddings_dim 100 --train_file ./"+data_dir+"/train --dev_file ./"+data_dir+"/dev --epochs 20")
        #os.system("java -Xmx1G -cp ./target/scala-2.12/*.jar edin.supertagger.MainTrain --model_dir ./"+model_dir+" --hyper_params_file ./configs/supertagger_model_desc.yaml --embedding_file ./glove/glove.6B.100d.txt --embeddings_lowercased true --train_file ./"+data_dir+"/train --dev_file ./"+data_dir+"/dev --epochs 20 --all_in_memory true --dynet-mem 4000")
        os.chdir('../')
        self.nothing(self.mainWindow, "Training of Supertagger Complete!")

    def loadLstmSupertagger(self):
        if self.confirmLoadTaggerWindow != None:
            self.destroyWindow(self.confirmLoadTaggerWindow, 'confirmLoadTaggerWindow')
        if self.supertagger != None:
            self.nothing(self.mainWindow, "Supertagger already loaded!\n(You must quit and relaunch Autobank to load another supertagger..)", width=600, height=71)
            return
        self.confirmLoadTaggerWindow = Toplevel(self.mainWindow)
        self.confirmLoadTaggerWindow.title("Confirm Load Supertagger")
        self.confirmLoadTaggerWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.confirmLoadTaggerWindow, 'confirmLoadTaggerWindow'))
        w=500
        h=55
        (x, y) = self.getCentrePosition(w, h)
        self.confirmLoadTaggerWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.confirmLoadTaggerWindow, text="What type of supertagger would you like to load?")
        label.pack()
        buttonFrame = Frame(self.confirmLoadTaggerWindow)
        buttonFrame.pack()
        reifiedButton = Button(buttonFrame, text="Reified Supertagger", command=lambda: self.loadLstmSupertagger2('reified_model'))
        reifiedButton.pack(side='left')
        abstractButton = Button(buttonFrame, text="Abstract Supertagger", command=lambda: self.loadLstmSupertagger2('abstract_model'))
        abstractButton.pack(side='left')
        cancelButton = Button(buttonFrame, text="Cancel", command=lambda: self.destroyWindow(self.confirmLoadTaggerWindow, 'confirmLoadTaggerWindow'))
        cancelButton.pack(side='left')

    def loadLstmSupertagger2(self, model_dir):
        if model_dir.split("_")[0] == 'abstract':
            model_phrase = 'an abstract'
        else:
            model_phrase = 'a reified'
        if self.confirmLoadTaggerWindow != None:
            self.destroyWindow(self.confirmLoadTaggerWindow, 'confirmLoadTaggerWindow')
        if 'SuperSuperTagger' not in os.listdir('./') or model_dir not in os.listdir('./SuperSuperTagger'):
            self.nothing(self.mainWindow, "You must train "+model_phrase+" supertagger first!", width=400)
            return
        self.supertagger = Supertagger.Supertagger('./SuperSuperTagger/'+model_dir)
        self.supertagger.model_dir = model_dir
        if 'abstract' in model_dir:
            self.supertagger.data_dir = 'abstract_data'
        else:
            self.supertagger.data_dir = 'reified_data'
        self.nothing(self.mainWindow, "Supertagger Loaded Successfully!")
    
    def astar_parse(self, k=40, tag_dict_threshold=3, seed_tag_dict_threshold=3):
        reload(astar_mg)
        if tag_dict_threshold != None:
            tag_dict = json.load(open('./SuperSuperTagger/'+self.supertagger.data_dir+'/tag_dict'))
        if seed_tag_dict_threshold != None:
            seed_tag_dict = json.load(open('./SuperSuperTagger/'+self.supertagger.data_dir+'/seed_tag_dict'))
        print '\nSupertagging sentence: '+(' ').join(self.test_words)+'\n'
        best_k = self.supertagger.bestK(self.test_words, k, aux_tags=None)
        print 'Supertagging successful..'
        best_k_file = open('./SuperSuperTagger/'+self.supertagger.model_dir+'/best_k', 'w')
        for i in range(len(self.test_words)):
            best_k_file.write(self.test_words[i]+'\t'+'X'+'\t')
            for tag in best_k[i][:-1]:
                best_k_file.write(tag[0]+" "+str(tag[1])+'\t')
            best_k_file.write(best_k[i][-1][0]+" "+str(best_k[i][-1][1]))
            if i != len(self.test_words)-1:
                best_k_file.write('\n')
        best_k_file.close()
        best_k_file = open('./SuperSuperTagger/'+self.supertagger.model_dir+'/best_k')
        REF_MGST_table = json.load(open('SuperSuperTagger/'+self.supertagger.data_dir+'/REF_MGST_table'))
        supertag_lists = []
        for line in best_k_file:
            supertag_list = [st for st in line.split('\t')[2:] if st[0] != '<' and line != '\n']
            if supertag_list == []:
                continue
            if supertag_list[-1][-1] == '\n':
                supertag_list[-1] = supertag_list[-1][:-1]
            supertag_lists.append(supertag_list)
        best_k_file.close()
        supertags = []
        for INDEX in range(len(self.test_words)):
            for supertag in supertag_lists[INDEX]:
                parts = supertag.split(' ')
                supertags.append([REF_MGST_table[parts[0]], float(parts[1]), INDEX])
        new_supertags = []
        word_tagged = {}
        for MGcat in supertags:
            MGcat = copy.deepcopy(MGcat)
            INDEX = MGcat[2]
            if INDEX not in word_tagged:
                word_tagged[INDEX] = False
            st = MGcat[0]
            if type(st[0][0][0]) != type([]) and type(st[0][0][0]) != type(()):
                if st[0][0] == 'OVERT_WORD':
                    overt_cat = copy.deepcopy(st)
                    st[0] = (self.test_words[INDEX], st[0][1], st[0][2])
                st[2] = INDEX
            else:
                for link in st:
                    if link[0][0][0] == 'OVERT_WORD':
                        overt_cat = copy.deepcopy(link[0])
                        link[0][0] = (self.test_words[INDEX], link[0][0][1], link[0][0][2])
                    if link[0][2] == None:
                        link[0][2] = INDEX
                    if link[2][0][0] == 'OVERT_WORD':
                        overt_cat = copy.deepcopy(link[2])
                        link[2][0] = (self.test_words[INDEX], link[2][0][1], link[2][0][2])
                    if link[2][2] == None:
                        link[2][2] = INDEX
            include_supertag = True
            seed_threshold_met = False
            if seed_tag_dict_threshold != None:
                if self.test_words[INDEX] in seed_tag_dict and seed_tag_dict[self.test_words[INDEX]][0] >= seed_tag_dict_threshold:
                    seed_threshold_met = True
                    if overt_cat[0] not in seed_tag_dict[self.test_words[INDEX]]:
                        include_supertag = False
            if tag_dict_threshold != None and not seed_threshold_met:
                if self.test_words[INDEX] in tag_dict and tag_dict[self.test_words[INDEX]][0] >= tag_dict_threshold:
                    if overt_cat[0] not in tag_dict[self.test_words[INDEX]]:
                        include_supertag = False
            if include_supertag:
                word_tagged[INDEX] = True
                new_supertags.append(MGcat)
        #now we renormalize, as some supertags have been removed (because of both k and the tag_dict)
        totals = {}
        for st in new_supertags:
            INDEX = st[2]
            if INDEX not in totals:
                totals[INDEX] = st[1]
            else:
                totals[INDEX] += st[1]
        for st in new_supertags:
            INDEX = st[2]
            del(st[2])
            new_prob = (1/totals[INDEX])*st[1]
            st[1] = new_prob
        supertags = new_supertags
        skip_parse = False
        for index in word_tagged:
            if not word_tagged[index]:
                self.derivation_bracketings = []
                skip_parse = True
                break
        if not skip_parse:
            try:
                with timeout(self.parserSettings['timeout_seconds']):
                    start_time = default_timer()
                    (parse_time, self.derivation_bracketings, self.derived_bracketings, self.xbar_bracketings, self.xbar_trees, self.subcat_derivation_bracketings, self.subcat_full_derivation_bracketings, self.full_derivation_bracketings, lex_scores) = astar_mg.main(sentence=(' ').join(self.test_words), show_trees=False, print_expressions=False, return_bracketings=True, return_xbar_trees=True, allowMoreGoals=True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], limitRightwardMove=False, supertags=supertags, start_time=start_time, MAXMOVEDIST=None)
                    end_time = default_timer() - start_time
            except Exception as e:
                print "Error!!"
                self.derivation_bracketings = []
        else:
            print "\nNo tags were assigned to word at index position "+str(index)+" owing to the tag dict threshold!\n"
        if len(self.derivation_bracketings) == 0:
            print "\nNo parses discovered!\n"
            self.noParses()
            return False
        else:
            if len(self.derivation_bracketings) > 1:
                text = " parses discovered.\n"
            else:
                text = " parse discovered.\n"
            print "\n"+str(len(self.derivation_bracketings))+text
        got_PTB_DEPS = False
        self.xbar_trees = []
        self.derivation_trees = []
        self.subcat_derivation_trees = []
        self.subcat_full_derivation_trees = []
        self.full_derivation_trees = []
        self.derived_trees = []
        for i in range(len(self.xbar_bracketings)):
            db = cky_mg.fix_coord_annotation(self.derivation_bracketings[i])
            sdb = cky_mg.fix_coord_annotation(self.subcat_derivation_bracketings[i])
            sfdb = cky_mg.fix_coord_annotation(self.subcat_full_derivation_bracketings[i])
            fdb = cky_mg.fix_coord_annotation(self.full_derivation_bracketings[i])
            while "  " in sfdb:
                sfdb = re.sub("  ", " ", sfdb, count=10000)
            while "  " in sfdb:
                fdb = re.sub("  ", " ", sfdb, count=10000)
            try:
                xbar_tree = Tree.parse(self.xbar_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derivation_tree = Tree.parse(db, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_derivation_tree = Tree.parse(sdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_full_derivation_tree = Tree.parse(sfdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                full_derivation_tree = Tree.parse(fdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derived_tree = Tree.parse(self.derived_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            except AttributeError:
                xbar_tree = Tree.fromstring(self.xbar_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derivation_tree = Tree.fromstring(db, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_derivation_tree = Tree.fromstring(sdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_full_derivation_tree = Tree.fromstring(sfdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                full_derivation_tree = Tree.fromstring(fdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derived_tree = Tree.fromstring(self.derived_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            self.xbar_trees.append(xbar_tree)
            self.derivation_trees.append(derivation_tree)
            self.subcat_derivation_trees.append(subcat_derivation_tree)
            self.subcat_full_derivation_trees.append(subcat_full_derivation_tree)
            self.full_derivation_trees.append(full_derivation_tree)
            self.derived_trees.append(derived_tree)
        self.trees = self.xbar_trees
        self.refreshSpins()
        self.showtrees(treeType='MG')
        return True

    def cycleParses(self, direction):
        file_list = []
        if self.corpusType == 'seed':
            corpus_folder = self.seed_folder
        else:
            corpus_folder = self.auto_folder
        for section_folder in sorted(os.listdir(corpus_folder)):
            if section_folder not in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']:
                continue
            tuples = []
            for FILE in sorted(os.listdir(corpus_folder+"/"+section_folder)):
                if FILE == '.DS_Store':
                    continue
                file_num = int(FILE.split("_")[1].split(".")[0])
                tuples.append((FILE, file_num))
            sorted_tuples = sorted(tuples, key=lambda tup: tup[1])
            for tup in sorted_tuples:
                parses = json.load(open(corpus_folder+"/"+section_folder+"/"+tup[0]))
                line_nums = []
                for parse in parses:
                    line_nums.append(int(parse))
                line_nums.sort()
                for line_num in line_nums:
                    file_list.append([corpus_folder+"/"+section_folder+"/"+tup[0], line_num])
        if 'new_parses' in os.listdir(corpus_folder):
            for FILE in sorted(os.listdir(corpus_folder+"/new_parses/")):
                if FILE != '.DS_Store':
                    file_list.append([corpus_folder+"/new_parses/"+FILE, 0])
        current_parse_index = file_list.index(self.seed)
        try:
            if direction == 'forward':
                new_parse = file_list[current_parse_index+1]
            else:
                if current_parse_index == 0:
                    new_parse = file_list[current_parse_index]
                else:
                    new_parse = file_list[current_parse_index-1]
        except IndexError:
            new_parse = file_list[current_parse_index]
        self.openFileFolder(catSearch=False, fileSearch=False, cycleParse=new_parse)
        

    def saveTree1(self, seeds=None):
        if self.saveTreeWindow != None:
            self.destroyWindow(self.saveTreeWindow, 'saveTreeWindow')
        self.saveTreeWindow = Toplevel(self.mainWindow)
        self.saveTreeWindow.title("Confirm Save New Tree")
        self.saveTreeWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.saveTreeWindow, 'saveTreeWindow'))
        w=300
        h=90
        (x, y) = self.getCentrePosition(w, h)
        self.saveTreeWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.saveTreeWindow, text="Please enter a file name for this tree:")
        label.pack()
        self.saveTreeEntry = Entry(self.saveTreeWindow, width=30)
        self.saveTreeEntry.pack()
        self.saveTreeEntry.insert(END, self.untokenizedTestSentence)
        self.saveTreeEntry.focus_set()
        buttonFrame = Frame(self.saveTreeWindow)
        buttonFrame.pack()
        saveButton = Button(buttonFrame, text="Save", command=lambda: self.saveTree2(seeds))
        saveButton.pack(side='left')
        cancelButton = Button(buttonFrame, text="Cancel", command=lambda: self.destroyWindow(self.saveTreeWindow, 'saveTreeWindow'))
        cancelButton.pack(side='left')
        self.saveTreeEntry.bind('<Return>', lambda x: saveButton.invoke())

    def saveTree2(self, seeds):
        self.newSentFileName = self.saveTreeEntry.get().lower()
        if self.newSentFileName == '':
            self.nothing(self.mainWindow, "You must enter a name for this tree!")
            return
        for char in self.newSentFileName:
            if char not in string.ascii_letters and char not in string.digits and char != '_' and char not in ' ':
                self.nothing(self.mainWindow, "Please use only alphanumerics, spaces and underscores in filenames!", width = 500)
                return
        if "new_parses" not in os.listdir(self.seed_folder):
            os.mkdir(self.seed_folder+"/new_parses")
            os.mkdir(self.seed_folder+"/new_parse_strings")
        if self.newSentFileName in os.listdir(self.seed_folder+"/new_parses/"):
            self.nothing(self.mainWindow, "A file already exists with that name!")
            return
        with open(self.seed_folder+"/new_parse_strings/"+self.newSentFileName, 'w') as newSentenceFile:
            newSentenceFile.write(" ".join(self.test_words).lower())
        if self.saveTreeWindow != None:
            self.destroyWindow(self.saveTreeWindow, 'saveTreeWindow')
        if seeds == None:
            self.seeds = {}
            self.acceptParse(new_sent=True)
        else:
            self.seeds = seeds
            self.addedNewSentenceToSeeds = True
            self.acceptParse(new_sent=True, fromDerivationBuilder=True)
        self.untokenizedTestSentence = None

    def about(self):
        if self.aboutWindow != None:
            self.destroyWindow(self.aboutWindow, 'aboutWindow')
        self.aboutWindow = Toplevel(self.mainWindow)
        self.aboutWindow.title("Autobank")
        self.aboutWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.aboutWindow, 'aboutWindow'))
        w=500
        h=310
        (x, y) = self.getCentrePosition(w, h)
        self.aboutWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.aboutWindow, text="Autobank version: beta")
        label.pack()
        label = Label(self.aboutWindow, text="Minimalist Grammar and Treebank Building Software")
        label.pack()
        label = Label(self.aboutWindow, text="Created by John Torr, University of Edinburgh, 2015-2018")
        label.pack()
        label = Label(self.aboutWindow, text="Email: john.torr@cantab.net")
        label.pack()
        label = Label(self.aboutWindow, text="Funded by Nuance Communications Inc and the\nEngineering and Physical Sciences Research Council (EPSRC)")
        label.pack()
        label = Label(self.aboutWindow, text="All rights reserved")
        label.pack()
        label = Label(self.aboutWindow, text="-----------------------------------------------------------")
        label.pack()
        label = Label(self.aboutWindow, text="For citations please use the following paper:")
        label.pack()
        label = Label(self.aboutWindow, text='John Torr.  2017.  "Autobank: a semi-automatic annotation')
        label.pack()
        label = Label(self.aboutWindow, text='tool for developing deep Minimalist Grammar treebanks",')
        label.pack()
        label = Label(self.aboutWindow, text="in Proceedings of the EACL 2017 Software Demonstrations,")
        label.pack()
        label = Label(self.aboutWindow, text="Valencia, Spain, April 3-7 2017, pages 81–86")
        label.pack()
        okButton = Button(self.aboutWindow, text="Ok", command=lambda: self.destroyWindow(self.aboutWindow, 'nothingEntered'))
        okButton.pack()

    def newFullTreeWindow(self, treeType, trees=None, spin=None, mode=None, fromResults=True):
        n=1
        if treeType == 'MG':
            if trees == None:
                #for the PTB tree window we will feed this function a list with a single PTB tree
                #but for MG trees we need to choose the tree format, xbar, derivation etc
                try:
                    trees = self.trees
                except AttributeError:
                    self.nothing(self.mainWindow, "No MG parses to display!")
                    return
            if len(trees) > 1:
                try: n = int(spin.get())
                except Exception as e: n=1
            if mode == 'derivation':
                if fromResults:
                    if self.lowerTreeTypeDerivationSpin.get() == 'MG Derivation Cats+Subcat':
                        trees = [t[0] for t in trees]
                    elif self.lowerTreeTypeDerivationSpin.get() == 'MG Derivation Ops+Subcat':
                        trees = [t[1] for t in trees]
                    elif self.lowerTreeTypeDerivationSpin.get() == 'MG Derivation Cats':
                        trees = [t[2] for t in trees]
                    elif self.lowerTreeTypeDerivationSpin.get() == 'MG Derivation Ops':
                        trees = [t[3] for t in trees]
                    elif self.lowerTreeTypeDerivationSpin.get() == 'Xbar':
                        trees = [t[4] for t in trees]
                    elif self.lowerTreeTypeDerivationSpin.get() == 'MG Derived':
                        trees = [t[5] for t in trees]
                elif self.source == self.overtCatsBox:
                    try:
                        wstree = self.TREES[self.overtCatsBox.curselection()[0]][self.getDerivationTypeIndex(self.lowerTreeTypeDerivationSpin)]
                    except AttributeError:
                        wstree = self.TREES[self.overtCatsBox.curselection()[0]][0]
                title = "Partial MG tree "+str(n)+" for PTB tree: "+self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"  Ln: "+str(self.ptb_file_line_number+1)
            if self.newSentMode:
                title = "MG new sentence tree"
            elif self.mode == 'annotation':
                title = "MG tree "+str(n)+" for PTB tree: "+self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"  Ln: "+str(self.ptb_file_line_number+1)
            elif self.mode == 'viewer':
                title = "MG tree for PTB tree: "+self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"  Ln: "+str(self.ptb_file_line_number+1)
        elif treeType == 'PTB':
            if self.ptb_ccg_toggle == "ptb":
                title = "Penn Tree: "+self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"  Ln: "+str(self.ptb_file_line_number+1)
                trees = [self.PTBtree]
            elif self.ptb_ccg_toggle == "ccg":
                title = "CCGbank Tree: "+"CCGbank/"+self.section_folder+"/"+self.ptb_file.split(".")[0]+".ccg"+"  Ln: "+str(self.ptb_file_line_number+1)
                trees = [self.ccg_tree]
        fullTreeWindow = Toplevel(self.mainWindow)
        fullTreeWindow.title(title)
        fullTreeFrame = CanvasFrame(fullTreeWindow, highlightthickness=2, highlightbackground='black', bg='white',height=10)
        fullTreeWindow.geometry("10000x10000")
        fullTreeFrame.pack(fill=BOTH, expand=True)
        if mode == 'derivation' and not fromResults:
            treeWidget = TreeWidget(fullTreeFrame.canvas(), wstree, draggable=1, shapeable=1)
        else:
            treeWidget = TreeWidget(fullTreeFrame.canvas(), trees[n-1], draggable=1, shapeable=1)
        fullTreeFrame.add_widget(treeWidget, 10, 10)

    def addOvertCatShortcut(self, event):
        self.addOvertCat()

    def addCovertCatShortcut(self, event):
        self.addCovertCat()

    def viewDelCatsOvertShortcut(self, event):
        self.viewDelCats('overt')

    def viewDelCatsCovertShortcut(self, event):
        self.viewDelCats('covert')

    def fileManagerSeedShortcut(self, event):
        self.fileManager(self.seed_folder)

    def fileManagerUnannotatedShortcut(self, event):
        self.fileManager(self.ptb_folder)

    def fileManagerAutoShortcut(self, event):
        self.fileManager(self.auto_folder)

    def reparse_all_seedsShortcut(self, event):
        self.confirm_reparse_all_trees(corpus='Seeds')

    def reparse_all_autosShortcut(self, event):
        self.confirm_reparse_all_trees(corpus='Autos')

    def viewCorpusStatsShortcut(self, event):
        self.viewCorpusStats()

    def confirmConstructLexiconShortcut(self, event):
        self.confirmConstructLexicon()

    def displayParserSettingsShortcut(self, event):
        self.displayParserSettings()

    def annotateTestSentShortcut(self, event):
        self.newTestSent('annotate')
    
    def parseTestSentShortcut(self, event):
        self.newTestSent('parse')
    
    def trainLstmSupertaggerShortcut(self, event):
        self.trainLstmSupertagger()

    def loadLstmSupertaggerShortcut(self, event):
        self.loadLstmSupertagger()

    def exitnewSentModeShortcut(self, event):
        self.exitnewSentMode()

    def createBackupShortcut(self, event):
        self.createBackup()

    def updateDepMappingsShortcut(self, event):
        self.autoGenerateCorpus(updateDepMappingsOnly=True)

    def createBackup(self):
        backupFolderName = self.MGbankName+"_"+time.strftime("%d_%m_%Y")+"@"+time.strftime("%H.%M.%S")
        outerBackupFolderName = self.ptb_folder+'_corpusBackups'
        if self.ptb_folder+'_corpusBackups' not in os.listdir(os.getcwd()):
            os.mkdir(outerBackupFolderName)
        dst = outerBackupFolderName+"/"+backupFolderName
        try:
            shutil.rmtree(outerBackupFolderName+"/"+self.MGbankName+"_"+"LatestBackup")
        except Exception as e:
            x=0
        dst2 = outerBackupFolderName+"/"+self.MGbankName+"_"+"LatestBackup"
        os.mkdir(dst)
        shutil.copytree(self.ptb_folder+"_"+self.MGbankName+"Seed", dst+"/"+self.ptb_folder+"_"+self.MGbankName+"Seed")
        shutil.copytree(self.ptb_folder+"_"+self.MGbankName+"Auto", dst+"/"+self.ptb_folder+"_"+self.MGbankName+"Auto")
        shutil.copytree(self.ptb_folder+"_"+self.MGbankName+"Seed", dst2+"/"+self.ptb_folder+"_"+self.MGbankName+"Seed")
        shutil.copytree(self.ptb_folder+"_"+self.MGbankName+"Auto", dst2+"/"+self.ptb_folder+"_"+self.MGbankName+"Auto")
        self.nothing(self.mainWindow, "Data saved in: "+outerBackupFolderName+"/"+backupFolderName, width=525)

    def newTestSent(self, task):
        if task == 'annotate':
            foundOvertMGcat = False
            for entry in self.PosMappings:
                if self.PosMappings[entry] != [u'No categories available', u'']:
                    foundOvertMGcat = True
                    break
            if not foundOvertMGcat:
                self.nothing(self.mainWindow, 'You must add at least one overt MG category first!')
                return
        else:
            if task == 'parse':
                if not self.check_for_model_file():
                    self.nothing(self.mainWindow, 'No LSTM supertag model detected.. please train one first!')
                    return
                elif self.supertagger == None:
                    self.nothing(self.mainWindow, 'Please load the supertagger before parsing (New Sentence > load LSTM supertagger)', width=600)
                    return
        if self.testSentenceWindow != None:
            self.destroyWindow(self.testSentenceWindow, 'testSentenceWindow')
        self.testSentenceWindow = Toplevel(self.mainWindow)
        w = 500
        h = 160
        x, y = self.getCentrePosition(w, h)
        self.testSentenceWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.testSentenceWindow.protocol('WM_DELETE_WINDOW', lambda : self.destroyWindow(self.testSentenceWindow,'testSentenceWindow'))
        if task == 'annotate':
            self.testSentenceWindow.title('Annotate New Sentence')
        else:
            if task == 'parse':
                self.testSentenceWindow.title('Parse New Sentence')
        emptyLabel = Label(self.testSentenceWindow, text=' ')
        emptyLabel.pack()
        testSentenceInstructionFrame = Frame(self.testSentenceWindow)
        testSentenceInstructionFrame.pack(fill=X)
        emptyLabel = Label(self.testSentenceWindow, text=' ')
        emptyLabel.pack()
        if task == 'annotate':
            txt = '    Enter a new sentence to annotate:'
        else:
            if task == 'parse':
                txt = '    Enter a new sentence to be parsed:'
        instructionLabel = Label(testSentenceInstructionFrame, text=txt)
        instructionLabel.pack(side='left')
        testSentenceEntryFrame = Frame(self.testSentenceWindow)
        testSentenceEntryFrame.pack()
        self.testSentenceEntry = Entry(testSentenceEntryFrame, width=50)
        self.testSentenceEntry.pack(side='left')
        self.testSentenceEntry.focus_set()
        if self.untokenizedTestSentence != None:
            self.testSentenceEntry.insert(END, self.untokenizedTestSentence)
        emptyLabel = Label(self.testSentenceWindow, text=' ')
        emptyLabel.pack()
        buttonFrame = Frame(self.testSentenceWindow)
        buttonFrame.pack()
        if task == 'annotate':
            btnTxt = 'Annotate'
        else:
            if task == 'parse':
                btnTxt = 'Parse'
        annotateButton = Button(buttonFrame, text=btnTxt, command=lambda : self.startTestSession(task))
        annotateButton.pack(side=LEFT)
        self.testSentenceEntry.bind('<Return>', lambda x: annotateButton.invoke())
        closeButton = Button(buttonFrame, text='Close', command=lambda : self.destroyWindow(self.testSentenceWindow, 'testSentenceWindow'))
        closeButton.pack(side=LEFT)
        return

    def check_for_model_file(self):
        return True

    def startTestSession(self, task):
        self.test_words = tokenize.word_tokenize(self.testSentenceEntry.get())
        words_to_remove = []
        for i in range(len(self.test_words)):
            if self.test_words[i] in punctuation or self.test_words[i] == "''":
                words_to_remove.append(self.test_words[i])
        while len(words_to_remove) > 0:
            self.test_words.remove(words_to_remove[0])
            del(words_to_remove[0])
        for i in range(len(self.test_words)):
            self.test_words[i] = self.test_words[i].lower()
            try:
                float(self.test_words[i])
                self.test_words[i] = 'num'
            except ValueError:
                x=0
        self.untokenizedTestSentence = self.testSentenceEntry.get().strip().lower()
        if len(self.test_words) == 0:
            self.nothing(self.testSentenceWindow, "You must enter a new sentence!")
            return
        self.quit = False
        self.newSentMode = True
        self.task = task
        self.mainWindow.destroy()

    def exitnewSentMode(self):
        if self.match == 'checked':
            self.GUI_MATCH = False
        self.viewCurrentPTBtree()      

    def displayParserSettings(self):
        if self.parserSettingsWindow != None:
            self.destroyWindow(self.parserSettingsWindow, 'parserSettingsWindow')
        self.parserSettingsWindow = Toplevel(self.mainWindow)
        emptyLabel = Label(self.parserSettingsWindow, text=" ")
        emptyLabel.pack()
        w=430
        h=420
        (x, y) = self.getCentrePosition(w, h)
        self.parserSettingsWindow.geometry('%dx%d+%d+%d' % (w, h, x, y-75))
        self.parserSettingsWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.parserSettingsWindow, 'parserSettingsWindow'))
        self.parserSettingsWindow.title('Parser Settings')
        settingsFrame = Frame(self.parserSettingsWindow)
        settingsFrame.pack()
        timeOutFrame = Frame(settingsFrame)
        timeOutFrame.pack()
        timeoutLabel = Label(timeOutFrame, text="Parser timeout (secs): ")
        timeoutLabel.pack(side='left')
        self.timeoutEntry = Entry(timeOutFrame, width=5)
        self.timeoutEntry.pack(side='left')
        self.timeoutEntry.insert(END, self.parserSettings['timeout_seconds'])
        self.timeoutEntry.focus_set()
        emptyLabel = Label(settingsFrame, text=" ")
        emptyLabel.pack()
        self.constrainMoveWithPTB = IntVar(value=self.parserSettings['constrainMoveWithPTB'])
        self.constrainConstWithPTBCCG = IntVar(value=self.parserSettings['constrainConstWithPTBCCG'])
        constrainMoveWithPTBcheckBox = Checkbutton(settingsFrame, text='constrain Move with PTB', variable=self.constrainMoveWithPTB)
        constrainMoveWithPTBcheckBox.pack()
        constrainConstWithPTBCCGcheckBox = Checkbutton(settingsFrame, text='constrain constituencies with PTB/CCGbank', variable=self.constrainConstWithPTBCCG)
        constrainConstWithPTBCCGcheckBox.pack()
        checkboxFrame = Frame(settingsFrame)
        checkboxFrame.pack(fill=X, expand=True)
        useAllNullFrame = Frame(checkboxFrame)
        useAllNullFrame.pack()
        self.useAllNull = IntVar(value=self.parserSettings['useAllNull'])
        self.skipRel = IntVar(value=self.parserSettings['skipRel'])
        self.skipPro = IntVar(value=self.parserSettings['skipPro'])
        useAllNullCheckBox = Checkbutton(useAllNullFrame, text='use all available null categories   ', variable=self.useAllNull)
        useAllNullCheckBox.pack()
        skipRelCheckBox = Checkbutton(useAllNullFrame, text='do not use relativizers', variable=self.skipRel)
        skipRelCheckBox.pack()
        #need to add an empty line to make everything neat
        skipProCheckBox = Checkbutton(useAllNullFrame, text='do not use [pro-x] categories', variable=self.skipPro)
        skipProCheckBox.pack()
        self.printPartialAnalyses = IntVar(value=self.parserSettings['printPartialAnalyses'])
        printPartialAnalysesCheckBox = Checkbutton(useAllNullFrame, text='print partial analyses to the console', variable=self.printPartialAnalyses)
        printPartialAnalysesCheckBox.pack()
        emptyLabel = Label(self.parserSettingsWindow, text=" ")
        emptyLabel.pack()
        self.parserTypeLabel = Label(self.parserSettingsWindow, text="Parser Type")
        self.parserTypeLabel.pack()
        radioFrame = Frame(self.parserSettingsWindow)
        radioFrame.pack()
        self.parserVar = Variable()
        self.parserVar.set(self.parserSettings['parserSetting'])
        basicOnly = Radiobutton(radioFrame, text='Use only the basic parser', variable=self.parserVar, value='basicOnly')
        basicOnly.grid(sticky=W)
        fullFirst = Radiobutton(radioFrame, text='Use the full parser', variable=self.parserVar, value='fullFirst')
        fullFirst.grid(sticky=W)
        basicAndRight = Radiobutton(radioFrame, text='Basic parser + null extraposers', variable=self.parserVar, value='basicAndRight')
        basicAndRight.grid(sticky=W)
        basicAndTough = Radiobutton(radioFrame, text='Basic parser + tough movement', variable=self.parserVar, value='basicAndTough')
        basicAndTough.grid(sticky=W)
        emptyLabel = Label(self.parserSettingsWindow, text=" ")
        emptyLabel.pack()
        buttonFrame = Frame(self.parserSettingsWindow)
        buttonFrame.pack()
        closeButton = Button(buttonFrame, text='close', command=lambda: self.destroyWindow(self.parserSettingsWindow, 'parserSettingsWindow'))
        closeButton.pack(side='left')
        saveButton = Button(buttonFrame, text='apply and save', command=self.saveSettings)
        saveButton.pack(side='left')
        emptyLabel = Label(self.parserSettingsWindow, text=" ")
        emptyLabel.pack()

    def saveFolders(self):
        for section_folder in self.foldersCheckboxVars:
            self.use_section_folder[section_folder] = self.foldersCheckboxVars[section_folder][0].get()
        self.parserSettings['section_folders'] = self.use_section_folder
        with open(self.seed_folder+'/parserSettings', 'w') as parserSettingsFile:
            json.dump(self.parserSettings, parserSettingsFile)
        self.destroyWindow(self.useSectionFoldersWindow, 'useSectionFoldersWindow')

    def saveSettings(self):
        try:
            timeout_seconds = int(self.timeoutEntry.get().strip())
        except ValueError:
            self.nothing(self.parserSettingsWindow, "You must enter seconds as an integer!")
            return
        self.parserSettings['parserSetting'] = self.parserVar.get()
        self.parserSettings['timeout_seconds'] = timeout_seconds
        self.parserSettings['printPartialAnalyses'] = self.printPartialAnalyses.get()
        self.parserSettings['useAllNull'] = self.useAllNull.get()
        self.parserSettings['skipRel'] = self.skipRel.get()
        self.parserSettings['skipPro'] = self.skipPro.get()
        self.parserSettings['constrainMoveWithPTB'] = self.constrainMoveWithPTB.get()
        self.parserSettings['constrainConstWithPTBCCG'] = self.constrainConstWithPTBCCG.get()
        with open(self.seed_folder+'/parserSettings', 'w') as parserSettingsFile:
            json.dump(self.parserSettings, parserSettingsFile)
        self.destroyWindow(self.parserSettingsWindow, 'parserSettingsWindow')

    def fileManager(self, folder, catSearch=None, fileSearch=False, files=None, cat=None, searchItem=None):
        if catSearch or fileSearch:
            search = True
        else:
            search = False
        if not catSearch:
            self.fileManagerFolder = folder
            if self.fileManagerFolder == self.seed_folder:
                self.corpusType = 'seed'
            elif self.fileManagerFolder == self.auto_folder:
                self.corpusType = 'auto'
            elif self.fileManagerFolder == self.ptb_folder:
                self.corpusType = 'ptb'
        else:
            self.corpusType = folder[:-1]
        if self.fileManagerWindow != None:
            self.destroyWindow(self.fileManagerWindow, 'fileManagerWindow')
        self.fileManagerWindow = Toplevel(self.mainWindow)
        if not catSearch and not fileSearch:
            self.fileManagerWindow.title(self.fileManagerFolder)
            folderContents = self.getFolderContents()
            self.fileLevel = self.checkLevel()
        elif fileSearch:
            if self.fileManagerFolder == self.seed_folder:
                corpus = "Seed"
            elif self.fileManagerFolder == self.auto_folder:
                corpus = "Auto"
            if self.length != None:
                insert = " of length "+str(self.length)+" "
            else:
                insert = " "
            if len(files) > 1:
                self.fileManagerWindow.title(corpus.lower()+" parses"+insert+"containing search item: "+searchItem+"  ("+str(len(files))+" matches)")
            elif len(files) == 1:
                self.fileManagerWindow.title(corpus.lower()+" parses"+insert+"containing search item: "+searchItem+"  (1 match)")
            else:
                self.fileManagerWindow.title(corpus.lower()+" parses"+insert+"containing search item: "+searchItem+"  (0 matches)")
            folderContents = [f[0]+"          Ln: "+str(f[1]+1) for f in files]
        else:
            self.fileManagerWindow.title(self.corpusType+" parses using MG cat: "+cat)
            folderContents = [f[0]+"          Ln: "+str(f[1]+1) for f in files]
        self.fileManagerWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.fileManagerWindow, 'fileManagerWindow'))
        fileManagerFrame = Frame(self.fileManagerWindow)
        fileManagerFrame.pack()
        scrollbarV = Scrollbar(fileManagerFrame, orient=VERTICAL)
        scrollbarH = Scrollbar(fileManagerFrame, orient=HORIZONTAL)
        width = 70
        height = 26
        self.fileBox = Listbox(fileManagerFrame, yscrollcommand=scrollbarV.set, width = width, height = height)
        self.fileBox.pack(side=LEFT)
        self.fileBox.bind('<Double-1>', lambda x: self.openFileFolder(catSearch, fileSearch))
        scrollbarV.config(command=self.fileBox.yview)
        scrollbarV.pack(side=RIGHT, fill=Y)
        for item in folderContents: self.fileBox.insert(END, item)
        buttonFrame = Frame(self.fileManagerWindow)
        buttonFrame.pack()
        buttonFrame2 = Frame(self.fileManagerWindow)
        buttonFrame2.pack()
        self.exitButton = Button(buttonFrame, text="exit", command=lambda: self.destroyWindow(self.fileManagerWindow, 'fileManagerWindow'))
        self.exitButton.pack(side='left')
        if not catSearch and not fileSearch:
            self.backButton = Button(buttonFrame, text='<', command=self.back)
            self.backButton.pack(side='left')
            self.backButton.config(state=DISABLED)
        self.openFileButton = Button(buttonFrame, text="open", command=lambda: self.openFileFolder(catSearch, fileSearch))
        self.openFileButton.pack(side='left')
        if self.corpusType == 'seed':
            self.removingFromAutos = False
            self.removeParseButton = Button(buttonFrame, text='remove from seeds', command=lambda: self.removeParse(search=search, folder='seeds'))
            self.removeParseButton.pack(side='left')
            if not catSearch and not fileSearch:
                self.removeParseButton.config(state=DISABLED)
        elif self.corpusType == 'auto':
            self.removingFromAutos = True
            self.removeParseButton = Button(buttonFrame, text='remove from autos', command=lambda: self.removeParse(search=search, folder='autos'))
            self.removeParseButton.pack(side='left')
            if not catSearch and not fileSearch:
                self.removeParseButton.config(state=DISABLED)
                self.moveParseButton = Button(buttonFrame, text='move to seeds', command=lambda: self.moveParse('fileBox'))
                self.moveParseButton.pack(side='left')
                self.moveParseButton.config(state=DISABLED)
        if not catSearch and not folder == self.ptb_folder:
            searchFilesLabel = Label(buttonFrame2, text="reg ex:")
            searchFilesLabel.pack(side=LEFT)
            self.searchFilesEntry = Entry(buttonFrame2, width = 20)
            self.searchFilesEntry.pack(side='left')
            emptyLabel = Label(buttonFrame2, text=" ")
            emptyLabel.pack(side=LEFT)
            lengthLabel = Label(buttonFrame2, text="length:")
            lengthLabel.pack(side=LEFT)
            self.lengthEntry = Entry(buttonFrame2, width = 4)
            self.lengthEntry.pack(side='left')
            emptyLabel = Label(buttonFrame2, text=" ")
            emptyLabel.pack(side=LEFT)
            self.searchFilesButton = Button(buttonFrame2, text='search files', command=lambda: self.searchFiles(folder))
            self.searchFilesButton.pack(side='left')
            self.searchFilesEntry.focus_set()
            self.searchFilesEntry.bind('<Return>', lambda x: self.searchFilesButton.invoke())
            self.lengthEntry.bind('<Return>', lambda x: self.searchFilesButton.invoke())
            self.clearSearchButton = Button(buttonFrame2, text='clear search', command=lambda: self.clearFileSearch())
            self.clearSearchButton.pack(side='left')

    def clearFileSearch(self):
        self.fileManager(self.fileManagerFolder)

    def searchFiles(self, folder):
        lengthError = False
        try:
            self.length = int(self.lengthEntry.get().strip())
        except ValueError:
            lengthError = True
        if self.lengthEntry.get().strip() == '':
            self.length = None
            lengthError = False
        elif not lengthError and self.length < 1:
            lengthError = True
        if lengthError:
            self.nothing(self.mainWindow, "Length entry must be empty or contain a positive integer!", width = 425)
            return
        searchItem = self.searchFilesEntry.get().strip()
        try:
            regexp = re.compile(searchItem)
        except Exception:
            regexp = False
        if self.corpusType == 'seed':
            root = self.seed_folder
        elif self.corpusType == 'auto':
            root = self.auto_folder
        files = []
        count = 0
        for section_folder in sorted(os.listdir(root)):
            if section_folder != 'new_parses' and section_folder in non_section_folders:
                continue
            for FILE in sorted(os.listdir(root+"/"+section_folder)):
                if FILE == '.DS_Store':
                    continue
                if section_folder != "new_parses":
                    ptb_file = open(self.ptb_folder+"/"+section_folder+"/"+FILE)
                    ptb_bracketings = []
                    for line in ptb_file:
                        ptb_bracketings.append(line)
                    sentence_file = open(self.ptb_folder+"_strings/"+section_folder+"/"+FILE+"_strings")
                else:
                    sentence_file = open(self.seed_folder+"/new_parse_strings/"+FILE)
                sentences = []
                for sentence in sentence_file:
                    sentences.append(sentence)
                parses = json.load(open(root+"/"+section_folder+"/"+FILE))
                for parse in parses:
                    line_number = int(parse)
                    sentence = sentences[line_number]
                    lengthSentence = len(sentence.split(" "))
                    if self.length == None or self.length == lengthSentence:
                        if section_folder != "new_parses":
                            ptb_bracketing = ptb_bracketings[line_number]
                            strippedBracketing = re.sub('{.*?}', '', ptb_bracketing, count=10000)
                            if (searchItem.lower() in sentence.lower() or searchItem in ptb_bracketing) or (regexp and (re.search(searchItem, sentence) or re.search(searchItem, strippedBracketing))):
                                files.append([root+"/"+section_folder+"/"+FILE, line_number])
                        else:
                            if (searchItem.lower() in sentence.lower()) or (regexp and (re.search(searchItem, sentence))):
                                files.append([root+"/"+section_folder+"/"+FILE, line_number])
        self.fileManager(folder=folder, fileSearch=True, files=files, searchItem=searchItem)

    def checkLevel(self):
        for item in sorted(os.listdir(self.fileManagerFolder)):
            if item != '.DS_Store':
                if os.path.isdir(self.fileManagerFolder+"/"+item):
                    self.fileLevel = False
                    break
                else:
                    self.fileLevel = True
                    break

    def getFolderContents(self):
        folderContents = sorted(os.listdir(self.fileManagerFolder))
        itemsToDelete = []
        for item in folderContents:
            if item != 'new_parses' and item in non_section_folders:
                itemsToDelete.append(item)
        while len(itemsToDelete) > 0:
            folderContents.remove(itemsToDelete[0])
            del(itemsToDelete[0])
        return folderContents

    def removeParse(self, search=False, folder=None):
        try:
            objectToOpen = self.fileBox.get(self.fileBox.curselection())
        except TclError:
            return
        if search:
            folder = objectToOpen.split("Ln:")[0].strip()
            line = int(objectToOpen.split("Ln:")[1].strip())
            self.seed = [folder, line-1]
        else:
            self.seed = [self.fileManagerFolder, int(objectToOpen)-1]
        self.removingFromFilebox = True
        if 'seed' in folder.lower():
            self.confirm(confirmWhat='removeParse', message = "Are you sure you wish to remove\nthis parse from the seed set?")
        elif 'auto' in folder.lower():
            self.confirm(confirmWhat='removeParse', message = "Are you sure you wish to remove\nthis parse from the auto set?")

    def moveParse(self, fromWhere):
        #moves a bracketing from the autogenerated set to the seed set.. this is a quick way for the
        #user to increase the size of the corpus..
        try:
            if fromWhere == 'fileBox':
                objectToOpen = self.fileBox.get(self.fileBox.curselection())
            elif fromWhere == 'view':
                objectToOpen = self.seed[1]+1
        except TclError:
            return
        try:
            if self.seed != None:
                old_seed = copy.deepcopy(self.seed)
            else:
                old_seed = None
        except AttributeError:
            old_seed = None
        seed_path = self.seed_folder+"/"+"/".join(self.fileManagerFolder.split("/")[1:])
        self.seed = [seed_path, int(objectToOpen)-1]
        self.autoParse = [self.fileManagerFolder, int(objectToOpen)-1]
        self.movingFromFilebox = True
        MGparses = json.load(open(self.fileManagerFolder))[str(self.autoParse[1])]
        subcat_derivation_bracketing = MGparses[0]
        subcat_derivation_tree = gen_derived_tree.gen_derivation_tree(subcat_derivation_bracketing)
        MGcats = autobank.get_MGcats(subcat_derivation_tree, [])
        nullMGcats = autobank.get_null_MGcats(subcat_derivation_tree, [])
        #we will now check to make sure that all cats in the new parse are in CatTreeMappings
        #and if they are not then we abort and the user must add the tree manually.. this
        #is to avoid any conflicts where we end up adding a slightly altered version of a category
        #that is still in the system..
        CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
        try:
            nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
        except IOError:
            nullCatTreeMappings = {}
        for cat in set(MGcats):
            if cat not in CatTreeMappings:
                self.seed = old_seed
                self.autoParse = None
                self.nothing(self.mainWindow, "\nThe overt MG category:\n\n"+cat+"\n\nis currently not used in any seed parses. Please check\nfor near duplicates and annotate this tree manually.\n", width=1000, height=170)
                return
        for cat in set(nullMGcats):
            if cat not in nullCatTreeMappings:
                self.seed = old_seed
                self.autoParse = None
                self.nothing(self.mainWindow, "\nThe covert MG category:\n\n"+cat+"\n\nis currently not used in any seed parses. Please check\nfor near duplicates and annotate this tree manually.\n", width=1000, height=170)
                return
        self.confirm(confirmWhat='moveParse', message = "Are you sure you wish to move\nthis parse from the auto set into\nthe seed set?")

    def moveParse2(self):
        #we need to get a count for the terminals in the moved tree so that this can be
        #decremented in autobank
        self.treeMoved = True
        ptbFile = self.ptb_folder+"/"+"/".join(self.seed[0].split("/")[1:])
        ptbBracketing = self.getPTBbracketing(ptbFile, self.seed)
        terminals = autobank.build_tree(ptbBracketing)[2]
        if self.counts[self.seedSentLen] == 1 and len(terminals) == self.seedSentLen:
            self.GUI_MATCH = False
        self.lenMovedTree = len(terminals)
        MGparses = json.load(open(self.fileManagerFolder))[str(self.autoParse[1])]
        subcat_derivation_bracketing = MGparses[0]
        subcat_derivation_tree = gen_derived_tree.gen_derivation_tree(subcat_derivation_bracketing)
        MGcats = autobank.get_MGcats(subcat_derivation_tree, [])
        nullMGcats = autobank.get_null_MGcats(subcat_derivation_tree, [])
        #no need to call removeTree as it can never be the case that there is a tree in the
        #seeds which is also in the auto set as whenever we label a new seed that tree is automatically
        #deleted from the autoset if it is in there..
        CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
        TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
        try:
            nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
        except IOError:
            nullCatTreeMappings = {}
        try:
            nullTreeCatMappings = json.load(open(self.seed_folder+"/"+'nullTreeCatMappings'))
        except IOError:
            nullTreeCatMappings = {}
        autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
        autoTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
        try:
            autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
        except IOError:
            autoNullCatTreeMappings = {}
        try:
            autoNullTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoNullTreeCatMappings'))
        except IOError:
            autoNullTreeCatMappings = {}
        self.seed[0] = self.seed[0].encode('utf8')
        TreeCatMappings[str(self.seed)] = copy.deepcopy(MGcats)
        nullTreeCatMappings[str(self.seed)] = copy.deepcopy(nullMGcats)
        if str(self.autoParse) in autoTreeCatMappings:
            del(autoTreeCatMappings[str(self.autoParse)])
        if str(self.autoParse) in autoNullTreeCatMappings:
            del(autoNullTreeCatMappings[str(self.autoParse)])
        for cat in MGcats:
            if cat in autoCatTreeMappings:
                if self.autoParse in autoCatTreeMappings[cat]:
                    autoCatTreeMappings[cat].remove(self.autoParse)
                    if len(autoCatTreeMappings[cat]) == 0:
                        del(autoCatTreeMappings[cat])
        for cat in nullMGcats:
            if cat in autoNullCatTreeMappings:
                if self.autoParse in autoNullCatTreeMappings[cat]:
                    autoNullCatTreeMappings[cat].remove(self.autoParse)
                    if len(autoNullCatTreeMappings[cat]) == 0:
                        del(autoNullCatTreeMappings[cat])
        #we will now check to make sure that all cats in the new parse are are CatTreeMappings
        #and if they are not then we abort and the user must add the tree manually.. this
        #is to avoid any conflicts where we end up adding a slightly altered version of a category
        #that is still in the system..
        for cat in set(MGcats):
            if cat in CatTreeMappings:
                CatTreeMappings[cat].append(copy.deepcopy(self.seed))
            else:
                CatTreeMappings[cat] = [copy.deepcopy(self.seed)]
                self.overtCatComments[cat] = ""
        for cat in set(nullMGcats):
            if cat in nullCatTreeMappings:
                nullCatTreeMappings[cat].append(copy.deepcopy(self.seed))
            else:
                nullCatTreeMappings[cat] = [copy.deepcopy(self.seed)]
        with open(self.seed_folder+"/"+'CatTreeMappings', 'w') as CatTreeMappingsFile:
            json.dump(CatTreeMappings, CatTreeMappingsFile)
        with open(self.seed_folder+"/"+'TreeCatMappings', 'w') as TreeCatMappingsFile:
            json.dump(TreeCatMappings, TreeCatMappingsFile)
        with open(self.seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
            json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
        with open(self.seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
            json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
        with open(self.auto_folder+"/"+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
            json.dump(autoCatTreeMappings, autoCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
            json.dump(autoTreeCatMappings, autoTreeCatMappingsFile)
        with open(self.auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
            json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
            json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)
        with open(self.seed_folder+"/"+'overtCatComments', 'w') as overtCatCommentsFile:
            json.dump(self.overtCatComments, overtCatCommentsFile)
        #now update PosMappings.. if the MGcat used in the auto parse has been modified since the parse
        #was created we have no way of knowing that so both categories will appear in PosMappings, which is fine..
        i=-1
        for MGcat in MGcats:
            i+=1
            ptbCat = terminals[i].mother.name.split("-")[0]
            if MGcat not in self.PosMappings[ptbCat]:
                self.PosMappings[ptbCat].append(MGcat)
                if 'No categories available' in self.PosMappings[ptbCat]:
                    self.PosMappings[ptbCat].remove('No categories available')
        with open(self.seed_folder+"/"+'PosMappings', 'w') as PosMappingsFile:
            json.dump(self.PosMappings, PosMappingsFile)
        #we can't update the covert lexicons as we have no way of knowing which covert
        #lexicon each of the null catgories in the auto parse should be entered into..
        #but this is ok.. the chances are that the null categories are still in the system
        #but the user can just add them if not anyway.
        #now remove the parse from the auto corpus
        auto_set = json.load(open(self.autoParse[0]))
        del(auto_set[str(self.autoParse[1])])
        with open(self.autoParse[0], 'w') as auto_set_file:
            json.dump(auto_set, auto_set_file)
        #now add the parse into the seed corpus
        try:
            seed_set = json.load(open(self.seed[0]))
        except IOError:
            seed_set = {}
        seed_set[str(self.seed[1])] = MGparses
        with open(self.seed[0], 'w') as seed_set_file:
            json.dump(seed_set, seed_set_file)
        self.viewCurrentPTBtree()
        
    def back(self):
        self.openFileButton.config(text="open", state=NORMAL)
        if self.corpusType == 'seed':
            self.removeParseButton.config(state=DISABLED)
        elif self.corpusType == 'auto':
            self.moveParseButton.config(state=DISABLED)
            self.removeParseButton.config(state=DISABLED)
        self.fileManagerFolder = "/".join(self.fileManagerFolder.split("/")[0:-1])
        self.fileManagerWindow.title(self.fileManagerFolder)
        if self.fileManagerFolder == self.seed_folder or self.fileManagerFolder == self.auto_folder or self.fileManagerFolder == self.ptb_folder:
            self.backButton.config(state=DISABLED)
        self.fileBox.delete(0, END)
        folderContents = self.getFolderContents()
        for item in folderContents: self.fileBox.insert(END, item)

    def openFileFolder(self, catSearch=False, fileSearch=False, cycleParse=False):
        if cycleParse:
            self.fileManagerFolder = cycleParse[0]
            objectToOpen = str(cycleParse[1]+1)
        else:
            try:
                objectToOpen = self.fileBox.get(self.fileBox.curselection())
                PATH = objectToOpen.split(" ")[0]
                if fileSearch:
                    self.fileManagerFolder = self.fileBox.get(self.fileBox.curselection()).split(" ")[0]
            except TclError:
                return
            self.fileBox.delete(0, END)
        if not cycleParse and not catSearch and not fileSearch and os.path.isdir(self.fileManagerFolder+"/"+objectToOpen):
            self.fileManagerFolder = self.fileManagerFolder+"/"+objectToOpen
            self.fileManagerWindow.title(self.fileManagerFolder)
            self.checkLevel()
            folderContents = self.getFolderContents()
            for item in folderContents: self.fileBox.insert(END, item)
            self.backButton.config(state=NORMAL)
        elif not cycleParse and not catSearch and not fileSearch and os.path.isfile(self.fileManagerFolder+"/"+objectToOpen):
            self.fileManagerWindow.title(self.fileManagerFolder+"/"+objectToOpen)
            self.openFileButton.config(text="view")
            self.fileManagerFolder = self.fileManagerFolder+"/"+objectToOpen
            top_level_folder = self.fileManagerFolder.split("/")[0]
            if top_level_folder != self.ptb_folder:
                self.parses = json.load(open(self.fileManagerFolder))
            else:
                #the case where we are just jumping to an unannotated PTB tree
                self.parses = {}
                seed_path = self.seed_folder+"/"+"/".join(self.fileManagerFolder.split("/")[1:])
                try:
                    seeds = json.load(open(seed_path))
                except IOError:
                    seeds = {}
                i=-1
                for line in open(self.fileManagerFolder):
                    i+=1
                    if str(i) not in seeds:
                        self.parses[str(i+1)] = line
            if len(self.parses) > 0:
                sortedParses = []
                for item in self.parses:
                    #need to convert the parse numbers to int temporarily to sorted them..
                    sortedParses.append(int(item))
                sortedParses.sort()
                i=-1
                for item in sortedParses:
                    #now convert back to strings
                    i+=1
                    sortedParses[i] = str(sortedParses[i])
                if top_level_folder != self.ptb_folder:
                    for item in sortedParses: self.fileBox.insert(END, str(int(item)+1))
                else:
                    for item in sortedParses: self.fileBox.insert(END, item)
                if self.corpusType == 'seed':
                    self.removeParseButton.config(state = NORMAL)
                elif self.corpusType == 'auto':
                    self.removeParseButton.config(state = NORMAL)
                    self.moveParseButton.config(state = NORMAL)
            else:
                self.openFileButton.config(state = DISABLED)
        else:
            if self.corpusType in ['ptb', 'auto']:
                try:
                    self.ptbLineNum = int(objectToOpen)-1
                except ValueError:
                    objectToOpen = objectToOpen.split(" ")[-1]
                    self.ptbLineNum = int(objectToOpen)-1
                if not catSearch:
                    try:
                        ptbBracketings = open(self.ptb_folder+"/"+"/".join(self.fileManagerFolder.split("/")[1:]))
                    except Exception as e:
                        ptbBracketings = open(self.ptb_folder+"/"+"/".join(PATH.split("/")[1:]))
                else:
                    ptbBracketings = open(self.ptb_folder+"/"+"/".join(PATH.split("/")[1:]))
                i = -1
                for ptbBracketing in ptbBracketings:
                    i+=1
                    if i == self.ptbLineNum:
                        break
                terminals = autobank.build_tree(ptbBracketing)[2]
                self.newSeedSentLen = len(terminals)
            if self.corpusType == 'ptb':
                ptbBracketing = self.parses[str(self.ptbLineNum+1)]
                self.PTBbracketing = ptbBracketing
                terminals = autobank.build_tree(ptbBracketing)[2]
                self.seedSentLen = len(terminals)
                self.searchItem = ''
                self.startNewPtbSearch()
                return
            if not cycleParse:
                self.destroyWindow(self.fileManagerWindow, 'fileManagerWindow')
            if not catSearch and not fileSearch:
                seed = [self.fileManagerFolder, int(objectToOpen)-1]
            else:
                try:
                    folder = objectToOpen.split("Ln:")[0].strip()
                    line = int(objectToOpen.split("Ln:")[1].strip())
                except Exception as e:
                    folder = PATH
                    line = int(objectToOpen)
                seed = [folder, line-1]
            self.seed_line_num = seed[1]
            self.viewParse(seed)
        
    def removeTree(self):
        #first we need to get a count for the number of terminals in the tree to be
        #deleted so we can decrement the overall counts for this..
        if self.removingFromAutos == False and self.seed[0].split("/")[1] != "new_parses":
            ptbFile = self.ptb_folder+"/"+"/".join(self.seed[0].split("/")[1:])
            ptbBracketing = self.getPTBbracketing(ptbFile, self.seed)
            terminals = autobank.build_tree(ptbBracketing)[2]
            self.terminals = terminals
            self.lenRemovedTree = len(terminals)
        elif self.removingFromAutos:
            self.lenRemovedTree = 'removedFromAutos'
        elif self.seed[0].split("/")[1] == "new_parses":
            self.lenRemovedTree = "NewParseRemoved"
        seeds = json.load(open(self.seed[0]))
        if not self.removingFromAutos:
            CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
            TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
            try:
                nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
            except IOError:
                nullCatTreeMappings = {}
            try:
                nullTreeCatMappings = json.load(open(self.seed_folder+"/"+'nullTreeCatMappings'))
            except IOError:
                nullTreeCatMappings = {}
            mapping_tuples = [(TreeCatMappings, CatTreeMappings), (nullTreeCatMappings, nullCatTreeMappings)]
        else:
            autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
            autoTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
            try:
                autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
            except IOError:
                autoNullCatTreeMappings = {}
            try:
                autoNullTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoNullTreeCatMappings'))
            except IOError:
                autoNullTreeCatMappings = {}
            mapping_tuples = [(autoTreeCatMappings, autoCatTreeMappings), (autoNullTreeCatMappings, autoNullCatTreeMappings)]
        for (tcm, ctm) in mapping_tuples:
            if str([self.seed[0].encode('utf8'), self.seed[1]]) in tcm:
                for MGcat in set(tcm[str([self.seed[0].encode('utf8'), self.seed[1]])]):
                    ctm[MGcat].remove(self.seed)
                    if len(ctm[MGcat]) == 0:
                        del(ctm[MGcat])
                del(tcm[str([self.seed[0].encode('utf8'), self.seed[1]])])
        if not self.removingFromAutos:
            with open(self.seed_folder+"/"+'CatTreeMappings', 'w') as CatTreeMappingsFile:
                json.dump(CatTreeMappings, CatTreeMappingsFile)
            with open(self.seed_folder+"/"+'TreeCatMappings', 'w') as TreeCatMappingsFile:
                json.dump(TreeCatMappings, TreeCatMappingsFile)
            with open(self.seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
                json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
            with open(self.seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
                json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
        else:
            with open(self.auto_folder+"/"+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
                json.dump(autoCatTreeMappings, autoCatTreeMappingsFile)
            with open(self.auto_folder+"/"+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
                json.dump(autoTreeCatMappings, autoTreeCatMappingsFile)
            with open(self.auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
                json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
            with open(self.auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
                json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)
        seeds = json.load(open(self.seed[0]))
        del(seeds[str(self.seed[1])])
        self.treeRemoved = True
        if self.seed[0].split("/")[1] != "new_parses" and self.removingFromAutos == False and len(self.terminals) == self.seedSentLen:
            self.override_GUI_MATCH = True
        if seeds == {}:
            os.remove(self.seed[0])
        else:
            with open(self.seed[0], 'w') as seedsFile:
                json.dump(seeds, seedsFile)
        if self.seed[0].split("/")[1] == "new_parses":
            os.remove(self.seed_folder+"/new_parse_strings/"+self.seed[0].split("/")[2])
        self.viewCurrentPTBtree()

    def viewCorpusStats(self):
        cat_set = []
        null_cat_set = []
        auto_cat_set = []
        auto_null_cat_set = []
        self.extractOvertLexicon(fromCorporaStats=True, useAutos=True)
        seed_lexicon = json.load(open(self.seed_folder+"/"+'OvertLexicon'))
        auto_lexicon = json.load(open(self.auto_folder+"/"+'OvertLexicon'))
        word_type_count = 0
        word_form_count = 0
        auto_word_type_count = 0
        auto_word_form_count = 0
        for entry in seed_lexicon:
            word_form_count += 1
            for cat in seed_lexicon[entry]:
                word_type_count += 1
        for entry in auto_lexicon:
            auto_word_form_count += 1
            for cat in auto_lexicon[entry]:
                auto_word_type_count += 1
        try:
            CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
        except IOError:
            CatTreeMappings = {}
        try:
            nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
        except IOError:
            nullCatTreeMappings = {}
        for MG_cat in CatTreeMappings:
            if MG_cat in ['No categories available', '']:
                continue
            elif MG_cat not in cat_set:
                cat_set.append(MG_cat)
        for null_MG_cat in nullCatTreeMappings:
            if null_MG_cat not in null_cat_set:
                null_cat_set.append(null_MG_cat)
        try:
            autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
        except IOError:
            autoCatTreeMappings = {}
        try:
            autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
        except IOError:
            autoNullCatTreeMappings = {}
        for MG_cat in autoCatTreeMappings:
            if MG_cat in ['No categories available', '']:
                continue
            elif MG_cat not in auto_cat_set:
                auto_cat_set.append(MG_cat)
        for null_MG_cat in autoNullCatTreeMappings:
            if null_MG_cat not in auto_null_cat_set:
                auto_null_cat_set.append(null_MG_cat)
        relative_complement = 0
        for item in auto_null_cat_set:
            if item not in null_cat_set:
                relative_complement += 1
        for section_folder in sorted(os.listdir(self.auto_folder)):
            try:
                int(section_folder)
                if len(section_folder) == 2:
                    for f in sorted(os.listdir(self.auto_folder+"/"+section_folder)):
                        if f != '.DS_Store':
                            strings = []
                            for line in open('wsj_strings/'+section_folder+"/"+f+"_strings"):
                                strings.append(line)
            except ValueError:
                continue
        cat_set_size = len(cat_set)
        null_cat_set_size = len(null_cat_set)
        auto_cat_set_size = len(auto_cat_set)
        auto_null_cat_set_size = len(auto_null_cat_set)
        total_mgbank_lex_size = word_type_count + auto_word_type_count + null_cat_set_size + auto_null_cat_set_size
        try:
            TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
        except IOError:
            TreeCatMappings = {}
        seed_set_size = 0
        additional_seed_set_size = 0
        for entry in TreeCatMappings:
            if 'new_parses' in entry:
                additional_seed_set_size += 1
            else:
                seed_set_size += 1
        auto_set_size = self.get_auto_set_size()
        mgbank_size = auto_set_size+seed_set_size+additional_seed_set_size
        if self.statsWindow != None:
            self.destroyWindow(self.statsWindow, 'statsWindow')
        self.statsWindow = Toplevel(self.mainWindow)
        self.statsWindow.title("Corpora Stats")
        w=625
        h=635
        (x, y) = self.getCentrePosition(w, h)
        self.statsWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        outerInfoFrame = Frame(self.statsWindow)
        outerInfoFrame.pack()
        infoFrame = Frame(outerInfoFrame)
        infoFrame.pack(side=LEFT)
        statsFrame = Frame(outerInfoFrame)
        statsFrame.pack(side=RIGHT)
        emptyLabel = Label(statsFrame, text=" ")
        emptyLabel.grid(sticky=W)
        sourceSizeVal = Label(statsFrame, text=str(self.totalSourceTrees))
        sourceSizeVal.grid(sticky=W)
        sourceTokenSizeVal = Label(statsFrame, text=str(self.ptb_word_token_count))
        sourceTokenSizeVal.grid(sticky=W)
        try:
            seed_percentage = 100/self.totalSourceTrees*seed_set_size
        except ZeroDivisionError:
            seed_percentage = 0
        mg_bank_trees_percentage = 100/self.totalSourceTrees*(seed_set_size+auto_set_size)
        try:
            tokenPercentage = 100/self.ptb_word_token_count*self.seed_word_token_count
        except ZeroDivisionError:
            tokenPercentage = 0
        try:
            autoTokenPercentage = 100/self.ptb_word_token_count*self.auto_word_token_count
        except ZeroDivisionError:
            autoTokenPercentage = 0
        total_token_count_mgbank = self.seed_word_token_count + self.auto_word_token_count
        total_token_count_percentage = 100/self.ptb_word_token_count*total_token_count_mgbank
        seedSizeVal = Label(statsFrame, text=str(seed_set_size)+"  (%.2f" % seed_percentage+"% of PTB covered)")
        seedSizeVal.grid(sticky=W)
        AdditionalSeedSizeVal = Label(statsFrame, text=str(additional_seed_set_size))
        AdditionalSeedSizeVal.grid(sticky=W)
        wordCountVal = Label(statsFrame, text=str(word_type_count))
        wordCountVal.grid(sticky=W)
        wordFormCountVal = Label(statsFrame, text=str(word_form_count))
        wordFormCountVal.grid(sticky=W)
        tokenSetSizeVal = Label(statsFrame, text=str(self.seed_word_token_count)+"  (%.2f" % tokenPercentage+"% of PTB covered)")
        tokenSetSizeVal.grid(sticky=W)
        catSetSizeVal = Label(statsFrame, text=str(cat_set_size))
        catSetSizeVal.grid(sticky=W)
        nullCatSetSizeVal = Label(statsFrame, text=str(null_cat_set_size))
        nullCatSetSizeVal.grid(sticky=W)
        totalCatSetSizeVal = Label(statsFrame, text=str(null_cat_set_size+cat_set_size))
        totalCatSetSizeVal.grid(sticky=W)
        try:
            timeAndDate = json.load(open(self.auto_folder+"/timeAndDate"))
            timeAndDate = Label(statsFrame, text=timeAndDate['date']+" at "+timeAndDate['time'])
        except IOError:
            timeAndDate = Label(statsFrame, text="No auto set generated yet")
        timeAndDate.grid(sticky=W)
        try:
            auto_percentage = 100/self.totalSourceTrees*auto_set_size
        except ZeroDivisionError:
            auto_percentage = 0
        autoSizeVal = Label(statsFrame, text=str(auto_set_size)+"  (%.2f" % auto_percentage+"% of PTB covered)")
        autoSizeVal.grid(sticky=W)
        autoWordCountVal = Label(statsFrame, text=str(auto_word_type_count))
        autoWordCountVal.grid(sticky=W)
        autoWordFormCountVal = Label(statsFrame, text=str(auto_word_form_count))
        autoWordFormCountVal.grid(sticky=W)
        autoTokenSetSizeVal = Label(statsFrame, text=str(self.auto_word_token_count)+"  (%.2f" % autoTokenPercentage+"% of PTB covered)")
        autoTokenSetSizeVal.grid(sticky=W)
        autoCatSetSizeVal = Label(statsFrame, text=str(auto_cat_set_size))
        autoCatSetSizeVal.grid(sticky=W)
        autoNullCatSetSizeVal = Label(statsFrame, text=str(auto_null_cat_set_size))
        autoNullCatSetSizeVal.grid(sticky=W)
        autoTotalCatSetSizeVal = Label(statsFrame, text=str(auto_null_cat_set_size+auto_cat_set_size))
        autoTotalCatSetSizeVal.grid(sticky=W)
        relCompVal = Label(statsFrame, text=str(relative_complement))
        relCompVal.grid(sticky=W)
        totalMGbankCats = Label(statsFrame, text=str(relative_complement+null_cat_set_size+cat_set_size))
        totalMGbankCats.grid(sticky=W)
        totalWordCountVal = Label(statsFrame, text=str(auto_word_type_count+word_type_count))
        totalWordCountVal.grid(sticky=W)
        totalWordFormCountVal = Label(statsFrame, text=str(auto_word_form_count+word_form_count))
        totalWordFormCountVal.grid(sticky=W)
        totalMGbankLexSize = Label(statsFrame, text=str(total_mgbank_lex_size))
        totalMGbankLexSize.grid(sticky=W)
        totalMGbankTokenCount = Label(statsFrame, text=str(total_token_count_mgbank)+"  (%.2f" % total_token_count_percentage+"% of PTB covered)")
        totalMGbankTokenCount.grid(sticky=W)
        totalMGbankSize = Label(statsFrame, text=str(auto_set_size+additional_seed_set_size+seed_set_size)+"  (%.2f" % mg_bank_trees_percentage+"% of PTB covered)")
        totalMGbankSize.grid(sticky=W)
        sourceSizeFrame = Frame(infoFrame)
        sourceSizeFrame.grid(column=0, sticky=W)
        sourceSize = Label(sourceSizeFrame, text="Total number of PTB trees: ")
        sourceSize.pack(side=LEFT)
        sourceTokenSizeFrame = Frame(infoFrame)
        sourceTokenSizeFrame.grid(column=0, sticky=W)
        sourceTokenSize = Label(sourceTokenSizeFrame, text="Total word tokens in PTB: ")
        sourceTokenSize.pack(side=LEFT)
        seedSizeFrame = Frame(infoFrame)
        seedSizeFrame.grid(column=0, sticky=W)
        seedSize = Label(seedSizeFrame, text="Number of PTB trees in seed set: ")
        seedSize.pack(side=LEFT)
        additionalSeedSizeFrame = Frame(infoFrame)
        additionalSeedSizeFrame.grid(column=0, sticky=W)
        additionalSeedSize = Label(additionalSeedSizeFrame, text="Number of additional trees in seed set: ")
        additionalSeedSize.pack(side=LEFT)
        wordCountFrame = Frame(infoFrame)
        wordCountFrame.grid(column=0, sticky=W)
        wordCount = Label(wordCountFrame, text="Number of word types in seed set: ")
        wordCount.pack(side=LEFT)
        wordFormCountFrame = Frame(infoFrame)
        wordFormCountFrame.grid(column=0, sticky=W)
        wordFormCount = Label(wordFormCountFrame, text="Number of word forms in seed set: ")
        wordFormCount.pack(side=LEFT)
        tokenSetSizeFrame = Frame(infoFrame)
        tokenSetSizeFrame.grid(column=0, sticky=W)
        tokenSetSize = Label(tokenSetSizeFrame, text="Number of PTB word tokens covered by seed set: ")
        tokenSetSize.pack(side=LEFT)
        catSetSizeFrame = Frame(infoFrame)
        catSetSizeFrame.grid(column=0, sticky=W)
        catSetSize = Label(catSetSizeFrame, text="Number of overt categories in seed set: ")
        catSetSize.pack(side=LEFT)
        nullCatSetSizeFrame = Frame(infoFrame)
        nullCatSetSizeFrame.grid(column=0, sticky=W)
        nullCatSetSize = Label(nullCatSetSizeFrame, text="Number of null categories in seed set: ")
        nullCatSetSize.pack(side=LEFT)
        totalCatSetSizeFrame = Frame(infoFrame)
        totalCatSetSizeFrame.grid(column=0, sticky=W)
        totalCatSetSize = Label(totalCatSetSizeFrame, text="Total MG categories in seed set: ")
        totalCatSetSize.pack(side=LEFT)
        autoTimeFrame = Frame(infoFrame)
        autoTimeFrame.grid(column=0, sticky=W)
        autoTime = Label(autoTimeFrame, text="Last auto set generated on: ")
        autoTime.pack(side=LEFT)
        autoSizeFrame = Frame(infoFrame)
        autoSizeFrame.grid(column=0, sticky=W)
        autoSize = Label(autoSizeFrame, text="Number of trees in auto set: ")
        autoSize.pack(side=LEFT)
        autoWordCountFrame = Frame(infoFrame)
        autoWordCountFrame.grid(column=0, sticky=W)
        autoWordCount = Label(autoWordCountFrame, text="Number of word types in auto set: ")
        autoWordCount.pack(side=LEFT)
        autoWordFormCountFrame = Frame(infoFrame)
        autoWordFormCountFrame.grid(column=0, sticky=W)
        autoWordFormCount = Label(autoWordFormCountFrame, text="Number of word forms in auto set: ")
        autoWordFormCount.pack(side=LEFT)
        autoTokenSetSizeFrame = Frame(infoFrame)
        autoTokenSetSizeFrame.grid(column=0, sticky=W)
        autoTokenSetSize = Label(autoTokenSetSizeFrame, text="Number of PTB word tokens covered by auto set: ")
        autoTokenSetSize.pack(side=LEFT)
        autoCatSetSizeFrame = Frame(infoFrame)
        autoCatSetSizeFrame.grid(column=0, sticky=W)
        autoCatSetSize = Label(autoCatSetSizeFrame, text="Number of overt categories in auto set: ")
        autoCatSetSize.pack(side=LEFT)
        autoNullCatSetSizeFrame = Frame(infoFrame)
        autoNullCatSetSizeFrame.grid(column=0, sticky=W)
        autoNullCatSetSize = Label(autoNullCatSetSizeFrame, text="Number of null categories in auto set: ")
        autoNullCatSetSize.pack(side=LEFT)
        autoTotalCatSetSizeFrame = Frame(infoFrame)
        autoTotalCatSetSizeFrame.grid(column=0, sticky=W)
        autoTotalCatSetSize = Label(autoTotalCatSetSizeFrame, text="Total categories in auto set: ")
        autoTotalCatSetSize.pack(side=LEFT)
        relCompFrame = Frame(infoFrame)
        relCompFrame.grid(column=0, sticky=W)
        relComp = Label(relCompFrame, text="Null categories in auto set but not seed set: ")
        relComp.pack(side=LEFT)
        totalMGbankCatsFrame = Frame(infoFrame)
        totalMGbankCatsFrame.grid(column=0, sticky=W)
        totalMGbankCats = Label(totalMGbankCatsFrame, text="Total (overt and null) categories in MGbank: ")
        totalMGbankCats.pack(side=LEFT)
        totalWordCountFrame = Frame(infoFrame)
        totalWordCountFrame.grid(column=0, sticky=W)
        totalWordCount = Label(totalWordCountFrame, text="Total Number of overt word types in MGbank: ")
        totalWordCount.pack(side=LEFT)
        totalWordFormCountFrame = Frame(infoFrame)
        totalWordFormCountFrame.grid(column=0, sticky=W)
        totalWordFormCount = Label(totalWordFormCountFrame, text="Total Number of overt word forms in MGbank: ")
        totalWordFormCount.pack(side=LEFT)
        totalLexMGbankFrame = Frame(infoFrame)
        totalLexMGbankFrame.grid(column=0, sticky=W)
        totalLexMGbank = Label(totalLexMGbankFrame, text="Total size of (overt and null) MGbank lexicon: ")
        totalLexMGbank.pack(side=LEFT)
        totalTokenFrame = Frame(infoFrame)
        totalTokenFrame.grid(column=0, sticky=W)
        totalToken = Label(totalTokenFrame, text="Total number of PTB word tokens covered by MGbank: ")
        totalToken.pack(side=LEFT)
        totalTreesMGbankFrame = Frame(infoFrame)
        totalTreesMGbankFrame.grid(column=0, sticky=W)
        totalTreesMGbank = Label(totalTreesMGbankFrame, text="Total trees in MGbank: ")
        totalTreesMGbank.pack(side=LEFT)
        emptyLabel = Label(statsFrame, text=" ")
        emptyLabel.grid(sticky=W)
        buttonFrame = Frame(self.statsWindow)
        buttonFrame.pack()
        self.statsWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.statsWindow, 'statsWindow'))
        okButton = Button(buttonFrame, text="close", command=lambda: self.destroyWindow(self.statsWindow, 'statsWindow'))
        okButton.grid(columnspan=3)

    def get_auto_set_size(self):
        try:
            x=os.listdir(self.auto_folder)
        except OSError:
            return 0
        auto_set_size = 0
        for section_folder in sorted(os.listdir(self.auto_folder)):
            if section_folder not in autoFilesToIgnore:
                for auto_file in sorted(os.listdir(self.auto_folder+"/"+section_folder)):
                    if auto_file == '.DS_Store':
                        continue
                    auto_set = json.load(open(self.auto_folder+"/"+section_folder+"/"+auto_file))
                    auto_set_size+=len(auto_set)
        return auto_set_size

    def viewNextPTBtree(self):
        #this button allows the user to move onto the next PTB tree without adding a
        #parse for the previous one..
        self.quit = False
        self.nextPTBtree = True
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()

    def viewPreviousPTBtree(self):
        #first we have to determine the path and file name of the previous
        self.quit = False
        self.previousPTBtree = True
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()

    def go(self):
        #first we have to determine the path and file name of the previous
        self.quit = False
        try:
            self.goto = int(self.gotoEntry.get())
            if self.goto > int(self.counts[self.seedSentLen]) or self.goto < 1:
                self.nothing(self.mainWindow, "No tree found at that index!")
                return
        except ValueError:
            self.nothing(self.mainWindow, "You must enter an integer!")
            return
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()

    def startNewSearch(self):
        if self.searchItem == self.searchEntry.get():
            #if all that's changed is the string length in the search we don't want to
            #waste time doing a whole new search..
            self.sameSearch = True
        self.searchItem = self.searchEntry.get()
        self.search = True
        try:
            self.seedSentLen = int(self.seedSentLenVar.get().split(" ")[0])
        except ValueError:
            self.seedSentLen = self.seedSentLenVar.get().split(" ")[0]
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()

    def startNewPtbSearch(self):
        self.searchItem = ''
        if self.newSeedSentLen != None:
            #we have arrived here from the auto viewer
            self.seedSentLen = self.newSeedSentLen
            self.fileManagerFolder = self.ptb_folder+"/"+"/".join(self.fileManagerFolder.split("/")[1:])
        self.ptb_search = True
        #this function is called if the user is selecting a specific ptb tree to view using the
        #the fileManager system.. we have already set self.searchItem to '' and
        #self.seedSentLen to equal the length of the ptb tree we are searching for..
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()

    def viewCurrentPTBtree(self):
        #used for returning to annotation view when in seed set view mode
        self.quit = False
        self.currentPTBtree = True
        if self.override_GUI_MATCH == False and (self.match == False or self.match == 'False' or self.match == 'checked'):
            self.GUI_MATCH = False
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()
                                
    def mainWindowQuit(self):
        self.delete_empty_files_folders()
        self.quit = True
        self.start_auto_gen = False
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()

    def acceptParse(self, new_sent=False, fromDerivationBuilder=False):
        if not (new_sent and fromDerivationBuilder):
            if self.xbar_trees == []:
                self.noParse()
                return
            else:
                if self.parserSettings['useAllNull']:
                    self.parserSettingsLastParse += "UseAllNull"
                if self.parserSettings['skipRel']:
                    self.parserSettingsLastParse += "SkipRel"
                if self.parserSettings['skipPro']:
                    self.parserSettingsLastParse += "SkipPro"
                if len(self.xbar_bracketings) > 1:
                    self.mg_tree_index = int(self.spin.get())-1
                else:
                    self.mg_tree_index = 0
                #we must take this tree out of the auto set as we have just added it to the seed set..
                if not new_sent:
                    self.seeds[self.ptb_file_line_number] = (self.subcat_derivation_bracketings[self.mg_tree_index], self.xbar_bracketings[self.mg_tree_index], self.derived_bracketings[self.mg_tree_index], self.subcat_full_derivation_bracketings[self.mg_tree_index], self.derivation_bracketings[self.mg_tree_index], self.full_derivation_bracketings[self.mg_tree_index], self.parserSettingsLastParse)
                    try:
                        auto_set = json.load(open(self.auto_folder+"/"+self.section_folder+"/"+self.ptb_file))
                        updateAutoCTM = False
                        if str(self.ptb_file_line_number) in auto_set:
                            del(auto_set[str(self.ptb_file_line_number)])
                            updateAutoCTM = True
                        with open(self.auto_folder+"/"+self.section_folder+"/"+self.ptb_file, 'w') as auto_set_file:
                            json.dump(auto_set, auto_set_file)
                        if updateAutoCTM:
                            self.updateAutoMappings(self.auto_folder, self.section_folder, self.ptb_file, self.ptb_file_line_number)
                    except IOError:
                        x=0
                    if self.auto_folder not in os.listdir(os.getcwd()):
                        os.mkdir(self.auto_folder)
                    if self.counts[self.seedSentLen] == 1:
                        self.GUI_MATCH = False
                    self.test_words = None
                    self.untokenizedTestSentence = None
                else:
                    self.seeds["0"] = (self.subcat_derivation_bracketings[self.mg_tree_index], self.xbar_bracketings[self.mg_tree_index], self.derived_bracketings[self.mg_tree_index], self.subcat_full_derivation_bracketings[self.mg_tree_index], self.derivation_bracketings[self.mg_tree_index], self.full_derivation_bracketings[self.mg_tree_index], self.parserSettingsLastParse)
                    self.addedNewSentenceToSeeds = True
                    self.newSentMode = False
            self.MGcatsCopy = self.MGcatsCopy1
        else:
            self.newSentMode = False
            self.MGcatsCopy = self.MGcatsCopy2
            parser_setting_add = ""
            if self.parserSettings['useAllNull']:
                parser_setting_add+="UseAllNull"
            if self.parserSettings['skipRel']:
                parser_setting_add+="SkipRel"
            if self.parserSettings['skipPro']:
                parser_setting_add+="SkipPro"
            self.seeds = {'0': (self.seeds['0'][0], self.seeds['0'][1], self.seeds['0'][2], self.seeds['0'][3], self.seeds['0'][4], self.seeds['0'][5], self.seeds['0'][6]+parser_setting_add)}
        if self.match == 'checked':
            self.search = True
        self.mainWindow.destroy()

    def updateAutoMappings(self, auto_folder, section_folder, ptb_file, line_number):
        autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
        autoTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
        try:
            autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
        except IOError:
            autoNullCatTreeMappings = {}
        try:
            autoNullTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoNullTreeCatMappings'))
        except IOError:
            autoNullTreeCatMappings = {}
        PARSE = [auto_folder+'/'+section_folder+'/'+ptb_file, line_number]
        MGcats = autoTreeCatMappings[str(PARSE)]
        nullMGcats = autoNullTreeCatMappings[str(PARSE)]
        if str(PARSE) in autoTreeCatMappings:
            del(autoTreeCatMappings[str(PARSE)])
        if str(PARSE) in autoNullTreeCatMappings:
            del(autoNullTreeCatMappings[str(PARSE)])
        for cat in MGcats:
            if cat in autoCatTreeMappings:
                if PARSE in autoCatTreeMappings[cat]:
                    autoCatTreeMappings[cat].remove(PARSE)
                    if len(autoCatTreeMappings[cat]) == 0:
                        del(autoCatTreeMappings[cat])
        for cat in nullMGcats:
            if cat in autoNullCatTreeMappings:
                if PARSE in autoNullCatTreeMappings[cat]:
                    autoNullCatTreeMappings[cat].remove(PARSE)
                    if len(autoNullCatTreeMappings[cat]) == 0:
                        del(autoNullCatTreeMappings[cat])
        with open(self.auto_folder+"/"+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
            json.dump(autoCatTreeMappings, autoCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
            json.dump(autoTreeCatMappings, autoTreeCatMappingsFile)
        with open(self.auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
            json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
            json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)

    def showtrees(self, treeType, trees=None):
        if trees == None:
            #for the PTB tree window we will feed this function a list with a single PTB tree
            #but for MG trees we need to let the system toggle between xbar, derivation and derived trees
            #hence we set trees to a class attribute..
            trees = self.trees
        if treeType == 'MG' and len(trees)>1:
            try: n = int(self.spin.get())
            except ValueError: n=1
        else:
            n=1
        if treeType=='PTB':
            self.destroySpins()
            #we need to wipe the xbar tree window clean as we have a new PTB tree..
            self.freshXbarWindow()
            self.PTB_tframe.destroy()
            self.PTB_tframe = Frame(self.outer_PTB_tframe)
            self.PTB_tframe.pack(side='left', fill=BOTH, expand=True)
            if self.PTBtreeWidget != None:
                self.ptb_label.destroy()
            self.outerPTBTreeframe = Frame(self.PTB_tframe)
            self.outerPTBTreeframe.pack(fill=BOTH, expand=True)
            self.innerPTBTreeframe = Frame(self.outerPTBTreeframe)
            self.innerPTBTreeframe.pack(fill=BOTH, expand=True)
            self.PTBwindow = CanvasFrame(self.innerPTBTreeframe, highlightthickness=2, highlightbackground='black', bg='white',height=10)
            self.PTBwindow.pack(fill=BOTH, expand=True)
            sourceLabelFrame = Frame(self.PTB_tframe)
            sourceLabelFrame.pack()
            self.sourceLabel = Label(sourceLabelFrame, text='Source')
            self.sourceLabel.pack(side='left')
            self.ccg_btn_text = StringVar()
            ccg_button_inactive = False
            if "CCGbank" not in os.listdir(os.getcwd()):
                ccg_button_inactive = True
            else:
                ccg_parses = json.load(open("CCGbank/"+self.section_folder+"/"+self.ptb_file.split(".")[0]+".ccg"))
                if self.ptb_file.split(".")[0]+"."+str(self.ptb_file_line_number+1) not in ccg_parses:
                    ccg_button_inactive = True
                else:
                    self.ccg_bracketing = ccg_parses[self.ptb_file.split(".")[0]+"."+str(self.ptb_file_line_number+1)]
                    try:
                        self.ccg_tree = Tree.fromstring(self.ccg_bracketing, remove_empty_top_bracketing=True)
                    except AttributeError:
                        self.ccg_tree = Tree.parse(ccg_bracketing, remove_empty_top_bracketing=True)
            self.viewPTBbracketButton = Button(sourceLabelFrame, text='[...]', command=lambda: self.viewPTBbracketing())
            self.viewPTBbracketButton.pack(side='left')
            self.fullPTBTreeButton = Button(sourceLabelFrame, text='FS', command=lambda: self.newFullTreeWindow(treeType='PTB', trees=[self.PTBtree]))
            self.fullPTBTreeButton.pack(side='left')
            self.CCGTreeButton = Button(sourceLabelFrame, textvariable=self.ccg_btn_text, command=lambda: self.toggle_ccg_ptb_tree(trees[n-1]))
            self.CCGTreeButton.pack(side='left')
            self.ccg_btn_text.set("CCG")
            if ccg_button_inactive:
                self.CCGTreeButton.config(state=DISABLED)
            if self.mode == 'annotation' and (self.match == False or self.match == 'False' or self.match == 'checked'):
                self.viewPTBbracketButton.config(state=DISABLED)
                self.CCGTreeButton.config(state=DISABLED)
                self.fullPTBTreeButton.config(state=DISABLED)
            labelAndNavButtonFrame = Frame(self.PTB_tframe)
            labelAndNavButtonFrame.pack()
            if self.mode == 'annotation':
                gotoButton = Button(labelAndNavButtonFrame, text='Goto', command=self.go)
                gotoButton.pack(side='left')
                self.gotoEntry = Entry(labelAndNavButtonFrame, width = 4)
                if self.match == False or self.match == 'False' or self.match == 'checked':
                    self.gotoEntry.insert(END, '0')
                else:
                    self.gotoEntry.insert(END, str(self.displayIndex))
                self.gotoEntry.pack(side='left')
                sourceTreeNumLabel = Label(labelAndNavButtonFrame, text='/'+str(self.counts[self.seedSentLen]))
                sourceTreeNumLabel.pack(side='left')
                previous_button = Button(labelAndNavButtonFrame, text='<', command=self.viewPreviousPTBtree)
                previous_button.pack(side='left')
                next_button = Button(labelAndNavButtonFrame, text='>', command=self.viewNextPTBtree)
                next_button.pack(side='left')
                if self.match == False or self.match == 'False' or self.match == 'checked':
                    previous_button.config(state=DISABLED)
                    next_button.config(state=DISABLED)
                    gotoButton.config(state=DISABLED)
            if self.mode == 'annotation' and (self.match == False or self.match == 'False' or self.match == 'checked'):
                x=0
            else:
                self.ptb_label = Label(labelAndNavButtonFrame, text='  '+self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"   Ln: "+str(self.ptb_file_line_number+1)+"   ")
                self.ptb_label.pack(side='left')
            if self.mode == 'annotation':
                stringLensWithCounts = []
                for item in self.stringLengths:
                    newItem = str(item)+"     ("+str(self.counts[item])+")"
                    stringLensWithCounts.append(newItem)
                self.seedSentLenVar = StringVar(self.mainWindow)
                self.seedSentLenVar.set(self.seedSentLen)
                stringLenLabel = Label(labelAndNavButtonFrame, text=" Len:")
                stringLenLabel.pack(side='left')
                dropDown = OptionMenu(labelAndNavButtonFrame, self.seedSentLenVar, *stringLensWithCounts)
                dropDown.config(width=6)
                dropDown.pack(side='left')
                self.seedSentLenVar.trace('w', self.quickStartNewSearch)
                searchButton = Button(labelAndNavButtonFrame, text='Find', command=self.startNewSearch)
                searchButton.pack(side='left')
                self.searchEntry = Entry(labelAndNavButtonFrame, width = 15)
                self.searchEntry.insert(END, self.searchItem)
                self.searchEntry.pack(side='left')
                self.searchEntry.focus_set()
                self.searchEntry.bind('<Return>', lambda x: searchButton.invoke())
                clearButton = Button(labelAndNavButtonFrame, text='Clear', command=self.clearSearch)
                clearButton.pack(side='left')
            if self.mode == 'annotation':
                if self.match == False or self.match == 'False' or self.match == 'checked':
                    try:
                        blank_tree = Tree.fromstring('(())', remove_empty_top_bracketing=True)
                    except AttributeError:
                        blank_tree = Tree.parse('(())', remove_empty_top_bracketing=True)
                    self.PTBtreeWidget = TreeWidget(self.PTBwindow.canvas(), blank_tree, draggable=1, shapeable=1)
                else:
                    self.PTBtreeWidget = TreeWidget(self.PTBwindow.canvas(), trees[n-1], draggable=1, shapeable=1)                    
            else:
                self.PTBtreeWidget = TreeWidget(self.PTBwindow.canvas(), trees[n-1], draggable=1, shapeable=1)
            self.PTBwindow.add_widget(self.PTBtreeWidget, 10, 10)
        elif treeType=='MG':
            self.freshXbarWindow()
            self.MGtreeWidget = TreeWidget(self.XBARwindow.canvas(), trees[n-1], draggable=1, shapeable=1)
            self.XBARwindow.add_widget(self.MGtreeWidget, 10, 10)
        self.TASK = 'annotate'

    def toggle_ccg_ptb_tree(self, ptb_tree):
        self.innerPTBTreeframe.destroy()
        self.innerPTBTreeframe = Frame(self.outerPTBTreeframe)
        self.innerPTBTreeframe.pack(fill=BOTH, expand=True)
        self.PTBwindow = CanvasFrame(self.innerPTBTreeframe, highlightthickness=2, highlightbackground='black', bg='white',height=10)
        self.PTBwindow.pack(fill=BOTH, expand=True)
        if self.ptb_ccg_toggle == "ccg":
            self.PTBtreeWidget = TreeWidget(self.PTBwindow.canvas(), ptb_tree, draggable=1, shapeable=1)
            self.ptb_ccg_toggle = "ptb"
            self.ccg_btn_text.set("CCG")
        elif self.ptb_ccg_toggle == "ptb":
            self.PTBtreeWidget = TreeWidget(self.PTBwindow.canvas(), self.ccg_tree, draggable=1, shapeable=1)
            self.ptb_ccg_toggle = "ccg"
            self.ccg_btn_text.set("Penn")
        self.PTBwindow.add_widget(self.PTBtreeWidget, 10, 10)

    def quickStartNewSearch(self, x, y, z):
        #quickly just switches to sentences of a different word count as soon as the
        #drop down option for that sentence length is selected
        self.startNewSearch()

    def clearSearch(self):
        self.searchEntry.delete(0, 'end')
        self.startNewSearch()

    def most_common(self, lst):
        return max(set(lst), key=lst.count)

    def Parse(self, confirmedUntaggedWords=False, saveOption=True):
        reload(cky_mg)
        if confirmedUntaggedWords:
            if not self.newSentMode:
                self.autoGenerateCorpus(updateDepMappingsOnly=True)
        self.MGcatsCopy1 = copy.deepcopy([cat.get().split("     ")[0] for cat in self.MGcats])
        index = -1
        for cat in self.MGcatsCopy1:
            index += 1
            self.MGcatsCopy1[index] = cat.strip()
        if self.untaggedWordsWindow != None:
            self.destroyWindow(self.untaggedWordsWindow, 'untaggedWordsWindow')
        i = -1
        if self.newSentMode:
            words = self.test_words
            moveable_spans = None
            source_spans = None
            #as we have no PTB tree to check against we will just set vp_ellipsis to True so that
            #the system does not block [pro-v] categories..
            vp_ellipsis = True
        else:
            words = self.words
            PTB_TREE = autobank.build_tree(self.PTBbracketing)
            PTB_tree = PTB_TREE[0]
            terminals = PTB_TREE[2]
            vp_ellipsis = autobank.contains_vp_ellipsis(PTB_tree)
            if self.parserSettings["constrainMoveWithPTB"]:
                moveable_spans = []
                autobank.get_moveable_spans(PTB_tree, terminals, moveable_spans)
            else:
                moveable_spans = None
            if self.parserSettings["constrainConstWithPTBCCG"]:
                source_spans = []
                try:
                    ccg_parses = json.load(open("CCGbank/"+self.section_folder+"/"+self.ptb_file.split(".")[0]+".ccg"))
                    ccg_bracketing = ccg_parses[self.ptb_file.split(".")[0]+"."+str(self.ptb_file_line_number+1)]
                    ccg_tree = autobank.build_tree(ccg_bracketing)
                    ccg_terminals = ccg_tree[2]
                    autobank.set_indices(ccg_terminals)
                    ccg_tree = ccg_tree[0]
                except Exception as e:
                    ccg_tree = None
                autobank.get_source_spans(PTB_tree, ccg_tree, source_spans, terminals)
            else:
                source_spans = None
        for word in words:
            i+=1
            if self.MGcats[i].get().split("          ")[0] == 'No categories available':
                self.nothing(self.mainWindow, "'No categories available' selected for some PTB cat!")
                return
        if not self.newSentMode:
            terminals = autobank.build_tree(self.PTBbracketing)[2]
        if not confirmedUntaggedWords:
            i=-1
            for word in words:
                i+=1
                if self.MGcats[i].get().split("          ")[0] == '':
                    self.untaggedWords()
                    return
        if self.newSentMode:
            try:
                overtLexicon = json.load(open(self.seed_folder+'/OvertLexicon'))
            except IOError:
                overtLexicon = {}
            #for unseen gendered first names, in new sentence mode (i.e. when doing real parsing), we will use the category
            #associated most frequently with names of that gender in the overt lexicon
            if len(self.nameCats['male']) > 0:
                unseen_male_name_cat = self.most_common(self.nameCats['male'])
            else:
                unseen_male_name_cat = None
            if len(self.nameCats['female']) > 0:
                unseen_female_name_cat = self.most_common(self.nameCats['female'])
            else:
                unseen_female_name_cat = None
        self.destroySpins()
        self.destroyButtons()
        self.miniOvertLexicon = []
        #this method takes the MG categories that the words have been annotated with
        #and send these to cky_mg parser as a mini lexicon.. MG parse then uses these plus
        #all the covert categories in the system to try to parse the sentence..
        #first, we must build the lexicon
        i=-1
        for word in words:
            i+=1
            if self.newSentMode and self.MGcats[i].get() == '':
                added_male_name_cat = False
                added_female_name_cat = False
                if unseen_male_name_cat != None and word in male_names:
                    print "Adding most common cat for male names for unseen word: "+word
                    MGcat = unseen_male_name_cat
                    added_male_name_cat = True
                elif unseen_female_name_cat != None and word in female_names:
                    print "Adding most common cat for female names for unseen word: "+word
                    MGcat = unseen_female_name_cat
                    added_female_name_cat = True
                if added_male_name_cat or added_female_name_cat:
                    features = MGcat.split(" ")
                    lex = self.constructMGlexEntry(word, features)
                    if lex != None and lex not in self.miniOvertLexicon:
                        self.miniOvertLexicon.append([lex, i])
            if self.MGcats[i].get().split("          ")[0] == 'No categories available':
                self.nothing(self.mainWindow, "'No categories available' for some PTB cat!")
                return
            elif self.MGcats[i].get().split("          ")[0] == '':
                if not self.newSentMode:
                    terminal_cat = terminals[i].mother.truncated_name
                if self.newSentMode:
                    if word not in overtLexicon:
                        if not (added_female_name_cat or added_male_name_cat):
                            self.nothing(self.mainWindow, "No lexicon entry for unannotated word: "+word)
                            return
                    else:
                        for MGcat in overtLexicon[word]:
                            print "Adding MG cat for word: "+word
                            features = MGcat.split(" ")
                            #ignoring subcat frames for now and just adding everything to the lexicon..
                            lex = self.constructMGlexEntry(word, features)
                            if lex != None and lex not in self.miniOvertLexicon:
                                self.miniOvertLexicon.append([lex, i])
                else:
                    if ccg_terminals[i].mother.name in stats_table:
                        for MGcat in stats_table[ccg_terminals[i].mother.name]:
                            print "Adding MG cat for CCG cat:", ccg_terminals[i].mother.name
                            features = MGcat.split(" ")
                            #ignoring subcat frames for now and just adding everything to the lexicon..
                            lex = self.constructMGlexEntry(terminals[i].name.lower(), features)
                            if lex != None and lex not in self.miniOvertLexicon:
                                self.miniOvertLexicon.append([lex, i])
                    else:
                        if len(self.PosDepsMappings[terminal_cat]) == 0:
                            self.nothing(self.mainWindow, terminal_cat+" has no label in seed corpus!")
                            return
                        print "\nCCG category for '"+terminals[i].name+"' not currently paired with any MG categories in seeds.. using PTB caregory instead..."
                        for MGcat in self.PosDepsMappings[terminal_cat]:
                            features = MGcat.split(" ")
                            #ignoring subcat frames for now and just adding everything to the lexicon..
                            lex = self.constructMGlexEntry(terminals[i].name.lower(), features)
                            if lex != None and lex not in self.miniOvertLexicon:
                                self.miniOvertLexicon.append([lex, i])
            else:
                features = self.MGcats[i].get().split("          ")[0].split(" ")
                entry = self.constructMGlexEntry(word, features)
                if entry != None:
                    self.miniOvertLexicon.append([entry, i])
        try:
            self.parseMessage()
            if not self.newSentMode:
                print '\nParsing sentence: "'+" ".join(words)+'", in PTB file: '+self.ptb_file+' Ln: '+str(self.ptb_file_line_number+1)+'...\n'
            else:
                print '\nParsing sentence: "'+" ".join(words)+'"...'
            with timeout(self.parserSettings['timeout_seconds']):
                start_time = default_timer()
                if self.parserSettings['useAllNull']:
                    useAllNull = True
                else:
                    useAllNull = False
                if self.parserSettings['skipRel']:
                    skipRel = True
                else:
                    skipRel = False
                if self.parserSettings['skipPro']:
                    skipPro = True
                else:
                    skipPro = False
                if self.parserSettings['parserSetting'] == 'basicOnly':
                    (parse_time, self.derivation_bracketings, self.derived_bracketings, self.xbar_bracketings, self.XBAR_trees, self.subcat_derivation_bracketings, self.subcat_full_derivation_bracketings, self.full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=self.miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                elif self.parserSettings['parserSetting'] == 'fullFirst':
                    (parse_time, self.derivation_bracketings, self.derived_bracketings, self.xbar_bracketings, self.XBAR_trees, self.subcat_derivation_bracketings, self.subcat_full_derivation_bracketings, self.full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=self.miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                elif self.parserSettings['parserSetting'] == 'basicAndRight':
                    (parse_time, self.derivation_bracketings, self.derived_bracketings, self.xbar_bracketings, self.XBAR_trees, self.subcat_derivation_bracketings, self.subcat_full_derivation_bracketings, self.full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=self.miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                elif self.parserSettings['parserSetting'] == 'basicAndExcorp':
                    (parse_time, self.derivation_bracketings, self.derived_bracketings, self.xbar_bracketings, self.XBAR_trees, self.subcat_derivation_bracketings, self.subcat_full_derivation_bracketings, self.full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=self.miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                elif self.parserSettings['parserSetting'] == 'basicAndTough':
                    (parse_time, self.derivation_bracketings, self.derived_bracketings, self.xbar_bracketings, self.XBAR_trees, self.subcat_derivation_bracketings, self.subcat_full_derivation_bracketings, self.full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=self.miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                end_time = default_timer() - start_time
                print "Parsing complete.  Total processing time: ", autobank.time_taken(end_time)
            self.destroyWindow(self.parsing, 'parsing')
            self.parserSettingsLastParse = self.parserSettings['parserSetting']
        except IOError:#Exception as e:
            if self.parsing != None:
                self.destroyWindow(self.parsing, 'parsing')
            print "Error!!!"
            self.derivation_bracketings = []
        if len(self.derivation_bracketings) == 0:
            print "\nNo Parses discovered!\n"
            self.noParses()
            if self.newSentMode:
                self.save_button.config(state=DISABLED)
            return
        else:
            if len(self.derivation_bracketings) > 1:
                text = " parses discovered.\n"
            else:
                text = " parse discovered.\n"
            print "\n"+str(len(self.derivation_bracketings))+text
        if not self.newSentMode:
            self.add2SeedsButton = Button(self.mainButtonFrame, text='save', command=self.acceptParse)
            self.add2SeedsButton.pack(side='left')
            if confirmedUntaggedWords:
                #leave this on because trying to get the linear order of categories when you let the system try all cats is a total nightmare
                self.add2SeedsButton.config(state=DISABLED)
        elif saveOption:
            self.save_button.config(state=NORMAL)
        else:
            self.save_button.config(state=DISABLED)
        got_PTB_DEPS = False
        if len(self.derivation_bracketings) > 1:
            emptyLabel = Label(self.mainButtonFrame, text=" ")
            emptyLabel.pack(side='left')
            self.eliminateButton = Button(self.mainButtonFrame, text='reject', command=self.eliminateParse)
            self.eliminateButton.pack(side='left')
            self.eliminateEntry = Entry(self.mainButtonFrame, width = 7)
            self.eliminateEntry.pack(side='left')
            emptyLabel = Label(self.mainButtonFrame, text=" ")
            emptyLabel.pack(side='left')
            self.compareButton = Button(self.mainButtonFrame, text='compare', command=self.treeCompare)
            self.compareButton.pack(side='left')
            got_PTB_DEPS = True
            (PTB_deps, terminals, PTB_tree) = autobank.get_deps_terminals(self.PTBbracketing, False)
            #if not self.newSentMode:
                #self.autoSelectButton = Button(self.mainButtonFrame, text='best', command=lambda: self.autoSelectEliminateQuestion(self.XBAR_trees, PTB_deps, PTB_tree, terminals))
                #self.autoSelectButton.pack(side='left')
        if len(self.derivation_bracketings) > 0 and not self.newSentMode:
            if not got_PTB_DEPS:
                (PTB_deps, terminals, PTB_tree) = autobank.get_deps_terminals(self.PTBbracketing, False)
            self.scoreButton = Button(self.mainButtonFrame, text='score', command=lambda: self.displayTreeScore(self.XBAR_trees, PTB_deps, PTB_tree, terminals))
            self.scoreButton.pack(side='left')
        self.xbar_trees = []
        self.derivation_trees = []
        self.subcat_derivation_trees = []
        self.subcat_full_derivation_trees = []
        self.full_derivation_trees = []
        self.derived_trees = []
        for i in range(len(self.xbar_bracketings)):
            db = cky_mg.fix_coord_annotation(self.derivation_bracketings[i])
            sdb = cky_mg.fix_coord_annotation(self.subcat_derivation_bracketings[i])
            sfdb = cky_mg.fix_coord_annotation(self.subcat_full_derivation_bracketings[i])
            fdb = cky_mg.fix_coord_annotation(self.full_derivation_bracketings[i])
            while "  " in sfdb:
                sfdb = re.sub("  ", " ", sfdb, count=10000)
            while "  " in fdb:
                fdb = re.sub("  ", " ", fdb, count=10000)
            try:
                xbar_tree = Tree.parse(self.xbar_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derivation_tree = Tree.parse(db, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_derivation_tree = Tree.parse(sdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_full_derivation_tree = Tree.parse(sfdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                full_derivation_tree = Tree.parse(fdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derived_tree = Tree.parse(self.derived_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            except AttributeError:
                xbar_tree = Tree.fromstring(self.xbar_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derivation_tree = Tree.fromstring(db, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_derivation_tree = Tree.fromstring(sdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                subcat_full_derivation_tree = Tree.fromstring(sfdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                full_derivation_tree = Tree.fromstring(fdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                derived_tree = Tree.fromstring(self.derived_bracketings[i], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            self.xbar_trees.append(xbar_tree)
            self.derivation_trees.append(derivation_tree)
            self.subcat_derivation_trees.append(subcat_derivation_tree)
            self.subcat_full_derivation_trees.append(subcat_full_derivation_tree)
            self.full_derivation_trees.append(full_derivation_tree)
            self.derived_trees.append(derived_tree)
        self.trees = self.xbar_trees
        self.refreshSpins()
        self.showtrees(treeType='MG')

    def displayTreeScore(self, XBAR_trees, PTB_deps, PTB_tree, terminals):
        try:
            DepMappings = json.load(open(self.seed_folder+'/DepMappings'))
        except IOError:
            DepMappings = {}
        try:
            RevDepMappings = json.load(open(self.seed_folder+'/RevDepMappings'))
        except IOError:
            RevDepMappings = {}
        if len(XBAR_trees) > 1:
            pdb.set_trace()
            index = int(self.spin.get())-1
        else:
            index = 0
        (best_trees, max_matched_mg_chains, self.reifiedSentDepMappings, self.matched_mappings) = autobank.evaluate_trees([XBAR_trees[index]], PTB_deps, DepMappings, RevDepMappings, PTB_tree, terminals, True, None, None, None, None, None)
        if len(best_trees) == 1:
            self.nothing(self.mainWindow, "Dependency mapping score for this tree: "+str(max_matched_mg_chains), extraButton="Deps", height=58, width=475)

    def autoSelectEliminateQuestion(self, XBAR_trees, PTB_deps, PTB_tree, terminals):
        if self.autoSelectEliminateQuestionWindow != None:
            self.destroyWindow(self.autoSelectEliminateQuestion, 'autoSelectEliminateQuestionWindow')
        self.autoSelectEliminateQuestionWindow = Toplevel(self.mainWindow)
        self.autoSelectEliminateQuestionWindow.title('Confirm eliminate parses')
        self.autoSelectEliminateQuestionWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.autoSelectEliminateQuestionWindow, 'autoSelectEliminateQuestionWindow'))
        w=350
        h=90
        (x, y) = self.getCentrePosition(w, h)
        self.autoSelectEliminateQuestionWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.autoSelectEliminateQuestionWindow, text="Only the best trees will be retained.\nAll other trees will be eliminated.\nDo you wish to continue?")
        label.pack()
        buttonFrame = Frame(self.autoSelectEliminateQuestionWindow)
        buttonFrame.pack()
        confirmButton = Button(buttonFrame, text="continue", command=lambda: self.autoSelect(XBAR_trees, PTB_deps, True, self.subcat_derivation_bracketings, self.derivation_bracketings, PTB_tree, terminals))
        confirmButton.pack(side=LEFT)
        cancelButton = Button(buttonFrame, text="cancel", command=lambda: self.destroyWindow(self.autoSelectEliminateQuestionWindow, 'autoSelectEliminateQuestionWindow'))
        cancelButton.pack(side=LEFT)

    def untaggedWords(self):
        if self.untaggedWordsWindow != None:
            self.destroyWindow(self.untaggedWordsWindow, 'untaggedWordsWindow')
        self.untaggedWordsWindow = Toplevel(self.mainWindow)
        self.untaggedWordsWindow.title('Confirm parse with untagged words')
        self.untaggedWordsWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.untaggedWordsWindow, 'untaggedWordsWindow'))
        w=370
        h=55
        (x, y) = self.getCentrePosition(w, h)
        self.untaggedWordsWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.untaggedWordsWindow, text="You must select MG categories for all words!")
        label.pack()
        buttonFrame = Frame(self.untaggedWordsWindow)
        buttonFrame.pack()
        okButton = Button(buttonFrame, text="Ok", command=lambda: self.destroyWindow(self.untaggedWordsWindow, 'untaggedWordsWindow'))
        okButton.pack(side=LEFT)

    def autoSelect(self, xbar_trees, PTB_deps, eliminate, subcat_derivation_bracketings, derivation_bracketings, PTB_tree, terminals):
        trees_showed = False
        self.autoGenerateCorpus(updateDepMappingsOnly=True)
        if self.autoSelectEliminateQuestionWindow != None:
            self.destroyWindow(self.autoSelectEliminateQuestionWindow, 'autoSelectEliminateQuestionWindow')
        #allows the system to automatically choose among candidate MG parses for a single Penn tree during seed set annotation..
        try:
            DepMappings = json.load(open(self.seed_folder+'/DepMappings'))
        except IOError:
            DepMappings = {}
        try:
            RevDepMappings = json.load(open(self.seed_folder+'/RevDepMappings'))
        except IOError:
            RevDepMappings = {}
        (best_trees, max_matched_mg_chains) = autobank.evaluate_trees(xbar_trees, PTB_deps, DepMappings, RevDepMappings, PTB_tree, terminals, False, derivation_bracketings, subcat_derivation_bracketings, None, None)
        if len(best_trees) > 1:
            autobank.eliminate_trees_with_certain_subcats(best_trees, subcat_derivation_bracketings, xbar_trees)
        if len(best_trees) > 1:
            autobank.retain_trees_with_least_pro(best_trees, derivation_bracketings)
        #if len(best_trees) > 1: I have commented this out because where there are reflexives, we'd rather have movement than not
            #autobank.retain_trees_with_least_traces(best_trees, derived_bracketings)
        if len(best_trees) > 1:
            autobank.retain_smallest_trees(best_trees, derivation_bracketings)
        trees_to_delete = []
        entries_to_delete = []
        for entry in best_trees:
            if entry[2] != max_matched_mg_chains:
                entries_to_delete.append(entry)
        for entry in entries_to_delete:
            best_trees.remove(entry)
        auto_selected_tree = best_trees[0][0]
        if eliminate:
            for xbar_tree in xbar_trees:
                if xbar_tree not in [t[0] for t in best_trees]:
                    trees_to_delete.append(xbar_tree)
            for xbar_tree in trees_to_delete:
                parse_index = xbar_trees.index(xbar_tree)
                del(xbar_trees[parse_index])
                del(self.xbar_trees[parse_index])
                del(self.xbar_bracketings[parse_index])
                del(self.derivation_trees[parse_index])
                del(self.derivation_bracketings[parse_index])
                del(self.subcat_derivation_trees[parse_index])
                del(self.subcat_derivation_bracketings[parse_index])
                del(self.subcat_full_derivation_trees[parse_index])
                del(self.subcat_full_derivation_bracketings[parse_index])
                del(self.full_derivation_trees[parse_index])
                del(self.full_derivation_bracketings[parse_index])
                del(self.derived_trees[parse_index])
                del(self.derived_bracketings[parse_index])
                self.refreshSpins()
                self.showtrees(treeType='MG', trees=self.xbar_trees)
                trees_showed = True
                if len(self.derivation_bracketings) == 1:
                    self.compareButton.destroy()
                    #self.autoSelectButton.destroy()
                    #self.scoreButton.destroy()
                    if self.eliminateButton != None:
                        self.eliminateButton.destroy()
                        self.eliminateEntry.destroy()
        auto_selected_tree_index = str(xbar_trees.index(auto_selected_tree)+1)
        try:
            self.spin.delete(0,"end")
            self.spin.insert(0, auto_selected_tree_index)
        except AttributeError:
            x=0
        if not trees_showed:
            self.showtrees(treeType='MG')
        if len(best_trees) == 1:
            self.nothing(self.mainWindow, "1 best tree with "+str(max_matched_mg_chains)+" stored mappings matched!")
        elif len(best_trees) > 1:
            self.nothing(self.mainWindow, str(len(best_trees))+" best trees with "+str(max_matched_mg_chains)+" stored mappings matched!")

    def viewPTBbracketing(self):
        if self.PTBbracketingWindow != None:
            self.destroyWindow(self.PTBbracketingWindow, 'PTBbracketingWindow')
        self.PTBbracketingWindow = Toplevel(self.mainWindow)
        if self.ptb_ccg_toggle == 'ptb':
            self.PTBbracketingWindow.title(self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"    Ln: "+str(self.ptb_file_line_number+1))
        elif self.ptb_ccg_toggle == 'ccg':
            self.PTBbracketingWindow.title("CCGbank/"+self.section_folder+"/"+self.ptb_file.split(".")[0]+".ccg"+"    Ln: "+str(self.ptb_file_line_number+1))
        h=250
        w=600
        (x, y) = self.getCentrePosition(w, h)
        self.PTBbracketingWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        closeButtonFrame = Frame(self.PTBbracketingWindow)
        closeButtonFrame.pack()
        closeButton = Button(closeButtonFrame, text="Close", command=lambda: self.destroyWindow(self.PTBbracketingWindow, 'PTBbracketingWindow'))
        closeButton.pack()
        bracketingFrame = Frame(self.PTBbracketingWindow)
        bracketingFrame.pack(expand=YES, fill=BOTH)
        bracketingTextBox = Text(bracketingFrame)
        if self.ptb_ccg_toggle == 'ptb':
            bracketingTextBox.insert(END, self.PTBbracketing)
        elif self.ptb_ccg_toggle == 'ccg':
            bracketingTextBox.insert(END, self.ccg_bracketing)
        bracketingTextBox.pack(side=LEFT, expand=YES, fill=BOTH)
        scrollb = Scrollbar(bracketingFrame, command=bracketingTextBox.yview)
        scrollb.pack(side=RIGHT, fill=Y)
        bracketingTextBox['yscrollcommand'] = scrollb.set
        bracketingTextBox.see(END)

    def viewDepMappings(self, reifiedSentDepMappings, matched):
        if matched:
            mappings = self.matched_mappings
            if mappings == []:
                self.nothing(self.mainWindow, "No mappings matched any seed mappings!")
                return
        else:
            mappings = reifiedSentDepMappings
            if mappings == []:
                self.nothing(self.mainWindow, "No dependency mappings were detected!")
                return
        depMappingsText = ""
        for entry in mappings:
            for ENTRY in entry:
                depMappingsText+=ENTRY+"  --->  "+str(entry[ENTRY])+"\n\n"
        if self.depMappingsWindow != None:
            self.destroyWindow(self.depMappingsWindow, 'depMappingsWindow')
        self.depMappingsWindow = Toplevel(self.mainWindow)
        if self.mode == 'annotation':
            try:
                if matched:
                    title = "Matched Dependency Mappings for tree "+self.spin.get()
                else:
                    title = "Reified Dependency Mappings for tree "+self.spin.get()
            except AttributeError:
                if matched:
                    title = "Matched Dependency Mappings for tree 1"
                else:
                    title = "Reified Dependency Mappings for tree 1"
        elif self.mode == 'viewer':
            if matched:
                title = "Matched Dependency Mappings for tree "+self.seed_folder+"/"+self.section_folder+"/"+self.ptb_file+"   Ln: "+str(self.ptb_file_line_number+1)
            else:
                title = "Reified Dependency Mappings for tree "+self.seed_folder+"/"+self.section_folder+"/"+self.ptb_file+"   Ln: "+str(self.ptb_file_line_number+1)
        h=500
        w=700
        (x, y) = self.getCentrePosition(w, h)
        self.depMappingsWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.depMappingsWindow.title(title)
        closeButtonFrame = Frame(self.depMappingsWindow)
        closeButtonFrame.pack()
        closeButton = Button(closeButtonFrame, text="Close", command=lambda: self.destroyWindow(self.depMappingsWindow, 'depMappingsWindow'))
        closeButton.pack()
        DepMappingsFrame = Frame(self.depMappingsWindow)
        DepMappingsFrame.pack(expand=YES, fill=BOTH)
        DepMappingsTextBox = Text(DepMappingsFrame)
        DepMappingsTextBox.insert(END, depMappingsText)
        DepMappingsTextBox.pack(side=LEFT, expand=YES, fill=BOTH)
        scrollb = Scrollbar(DepMappingsFrame, command=DepMappingsTextBox.yview)
        scrollb.pack(side=RIGHT, fill=Y)
        DepMappingsTextBox['yscrollcommand'] = scrollb.set
        DepMappingsTextBox.see(END)
        
    def viewBracketing(self, upperOrLower):
        if upperOrLower == 'main':
            if self.mainBracketingWindow != None:
                self.destroyWindow(self.mainBracketingWindow, 'mainBracketingWindow')
            if len(self.xbar_bracketings) == 0:
                self.nothing(self.mainWindow, "No MG parses generated yet!")
                return
            elif len(self.xbar_bracketings) > 1:
                index = int(self.spin.get())-1
            else:
                index = 0
            treeTypeSpin = self.treeTypeSpin
            self.mainBracketingWindow = Toplevel(self.mainWindow)
            bracketingWindow = self.mainBracketingWindow
            if self.mode == 'annotation':
                try:
                    title = treeTypeSpin.get()+" Bracketing for tree "+self.spin.get()
                except AttributeError:
                    title = treeTypeSpin.get()+" Bracketing for tree 1"
            elif self.mode == 'viewer':
                title = treeTypeSpin.get()+" Bracketing for tree "+self.seed_folder+"/"+self.section_folder+"/"+self.ptb_file+"   Ln: "+str(self.ptb_file_line_number+1)
        if upperOrLower == 'upper':
            if self.upperBracketingWindow != None:
                self.destroyWindow(self.upperBracketingWindow, 'upperBracketingWindow')
            if len(self.xbar_bracketings) > 1:
                index = int(self.upperSpin.get())-1
            else:
                index = 0
            treeTypeSpin = self.upperTreeTypeSpin
            self.upperBracketingWindow = Toplevel(self.treeCompareWindow)
            bracketingWindow = self.upperBracketingWindow
            title = treeTypeSpin.get()+" Bracketing for tree "+self.upperSpin.get()
        if upperOrLower == 'lower':
            if self.lowerBracketingWindow != None:
                self.destroyWindow(self.lowerBracketingWindow, 'lowerBracketingWindow')
            if len(self.xbar_bracketings) > 1:
                index = int(self.lowerSpin.get())-1
            else:
                index = 0
            treeTypeSpin = self.lowerTreeTypeSpin
            self.lowerBracketingWindow = Toplevel(self.treeCompareWindow)
            bracketingWindow = self.lowerBracketingWindow
            title = treeTypeSpin.get()+" Bracketing for tree "+self.lowerSpin.get()
        if treeTypeSpin.get() == 'Xbar':
            bracketing = "("+re.sub("[^(\w|')]*\(", " (", self.xbar_bracketings[index][1:], count = 10000)
        elif treeTypeSpin.get() == 'MG Derivation Ops+Subcat':
            bracketing = "("+re.sub("[^(\w|')]*\(", " (", self.subcat_derivation_bracketings[index][1:], count = 10000)
        elif treeTypeSpin.get() == 'MG Derived':
            bracketing = "("+re.sub("[^(\w|')]*\(", " (", self.derived_bracketings[index][1:], count = 10000)
        elif treeTypeSpin.get() == 'MG Derivation Ops':
            bracketing = "("+re.sub("[^(\w|')]*\(", " (", self.derivation_bracketings[index][1:], count = 10000)
        elif treeTypeSpin.get() == 'MG Derivation Cats':
            bracketing = "("+re.sub("[^(\w|')]*\(", " (", self.full_derivation_bracketings[index][1:], count = 10000)
        elif treeTypeSpin.get() == 'MG Derivation Cats+Subcat':
            bracketing = "("+re.sub("[^(\w|')]*\(", " (", self.subcat_full_derivation_bracketings[index][1:], count = 10000)
        bracketing = re.sub('\) \(', ')(', bracketing, count=10000)
        h=250
        w=600
        (x, y) = self.getCentrePosition(w, h)
        if upperOrLower == 'upper':
            x+=200
        elif upperOrLower == 'lower':
            x-=200
        bracketingWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        bracketingWindow.title(title)
        closeButtonFrame = Frame(bracketingWindow)
        closeButtonFrame.pack()
        if upperOrLower == 'upper':
            closeButton = Button(closeButtonFrame, text="Close", command=lambda: self.destroyWindow(self.upperBracketingWindow, 'upperBracketingWindow'))
        if upperOrLower == 'lower':
            closeButton = Button(closeButtonFrame, text="Close", command=lambda: self.destroyWindow(self.lowerBracketingWindow, 'lowerBracketingWindow'))
        if upperOrLower == 'main':
            closeButton = Button(closeButtonFrame, text="Close", command=lambda: self.destroyWindow(self.mainBracketingWindow, 'mainBracketingWindow'))
        closeButton.pack()
        bracketingFrame = Frame(bracketingWindow)
        bracketingFrame.pack(expand=YES, fill=BOTH)
        bracketingTextBox = Text(bracketingFrame)
        bracketingTextBox.insert(END, bracketing)
        bracketingTextBox.pack(side=LEFT, expand=YES, fill=BOTH)
        scrollb = Scrollbar(bracketingFrame, command=bracketingTextBox.yview)
        scrollb.pack(side=RIGHT, fill=Y)
        bracketingTextBox['yscrollcommand'] = scrollb.set
        bracketingTextBox.see(END)
            
    def diff(self):
        if self.diffWindow != None:
            self.destroyWindow(self.diffWindow, 'diffWindow')
        if len(self.xbar_bracketings) > 1:
            self.upper_tree_index = int(self.upperSpin.get())-1
            self.lower_tree_index = int(self.lowerSpin.get())-1
        else:
            self.upper_tree_index = 0
            self.lower_tree_index = 0
        bracketingsToCompare = []
        treeTypes = []
        index = self.upper_tree_index
        for treeTypeSpin in [self.upperTreeTypeSpin, self.lowerTreeTypeSpin]:
            treeTypes.append(treeTypeSpin.get())
            if treeTypeSpin.get() == 'Xbar':
                bracketingsToCompare.append(self.xbar_bracketings[index])
            elif treeTypeSpin.get() == 'MG Derivation Ops+Subcat':
                bracketingsToCompare.append(self.subcat_derivation_bracketings[index])
            elif treeTypeSpin.get() == 'MG Derived':
                bracketingsToCompare.append(self.derived_bracketings[index])
            elif treeTypeSpin.get() == 'MG Derivation Cats+Subcat':
                bracketingsToCompare.append(self.subcat_full_derivation_bracketings[index])
            elif treeTypeSpin.get() == 'MG Derivation Ops':
                bracketingsToCompare.append(self.derivation_bracketings[index])
            elif treeTypeSpin.get() == 'MG Derivation Cats':
                bracketingsToCompare.append(self.full_derivation_bracketings[index])
            index = self.lower_tree_index
        if bracketingsToCompare[0] == bracketingsToCompare[1]:
            self.nothing(self.treeCompareWindow, "The two bracketings are identical!")
            return
        elif treeTypes[0] != treeTypes[1]:
            self.nothing(self.treeCompareWindow, "Set tree formats need to be the same before doing diff!")
            return
        diff = simplediff.diff(bracketingsToCompare[0], bracketingsToCompare[1])
        max_len = 0
        for item in diff:
            if len(item[1]) > max_len:
                max_len = len(item[1])
        self.diffWindow = Toplevel(self.treeCompareWindow)
        if 600 < max_len*10 < 1000:
            w = max_len*10
        elif max_len*10 < 600:
            w = 600
        else:
            w = 1000
        h=len(diff)*25
        if h > 600:
            h=600
        (x, y) = self.getCentrePosition(w, h)
        self.diffWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.diffWindow.title("Diff Viewer")
        diffButtonFrame = Frame(self.diffWindow)
        diffButtonFrame.pack()
        diffCloseButton = Button(diffButtonFrame, text="Close", command=lambda: self.destroyWindow(self.diffWindow, 'diffWindow'))
        diffCloseButton.pack()
        diffFrame = Frame(self.diffWindow)
        diffFrame.pack(expand=YES, fill=BOTH)
        diffTextBox = Text(diffFrame)
        diffTextBox.pack(side=LEFT, expand=YES, fill=BOTH)
        scrollb = Scrollbar(diffFrame, command=diffTextBox.yview)
        scrollb.pack(side=RIGHT, fill=Y)
        diffTextBox['yscrollcommand'] = scrollb.set
        for item in diff:
            diffTextBox.insert(END, item[0]+"     "+item[1]+"\n")
        print ""
        #there is a bug with the Tkinter textbox that very occasionally certain diffs
        #just don't display.. for this reason, we also print the diff to the console
        for line in diff:
            print line[0]+"     "+line[1]
        print ""
        diffTextBox.see(END)
        
    def changeUpperTreeType(self):
        if self.upperTreeTypeSpin.get() == 'Xbar':
            self.upperTrees = self.xbar_trees
        elif self.upperTreeTypeSpin.get() == 'MG Derivation Ops+Subcat':
            self.upperTrees = self.subcat_derivation_trees
        elif self.upperTreeTypeSpin.get() == 'MG Derived':
            self.upperTrees = self.derived_trees
        elif self.upperTreeTypeSpin.get() == 'MG Derivation Cats+Subcat':
            self.upperTrees = self.subcat_full_derivation_trees
        elif self.upperTreeTypeSpin.get() == 'MG Derivation Ops':
            self.upperTrees = self.derivation_trees
        elif self.upperTreeTypeSpin.get() == 'MG Derivation Cats':
            self.upperTrees = self.full_derivation_trees
        self.showUpperTrees()

    def changeLowerTreeType(self):
        if self.lowerTreeTypeSpin.get() == 'Xbar':
            self.lowerTrees = self.xbar_trees
        elif self.lowerTreeTypeSpin.get() == 'MG Derivation Ops+Subcat':
            self.lowerTrees = self.subcat_derivation_trees
        elif self.lowerTreeTypeSpin.get() == 'MG Derived':
            self.lowerTrees = self.derived_trees
        elif self.lowerTreeTypeSpin.get() == 'MG Derivation Cats+Subcat':
            self.lowerTrees = self.subcat_full_derivation_trees
        elif self.lowerTreeTypeSpin.get() == 'MG Derivation Ops':
            self.lowerTrees = self.derivation_trees
        elif self.lowerTreeTypeSpin.get() == 'MG Derivation Cats':
            self.lowerTrees = self.full_derivation_trees
        self.showLowerTrees()

    def derivationBuilder(self):
        if self.newSentMode:
            words = self.test_words
            self.moveable_spans = None
            self.source_spans = None
        else:
            words = self.words
            PTB_TREE = autobank.build_tree(self.PTBbracketing)
            PTB_tree = PTB_TREE[0]
            terminals = PTB_TREE[2]
            if self.parserSettings["constrainMoveWithPTB"]:
                self.moveable_spans = []
                autobank.get_moveable_spans(PTB_tree, terminals, self.moveable_spans)
            else:
                self.moveable_spans = None
            if self.parserSettings["constrainConstWithPTBCCG"]:
                self.source_spans = []
                try:
                    ccg_parses = json.load(open("CCGbank/"+self.section_folder+"/"+self.ptb_file.split(".")[0]+".ccg"))
                    ccg_bracketing = ccg_parses[self.ptb_file.split(".")[0]+"."+str(self.ptb_file_line_number+1)]
                    ccg_tree = autobank.build_tree(ccg_bracketing)
                    ccg_terminals = ccg_tree[2]
                    autobank.set_indices(ccg_terminals)
                    ccg_tree = ccg_tree[0]
                except Exception as e:
                    ccg_tree = None
                autobank.get_source_spans(PTB_tree, ccg_tree, self.source_spans, terminals)
                source_spans = {}
                for i in range(len(self.words)):
                    source_spans[i] = []
                for span in self.source_spans:
                    source_spans[span[0]].append(span)
                self.source_spans = source_spans
            else:
                self.source_spans = None
        self.MGcatsCopy2 = copy.deepcopy([cat.get().split("     ")[0] for cat in self.MGcats])
        index = -1
        for cat in self.MGcatsCopy2:
            #for some weird reason, the drop down menu sometimes returns an extra space at the start of a category
            #the next lines fixe this
            index+=1
            self.MGcatsCopy2[index] = cat.strip()
        self.statePath = []
        self.statePointer = 0
        self.TREES = None
        self.expressions = None
        self.resultsExpressionList = None
        self.axiomIndices = None
        self.arg1 = None
        self.arg2 = None
        self.nextArg = 'arg1'
        self.oldCurselection = -1
        self.oldSource = None
        self.nextArg = 'arg1'
        if self.derivationWindow != None:
            self.destroyWindow(self.derivationWindow, 'derivationWindow')
        if self.treeCompareWindow != None:
            self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')
        i = -1
        if self.newSentMode:
            words = self.test_words
        else:
            words = self.words
        for word in words:
            i+=1
            if self.MGcats[i].get().split("          ")[0] == '' or self.MGcats[i].get().split("          ")[0] == 'No categories available':
                self.nothing(self.viewDelCatsWindow, "Select MG categories for all words first!")
                return
        self.derivationWindow = Toplevel(self.mainWindow)
        self.derivationWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.derivationWindow, 'derivationWindow'))
        self.derivationWindow.title("Derivation Builder")
        w=10000
        h=10000
        (x, y) = self.getCentrePosition(w, h)
        self.derivationWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        if not self.newSentMode:
            self.ptbLabel = Label(self.derivationWindow, text=self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"    Ln: "+str(self.ptb_file_line_number+1)+'    "'+" ".join(self.words)+'"')
            self.ptbLabel.pack()
        else:
            sentence = "New sentence:   "+" ".join([w for w in self.test_words])
            self.testLabel = Label(self.derivationWindow, text=sentence)
            self.testLabel.pack()
        self.derivationFrame = Frame(self.derivationWindow)
        self.derivationFrame.pack(fill=BOTH, expand=True)
        self.workspaceFrame = Frame(self.derivationFrame)
        self.workspaceFrame.pack(side=LEFT)
        workspaceLabel = Label(self.workspaceFrame, text="Numeration/Workspace")
        workspaceLabel.pack()
        globalTreeFrame = Frame(self.derivationFrame)
        globalTreeFrame.pack(side=RIGHT, fill=BOTH, expand=True)
        self.overtCats = []
        i = -1
        self.TREES = []
        #self.TREES will hold the nltk tree objects for each item in the overt axioms and derived categories listbox
        for word in words:
            i+=1
            cat = self.MGcats[i].get().strip()
            overtCat = word+" "+cat.split("          ")[0]
            self.overtCats.append(overtCat)
            try:
                derivation_tree = Tree.parse('('+overtCat+')', remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            except AttributeError:
                derivation_tree = Tree.fromstring('('+overtCat+')', remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            self.TREES.append([derivation_tree])
        overtCatsANDscroll = Frame(self.workspaceFrame)
        overtCatsANDscroll.pack(fill=Y, expand=True)
        scrollbarV = Scrollbar(overtCatsANDscroll, orient=VERTICAL)
        width = 50
        self.overtCatsBox = Listbox(overtCatsANDscroll, yscrollcommand=scrollbarV.set, width = width, height = 13)
        self.overtCatsBox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbarV.config(command=self.overtCatsBox.yview)
        scrollbarV.pack(side=RIGHT, fill=Y)
        self.expressions = []
        self.axiomIndices = []
        index = -1
        for item in self.overtCats:
            index+=1
            #these mappings will be used to retrieve the indices in the string of the axioms
            #and are needed because things will be reduced and added to the workspace, meaning that
            #the original indices are messed up.. the key is the original index, the value the new index
            self.overtCatsBox.insert(END, " "+item)
            (word, features) = (item.split(" ")[0], item.split(" ")[1:])
            entry = self.constructMGlexEntry(word, features)
            if 'conj' in entry[2]:
                separator = ":\u0305:\u0305"
            else:
                separator = "::"
            start = index
            end = index + 1
            span = [start, end]
            expression = cky_mg.Expression(cat_feature = entry[2], head_string = entry[0], head_features = entry[1], head_span = span, separator=separator)
            self.expressions.append(expression)
            self.axiomIndices.append(index)
        self.statePath.append([copy.deepcopy(self.overtCats), [e for e in self.expressions], copy.deepcopy(self.axiomIndices), [t for t in self.TREES]])
        self.overtCatsBox.bind('<Double-1>', lambda x: self.doubleClick(self.overtCatsBox))
        self.overtCatsBox.bind('<<ListboxSelect>>', lambda x: self.displayWorkspaceTree(self.overtCatsBox))
        overtCatsLabel = Label(self.workspaceFrame, text="Overt axioms and derived categories")
        overtCatsLabel.pack()
        self.ovComments = StringVar()
        Label(self.workspaceFrame, textvariable=self.ovComments).pack()
        exportButtonFrame1 = Frame(self.workspaceFrame)
        exportButtonFrame1.pack()
        exportButton1_1 = Button(exportButtonFrame1, text='Export to Arg1', command=lambda: self.exportArg(self.overtCatsBox, 'arg1'))
        exportButton1_1.pack(side=LEFT)
        exportButton1_2 = Button(exportButtonFrame1, text='Export to Arg2', command=lambda: self.exportArg(self.overtCatsBox, 'arg2'))
        exportButton1_2.pack(side=LEFT)
        copyCatButton = Button(exportButtonFrame1, text='duplicate', command=self.copyCat)
        copyCatButton.pack(side=LEFT)
        self.undoButton = Button(exportButtonFrame1, text='undo', command=self.undo)
        self.undoButton.pack(side=LEFT)
        self.undoButton.config(state=DISABLED)
        self.addToSeedsButton = Button(exportButtonFrame1, text='save', command=lambda: self.confirmAddDerivationToSeeds())
        self.addToSeedsButton.pack(side=LEFT)
        self.addToSeedsButton.config(state=DISABLED)
        emptyLabel = Label(self.workspaceFrame, text=" ")
        emptyLabel.pack()
        covertCats = self.updateCovertRefs()[0]                  
        covertCats.sort()
        covertCatsANDscroll = Frame(self.workspaceFrame)
        covertCatsANDscroll.pack()
        scrollbarV = Scrollbar(covertCatsANDscroll, orient=VERTICAL)
        width = 50
        self.covertCatsBox = Listbox(covertCatsANDscroll, yscrollcommand=scrollbarV.set, width = width, height = 13)
        self.covertCatsBox.pack(side=LEFT, fill=BOTH, expand=True)
        self.covertCatsBox.bind('<<ListboxSelect>>', lambda x: self.displayWorkspaceTree(self.covertCatsBox))
        self.covertCatsBox.bind('<Double-1>', lambda x: self.doubleClick(self.covertCatsBox))
        scrollbarV.config(command=self.covertCatsBox.yview)
        scrollbarV.pack(side=RIGHT, fill=Y)
        for item in covertCats: self.covertCatsBox.insert(END, " "+item)
        covertCatsLabelFrame = Frame(self.workspaceFrame)
        covertCatsLabelFrame.pack()
        covertCatsLabel = Label(covertCatsLabelFrame, text="Covert axioms")
        covertCatsLabel.pack(side=LEFT)
        searchButton = Button(covertCatsLabelFrame, text="s", command = lambda: self.viewDelCats(catType='covert', fromDerivation=True))
        searchButton.pack(side=LEFT)
        self.covComments = StringVar()
        Label(self.workspaceFrame, textvariable=self.covComments).pack()
        exportButtonFrame2 = Frame(self.workspaceFrame)
        exportButtonFrame2.pack()
        exportButton2_1 = Button(exportButtonFrame2, text='Export to Arg1', command=lambda: self.exportArg(self.covertCatsBox, 'arg1'))
        exportButton2_1.pack(side=LEFT)
        exportButton2_2 = Button(exportButtonFrame2, text='Export to Arg2', command=lambda: self.exportArg(self.covertCatsBox, 'arg2'))
        exportButton2_2.pack(side=LEFT)
        closeButton = Button(globalTreeFrame, text='close', command=lambda: self.destroyWindow(self.derivationWindow, 'derivationWindow'))
        closeButton.pack()
        upperTreeFrame = Frame(globalTreeFrame)
        upperTreeFrame.pack(fill=BOTH, expand=True)
        mergeButtonFrame = Frame(globalTreeFrame)
        mergeButtonFrame.pack()
        self.outerLowerTreeFrame = Frame(globalTreeFrame)
        self.outerLowerTreeFrame.pack(fill=BOTH, expand=True)
        self.innerLowerTreeFrame = Frame(self.outerLowerTreeFrame)
        self.innerLowerTreeFrame.pack(fill=BOTH, expand=True)
        self.innerInnerLowerTreeFrame = Frame(self.innerLowerTreeFrame, relief=SUNKEN)
        self.innerInnerLowerTreeFrame.pack(fill=BOTH, expand=True)
        self.lowerOuterTreeSpinFrame = Frame(self.innerLowerTreeFrame)
        self.lowerOuterTreeSpinFrame.pack()
        lowerTreeButtonFrame = Frame(self.outerLowerTreeFrame)
        lowerTreeButtonFrame.pack()
        self.acceptButton = Button(lowerTreeButtonFrame, text="Accept Result", command=self.acceptResult)
        self.acceptButton.pack()
        self.acceptButton.config(state=DISABLED)
        self.lowerTreeLabelText = StringVar()
        self.lowerTreeLabelText.set(" ")
        self.freshLowerTreeWindow()
        self.freshLowerDerivationSpins()
        self.outerArg1TreeFrame = Frame(upperTreeFrame)
        self.outerArg1TreeFrame.pack(side=LEFT, fill=BOTH, expand=True)
        self.innerArg1TreeFrame = Frame(self.outerArg1TreeFrame)
        self.innerArg1TreeFrame.pack(fill=BOTH, expand=True)
        self.innerInnerArg1TreeFrame = Frame(self.innerArg1TreeFrame)
        self.innerInnerArg1TreeFrame.pack(fill=BOTH, expand=True)
        self.freshArg1TreeWindow()
        self.arg1Label = Label(self.outerArg1TreeFrame, text="Arg1")
        self.arg1Label.pack()
        self.arg1SpinFrame = Frame(self.outerArg1TreeFrame)
        self.arg1SpinFrame.pack()
        emptyLabel = Label(self.arg1SpinFrame, text="")
        emptyLabel.pack(side=LEFT)
        saturateButton = Button(mergeButtonFrame, text='Saturate(Arg1)',  command=lambda: self.saturate(1))
        saturateButton.pack(side=LEFT)
        moveButton1 = Button(mergeButtonFrame, text="Move(Arg1)", command=lambda: self.move(1))
        moveButton1.pack(side=LEFT)
        emptyLabel = Label(mergeButtonFrame, text="             ")
        emptyLabel.pack(side=LEFT)
        mergeButton = Button(mergeButtonFrame, text="Merge(Arg1,Arg2)", command=self.merge)
        mergeButton.pack(side=LEFT)
        emptyLabel = Label(mergeButtonFrame, text="           ")
        emptyLabel.pack(side=LEFT)
        self.outerArg2TreeFrame = Frame(upperTreeFrame)
        self.outerArg2TreeFrame.pack(side=LEFT, fill=BOTH, expand=True)
        self.innerArg2TreeFrame = Frame(self.outerArg2TreeFrame)
        self.innerArg2TreeFrame.pack(fill=BOTH, expand=True)
        self.innerInnerArg2TreeFrame = Frame(self.innerArg2TreeFrame)
        self.innerInnerArg2TreeFrame.pack(fill=BOTH, expand=True)
        self.freshArg2TreeWindow()
        self.arg2Label = Label(self.outerArg2TreeFrame, text="Arg2")
        self.arg2Label.pack()
        self.arg2SpinFrame = Frame(self.outerArg2TreeFrame)
        self.arg2SpinFrame.pack()
        emptyLabel = Label(self.arg2SpinFrame, text="")
        emptyLabel.pack(side=LEFT)
        moveButton2 = Button(mergeButtonFrame, text="Move(Arg2)", command=lambda: self.move(2))
        moveButton2.pack(side=LEFT)
        saturateButton = Button(mergeButtonFrame, text='Saturate(Arg2)', command=lambda: self.saturate(2))
        saturateButton.pack(side=LEFT)

    def saturate(self, argNum):
        if argNum == 1 and self.arg1 == None:
            self.nothing(self.derivationWindow, "Arg1 is currently empty!")
            return
        elif argNum == 2 and self.arg2 == None:
            self.nothing(self.derivationWindow, "Arg2 is currently empty!")
            return
        self.resultsBracketings = None
        self.resultsTrees = None
        self.freshLowerTreeWindow()
        self.lowerTreeLabelText.set("")
        self.resultsExpressionList = []
        if argNum == 1:
            self.arg2 = None
            self.arg2CatsBox = None
            self.arg2CatsBoxIndex = None
            saturated_expression = cky_mg.type_saturate(self.arg1)
            self.freshArg2TreeWindow()
            self.destroyArg2Spin()
            self.resultText = "Result of Saturate(Arg1)"
        elif argNum == 2:
            self.arg1 = None
            self.arg1CatsBox = None
            self.arg1CatsBoxIndex = None
            saturated_expression = cky_mg.type_saturate(self.arg2)
            self.freshArg1TreeWindow()
            self.destroyArg1Spin()
            self.resultText = "Result of Saturate(Arg2)"
        self.oldCurselection = -1
        self.oldSource = None
        if saturated_expression == None:
            self.nothing(self.derivationWindow, "You cannot type-saturate that category!")
            return
        self.resultsExpressionList = [saturated_expression]
        exp_to_remove = []
        for item in self.resultsExpressionList:
            if not cky_mg.add_to_agenda(item, agenda=None, sentence_length=None, returnToAutobank=True):
                exp_to_remove.append(item)
        for item in exp_to_remove:
            self.resultsExpressionList.remove(item)
        if len(self.resultsExpressionList) == 0:
            self.nothing(self.derivationWindow, "The Move(Arg"+str(argNum)+") operation failed!")
            return
        self.getResultsBracketings()
        self.getResultsTrees()
        self.displayResultTrees()

    def move(self, argNum):
        if argNum == 1 and self.arg1 == None:
            self.nothing(self.derivationWindow, "Arg1 is currently empty!")
            return
        elif argNum == 2 and self.arg2 == None:
            self.nothing(self.derivationWindow, "Arg2 is currently empty!")
            return
        self.resultsBracketings = None
        self.resultsTrees = None
        self.freshLowerTreeWindow()
        self.lowerTreeLabelText.set("")
        self.resultsExpressionList = []
        if argNum == 1:
            self.arg2 = None
            self.arg2CatsBox = None
            self.arg2CatsBoxIndex = None
            trigger_item = self.arg1
            self.freshArg2TreeWindow()
            self.destroyArg2Spin()
            self.resultText = "Result of Move(Arg1)"
        elif argNum == 2:
            self.arg1 = None
            self.arg1CatsBox = None
            self.arg1CatsBoxIndex = None
            trigger_item = self.arg2
            self.freshArg1TreeWindow()
            self.destroyArg1Spin()
            self.resultText = "Result of Move(Arg2)"
        self.oldCurselection = -1
        self.oldSource = None
        failure_messages = []
        if '+' in trigger_item.head_chain.features[0]:
            cky_mg.move(trigger_item = trigger_item, agenda = None, direction = 'left', resultsExpressionList=self.resultsExpressionList, failure_messages=failure_messages)
        elif trigger_item.head_chain.features[0] in ['=d', '=D', '=q', '=Q']:
            cky_mg.move(trigger_item = trigger_item, agenda = None, direction = 'left', resultsExpressionList=self.resultsExpressionList, failure_messages=failure_messages)
        elif re.sub('{.*?}', '', trigger_item.head_chain.features[0]) in extraposition_hosts:
            cky_mg.move(trigger_item = trigger_item, agenda = None, direction = 'right', resultsExpressionList=self.resultsExpressionList, failure_messages=failure_messages)
        exp_to_remove = []
        #we need to make sure none of the expressions violates shortest move
        for item in self.resultsExpressionList:
            if not cky_mg.add_to_agenda(item, agenda=None, sentence_length=None, returnToAutobank=True, failure_messages=failure_messages):
                exp_to_remove.append(item)
        for item in exp_to_remove:
            self.resultsExpressionList.remove(item)
        if len(self.resultsExpressionList) == 0:
            failure_messages = list(set(failure_messages))
            if len(failure_messages) == 0:
                self.nothing(self.derivationWindow, "The Move(Arg"+str(argNum)+") operation failed!")
            elif len(failure_messages) > 1:
                self.nothing(self.derivationWindow, "The Move(Arg"+str(argNum)+") operation failed for the following reasons:", height=200, width=600, failure_messages=failure_messages)
            elif len(failure_messages) == 1:
                self.nothing(self.derivationWindow, "The Move(Arg"+str(argNum)+") operation failed for the following reason:", height=200, width=600, failure_messages=failure_messages)
            return
        self.getResultsBracketings()
        self.getResultsTrees()
        self.displayResultTrees()
                
    def copyCat(self):
        try:
            newCat = copy.deepcopy(self.overtCatsBox.get(self.overtCatsBox.curselection()[0]))
        except IndexError:
            self.nothing(self.derivationWindow, "You must select a category to duplicate!")
            return
        #get the index of the original cat (not a dupe) which is identical to the new cat and the first in the list
        #hence will be the index stopped at first
        index = -1
        for exp in self.expressions:
            index += 1
            if cky_mg.expressions_identical(exp, self.expressions[self.overtCatsBox.curselection()[0]]):
                break
        self.overtCats.append(newCat[1:])
        self.TREES.append(self.TREES[index])
        self.expressions.append(self.expressions[index])
        while self.overtCatsBox.get(0):
            self.overtCatsBox.delete(0)
        for item in self.overtCats:
            self.overtCatsBox.insert(END, " "+item)
        self.statePath.append([copy.deepcopy(self.overtCats), [e for e in self.expressions], copy.deepcopy(self.axiomIndices), [t for t in self.TREES]])
        self.undoButton.config(state=NORMAL)

    def confirmAddDerivationToSeeds(self):
        if self.confirmAddDerivationToSeedsWindow != None:
            self.destroyWindow(self.confirmAddDerivationToSeedsWindow, 'confirmAddDerivationToSeedsWindow')
        self.confirmAddDerivationToSeedsWindow = Toplevel(self.derivationWindow)
        self.confirmAddDerivationToSeedsWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.confirmAddDerivationToSeedsWindow, 'confirmAddDerivationToSeedsWindow'))
        w=425
        h = 70
        (x, y) = self.getCentrePosition(w, h)
        self.confirmAddDerivationToSeedsWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.confirmAddDerivationToSeedsWindow, text="Before continuing, please ensure the parser settings are set\ncorrectly for this parse, otherwise later reparsing may fail.")
        label.pack()
        buttonFrame = Frame(self.confirmAddDerivationToSeedsWindow)
        buttonFrame.pack()
        confirmButton = Button(buttonFrame, text="Save", command=lambda: self.addDerivationToSeeds())
        confirmButton.pack(side=LEFT)
        cancelButton = Button(buttonFrame, text="Cancel", command=lambda: self.destroyWindow(self.confirmAddDerivationToSeedsWindow, 'confirmAddDerivationToSeedsWindow'))
        cancelButton.pack(side=LEFT)

    def addDerivationToSeeds(self):
        try:
            if self.overtCatsBox.curselection()[0] != 0:
                raise Exception("oops, goal index is not 0")
        except Exception as e:
            self.nothing(self.derivationWindow, "Please select the category first!")
            return
        goalList = []
        if self.newSentMode:
            self.words = self.test_words
        cky_mg.check_goal(self.expressions[0], None, len(self.words), goalList=goalList, from_autobankGUI=True)
        if goalList == []:
            self.nothing(self.derivationWindow, "That is not a suitable goal item!")
            return
        self.resultsExpressionList = goalList
        self.getResultsBracketings()
        goalBracketings = self.resultsBracketings[0]
        sfdb = goalBracketings[0]
        sdb = goalBracketings[1]
        fdb = goalBracketings[2]
        db = goalBracketings[3]
        xbar = goalBracketings[4]
        derived = goalBracketings[5]
        if not self.newSentMode:
            self.seeds[self.ptb_file_line_number] = (sdb, xbar, derived, sfdb, db, fdb, self.parserSettings['parserSetting'])
            try:
                auto_set = json.load(open(self.auto_folder+"/"+self.section_folder+"/"+self.ptb_file))
                updateAutoCTM = False
                if str(self.ptb_file_line_number) in auto_set:
                    del(auto_set[str(self.ptb_file_line_number)])
                    updateAutoCTM = True
                with open(self.auto_folder+"/"+self.section_folder+"/"+self.ptb_file, 'w') as auto_set_file:
                    json.dump(auto_set, auto_set_file)
                if updateAutoCTM:
                    self.updateAutoMappings(self.auto_folder, self.section_folder, self.ptb_file, self.ptb_file_line_number)
            except IOError:
                x=0
            if self.counts[self.seedSentLen] == 1:
                self.GUI_MATCH = False
        else:
            seeds = {"0":(sdb, xbar, derived, sfdb, db, fdb, self.parserSettings['parserSetting'])}
            self.saveTree1(seeds=seeds)
            return
        if self.auto_folder not in os.listdir(os.getcwd()):
            os.mkdir(self.auto_folder)
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.MGcatsCopy = self.MGcatsCopy2
        self.mainWindow.destroy()
        
    def doubleClick(self, box):
        if self.arg1 == None:
            self.exportArg(box, 'arg1')
            self.nextArg = 'arg2'
        elif self.arg2 == None:
            self.exportArg(box, 'arg2')
            self.nextArg = 'arg1'
        elif self.nextArg == 'arg1':
            self.exportArg(box, 'arg1')
            self.nextArg = 'arg2'
        elif self.nextArg == 'arg2':
            self.exportArg(box, 'arg2')
            self.nextArg = 'arg1'

    def undo(self):
        self.destroyArg1Spin()
        self.destroyArg2Spin()
        self.destroyLowerDerivationSpins()
        self.arg1 = None
        self.arg2 = None
        self.resultText = ""
        self.lowerTreeLabelText.set(self.resultText)
        self.overtCats = copy.deepcopy(self.statePath[-2][0])
        self.expressions = [exp for exp in self.statePath[-2][1]]
        self.axiomIndices = copy.deepcopy(self.statePath[-2][2])
        self.TREES = copy.deepcopy(self.statePath[-2][3])
        del(self.statePath[-1])
        if len(self.statePath) == 1:
            self.undoButton.config(state=DISABLED)
        while self.overtCatsBox.get(0):
            self.overtCatsBox.delete(0)
        for item in self.overtCats:
            self.overtCatsBox.insert(END, " "+item)
        self.freshArg1TreeWindow()
        self.freshArg2TreeWindow()
        self.freshLowerTreeWindow()
        if len(self.overtCats) > 1:
            self.addToSeedsButton.config(state=DISABLED)

    def acceptResult(self):
        if self.arg1CatsBoxIndex != None:
            del(self.overtCats[self.arg1CatsBoxIndex])
            del(self.TREES[self.arg1CatsBoxIndex])
            del(self.expressions[self.arg1CatsBoxIndex])
            try:
                del(self.axiomIndices[self.arg1CatsBoxIndex])
            except IndexError:
                x=0
            if self.arg2CatsBoxIndex != None and self.arg1CatsBoxIndex < self.arg2CatsBoxIndex:
                self.arg2CatsBoxIndex -= 1
            self.arg1CatsBoxIndex = None
            self.arg1CatsBox = None
            self.arg1 = None
            self.freshArg1TreeWindow()
            self.freshArg2TreeWindow()
            self.destroyArg1Spin()
        if self.arg2CatsBoxIndex != None:
            del(self.overtCats[self.arg2CatsBoxIndex])
            del(self.TREES[self.arg2CatsBoxIndex])
            del(self.expressions[self.arg2CatsBoxIndex])
            try:
                del(self.axiomIndices[self.arg2CatsBoxIndex])
            except IndexError:
                x=0
            self.arg2CatsBoxIndex = None
            self.arg2CatsBox = None
            self.arg2 = None
            self.freshArg2TreeWindow()
            self.freshArg1TreeWindow()
            self.destroyArg2Spin()
        try:
            #there are mutliple results
            self.resultsBracketings = self.resultsBracketings[int(self.derivationSpin.get())-1]
            self.TREES.append(self.resultsTrees[int(self.derivationSpin.get())-1])
            self.expressions.append(self.resultsExpressionList[int(self.derivationSpin.get())-1])
        except AttributeError:
            #there is only one result
            self.resultsBracketings = self.resultsBracketings[0]
            self.TREES.append(self.resultsTrees[0])
            self.expressions.append(self.resultsExpressionList[0])
        resultCat = self.resultsBracketings[0].split(")")[0].split("(")[1]
        self.overtCats.append(resultCat)
        while self.overtCatsBox.get(0):
            self.overtCatsBox.delete(0)
        for item in self.overtCats:
            self.overtCatsBox.insert(END, " "+item)
        if len(self.overtCats) == 1:
            self.addToSeedsButton.config(state=NORMAL)
        self.oldCurselection = -1
        self.oldSource = None
        self.freshLowerTreeWindow()
        self.acceptButton.config(state=DISABLED)
        self.destroyLowerDerivationSpins()
        self.resultText = ""
        self.lowerTreeLabelText.set(self.resultText)
        self.resultsTrees = None
        self.RESULTSbracketings = self.resultsBracketings
        self.resultsBracketings = None
        self.resultsExpessionList = None
        self.arg1 = None
        self.arg2 = None
        self.nextArg = 'arg1'
        self.statePath.append([copy.deepcopy(self.overtCats), [e for e in self.expressions], copy.deepcopy(self.axiomIndices), [t for t in self.TREES]])
        self.undoButton.config(state=NORMAL)

    def merge(self):
        if self.arg1 == None or self.arg2 == None:
            self.nothing(self.derivationWindow, "Both Arg1 and Arg2 must be filled!")
            return
        failure_messages = []
        #the parser only performs relativized smc checks prior to entering something into the chart so that
        #move can be fed straight after merge without triggering rsmc (strict smc applies immediately)
        #so we have to do the rsmc check from here because we are not putting things in the chart
        (arg1_smc_violation, arg1_smc_relativized) = cky_mg.smc_violation(self.arg1, True, failure_messages)
        (arg2_smc_violation, arg2_smc_relativized) = cky_mg.smc_violation(self.arg2, True, failure_messages)
        abort = False
        if arg1_smc_violation:
            if arg1_smc_relativized:
                smc_type = 'Type-based'
            else:
                smc_type = 'Classical'
            failure_messages.append("("+smc_type+") Shortest Move Constraint violation (Arg1).")
            abort = True
        if arg2_smc_violation:
            if arg2_smc_relativized:
                smc_type = 'Type-based'
            else:
                smc_type = 'Classical'
            failure_messages.append("("+smc_type+") Shortest Move Constraint violation (Arg2).")
            abort = True
        if abort:
            if len(failure_messages) == 1:
                self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed for the following reason:", height=200, width=600, failure_messages=failure_messages)
            else:
                self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed for the following reasons:", height=200, width=600, failure_messages=failure_messages)
            return
        arg1_copy = cky_mg.copy_expression(self.arg1)
        arg1_copy.pointers = self.arg1.pointers
        arg2_copy = cky_mg.copy_expression(self.arg2)
        arg2_copy.pointers = self.arg2.pointers
        if '=' in self.arg1.head_chain.features[0] or '≈' in self.arg1.head_chain.features[0]:
            selector = self.arg1
        elif '=' in self.arg2.head_chain.features[0] or '≈' in self.arg2.head_chain.features[0]:
            selector = self.arg2
        else:
            failure_messages.append("No selector detected for attempted merge operation.")
            self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed for the following reason:", height=200, width=600, failure_messages=failure_messages)
            return
        try:
            if cat_feature_pattern.search(re.sub('{.*}', '', self.arg1.head_chain.features[0])).group(0).upper() != cat_feature_pattern.search(re.sub('{.*}', '', self.arg2.head_chain.features[0])).group(0).upper():
                failure_messages.append("Selector/Selectee feature mismatch.")
                self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed for the following reason:", height=200, width=600, failure_messages=failure_messages)
                return
        except AttributeError:
            failure_messages.append("Selector/Selectee feature mismatch.")
            self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed for the following reason:", height=200, width=600, failure_messages=failure_messages)
            return
        self.resultsBracketings = None
        self.resultsTrees = None
        self.resultsExpressionList = None
        self.freshLowerTreeWindow()
        self.lowerTreeLabelText.set("")
        self.resultsExpressionList = []
        failure_messages = []
        cky_mg.merge(self.arg1, self.arg2, None, len(self.words), self.resultsExpressionList, failure_messages, ss=self.source_spans, ms=self.moveable_spans)
        #in case the parser returned any persistant selectees, we need to pass these back in
        #(the parser usually creates these before doing merge and puts them back into the agenda
        #from where they then reeneter cky_mg.merge() marked as persistant and then undergo merge..
        for exp in self.resultsExpressionList:
            if type(exp) == type((0,)):
                cky_mg.merge(selector, exp[0], None, len(self.words), self.resultsExpressionList, failure_messages)
        exp_to_remove = []
        self.arg1 = arg1_copy
        self.arg2 = arg2_copy
        for exp in self.resultsExpressionList:
            if type(exp) == type((0,)):
                exp_to_remove.append(exp)
        while len(exp_to_remove) > 0:
            self.resultsExpressionList.remove(exp_to_remove[0])
            del(exp_to_remove[0])
        exp_to_remove = []
        #we need to make sure none of the expressions violates shortest move
        for item in self.resultsExpressionList:
            if not cky_mg.add_to_agenda(item, agenda=None, sentence_length=None, returnToAutobank=True, failure_messages=failure_messages):
                exp_to_remove.append(item)
        for item in exp_to_remove:
            self.resultsExpressionList.remove(item)
        failure_messages = list(set(failure_messages))
        if len(self.resultsExpressionList) == 0:
            if len(failure_messages) == 0:
                self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed!")
            elif len(failure_messages) > 1:
                self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed for the following reasons:", height=200, width=600, failure_messages=failure_messages)
            elif len(failure_messages) == 1:
                self.nothing(self.derivationWindow, "The Merge(Arg1,Arg2) operation failed for the following reason:", height=200, width=600, failure_messages=failure_messages)
            return
        self.getResultsBracketings()
        self.resultText = "Result of Merge(Arg1,Arg2)"
        self.oldCurselection = -1
        self.oldSource = None
        self.getResultsTrees()
        self.displayResultTrees()

    def getResultsTrees(self):
        self.resultsTrees = []
        for bracketingSet in self.resultsBracketings:
            self.resultsTrees.append([])
            for bracketing in bracketingSet:
                try:
                    tree = Tree.parse(bracketing, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                except AttributeError:
                    tree = Tree.fromstring(bracketing, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
                self.resultsTrees[-1].append(tree)

    def getResultsBracketings(self):
        self.resultsBracketings = []
        derivation_bracketings = []
        for EXP in self.resultsExpressionList:
            exp = cky_mg.copy_expression(EXP)
            exp.pointers = EXP.pointers
            #we use a copy of the expression because the generate_derivation_bracketing function
            #actually alters the expression, e.g. adding back in subcat features to the main features..
            (subcat_derivation_bracketing, subcat_full_derivation_bracketing) = cky_mg.generate_derivation_bracketing(exp, from_derivation_builder=True)
            subcat_derivation_bracketing = re.sub('@COMMA@', ';', subcat_derivation_bracketing, count = 10000)
            subcat_full_derivation_bracketing = re.sub('@COMMA@', ';', subcat_full_derivation_bracketing, count = 10000)
            derivation_bracketing = re.sub('{.*?}', '', subcat_derivation_bracketing, count = 100000)
            full_derivation_bracketing = re.sub('{.*?}', '', subcat_full_derivation_bracketing, count = 100000)
            sdb = " ".join("".join(cky_mg.fix_coord_annotation(subcat_derivation_bracketing).split(" ")).split("##"))
            db = " ".join("".join(cky_mg.fix_coord_annotation(derivation_bracketing).split(" ")).split("##"))
            sfdb = " ".join("".join(cky_mg.fix_coord_annotation(subcat_full_derivation_bracketing).split(" ")).split("##"))
            fdb = " ".join("".join(cky_mg.fix_coord_annotation(full_derivation_bracketing).split(" ")).split("##"))
            derivation_bracketings.append([sfdb, sdb, fdb, db])
            (derived_bracketing, xbar_bracketing) = gen_derived_tree.main(derivation_bracketings[-1][1], allowMoreGoals=True, allowOnlyGoals=False)
            self.resultsBracketings.append([])
            for bracketing in derivation_bracketings[-1]:
                self.resultsBracketings[-1].append(bracketing)
            self.resultsBracketings[-1] += [xbar_bracketing, derived_bracketing]
        
    def displayWorkspaceTree(self, source):
        self.source = source
        if source == self.covertCatsBox:
            self.showCovertComments(self.covertCatsBox, self.covComments)
            self.ovComments.set('')
        elif source == self.overtCatsBox:
            self.covComments.set('')
            ind = -1
            for char in self.overtCatsBox.get(self.overtCatsBox.curselection()).strip():
                ind+=1
                if char == ':':
                    break
            cat = self.overtCatsBox.get(self.overtCatsBox.curselection()).strip()[ind:]
            try:
                self.ovComments.set(self.overtCatComments[cat])
            except KeyError:
                self.ovComments.set('')
        self.resultsBracketings = None
        self.resultsTrees = None
        self.resultsExpressionList = None
        if source == self.oldSource:
            #single click is pre-empting double click and for some reason parts of this
            #function prevent double click from ever being executed..
            #to get around this, if the selection on the single click is the
            #same as the previous selection, we will just ignore the single click..
            try:
                if self.oldCurselection == source.curselection()[0]:
                    return
            except IndexError:
                #null lexicons are all empty
                return
        self.oldSource = source
        try:
            self.oldCurselection = source.curselection()[0]
        except IndexError:
            #null lexicons are all empty
            return
        self.acceptButton.config(state=DISABLED)
        if source == self.overtCatsBox:
            TREES = self.TREES
            index = source.curselection()[0]
        elif source == self.covertCatsBox:
            covertCat = source.get(source.curselection()).split("     ")[0][1:]
            try:
                derivation_tree = Tree.parse('('+covertCat+')', remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            except AttributeError:
                derivation_tree = Tree.fromstring('('+covertCat+')', remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            TREES = [[derivation_tree]]
            index = 0
        self.lowerTreeLabelText.set("Viewing Workspace Tree")
        self.freshLowerTreeWindow()
        self.freshLowerDerivationSpins()
        self.LowerTreeWidget = TreeWidget(self.lowerWindow.canvas(), TREES[index][0], draggable=1, shapeable=1)
        if len(TREES[index]) > 1:
            self.lowerTreeTypeDerivationSpin = Spinbox(self.lowerTreeSpinFrame, values=('MG Derivation Cats+Subcat', 'MG Derivation Ops+Subcat', 'MG Derivation Cats', 'MG Derivation Ops', 'Xbar', 'MG Derived'), command = self.changeWorkspaceTree, width=20)
            self.lowerTreeTypeDerivationSpin.pack(side='left')
            self.lowerFullMGTreeButton = Button(self.lowerTreeSpinFrame, text='FS', command=lambda: self.newFullTreeWindow(treeType='MG', trees=TREES, spin=None, mode='derivation', fromResults=False))
            self.lowerFullMGTreeButton.pack(side='left')
        self.lowerWindow.add_widget(self.LowerTreeWidget, 10, 10)

    def displayResultTrees(self):
        self.acceptButton.config(state=NORMAL)
        self.lowerTreeLabelText.set(self.resultText)
        self.freshLowerTreeWindow()
        self.freshLowerDerivationSpins()
        self.LowerTreeWidget = TreeWidget(self.lowerWindow.canvas(), self.resultsTrees[0][0], draggable=1, shapeable=1)
        self.lowerWindow.add_widget(self.LowerTreeWidget, 10, 10)

    def changeResultTree(self):
        self.freshLowerTreeWindow()
        try:
            #there are mutliple results
            self.LowerTreeWidget = TreeWidget(self.lowerWindow.canvas(), self.resultsTrees[int(self.derivationSpin.get())-1][self.getDerivationTypeIndex(self.lowerTreeTypeDerivationSpin)], draggable=1, shapeable=1)
        except AttributeError:
            #there is only one result
            self.LowerTreeWidget = TreeWidget(self.lowerWindow.canvas(), self.resultsTrees[0][self.getDerivationTypeIndex(self.lowerTreeTypeDerivationSpin)], draggable=1, shapeable=1)
        self.lowerWindow.add_widget(self.LowerTreeWidget, 10, 10)

    def changeWorkspaceTree(self):
        self.freshLowerTreeWindow()
        try:
            self.LowerTreeWidget = TreeWidget(self.lowerWindow.canvas(), self.TREES[self.overtCatsBox.curselection()[0]][self.getDerivationTypeIndex(self.lowerTreeTypeDerivationSpin)], draggable=1, shapeable=1)
        except AttributeError:
            self.LowerTreeWidget = TreeWidget(self.lowerWindow.canvas(), self.TREES[self.overtCatsBox.curselection()[0]][0], draggable=1, shapeable=1)
        self.lowerWindow.add_widget(self.LowerTreeWidget, 10, 10)

    def getDerivationTypeIndex(self, spin):
        if spin.get() == 'MG Derivation Cats+Subcat':
            return 0
        elif spin.get() == 'MG Derivation Ops+Subcat':
            return 1
        elif spin.get() == 'MG Derivation Cats':
            return 2
        elif spin.get() == 'MG Derivation Ops':
            return 3
        elif spin.get() == 'Xbar':
            return 4
        elif spin.get() == 'MG Derived':
            return 5

    def destroyLowerDerivationSpins(self):
        if self.derivationSpin != None:
            self.derivationSpin.destroy()
            self.derivationSpin = None
            self.derivationSpinLabel.destroy()
            self.ofXlabelDerivation.destroy()
        if self.lowerTreeTypeDerivationSpin != None:
            self.lowerTreeTypeDerivationSpin.destroy()
            self.lowerTreeTypeDerivationSpin = None
            self.lowerFullMGTreeButton.destroy()
            self.lowerFullMGTreeButton = None

    def freshLowerTreeWindow(self):
        if self.innerInnerInnerLowerTreeFrame != None:
            self.innerInnerInnerLowerTreeFrame.destroy()
        self.innerInnerInnerLowerTreeFrame = Frame(self.innerInnerLowerTreeFrame, relief=SUNKEN)
        self.innerInnerInnerLowerTreeFrame.pack(fill=BOTH, expand=True)
        self.lowerWindow = CanvasFrame(self.innerInnerInnerLowerTreeFrame, highlightthickness=2, highlightbackground='black', bg='white', height=10)
        self.lowerWindow.pack(fill=BOTH, expand=True)
        if self.lowerTreeLabel != None:
            self.lowerTreeLabel.destroy()
        self.lowerTreeLabel = Label(self.innerInnerInnerLowerTreeFrame, textvariable=self.lowerTreeLabelText)
        self.lowerTreeLabel.pack()

    def freshLowerDerivationSpins(self):
        self.destroyDerivationSpinsFrame()
        self.lowerTreeSpinFrame = Frame(self.lowerOuterTreeSpinFrame)
        self.lowerTreeSpinFrame.pack()
        emptyLabel = Label(self.lowerTreeSpinFrame, text="")
        emptyLabel.pack(side=LEFT)
        if self.resultsTrees != None:
            if self.derivationSpin != None:
                self.destroyLowerDerivationSpins()
            if len(self.resultsTrees) > 1:
                var = StringVar(self.derivationWindow)
                var.set('1')
                self.derivationSpinLabel = Label(self.lowerTreeSpinFrame, text="Possible Result:")
                self.derivationSpinLabel.pack(side=LEFT)
                self.derivationSpin = Spinbox(self.lowerTreeSpinFrame, from_=1, to=len(self.resultsTrees), textvariable=var,
                command = lambda: self.changeResultTree(), width=3)
                self.derivationSpin.pack(side='left')
                self.ofXlabelDerivation = Label(self.lowerTreeSpinFrame, text="of %d" % len(self.resultsTrees))
                self.ofXlabelDerivation.pack(side='left')
            else:
                self.derivationSpin = None
            if len(self.resultsTrees[0]) > 1:
                self.lowerTreeTypeDerivationSpin = Spinbox(self.lowerTreeSpinFrame, values=('MG Derivation Cats+Subcat', 'MG Derivation Ops+Subcat', 'MG Derivation Cats', 'MG Derivation Ops', 'Xbar', 'MG Derived'), command = lambda: self.changeResultTree(), width=20)
                self.lowerTreeTypeDerivationSpin.pack(side='left')
            self.lowerFullMGTreeButton = Button(self.lowerTreeSpinFrame, text='FS', command=lambda: self.newFullTreeWindow(treeType='MG', trees=self.resultsTrees, spin=self.derivationSpin, mode='derivation'))
            self.lowerFullMGTreeButton.pack(side='left')

    def destroyDerivationSpinsFrame(self):
        if self.lowerTreeSpinFrame != None:
            self.lowerTreeSpinFrame.destroy()
            self.lowerTreeSpinFrame = None
            self.derivationSpin = None
            self.lowerTreeTypeDerivationSpin = None
            
    def displayArg1Tree(self, source):
        self.destroyArg1Spin()
        if source == self.overtCatsBox:
            TREES = self.TREES
            index = source.curselection()[0]
        elif source in [self.covertCatsBox, self.catsBox]:
            covertCat = source.get(source.curselection()).split("     ")[0][1:]
            try:
                derivation_tree = Tree.parse('('+covertCat+')', remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            except AttributeError:
                derivation_tree = Tree.fromstring('('+covertCat+')', remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            TREES = [[derivation_tree]]
            index = 0
        self.freshArg1TreeWindow()
        self.arg1TreeWidget = TreeWidget(self.arg1Window.canvas(), TREES[index][0], draggable=1, shapeable=1)
        if len(TREES[index]) > 1:
            self.arg1TreeTypeSpin = Spinbox(self.arg1SpinFrame, values=('MG Derivation Cats+Subcat', 'MG Derivation Ops+Subcat', 'MG Derivation Cats', 'MG Derivation Ops', 'Xbar', 'MG Derived'), command = self.changeArg1TreeType, width=20)
            self.arg1TreeTypeSpin.pack(side='left')
        self.arg1Window.add_widget(self.arg1TreeWidget, 10, 10)
        if source == self.catsBox:
            self.destroyWindow(self.viewDelCatsWindow, 'viewDelCatsWindow')

    def changeArg1TreeType(self):
        self.freshArg1TreeWindow()
        if self.arg1CatsBox == self.overtCatsBox:
            self.arg1TreeWidget = TreeWidget(self.arg1Window.canvas(), self.TREES[self.arg1CatsBoxIndex][self.getDerivationTypeIndex(self.arg1TreeTypeSpin)], draggable=1, shapeable=1)
        self.arg1Window.add_widget(self.arg1TreeWidget, 10, 10)

    def changeArg2TreeType(self):
        self.freshArg2TreeWindow()
        if self.arg2CatsBox == self.overtCatsBox:
            self.arg2TreeWidget = TreeWidget(self.arg2Window.canvas(), self.TREES[self.arg2CatsBoxIndex][self.getDerivationTypeIndex(self.arg2TreeTypeSpin)], draggable=1, shapeable=1)
        self.arg2Window.add_widget(self.arg2TreeWidget, 10, 10)

    def freshArg1TreeWindow(self):
        if self.innerInnerArg1TreeFrame != None:
            self.innerInnerArg1TreeFrame.destroy()
        self.innerInnerArg1TreeFrame = Frame(self.innerArg1TreeFrame, relief=SUNKEN)
        self.innerInnerArg1TreeFrame.pack(fill=BOTH, expand=True)
        self.arg1Window = CanvasFrame(self.innerInnerArg1TreeFrame, highlightthickness=2, highlightbackground='black', bg='white', height=10)
        self.arg1Window.pack(fill=BOTH, expand=True)

    def displayArg2Tree(self, source):
        self.destroyArg2Spin()
        if source == self.overtCatsBox:
            TREES = self.TREES
            index = source.curselection()[0]
        elif source in [self.covertCatsBox, self.catsBox]:
            covertCat = source.get(source.curselection()).split("     ")[0][1:]
            try:
                derivation_tree = Tree.parse('('+covertCat, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            except AttributeError:
                derivation_tree = Tree.fromstring('('+covertCat+')', remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)
            TREES = [[derivation_tree]]
            index = 0
        self.freshArg2TreeWindow()
        self.arg2TreeWidget = TreeWidget(self.arg2Window.canvas(), TREES[index][0], draggable=1, shapeable=1)
        if len(TREES[index]) > 1:
            self.arg2TreeTypeSpin = Spinbox(self.arg2SpinFrame, values=('MG Derivation Cats+Subcat', 'MG Derivation Ops+Subcat', 'MG Derivation Cats', 'MG Derivation Ops', 'Xbar', 'MG Derived'), command = self.changeArg2TreeType, width=20)
            self.arg2TreeTypeSpin.pack(side='left')
        self.arg2Window.add_widget(self.arg2TreeWidget, 10, 10)
        if source == self.catsBox:
            self.destroyWindow(self.viewDelCatsWindow, 'viewDelCatsWindow')

    def destroyArg2Spin(self):
        if self.arg2TreeTypeSpin != None:
            self.arg2TreeTypeSpin.destroy()

    def destroyArg1Spin(self):
        if self.arg1TreeTypeSpin != None:
            self.arg1TreeTypeSpin.destroy()

    def freshArg2TreeWindow(self):
        if self.innerInnerArg2TreeFrame != None:
            self.innerInnerArg2TreeFrame.destroy()
        self.innerInnerArg2TreeFrame = Frame(self.innerArg2TreeFrame, relief=SUNKEN)
        self.innerInnerArg2TreeFrame.pack(fill=BOTH, expand=True)
        self.arg2Window = CanvasFrame(self.innerInnerArg2TreeFrame, highlightthickness=2, highlightbackground='black', bg='white', height=10)
        self.arg2Window.pack(fill=BOTH, expand=True)

    def exportArg(self, box, arg):
        try:
            box.get(box.curselection()[0]).split(" ")
        except Exception as e:
            self.nothing(self.derivationWindow, "You must select a category to export!")
            return
        if box in [self.covertCatsBox, self.catsBox] or self.expressions[box.curselection()[0]] == None:
            #if we are taking an axiom from either the overt or covert cats box, then we need
            #to construct an MG lexical entry recognizable by the parser and convert this into
            #the expression object used by the parser..
            #there's an extra space behind the words when I do split, hence we take 1 as the word and 2: as the features below..
            if box == self.overtCatsBox:
                (word, features) = (box.get(box.curselection()[0]).split(" ")[1], box.get(box.curselection()[0]).split(" ")[2:])
            else:
                features = []
                FEATURES = box.get(box.curselection()[0]).split("     ")[0]
                features = FEATURES.split(" ")[2:]
                word = FEATURES.split(" ")[1]
            entry = self.constructMGlexEntry(word, features)
            if box == self.overtCatsBox:
                start = self.axiomIndices[box.curselection()[0]]
                end = start+1
                span = [start, end]
            else:
                #this is a covert category, so it has a null span
                span = [[],[]]
            if 'conj' in entry[2]:
                separator = ":\u0305:\u0305"
            else:
                separator = "::"
            expression = cky_mg.Expression(cat_feature = entry[2], head_string = entry[0], head_features = entry[1], head_span = span, separator=separator)
        else:
            expression = self.expressions[box.curselection()[0]]
        if arg == 'arg1':
            self.arg1 = cky_mg.copy_expression(expression)
            self.arg1.pointers = expression.pointers
            self.displayArg1Tree(box)
            if box == self.overtCatsBox:
                self.arg1CatsBox = self.overtCatsBox
                self.arg1CatsBoxIndex = box.curselection()[0]
            elif box == self.covertCatsBox:
                self.arg1CatsBox = self.covertCatsBox
                self.arg1CatsBoxIndex = None
            elif box == self.catsBox:
                self.arg1CatsBox = self.catsBox
                self.arg1CatsBoxIndex = None
        elif arg == 'arg2':
            self.arg2 = cky_mg.copy_expression(expression)
            self.arg2.pointers = expression.pointers
            self.displayArg2Tree(box)
            if box == self.overtCatsBox:
                self.arg2CatsBox = self.overtCatsBox
                self.arg2CatsBoxIndex = box.curselection()[0]
            elif box in [self.covertCatsBox, self.catsBox]:
                self.arg2CatsBox = self.covertCatsBox
                self.arg2CatsBoxIndex = None
            elif box == self.catsBox:
                self.arg2CatsBox = self.catsBox
                self.arg2CatsBoxIndex = None

    def treeCompare(self):
        self.upperTrees = self.xbar_trees
        self.lowerTrees = self.xbar_trees
        if self.treeCompareWindow != None:
            self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')
        if self.derivationWindow != None:
            self.destroyWindow(self.derivationWindow, 'derivationWindow')
        self.treeCompareWindow = Toplevel(self.mainWindow)
        self.treeCompareWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow'))
        self.treeCompareWindow.title("Tree Comparison Viewer")
        w=10000
        h=10000
        (x, y) = self.getCentrePosition(w, h)
        self.treeCompareWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.ptbLabel = Label(self.treeCompareWindow, text="Source Tree File:  "+self.ptb_folder+"/"+self.section_folder+"/"+self.ptb_file+"    Ln: "+str(self.ptb_file_line_number+1))
        self.ptbLabel.pack()
        self.globaltframeSBS = Frame(self.treeCompareWindow)
        self.globaltframeSBS.pack(side='left', fill=BOTH, expand=True)
        self.uppertframeSBS = Frame(self.globaltframeSBS)
        self.uppertframeSBS.pack(fill=BOTH, expand=True)
        self.upperSpinFrameSBS = Frame(self.globaltframeSBS)
        self.upperSpinFrameSBS.pack()
        upperVar = StringVar(self.treeCompareWindow)
        try:
            upperVar.set(self.spin.get())
        except AttributeError:
            upperVar.set('1')
        self.upperSpin = Spinbox(self.upperSpinFrameSBS, from_=1, to=len(self.xbar_trees), textvariable=upperVar,
            command = lambda: self.showUpperTrees(), width=3)
        if len(self.xbar_trees) > 1: self.upperSpin.pack(side='left')
        self.upperOfXlabel = Label(self.upperSpinFrameSBS, text="of %d" % len(self.xbar_trees))
        if len(self.xbar_trees) > 1: self.upperOfXlabel.pack(side='left')
        self.upperTreeTypeSpin = Spinbox(self.upperSpinFrameSBS, values=('Xbar', 'MG Derived', 'MG Derivation Ops+Subcat', 'MG Derivation Cats+Subcat', 'MG Derivation Ops', 'MG Derivation Cats'), command = self.changeUpperTreeType, width=20)
        self.upperTreeTypeSpin.pack(side='left')
        self.upperViewBracketButton = Button(self.upperSpinFrameSBS, text='[...]', command=lambda: self.viewBracketing('upper'))
        self.upperViewBracketButton.pack(side='right')
        self.mainButtonFrameSBS = Frame(self.globaltframeSBS)
        self.mainButtonFrameSBS.pack()
        diffButton = Button(self.mainButtonFrameSBS, text='diff', command=self.diff)
        diffButton.pack(side='left')
        if not simpleDiffEnabled:
            diffButton.config(state=DISABLED)
        closeButton = Button(self.mainButtonFrameSBS, text='close', command=lambda: self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow'))
        closeButton.pack(side='left')
        self.lowertframeSBS = Frame(self.globaltframeSBS)
        self.lowertframeSBS.pack(fill=BOTH, expand=True)
        self.outer_upper_tframeSBS = Frame(self.uppertframeSBS, relief=SUNKEN)
        self.outer_upper_tframeSBS.pack(side='left', fill=BOTH, expand=True)
        self.upper_tframeSBS = Frame(self.outer_upper_tframeSBS, relief=SUNKEN)
        self.upper_tframeSBS.pack(side='left', fill=BOTH, expand=True)
        self.upperWindowSBS = CanvasFrame(self.upper_tframeSBS, highlightthickness=2, highlightbackground='black', bg='white',height=10)#1
        self.upperWindowSBS.pack(fill=BOTH, expand=True)
        self.outer_lower_tframeSBS = Frame(self.lowertframeSBS, relief=SUNKEN)
        self.outer_lower_tframeSBS.pack(side='left', fill=BOTH, expand=True)
        self.lower_tframeSBS = Frame(self.outer_lower_tframeSBS, relief=SUNKEN)
        self.lower_tframeSBS.pack(side='left', fill=BOTH, expand=True)
        self.lowerWindowSBS = CanvasFrame(self.lower_tframeSBS, highlightthickness=2, highlightbackground='black', bg='white',height=10)#1
        self.lowerWindowSBS.pack(fill=BOTH, expand=True)
        var = StringVar(self.treeCompareWindow)
        self.lowerSpinFrameSBS = Frame(self.globaltframeSBS)
        self.lowerSpinFrameSBS.pack()
        lowerVar = StringVar(self.treeCompareWindow)
        try:
            lowerVar.set(self.spin.get())
        except AttributeError:
            lowerVar.set('1')
        self.lowerSpin = Spinbox(self.lowerSpinFrameSBS, from_=1, to=len(self.xbar_trees), textvariable=lowerVar,
            command = lambda: self.showLowerTrees(), width=3)
        if len(self.xbar_trees) > 1: self.lowerSpin.pack(side='left')
        self.lowerOfXlabel = Label(self.lowerSpinFrameSBS, text="of %d" % len(self.xbar_trees))
        if len(self.xbar_trees) > 1: self.lowerOfXlabel.pack(side='left')
        self.lowerTreeTypeSpin = Spinbox(self.lowerSpinFrameSBS, values=('Xbar', 'MG Derived', 'MG Derivation Ops+Subcat', 'MG Derivation Cats+Subcat', 'MG Derivation Ops', 'MG Derivation Cats'), command = self.changeLowerTreeType, width=20)
        self.lowerTreeTypeSpin.pack(side='left')
        self.lowerViewBracketButton = Button(self.lowerSpinFrameSBS, text='[...]', command=lambda: self.viewBracketing('lower'))
        self.lowerViewBracketButton.pack(side='right')
        self.showUpperTrees()
        self.showLowerTrees()

    def showUpperTrees(self):
        try:
            index = int(self.upperSpin.get())-1
        except ValueError:
            index=0
        self.freshUpperWindow()
        self.upperMGtreeWidget = TreeWidget(self.upperWindowSBS.canvas(), self.upperTrees[index], draggable=1, shapeable=1)
        self.upperWindowSBS.add_widget(self.upperMGtreeWidget, 10, 10)

    def showLowerTrees(self):
        try:
            index = int(self.lowerSpin.get())-1
        except ValueError:
            n=0
        self.freshLowerWindow()
        self.lowerMGtreeWidget = TreeWidget(self.lowerWindowSBS.canvas(), self.lowerTrees[index], draggable=1, shapeable=1)
        self.lowerWindowSBS.add_widget(self.lowerMGtreeWidget, 10, 10)

    def freshUpperWindow(self):
        if self.upper_tframeSBS != None:
            self.upper_tframeSBS.destroy()
        self.upper_tframeSBS = Frame(self.outer_upper_tframeSBS, relief=SUNKEN)
        self.upper_tframeSBS.pack(side='left', fill=BOTH, expand=True)
        self.upperWindowSBS = CanvasFrame(self.upper_tframeSBS, highlightthickness=2, highlightbackground='black', bg='white', height=10)
        self.upperWindowSBS.pack(fill=BOTH, expand=True)

    def freshLowerWindow(self):
        if self.lower_tframeSBS != None:
            self.lower_tframeSBS.destroy()
        self.lower_tframeSBS = Frame(self.outer_lower_tframeSBS, relief=SUNKEN)
        self.lower_tframeSBS.pack(side='left', fill=BOTH, expand=True)
        self.lowerWindowSBS = CanvasFrame(self.lower_tframeSBS, highlightthickness=2, highlightbackground='black', bg='white', height=10)
        self.lowerWindowSBS.pack(fill=BOTH, expand=True)

    def parseMessage(self):
        #this message won't display for some reason.. its not critical.. but I'm changing the message
        #as it only displays when the parser is interrupted
        if self.parsing != None:
            self.destroyWindow(self.parsing, 'parsing')
        self.parsing = Toplevel(self.mainWindow)
        self.parsing.title = ("")
        self.parsing.grab_set()
        self.parsing.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.parsing, 'parsing'))
        w=300
        h=40
        (x, y) = self.getCentrePosition(w, h)
        self.parsing.geometry('%dx%d+%d+%d' % (w, h, x, y))
        timeout = str(self.parserSettings['timeout_seconds'])
        parserSetting = self.parserSettings['parserSetting']
        label = Label(self.parsing, text="The parser has been interrupted...")
        label.pack()

    def eliminateParse(self):
        if len(self.xbar_trees) == 1:
            self.viewNextPTBtree()
        if self.eliminateEntry.get() == '':
            parse_index = int(self.spin.get())-1
            del(self.xbar_trees[parse_index])
            del(self.XBAR_trees[parse_index])
            del(self.xbar_bracketings[parse_index])
            del(self.derivation_trees[parse_index])
            del(self.derivation_bracketings[parse_index])
            del(self.subcat_derivation_trees[parse_index])
            del(self.subcat_derivation_bracketings[parse_index])
            del(self.subcat_full_derivation_trees[parse_index])
            del(self.subcat_full_derivation_bracketings[parse_index])
            del(self.full_derivation_trees[parse_index])
            del(self.full_derivation_bracketings[parse_index])
            del(self.derived_trees[parse_index])
            del(self.derived_bracketings[parse_index])
        else:
            search_item = self.eliminateEntry.get().encode('utf8')
            search_item2 = re.sub(" \(", "(", search_item, count=100000)
            xbar_to_delete = []
            derived_to_delete = []
            sfdb_to_delete = []
            sdb_to_delete = []
            db_to_delete = []
            fdb_to_delete = []
            for i in range(len(self.xbar_trees)):
                if search_item in self.xbar_bracketings[i] or search_item in self.derivation_bracketings[i] or search_item in self.subcat_derivation_bracketings[i] or search_item in self.subcat_full_derivation_bracketings[i] or search_item in self.full_derivation_bracketings[i] or search_item in self.derived_bracketings[i] or search_item2 in self.xbar_bracketings[i] or search_item2 in self.derivation_bracketings[i] or search_item2 in self.subcat_derivation_bracketings[i] or search_item2 in self.subcat_full_derivation_bracketings[i] or search_item2 in self.full_derivation_bracketings[i] or search_item2 in self.derived_bracketings[i]:
                    xbar_to_delete.append((self.xbar_bracketings[i], self.xbar_trees[i]))
                    derived_to_delete.append((self.derived_bracketings[i], self.derived_trees[i]))
                    sfdb_to_delete.append((self.subcat_full_derivation_bracketings[i], self.subcat_full_derivation_trees[i]))
                    sdb_to_delete.append((self.subcat_derivation_bracketings[i], self.subcat_derivation_trees[i]))
                    db_to_delete.append((self.derivation_bracketings[i], self.derivation_trees[i]))
                    fdb_to_delete.append((self.full_derivation_bracketings[i], self.full_derivation_trees[i]))
            if len(xbar_to_delete) == len(self.xbar_trees):
                self.nothing(self.mainWindow, "That would eliminate all candidate trees!")
                return
            for i in range(len(xbar_to_delete)):
                self.xbar_trees.remove(xbar_to_delete[i][1])
                self.xbar_bracketings.remove(xbar_to_delete[i][0])
                self.derivation_trees.remove(db_to_delete[i][1])
                self.derivation_bracketings.remove(db_to_delete[i][0])
                self.subcat_derivation_trees.remove(sdb_to_delete[i][1])
                self.subcat_derivation_bracketings.remove(sdb_to_delete[i][0])
                self.subcat_full_derivation_trees.remove(sfdb_to_delete[i][1])
                self.subcat_full_derivation_bracketings.remove(sfdb_to_delete[i][0])
                self.full_derivation_trees.remove(fdb_to_delete[i][1])
                self.full_derivation_bracketings.remove(fdb_to_delete[i][0])
                self.derived_trees.remove(derived_to_delete[i][1])
                self.derived_bracketings.remove(derived_to_delete[i][0])
        self.refreshSpins()
        self.showtrees(treeType='MG', trees=self.xbar_trees)
        if len(self.derivation_bracketings) == 1:
            self.compareButton.destroy()
            #if not self.newSentMode:
                #self.autoSelectButton.destroy()
                #self.scoreButton.destroy()
            if self.eliminateButton != None:
                self.eliminateButton.destroy()
                self.eliminateEntry.destroy()
        else:
            if self.eliminateEntry.get() != '':
                self.eliminateEntry.delete(0, 'end')

    def refreshSpins(self):
        self.trees = self.xbar_trees
        var = StringVar(self.mainWindow)
        if len(self.xbar_trees) == 1:
            if self.eliminateButton != None:
                self.eliminateButton.destroy()
                self.eliminateEntry.destroy()
                self.eliminateButton = None
                self.eliminateEntry = None
        try:
            var.set(self.spin.get())
        except AttributeError:
            var.set('1')
        if self.spin != None:
            self.destroySpins()
        if len(self.xbar_trees) > 1:
            self.emptyLabel = Label(self.mainButtonFrame, text=" ")
            self.emptyLabel.pack(side='left')
            self.spin = Spinbox(self.mainButtonFrame, from_=1, to=len(self.xbar_trees), textvariable=var,
            command = lambda: self.showtrees(treeType='MG'), width=3)
            self.spin.pack(side='left')
            self.ofXlabel = Label(self.mainButtonFrame, text="of %d" % len(self.xbar_trees))
            self.ofXlabel.pack(side='left')
        else:
            self.spin = None
        self.treeTypeSpin = Spinbox(self.mainButtonFrame, values=('Xbar', 'MG Derived', 'MG Derivation Ops+Subcat', 'MG Derivation Cats+Subcat', 'MG Derivation Ops', 'MG Derivation Cats'), command = self.changeTreeType, width=20)
        self.treeTypeSpin.pack(side='left')

    def constructMGlexEntry(self, word, features):
        #because in the derive mode we will need to send derived items off to the functions
        #of the parser, this function has been expanded to create not just axioms (ie lexical items)
        #but also larger constituents
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
            #we need to strip ofdef constructf the subcat stuff as this contains + and - not relevant to the
            #determining if this is a selectee feature and also not included on the PoS feature..
            FEATURE = re.sub('{.*}', '', feature)
            #need to get rid of the pesky unicode introduced by the coord separator,
            #as the parser doesn't like it
            try:
                selectee_feature = FEATURE.encode('ascii').lower()
            except UnicodeEncodeError:
                #doesn't work for adjuncts 
                selectee_feature = FEATURE.lower()
            not_cat = not_selectee_feature.search(selectee_feature)
            if not_cat or u'\u2248' in selectee_feature or u'\\u2248' in selectee_feature:
                selectee_feature = ''
            else:
                break
        #determine whether this is a coordinator
        if features[0] in [u':\u0305:\u0305', u':\u0305', u':\\u0305:\\u0305', u':\\u0305']:
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

    def destroySpins(self):
        if self.treeTypeSpin != None:
            self.treeTypeSpin.destroy()
            self.treeTypeSpin = None
        if self.spin != None:
            self.spin.destroy()
            self.emptyLabel.destroy()
            self.ofXlabel.destroy()
            self.spin = None

    def destroyButtons(self):
        if self.add2SeedsButton != None:
            self.add2SeedsButton.destroy()
            self.add2SeedsButton = None
        if self.eliminateButton != None:
            self.eliminateButton.destroy()
            self.eliminateButton = None
            self.eliminateEntry.destroy()
            self.eliminateEntry = None
        if self.compareButton != None:
            self.compareButton.destroy()
            self.compareButton = None
            if not self.newSentMode:
                #self.autoSelectButton.destroy()
                #self.autoSelectButton = None
                self.scoreButton.destroy()
                self.scoreButton = None
    def changeTreeType(self):
        if self.treeTypeSpin.get() == 'Xbar':
            self.trees = self.xbar_trees
        elif self.treeTypeSpin.get() == 'MG Derivation Ops+Subcat':
            self.trees = self.subcat_derivation_trees
        elif self.treeTypeSpin.get() == 'MG Derived':
            self.trees = self.derived_trees
        elif self.treeTypeSpin.get() == 'MG Derivation Ops':
            self.trees = self.derivation_trees
        elif self.treeTypeSpin.get() == 'MG Derivation Cats+Subcat':
            self.trees = self.subcat_full_derivation_trees
        elif self.treeTypeSpin.get() == 'MG Derivation Cats':
            self.trees = self.full_derivation_trees
        self.showtrees(treeType='MG')

    def destroyWindow(self, window, windowName):
        if windowName == 'addOvertCatWindow':
            self.addOvertCatWindow = None
        elif windowName == 'addCovertCatWindow':
            self.checkedForDuplicates = False
            self.covertLexUpdated = False
            self.addCovertCatWindow = None
        elif windowName == 'alreadyExistsWindow':
            self.alreadyExistsWindow = None
        elif windowName == 'nothingEntered':
            self.nothingEntered = None
        elif windowName == 'noParsesFound':
            self.noParsesFound = None
        elif windowName == 'noParse2accept':
            self.noParse2accept = None
        elif windowName == 'viewDelCatsWindow':
            self.viewDelCatsWindow = None
        elif windowName == 'confirmWindow':
            self.confirmWindow = None
        elif windowName == 'confirmReparseWindow':
            self.confirmReparseWindow = None
        elif windowName == 'statsWindow':
            self.statsWindow = None
        elif windowName == 'overwriteWindow':
            self.overwriteWindow = None
        elif windowName == 'treeCompareWindow':
            self.treeCompareWindow = None
        elif windowName == 'modifyWindow':
            self.modifyWindow = None
        elif windowName == 'diffWindow':
            self.diffWindow = None
        elif windowName == 'upperBracketingWindow':
            self.upperBracketingWindow = None
        elif windowName == 'lowerBracketingWindow':
            self.lowerBracketingWindow = None
        elif windowName == 'mainBracketingWindow':
            self.mainBracketingWindow = None
        elif windowName == 'parsing':
            self.parsing = None
        elif windowName == 'PTBbracketingWindow':
            self.PTBbracketingWindow = None
        elif windowName == 'viewParseQuestionWindow':
            self.viewParseQuestionWindow = None
        elif windowName == 'fileManagerWindow':
            self.fileManagerWindow = None
        elif windowName == 'parserSettingsWindow':
            self.parserSettingsWindow = None
        elif windowName == 'derivationWindow':
            self.derivationWindow = None
        elif windowName == 'testSentenceWindow':
            self.testSentenceWindow = None
        elif windowName == 'untaggedWordsWindow':
            self.untaggedWordsWindow = None
        elif windowName == 'autoSelectEliminateQuestionWindow':
            self.autoSelectEliminateQuestionWindow = None
        elif windowName == 'constructLexiconWindow':
            self.constructLexiconWindow = None
        elif windowName == 'confirmDeleteTreeWindow':
            self.confirmDeleteTreeWindow = None
        elif windowName == 'saveTreeWindow':
            self.saveTreeWindow = None
        elif windowName == 'aboutWindow':
            self.aboutWindow = None
        elif windowName == 'confirmAddDerivationToSeedsWindow':
            self.confirmAddDerivationToSeedsWindow = None
        elif windowName == 'useSectionFoldersWindow':
            self.useSectionFoldersWindow = None
        elif windowName == 'depMappingsWindow':
            self.depMappingsWindow = None
        elif windowName == 'useAutosWindow':
            self.useAutosWindow = None
        elif windowName == 'overwriteDerivationWindow':
            self.overwriteDerivationWindow = None
        elif windowName == 'confirmTrainTaggerWindow':
            self.confirmTrainTaggerWindow = None
        elif windowName == 'confirmLoadTaggerWindow':
            self.confirmLoadTaggerWindow = None
        #whenever we destroy any window, all error windows will close.. therefore
        #we should always reset nothingEntered to = None
        if self.nothingEntered != None:
            self.nothingEntered.destroy()
            self.nothingEntered = None
        window.destroy()
        window = None

    def function():
        print "hello world!"

    def refreshTerminals(self):
        self.oldCatIsNewCat = False
        self.oldAndNewPosMappingsMatch = False
        self.oldAndNewPosMappingsSubset = False
        self.covertLexUpdated = False
        self.checkedForDuplicates = False
        if self.terminals_frame != None:
            self.terminals_frame.destroy()
        if self.TASK != 'parse':
            self.terminals_frame = Frame(self.TERMINALS_FRAME)
            self.terminals_frame.pack(expand=True, fill='both')
            self.TERMINALS = TerminalsFrame(self.terminals_frame)
            self.TERMINALS.pack(side='left', expand=True)
            self.TERMINALS.frame.bind("<Configure>", self.TERMINALS.onFrameConfigure)
            self.MGcats = []
            if self.newSentMode:
                foundOvertMGcat = False
                #if this is a test sentence, then we have no PTB categories
                #as we have no PTB parse, so all MGcats must be made available in the
                #drop-downs for each word..
                words = self.test_words
                allMGcats = []
                for ptbCat in self.PosMappings:
                    for MGcat in self.PosMappings[ptbCat]:
                        if MGcat not in ["", "No categories available"]:
                            foundOvertMGcat = True
                            if MGcat not in allMGcats:
                                allMGcats.append(MGcat)
                if not foundOvertMGcat:
                    self.viewCurrentPTBtree()
            else:
                words = self.words
            if (self.match != False and self.match != 'False' and self.match != 'checked') or self.newSentMode:
                for i in range(len(words)):
                    t = Frame(self.TERMINALS.frame)
                    t.grid(column=0, sticky=W)
                    self.MGcats.append(StringVar(self.mainWindow))
                    word = Label(t, text="  "+words[i]+" ")
                    word.pack(side='left')
                    if self.newSentMode:
                        MGcatOptions = allMGcats
                        if '' not in MGcatOptions:
                            MGcatOptions.insert(0, '')
                    else:
                        MGcatOptions = copy.deepcopy(self.PosMappings[self.PTBPosTags[i]])
                    ind = -1
                    MGcatOptions.sort()
                    for MGcat in MGcatOptions:
                        ind+=1
                        try:
                            comments = self.overtCatComments[MGcat]
                        except KeyError:
                            comments = ''
                        if comments == '':
                            MGcatOptions[ind] = MGcat
                        else:
                            MGcatOptions[ind] = MGcat+"          ("+comments+")"
                    drop_down = OptionMenu(t, self.MGcats[i], *MGcatOptions)
                    drop_down.config(width=30)
                    drop_down.pack(side='left')
                    dropDownVar = self.MGcats[i]
                    searchButton = Button(t, text="s", command=partial(self.viewDelCats, catType='overt', dropDownVar=dropDownVar, MGcatOptions=MGcatOptions), relief=RAISED)
                    searchButton.pack(side='left')
                    if len(MGcatOptions) == 2 and MGcatOptions[1] == "No categories available":
                        searchButton.config(state=DISABLED)
                    emptyLabel = Label(t, text=" ")
                    emptyLabel.pack(side='left')
            self.TERMINALS.frame.bind("<Configure>", self.TERMINALS.onFrameConfigure)

    def viewDelCats(self, catType, dropDownVar=False, MGcatOptions=None, fromDerivation=False):
        self.dropDownVar = dropDownVar
        if not dropDownVar and not fromDerivation:
            try:
                nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
            except IOError:
                nullCatTreeMappings = {}
        if self.viewDelCatsWindow != None:
            self.destroyWindow(self.viewDelCatsWindow, 'viewDelCatsWindow')
        if self.addOvertCatWindow != None:
            self.destroyWindow(self.addOvertCatWindow, 'addOvertCatWindow')
        if self.addCovertCatWindow != None:
            self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow')
        self.cats = []
        if dropDownVar:
            viewDelCatsWindowTitle = "Overt MG category selection"
            self.cats = MGcatOptions[1:]
        elif fromDerivation:
            viewDelCatsWindowTitle = "Covert MG category selection"
            self.updateSelfDotCats(catType)
        else:
            if catType == 'overt':
                viewDelCatsWindowTitle = "View/delete/modify overt MG categories"
            elif catType == 'covert':
                viewDelCatsWindowTitle = "View/delete/modify covert MG categories"
            self.updateSelfDotCats(catType)                 
        if len(self.cats) == 0:
            self.nothing(self.mainWindow, "There are currently no "+catType+" categories in the system!")
            return
        if not fromDerivation:
            self.viewDelCatsWindow = Toplevel(self.mainWindow)
        else:
            self.viewDelCatsWindow = Toplevel(self.derivationWindow)
        self.viewDelCatsWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.viewDelCatsWindow, 'viewDelCatsWindow'))
        self.viewDelCatsWindow.title(viewDelCatsWindowTitle)
        catsANDscroll = Frame(self.viewDelCatsWindow)
        catsANDscroll.pack()
        scrollbarV = Scrollbar(catsANDscroll, orient=VERTICAL)
        scrollbarH = Scrollbar(catsANDscroll, orient=HORIZONTAL)
        width = 140
        self.catsBox = Listbox(catsANDscroll, yscrollcommand=scrollbarV.set, width = width, height = 10)
        self.catsBox.pack(side=LEFT)
        scrollbarV.config(command=self.catsBox.yview)
        scrollbarV.pack(side=RIGHT, fill=Y)
        for item in self.cats: self.catsBox.insert(END, " "+item)
        if catType == 'overt':
            self.overtComments = StringVar()
            Label(self.viewDelCatsWindow, textvariable=self.overtComments).pack()
        elif catType == 'covert':
            self.covertComments = StringVar()
            Label(self.viewDelCatsWindow, textvariable=self.covertComments).pack()
        buttonBox = Frame(self.viewDelCatsWindow)
        buttonBox.pack()
        closeButton = Button(buttonBox, text="Close", command=lambda: self.destroyViewDelCatsWindow(self.viewDelCatsWindow, 'viewDelCatsWindow'))
        closeButton.pack(side=LEFT)
        if not dropDownVar and not fromDerivation:
            deleteButton = Button(buttonBox, text="Delete", command=lambda: self.confirm(catType=catType, confirmWhat='delCats', message=None))
            deleteButton.pack(side=LEFT)
            modifyButton = Button(buttonBox, text="Modify", command=lambda: self.modifyCats(self.catsBox.curselection(), catType))
            modifyButton.pack(side=LEFT)
            seedsButton = Button(buttonBox, text="View Seeds", command=lambda: self.viewCatParses(catType, folder='seeds'))
            seedsButton.pack(side=LEFT)
            autosButton = Button(buttonBox, text="View Autos", command=lambda: self.viewCatParses(catType, folder='autos'))
            autosButton.pack(side=LEFT)
        elif fromDerivation:
            exportArg1Button = Button(buttonBox, text="Export to Arg1", command=lambda: self.exportArg(self.catsBox, 'arg1'))
            exportArg1Button.pack(side=LEFT)
            exportArg2Button = Button(buttonBox, text="Export to Arg2", command=lambda: self.exportArg(self.catsBox, 'arg2'))
            exportArg2Button.pack(side=LEFT)
        else:
            selectButton = Button(buttonBox, text="Select", command=lambda: self.autoSelectDropDown(dropDownVar))
            selectButton.pack(side=LEFT)
        emptyLabel = Label(buttonBox, text = "   ")
        emptyLabel.pack(side=LEFT)
        searchButton = Button(buttonBox, text='Search comments:', command=lambda: self.startNewCommentSearch(catType))
        searchButton.pack(side=LEFT)
        self.commentSearchEntry = Entry(buttonBox, width = 20)
        self.commentSearchEntry.pack(side=LEFT)
        self.commentSearchEntry.focus_set()
        self.commentSearchEntry.bind('<Return>', lambda x: searchButton.invoke())
        self.featureSearchEntry = Entry(buttonBox, width = 16)#creatig this here and packing later to get the traversal order right for the tab key
        emptyLabel = Label(buttonBox, text = "  ")
        emptyLabel.pack(side=LEFT)
        clearButton = Button(buttonBox, text='Clear Searches', command=lambda: self.startNewCommentSearch(catType, True))
        clearButton.pack(side=LEFT)
        emptyLabel = Label(buttonBox, text = "  ")
        emptyLabel.pack(side=LEFT)
        featureSearchButton = Button(buttonBox, text='Search features:', command=lambda: self.startNewFeatureSearch(catType))
        featureSearchButton.pack(side=LEFT)
        self.featureSearchEntry.pack(side=LEFT)
        self.featureSearchEntry.bind('<Return>', lambda x: featureSearchButton.invoke())
        #two commands which define what happens upon single and double clicks..
        if not dropDownVar and not fromDerivation:
            self.catsBox.bind('<Double-1>', lambda x: modifyButton.invoke())
        elif fromDerivation:
            self.catsBox.bind('<Double-1>', lambda x: self.doubleClick(self.catsBox))
        else:
            self.catsBox.bind('<Double-1>', lambda x: selectButton.invoke())
        try:
            if catType == 'overt':
                self.catsBox.bind('<<ListboxSelect>>', lambda x: self.showOvertComments(self.catsBox, self.overtComments))
            elif catType == 'covert':
                self.catsBox.bind('<<ListboxSelect>>', lambda x: self.showCovertComments(self.catsBox, self.covertComments))
        except TclError:
            x=0
        emptyBox = Frame(self.viewDelCatsWindow)
        emptyBox.pack()
        emptyLabel = Label(emptyBox, text = "   ")
        emptyLabel.pack()

    def autoSelectDropDown(self, dropDownVar):
        try:
            dropDownVar.set(self.catsBox.get(self.catsBox.curselection()))
        except TclError:
            self.nothing(self.viewDelCatsWindow, "You must click on a category first!")
            return
        self.destroyWindow(self.viewDelCatsWindow, 'viewDelCatsWindow')

    def viewCatParses(self, catType, folder):
        try:
            cat = self.catsBox.get(self.catsBox.curselection()).strip().split("    ")[0]
        except TclError:
            self.nothing(self.viewDelCatsWindow, "You must select a category first!")
            return
        if catType == 'overt':
            if folder == 'seeds':
                CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
            elif folder == 'autos':
                if 'autoCatTreeMappings' not in os.listdir(self.auto_folder):
                    autobank.buildAutoMappings(self.auto_folder, self.seed_folder)
                CatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
        elif catType == 'covert':
            if folder == 'seeds':
                CatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
            elif folder == 'autos':
                if 'autoNullCatTreeMappings' not in os.listdir(self.auto_folder):
                    autobank.buildAutoMappings(self.auto_folder, self.seed_folder)
                CatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
        if cat in CatTreeMappings:
            CatTreeMappings[cat].sort()
            parses = CatTreeMappings[cat]
            self.fileManager(folder=folder, catSearch=True, files=parses, cat=cat)
        else:
            self.nothing(self.viewDelCatsWindow, "No "+folder[:-1]+" parses are associated with that category!")
            return

    def updateSelfDotCats(self, catType):
        self.cats = []
        if catType == 'overt':
            #now we gather the list of MG categories..
            for PTBentry in self.PosMappings:
                for MGentry in self.PosMappings[PTBentry]:
                    if MGentry not in ['', 'No categories available']:
                        try:
                            comments = self.overtCatComments[MGentry]
                        except KeyError:
                            comments = ''
                        if comments == '':
                            if MGentry not in self.cats:
                                self.cats.append(MGentry)
                        else:
                            if MGentry+"          "+"("+comments+")" not in self.cats:
                                self.cats.append(MGentry+"          "+"("+comments+")")
        elif catType == 'covert':
            #we need covertRefs so we can easily choose which covert lexical entry to delete or modify
            #in the covert lexicon files
            viewDelCatsWindowTitle = "View/delete/modify covert MG categories"
            (self.cats, self.covertRefs) = self.updateCovertRefs()
        self.cats.sort()

    def startNewFeatureSearch(self, catType):
        if catType == 'overt':
            self.overtComments.set("")
        elif catType == 'covert':
            self.covertComments.set("")
        if not self.dropDownVar:
            self.updateSelfDotCats(catType)
        self.commentSearchEntry.delete(0, 'end')
        cats = self.cats
        self.catsBox.delete(0, END)
        searchFeatures = self.sortSubCat(self.featureSearchEntry.get().strip().split(" "), return_list=True)
        newCats = []
        target = len(searchFeatures)
        for CAT in cats:
            numFeaturesMatched = 0
            cat = CAT.split("    ")[0].strip().split(" ")
            del(cat[0])
            if catType == 'covert':
                del(cat[0])
            for FEATURE in searchFeatures:
                if len(FEATURE) > 2 and FEATURE[0] in ["'", '"'] and FEATURE[-1] in ["'", '"']:
                    FEATURE = FEATURE[1:-1]
                    whole = True
                else:
                    whole = False
                original_FEATURE = FEATURE
                sf = []
                if re.search('{.*}', FEATURE):
                    sf = re.search('{.*}', FEATURE).group(0)[1:-1].split(".")
                    FEATURE = re.sub('{.*}', '', FEATURE)
                else:
                    sf = []
                for feature in cat:
                    SF = []
                    subcat_match = True
                    original_feature = feature
                    if re.search('{.*}', feature):
                        SFString = re.search('{.*}', feature).group(0)[1:-1]
                        SF = SFString.split(".")
                        for F in SF:
                            subcats_to_add = []
                            if re.search('^[+-]\[.*?\]', F):
                                optional_subcat = re.search('[+-]\[.*?\]', F).group(0)
                                optional_subcat = re.sub('\[', '', optional_subcat)
                                optional_subcat = re.sub('\]', '', optional_subcat)
                                optional_subcat = re.sub('\.', '|', optional_subcat, count=100)
                                sign = optional_subcat[0]
                                optional_subcat = optional_subcat[1:].split("|")
                                for f in optional_subcat:
                                    subcats_to_add.append(sign+f)
                            elif re.search('^\[.*?\]', F):
                                optional_subcat = re.search('\[.*?\]', F).group(0)
                                optional_subcat = re.sub('\[', '', optional_subcat)
                                optional_subcat = re.sub('\]', '', optional_subcat)
                                optional_subcat = re.sub('\.', '|', optional_subcat, count=100)
                                optional_subcat = optional_subcat.split("|")
                                for f in optional_subcat:
                                    subcats_to_add.append(f)
                            for f in subcats_to_add:
                                SF.append(f)
                        SUBCAT_MATCH = False
                        if "." in original_FEATURE:
                            SUBCAT_MATCH = True
                            #as well as checking subcat features inside whole features, the user
                            #has the option just to type the subcat features separated by dots
                            #this bit of code checks for a match on this
                            SUBCAT_FEATURES = original_FEATURE.split(".")
                            for SUBCAT_FEATURE in SUBCAT_FEATURES:
                                if SUBCAT_FEATURE not in SF:
                                    SUBCAT_MATCH = False
                                    break
                        if original_FEATURE in SF or SUBCAT_MATCH:
                            cat.remove(feature)
                            numFeaturesMatched += 1
                            break
                        feature = re.sub('{.*}', '', feature)
                        #we don't have to find every subcat_feature of a category in the system but all the ones
                        #we specify in the search must match.. so in a subset relation
                        for subcat_feature in sf:
                            if subcat_feature not in SF:
                                subcat_match = False
                                break
                    else:
                        if len(sf) > 0:
                            subcat_match = False
                    if whole:
                        if original_FEATURE == original_feature:
                            cat.remove(original_feature)
                            numFeaturesMatched += 1
                            break
                    elif FEATURE == feature and subcat_match:
                        cat.remove(original_feature)
                        numFeaturesMatched += 1
                        break
            if numFeaturesMatched == target:
                newCats.append(CAT)
        for item in newCats: self.catsBox.insert(END, " "+item)

    def startNewCommentSearch(self, catType, clear=False):
        if catType == 'overt':
            self.overtComments.set("")
        elif catType == 'covert':
            self.covertComments.set("")
        if not self.dropDownVar:
            self.updateSelfDotCats(catType)
        cats = self.cats
        self.featureSearchEntry.delete(0, 'end')
        self.catsBox.delete(0, END)
        searchItem = self.commentSearchEntry.get().strip().lower()
        #we don't want to separate apostrophe's from words during tokenization of the search item as we want to be able to search for whole words only
        searchItem = re.sub("'", "APOSTROPHE", searchItem, count = 100)
        searchItem = re.sub('"', "APOSTROPHE", searchItem, count = 100)
        searchSentence = tokenize.word_tokenize(searchItem)
        wordsToDelete = []
        for word in searchSentence:
            if word in [',', '.']:
                wordsToDelete.append(word)
        for word in wordsToDelete:
            searchSentence.remove(word)
        if clear or searchSentence == []:
            self.commentSearchEntry.delete(0, 'end')
            for item in cats: self.catsBox.insert(END, " "+item)
        else:
            newCats = []
            if catType == 'overt':
                for CAT in cats:
                    cat = CAT.split("    ")[0].strip()
                    try:
                        comment = tokenize.word_tokenize(self.overtCatComments[cat].lower())
                    except Exception as e:
                        self.nothing(self.mainWindow, "Oops, an error corrupted some files!..  Please restore from last backup or fix files manually.", width=600)
                        return
                    wordsToDelete = []
                    for word in comment:
                        if word in [',', '.']:
                            wordsToDelete.append(word)
                    for word in wordsToDelete:
                        comment.remove(word)
                    includeCat = True
                    for word in searchSentence:
                        if len(word) >= 21 and word[:10] == 'APOSTROPHE' and word[-10:] == 'APOSTROPHE':
                            whole = True
                        else:
                            whole = False
                        word = re.sub("APOSTROPHE", "", word, count = 1000)
                        if word == '':
                            continue
                        located_word = False
                        for WORD in comment:
                            WORD = re.sub("'", "", WORD, count = 1000)
                            WORD = re.sub('"', "", WORD, count = 1000)
                            #we want to be able to match partial words, e.g. 'pres' in search sentence should match 'present' in comment
                            if (not whole and word in WORD) or (whole and word == WORD):
                                located_word = True
                                break
                        if not located_word:
                            includeCat = False
                            break
                    if includeCat:
                        newCats.append(CAT)
            elif catType == 'covert':
                for CAT in cats:
                    cat = CAT.split("     ")[0].strip()
                    lexicon = CAT.split("     ")[1].strip()[1:-1]
                    for entry in self.covertCatComments[cat]:
                        if entry[0] == lexicon:
                            comment = tokenize.word_tokenize(entry[1].lower())
                            wordsToDelete = []
                            for word in comment:
                                if word in [',', '.']:
                                    wordsToDelete.append(word)
                            for word in wordsToDelete:
                                comment.remove(word)
                    includeCat = True
                    for word in searchSentence:
                        if len(word) >= 21 and word[:10] == 'APOSTROPHE' and word[-10:] == 'APOSTROPHE':
                            whole = True
                        else:
                            whole = False
                        word = re.sub("APOSTROPHE", "", word, count = 1000)
                        if word == '':
                            continue
                        located_word = False
                        for WORD in comment:
                            WORD = re.sub("'", "", WORD, count = 1000)
                            WORD = re.sub('"', "", WORD, count = 1000)
                            #we want to be able to match partial words, e.g. 'pres' in search sentence should match 'present' in comment
                            if (not whole and word in WORD) or (whole and word == WORD):
                                located_word = True
                                break
                        if not located_word:
                            includeCat = False
                            break
                    if includeCat:
                        newCats.append(CAT)
            for item in newCats: self.catsBox.insert(END, " "+item)

    def showCovertComments(self, catsBox, covertComments):
        try:
            for entry in self.covertCatComments[catsBox.get(catsBox.curselection()).split("     ")[0].strip()]:
                if entry[0] == catsBox.get(catsBox.curselection()).split("     ")[1].strip()[1:-1]:
                    covertComments.set(entry[1])
                    return
        except TclError:
            #null lexicons are all empty
            return

    def showOvertComments(self, catsBox, overtComments):
        try:
            overtComments.set(self.overtCatComments[catsBox.get(self.catsBox.curselection()).split("     ")[0].strip()])
        except TclError:
            #overt lexicon is empty
            return
        
    def updateCovertRefs(self):
        cats = []
        covertRefs = {}
        for f in self.covertLexicons:
            index = -1
            try:
                covLex = json.load(open(self.seed_folder+"/"+f[1]))
            except IOError:
                covLex = []
            for entry in covLex:
                index += 1
                word = entry[0]
                features = entry[1]
                if 'conj' in entry[2]:
                    typeSeparator = u':\u0305:\u0305'
                else:
                    typeSeparator = "::"
                MGentry = word+" "+typeSeparator+" "+" ".join(features)
                if MGentry not in covertRefs:                        
                    covertRefs[MGentry] = [(f[1], index)]
                else:
                    covertRefs[MGentry].append((f[1], index))
                cats.append(MGentry+u"     ("+unicode(f[1])+u")")
        return (cats, covertRefs)

    def confirmConstructLexicon(self):
        if self.constructLexiconWindow != None:
            self.destroyWindow(self.constructLexiconWindow, 'constructLexiconWindow')
        self.constructLexiconWindow = Toplevel(self.mainWindow)
        self.constructLexiconWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.constructLexiconWindow, 'constructLexiconWindow'))
        w=375
        h = 90
        (x, y) = self.getCentrePosition(w, h)
        self.constructLexiconWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.constructLexiconWindow, text="Do you wish to extract a lexicon for\nthe seeds only, or for the seeds and autos\n(two files will be generated)?")
        label.pack()
        buttonFrame = Frame(self.constructLexiconWindow)
        buttonFrame.pack()
        seedsButton = Button(buttonFrame, text="use seeds only", command=lambda: self.extractOvertLexicon(useAutos=False))
        seedsButton.pack(side=LEFT)
        seedsAutosButton = Button(buttonFrame, text="use seeds and autos", command=lambda: self.extractOvertLexicon(useAutos=True))
        seedsAutosButton.pack(side=LEFT)
        cancelButton = Button(buttonFrame, text="cancel", command=lambda: self.destroyWindow(self.constructLexiconWindow, 'constructLexiconWindow'))
        cancelButton.pack(side=LEFT)

    def extractTaggingModels(self, MGtagType, gen_new_ccg_trees=False, ns=False, buildCorpusOnly=False, train=None, dev=None, test=None):
        #for words seen at least k times in the data, we only allow tags to be assigned to them which were seen with that word in the data
        #the tag dictionaries keep track of the number of times a word was seen and the tags it was seen with.
        tag_dict = {}
        seed_tag_dict = {}
        if MGtagType == 'atomic':
            print "\nExtracting Unigram Tag model: P(overt MG axiom|CCG supertag), and backoff models...\n"
        elif MGtagType == 'atomic_maxent':
            print "\nExtracting MaxEnt Tag model: P(overt MG axiom|word,CCG supertag,context), and backoff models..."
            print "\nMaxEnt Supertagger Parameters:\n"
            print " --super-forward_beam_ratio "+str(self.super_forward_beam_ratio)
            print " --super-beam_ratio "+str(self.super_beam_ratio)
            print " --super-beam_width "+str(self.super_beam_width)
            print " --super-category_cutoff: "+str(self.super_category_cutoff)
            print " --super-rare_cutoff "+str(self.super_rare_cutoff)
            print " --super-tagdict_min "+str(self.super_tag_dict_min)+'\n'
            #the MG tags are very verbose, so we give them a unique id instead and the following tables keep track of this
            MGT_REF_table = {}
            REF_MGT_table = {}
            MGT_ref = 0
            #we will have 4 different backoff models for both the unigram and the maxEnt models.. the first
            #model uses the full CCG+PTB+function_tags tags.. model2 uses just the CCG+PTB tags, model 3 uses just the CCG tags, and model4 uses the PTB+function_tags only
            #(PTB tags include both the original PTB tag (often modified by my PTB pretransformPTB.py program) and Julia Hockenmaier's revised PTB tag)
            #There are some trees that propbank does not cover, so model2 deals with that
            #Then there are some CCG tags that are not covered by the seed set, so model4 deals with that
            #there is currently no model to deal with the intersection of these issues, i.e. where propbank
            #doesn't cover the tree and the CCG tag is not in the seed set.
            #the following are the tagged corpus files needed by the max ent model
            ccg_mg_tag_corpus_model1_file = open(self.seed_folder+'/'+'ccg_mg_tagged_corpus_model1', 'w')
            ccg_mg_tag_corpus_model2_file = open(self.seed_folder+'/'+'ccg_mg_tagged_corpus_model2', 'w')
            ccg_mg_tag_corpus_model3_file = open(self.seed_folder+'/'+'ccg_mg_tagged_corpus_model3', 'w')
            ccg_mg_tag_corpus_model4_file = open(self.seed_folder+'/'+'ccg_mg_tagged_corpus_model4', 'w')
        elif MGtagType == 'supertag':
            if not ns:
                print "\nExtracting Unigram Supertag models: P(MG supertag|CCG_PTB_Propbank tags)...\n"
            else:
                print "\nExtracting Unigram Supertag models: P(MG supertag_NS|CCG_PTB_Propbank tags)...\n"
        elif MGtagType == 'supertag_maxent':
            if not ns:
                print "\nExtracting MaxEnt Supertag models: P(MG supertag|word,CCG_PTB_Propbank tags,context)... (This may take a few minutes.)\n"
            else:
                print "\nExtracting MaxEnt Supertag models: P(MG supertag_NS|word,CCG_PTB_Propbank tags,context)... (This may take a few minutes.)\n"
            MGST_REF_table = {}
            REF_MGST_table = {}
            MGST_ref = 0
            if ns:
                NS = '_ns'
            else:
                NS = ''
            ccg_mg_supertag_corpus_model1_file = open(self.seed_folder+'/'+'ccg_mg_supertagged'+NS+'_model1_corpus', 'w')
            ccg_mg_supertag_corpus_model2_file = open(self.seed_folder+'/'+'ccg_mg_supertagged'+NS+'_model2_corpus', 'w')
            ccg_mg_supertag_corpus_model3_file = open(self.seed_folder+'/'+'ccg_mg_supertagged'+NS+'_model3_corpus', 'w')
            ccg_mg_supertag_corpus_model4_file = open(self.seed_folder+'/'+'ccg_mg_supertagged'+NS+'_model4_corpus', 'w')
        if MGtagType == 'supertag_lstm':
            if not buildCorpusOnly:
                #this function is sometimes called simply to construct the corpora which are then used to train
                #the lstm model from autobank.py
                if not ns:
                        print '\nExtracting LSTM Supertag models: P(MG supertag|word,context)... (This may take a few minutes.)\n'
                else:
                    print '\nExtracting LSTM Supertag models: P(MG supertag_NS|word,context)... (This may take a few minutes.)\n'
            #the supertags are very verbose, so we assign them a unique id and then use that for tagging
                    #the following look up tables allow us to convert back and forth between a supertag and its id
            MGST_REF_table = {}
            REF_MGST_table = {}
            #MGST_ref is the unique id
            MGST_ref = 0
            #ns or NS means 'no subcategorisation features' and refers to the more abstract MG supertags which lack features like NOM, -ACC, +PRES etc
            if ns:
                NS = '_ns'
                data_dir = 'abstract_data'
            else:
                NS = ''
                data_dir = 'reified_data'
            if data_dir not in os.listdir('SuperSuperTagger/'):
                os.mkdir('SuperSuperTagger/'+data_dir)
            mg_lstm_words_train_corpus_file = open('SuperSuperTagger/'+data_dir+'/train.words', 'w')
            mg_lstm_supertags_train_corpus_file = open('SuperSuperTagger/'+data_dir+'/train.tags', 'w')
            mg_lstm_words_dev_corpus_file = open('SuperSuperTagger/'+data_dir+'/dev.words', 'w')
            mg_lstm_supertags_dev_corpus_file = open('SuperSuperTagger/'+data_dir+'/dev.tags', 'w')
            mg_lstm_words_test_corpus_file = open('SuperSuperTagger/'+data_dir+'/test.words', 'w')
            mg_lstm_supertags_test_corpus_file = open('SuperSuperTagger/'+data_dir+'/test.tags', 'w')
            mg_lstm_corpus_file = open(self.seed_folder+'/'+'mg_lstm_corpus'+NS, 'w')
            current_lstm_words_file = mg_lstm_words_dev_corpus_file
            current_lstm_supertags_file = mg_lstm_supertags_dev_corpus_file
        elif MGtagType == 'hybrid':
            #hybrid refers to supertags lacking null c heads.. null c heads are just allowed into the chart as atomic types
            print "\nExtracting Unigram Hybrid models: P(MG hybrid_supertag|CCG_PTB_Propbank) tags)...\n"
        elif MGtagType == 'hybrid_maxent':
            #hybrid refers to supertags lacking null c heads.. null c heads are just allowed into the chart as atomic types
            if not ns:
                print "\nExtracting MaxEnt Hybrid models: P(MG hybrid_supertag|word,CCG_PTB_Propbank tags,context)... (This may take a few minutes.)\n"
            else:
                print "\nExtracting MaxEnt Hybrid models: P(MG hybrid_supertag_NS|word,CCG_PTB_Propbank tags,context)... (This may take a few minutes.)\n"
            MGHST_REF_table = {}
            REF_MGHST_table = {}
            MGHST_ref = 0
            if ns:
                NS = '_ns'
            else:
                NS = ''
            ccg_mg_supertag_corpus_model1_file = open(self.seed_folder+'/'+'ccg_mg_hybrid'+NS+'_model1_corpus', 'w')
            ccg_mg_supertag_corpus_model2_file = open(self.seed_folder+'/'+'ccg_mg_hybrid'+NS+'_model2_corpus', 'w')
            ccg_mg_supertag_corpus_model3_file = open(self.seed_folder+'/'+'ccg_mg_hybrid'+NS+'_model3_corpus', 'w')
            ccg_mg_supertag_corpus_model4_file = open(self.seed_folder+'/'+'ccg_mg_hybrid'+NS+'_model4_corpus', 'w')
        model1 = {}
        model2 = {}
        model3 = {}
        model4 = {}
        try:
            CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
        except IOError:
            CatTreeMappings = {}
        try:
            TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
        except IOError:
            TreeCatMappings = {}
        #we create lists of the different derivation tree bracketings
        sd_bracketings = []
        sfd_bracketings = []
        d_bracketings = []
        #PARSE_list contains the file name and index in the file of each parse
        PARSE_list = []
        if self.useAutos or MGtagType == 'supertag_lstm':
            #we will include the auotmatically generated MG trees in the corpora/models we are constructing
            for section_folder in sorted(os.listdir(self.auto_folder)):
                if section_folder not in autoFilesToIgnore:
                    for FILE in sorted(os.listdir(self.auto_folder+'/'+section_folder)):
                        if FILE != '.DS_Store':
                            autoParses = json.load(open(self.auto_folder+'/'+section_folder+'/'+FILE))
                            for parse in autoParses:
                                PARSE_list.append(str([self.auto_folder+'/'+section_folder+'/'+FILE, int(parse)]))
                                sd_bracketing = autoParses[parse][0]
                                sfd_bracketing = autoParses[parse][3]
                                d_bracketing = autoParses[parse][4]
                                sd_bracketings.append(sd_bracketing)
                                sfd_bracketings.append(sfd_bracketing)
                                d_bracketings.append(d_bracketing)
        PARSES = []
        for PARSE in TreeCatMappings:
            PARSES.append(PARSE)
        PARSES.sort()
        #now we add the hand annotated MG trees to the corpora which will be used to train the models
        for PARSE in PARSES:
            PARSE_list.append(PARSE)
            parse = ast.literal_eval(PARSE)
            MGFileName = parse[0].encode('ascii')
            seed_line_num = parse[1]
            if 'supertag' in MGtagType or 'hybrid' in MGtagType:
                seeds = json.load(open(MGFileName))
                sd_bracketing = seeds[str(seed_line_num)][0]
                sfd_bracketing = seeds[str(seed_line_num)][3]
                d_bracketing = seeds[str(seed_line_num)][4]
                sd_bracketings.append(sd_bracketing)
                sfd_bracketings.append(sfd_bracketing)
                d_bracketings.append(d_bracketing)
        bracketing_index = -1
        len_PARSE_list = len(PARSE_list)-1
        if MGtagType == 'supertag_lstm':
            test_start = int(dev / 100 * len(PARSE_list))
            train_start = int((dev + test) / 100 * len(PARSE_list))
        if "compressed_parses" not in os.listdir(os.getcwd()+"/CCGbank"):
            os.mkdir("CCGbank/compressed_parses/")
        for PARSE in PARSE_list:
            parse = ast.literal_eval(PARSE)
            MGFileName = parse[0].encode('ascii')
            lineNum = parse[1]
            bracketing_index += 1
            if 'supertag' in MGtagType or 'hybrid' in MGtagType:
                if 'supertag' in MGtagType:
                    ignore_null_c=False#to get hybrid tags for lstm data, change this to True and also change folder names for data
                elif 'hybrid' in MGtagType:
                    ignore_null_c=True
                supertags = self.extractMGsupertags(sfd_bracketings[bracketing_index], d_bracketings[bracketing_index], ignore_null_c=ignore_null_c, ns=ns)
                supertag_indices = {}
                supertag_index = -1
                for supertag in supertags:
                    cat_ids = {}
                    ID = 0
                    supertag_index += 1
                    if type(supertag) != type([]):
                        if supertag not in cat_ids:
                            cat_ids[supertag] = ID
                            ID += 1
                        (features, word) = self.get_derivation_tree_features(supertag, return_name_and_type=True)
                        if int(supertag.index) == -1:
                            raise Exception("Error! Null category detected that is not anchored to an overt category!")
                        else:
                            SUPERTAG = [self.constructMGlexEntry('OVERT_WORD', features), cat_ids[supertag], None]
                            supertags[supertag_index] = SUPERTAG
                            if word not in tag_dict:
                                tag_dict[word] = [1, list(SUPERTAG[0])]
                            elif list(SUPERTAG[0]) not in tag_dict[word]:
                                tag_dict[word].append(list(SUPERTAG[0]))
                                tag_dict[word][0] += 1
                            else:
                                tag_dict[word][0] += 1
                            if 'Seed' in MGFileName:
                                if word not in seed_tag_dict:
                                    seed_tag_dict[word] = [1, list(SUPERTAG[0])]
                                elif list(SUPERTAG[0]) not in seed_tag_dict[word]:
                                    seed_tag_dict[word].append(list(SUPERTAG[0]))
                                    seed_tag_dict[word][0] += 1
                                else:
                                    seed_tag_dict[word][0] += 1
                        #we need to keep track on the string indices of the supertags so we can match them to the CCG tags
                        supertag_indices[int(supertag.index)] = SUPERTAG
                        supertag = SUPERTAG
                    else:
                        for link in supertag:
                            if link[0] not in cat_ids:
                                cat_ids[link[0]] = ID
                                ID += 1
                            (features, word) = self.get_derivation_tree_features(link[0], return_name_and_type=True)
                            if link[0].index == -1:
                                link[0] = [self.constructMGlexEntry(word, features), cat_ids[link[0]], int(link[0].index)]
                            else:
                                supertag_indices[int(link[0].index)] = supertag
                                link[0] = [self.constructMGlexEntry('OVERT_WORD', features), cat_ids[link[0]], None]
                                if word not in tag_dict:
                                    tag_dict[word] = [1, list(link[0][0])]
                                elif list(link[0][0]) not in tag_dict[word]:
                                    tag_dict[word].append(list(link[0][0]))
                                    tag_dict[word][0] += 1
                                else:
                                    tag_dict[word][0] += 1
                                if 'Seed' in MGFileName:
                                    if word not in seed_tag_dict:
                                        seed_tag_dict[word] = [1, list(link[0][0])]
                                    elif list(link[0][0]) not in seed_tag_dict[word]:
                                        seed_tag_dict[word].append(list(link[0][0]))
                                        seed_tag_dict[word][0] += 1
                                    else:
                                        seed_tag_dict[word][0] += 1
                            if link[2] not in cat_ids:
                                cat_ids[link[2]] = ID
                                ID += 1
                            (features, word) = self.get_derivation_tree_features(link[2], return_name_and_type=True)
                            if link[2].index == -1:
                                link[2] = [self.constructMGlexEntry(word, features), cat_ids[link[2]], int(link[2].index)]
                            else:
                                supertag_indices[int(link[2].index)] = supertag
                                link[2] = [self.constructMGlexEntry('OVERT_WORD', features), cat_ids[link[2]], None]
                                if word not in tag_dict:
                                    tag_dict[word] = [1, list(link[2][0])]
                                elif list(link[2][0]) not in tag_dict[word]:
                                    tag_dict[word].append(list(link[2][0]))
                                    tag_dict[word][0] += 1
                                else:
                                    tag_dict[word][0] += 1
                                if 'Seed' in MGFileName:
                                    if word not in seed_tag_dict:
                                        seed_tag_dict[word] = [1, list(link[2][0])]
                                    elif list(link[2][0]) not in seed_tag_dict[word]:
                                        seed_tag_dict[word].append(list(link[2][0]))
                                        seed_tag_dict[word][0] += 1
                                    else:
                                        seed_tag_dict[word][0] += 1
                            #we don't want the order of the categories in each link or the order of the links in each supertag to
                            #mean anything or distinguish tags, so we will sort them here if necessary to make sure they are uniform
                            original_order = [link[0], link[2]]
                            new_order = sorted(original_order)
                            if new_order != original_order:
                                link0copy = link[0]
                                link1copy = link[1]
                                link2copy = link[2]
                                link3copy = link[3]
                                link[0] = link2copy
                                link[1] = link3copy
                                link[2] = link0copy
                                link[3] = link1copy
                        supertag.sort()
                    if (MGtagType == 'supertag_maxent' or MGtagType == 'supertag_lstm') and str(supertag) not in MGST_REF_table:
                        MGST_REF_table[str(supertag)] = 'MGST_' + str(MGST_ref)
                        if not buildCorpusOnly:
                            REF_MGST_table['MGST_' + str(MGST_ref)] = str(supertag)
                        else:
                            REF_MGST_table['MGST_' + str(MGST_ref)] = supertag
                        MGST_ref += 1
                    elif MGtagType == 'hybrid_maxent' and str(supertag) not in MGHST_REF_table:
                        MGHST_REF_table[str(supertag)] = 'MGHST_'+str(MGHST_ref)
                        REF_MGHST_table['MGHST_'+str(MGHST_ref)] = str(supertag)
                        MGHST_ref += 1
                supertag_duplicates = []
                #duplicates arise when the tree has ATB movement so a given lexical item appears twice in the derivation tree
                for st in supertags:
                    if supertags.count(st) > 1:
                        if [st, supertags.count(st)] not in supertag_duplicates:
                            supertag_duplicates.append([st, supertags.count(st)])
                for st in supertag_duplicates:
                    while st[1] > 1:
                        supertags.remove(st[0])
                        st[1] -= 1
            if self.auto_folder in PARSE_list[bracketing_index]:
                seeds = json.load(open(MGFileName))
                seed_line_num=str(parse[1])
                sd_bracketing = seeds[str(seed_line_num)][0]
                sd_tree = gen_derived_tree.gen_derivation_tree(sd_bracketing)
                MGcats = autobank.get_MGcats(sd_tree, [], returnWords=False)
                if 'atomic' in MGtagType:
                    sfd_bracketing = seeds[str(seed_line_num)][3]
                    d_bracketing = seeds[str(seed_line_num)][4]
                    d_tree = gen_derived_tree.gen_derivation_tree(d_bracketing)
                    sfd_tree = gen_derived_tree.gen_derivation_tree(sfd_bracketing)
                    indices_mappings = self.get_overt_indices(d_tree, sfd_tree)
                    sortedMGcats = []
                    for MGcat in MGcats:
                        sortedMGcats.append(None)
                    for mapping in indices_mappings:
                        sortedMGcats[int(indices_mappings[mapping])] = MGcats[int(mapping)]
                    MGcats = sortedMGcats
                    while None in MGcats:
                        MGcats.remove(None)
            else:
                MGcats = TreeCatMappings[PARSE]
            if ns:
                cat_index = -1
                for cat in MGcats:
                    cat_index += 1
                    MGcats[cat_index] = autobank.strip_features(MGcats[cat_index])
            if MGtagType == 'atomic_maxent':
                for MGcat in MGcats:
                    if MGcat not in MGT_REF_table:
                        MGT_REF_table[MGcat] = 'MGT_'+str(MGT_ref)
                        REF_MGT_table['MGT_'+str(MGT_ref)] = MGcat
                        MGT_ref += 1
            parts = MGFileName.split('/')
            if parts[1] == 'new_parses' and gen_new_ccg_trees:
                #we don't have a ccg tree for this sentence but if it is a subpart of the terminal string
                #at the frontier of a ccg tree we do have, then we'll extract the relevant ccg subtree and use that
                words = parts[2].split(" ")
                index = -1
                for word in words:
                    index += 1
                    try:
                        float(word)
                        words[index] = 'num'
                    except ValueError:
                        #sometimes we have a range of dates like 1956-1970.. we also want to make these 'num'
                        if "-" in word or "," in word:
                            chars = [char for char in word]
                            while "-" in chars:
                                chars.remove("-")
                            while "," in chars:
                                chars.remove(",")
                                new_word = "".join(chars)
                            try:
                                float(new_word)
                                words[index] = 'num'
                            except ValueError:
                                x=0
                trigrams = []
                index = -1
                for word in words:
                    index += 1
                    if index < len(words)-2:
                        trigrams.append(" ".join(words[index:index+3]))
                found_matching_sentence = False
                for section_folder in sorted(os.listdir(self.ptb_folder+"_strings/")):
                    if section_folder != '.DS_Store':
                        for stringFile in sorted(os.listdir(self.ptb_folder+"_strings/"+section_folder)):
                            if stringFile != '.DS_Store':
                                line_index = -1
                                for line in open(self.ptb_folder+"_strings/"+section_folder+"/"+stringFile):
                                    line_index += 1
                                    if found_matching_sentence:
                                        break
                                    line = line.lower()
                                    WORDS = [w for w in line.split(" ")]
                                    index = -1
                                    for word in WORDS:
                                        index += 1
                                        if word == "-" or word == "--":
                                            WORDS[index] = 'hyph'
                                            word = 'hyph'
                                        elif word == "$":
                                            WORDS[index] = 'dollar'
                                            word = 'dollar'
                                        elif word == "%":
                                            WORDS[index] = 'percent'
                                            word = 'percent'
                                        elif word == "'":
                                            WORDS[index] = "apostrophe"
                                            word = 'apostrophe'
                                        elif word == "/":
                                            WORDS[index] = "forslash"
                                            word = 'forslash'
                                        elif word == ":":
                                            WORDS[index] = "colon"
                                            word = 'colon'
                                        elif word == ";":
                                            WORDS[index] = "semicolon"
                                            word = 'semicolon'
                                        new_word = ""
                                        for char in word:
                                            if char not in string.punctuation:
                                                new_word+=char
                                        WORDS[index] = new_word
                                        word = new_word
                                        if "." in word:
                                            new_word = ""
                                            for char in word:
                                                if char != ".":
                                                    new_word+=char
                                            WORDS[index] = new_word
                                            word = new_word
                                        if "'" in word:
                                            new_word = ""
                                            for char in word:
                                                if char != "'":
                                                    new_word+=char
                                            WORDS[index] = new_word
                                            word = new_word
                                    while '' in WORDS:
                                        WORDS.remove('')
                                    if '\n' in WORDS[-1]: WORDS[-1] = WORDS[-1][:-1]
                                    all_words_found = True
                                    for word in words:
                                        if word not in WORDS:
                                            all_words_found = False
                                    if all_words_found:
                                        for trigram in trigrams:
                                            if trigram in line:
                                                partial_states = [[[0,0]]]
                                                complete_states = []
                                                while len(partial_states) > 0:
                                                    state_list = partial_states[0]
                                                    #each state in the state list is a tuple composed of an index pointing to a word in the new sentence and an index pointing to its corresponding word in the original sentence
                                                    while state_list[-1][1] < len(WORDS):
                                                        if words[state_list[-1][0]] == WORDS[state_list[-1][1]]:
                                                            if state_list[-1][1] == len(WORDS)-1:
                                                                partial_states.remove(state_list)
                                                                if len(state_list) == len(words):
                                                                    complete_states.append(state_list)
                                                                break
                                                            else:
                                                                if words[state_list[-1][0]] in WORDS[state_list[-1][1]+1:]:
                                                                    #if the same word is also found later on in the sentence, we create a new
                                                                    #state_list and store it to be completed later
                                                                    pos = WORDS[state_list[-1][1]+1:].index(words[state_list[-1][0]])
                                                                    new_state_list = copy.deepcopy(state_list)
                                                                    new_state_list[-1][1] = state_list[-1][1] + pos+1
                                                                    partial_states.append(new_state_list)
                                                                if len(state_list) == len(words):
                                                                    complete_states.append(state_list)
                                                                    partial_states.remove(state_list)
                                                                    break
                                                                state_list.append([state_list[-1][0]+1,state_list[-1][1]+1])
                                                        else:
                                                            state_list[-1][1] += 1
                                                            if state_list[-1][1] == len(WORDS):
                                                                partial_states.remove(state_list)
                                                ccg_trees = []
                                                tree_scores = []
                                                for state_list in complete_states:
                                                    ccgParses = json.load(open("CCGbank/"+section_folder+"/"+stringFile.split(".")[0]+".ccg"))
                                                    ID = stringFile.split(".")[0]+"."+str(line_index+1)
                                                    CCG_TREE = autobank.build_tree(ccgParses[ID])
                                                    ccg_tree = CCG_TREE[0]
                                                    ccg_terminals = CCG_TREE[2]
                                                    if len(ccg_terminals) == len(WORDS):
                                                        indices_to_retain = [ind[1] for ind in state_list]
                                                        ind = -1
                                                        for terminal in ccg_terminals:
                                                            ind += 1
                                                            if ind not in indices_to_retain:
                                                                current_node = terminal
                                                                while current_node.mother != None and len(current_node.mother.daughters) == 1:
                                                                    current_node = current_node.mother
                                                                current_node.mother.daughters.remove(current_node)
                                                        found_matching_sentence = True
                                                        ccg_trees.append(ccg_tree)
                                                        tree_scores.append(self.get_ccg_tree_score(ccg_trees[-1], 0))
                                                ccg_bracketing = ccg_tree.generate_bracketing(terminal_brackets=False)
                                                ccg_parse = {ID:ccg_bracketing}
                                                with open("CCGbank/compressed_parses/"+parts[2]+".ccg", "w") as ccg_file:
                                                    json.dump(ccg_parse, ccg_file)
                                                id_parts = ID.split(".")
                                                print "Matched PTB tree: "+id_parts[0]+ " ln: "+id_parts[1]+", with new sentence MG tree: "+parts[2]+"...\nCreated compressed CCG tree for this sentence to be used in all tagging models..\n"
                                                break
            elif parts[1] != 'new_parses':
                stringFile = open(self.ptb_folder+"_strings/"+parts[1]+'/'+parts[2]+"_strings")
                index = -1
                for line in stringFile:
                    index += 1
                    if index == lineNum:
                        words = line.split(" ")
                        break
            if parts[1] == 'new_parses':
                if parts[2]+".ccg" in os.listdir("CCGbank/compressed_parses/") or MGtagType == 'supertag_lstm':
                    words = parts[2].split(" ")
                    index = -1
                    for word in words:
                        index += 1
                        try:
                            float(word)
                            words[index] = 'num'
                        except ValueError:
                            #sometimes we have a range of dates like 1956-1970.. we also want to make these 'num'
                            if "-" in word or "," in word:
                                chars = [char for char in word]
                                while "-" in chars:
                                    chars.remove("-")
                                while "," in chars:
                                    chars.remove(",")
                                    new_word = "".join(chars)
                                try:
                                    float(new_word)
                                    words[index] = 'num'
                                except ValueError:
                                    x=0
                else:
                    continue
            while '' in words:
                words.remove('')
            index = -1
            for word in words:
                index+=1
                if word[-1] == '\n':
                    words[index] = word[:-1].lower()
                else:
                    words[index] = word.lower()
            if not MGtagType == 'supertag_lstm' and parts[1] == 'new_parses' and parts[2] + '.ccg' in os.listdir('CCGbank/compressed_parses/'):
                ccgParses = json.load(open('CCGbank/compressed_parses/' + parts[2] + '.ccg'))
                for ID in ccgParses:
                    break
            
            else:
                if not MGtagType == 'supertag_lstm' and parts[1] == 'new_parses':
                    continue
                else:
                    if not MGtagType == 'supertag_lstm':
                        ccgParses = json.load(open('CCGbank/' + parts[1] + '/' + parts[2].split('.')[0] + '.ccg'))
                        ID = parts[2].split('.')[0] + '.' + str(lineNum + 1)
            if not MGtagType == 'supertag_lstm':
                CCGPARSES = []
                if ID in ccgParses:
                    CCGPARSES.append(ID)
                CCGPARSES.sort()
                for ID in CCGPARSES:
                    ccg_terminals = autobank.build_tree(ccgParses[ID])[2]
                    if len(ccg_terminals) == len(MGcats):
                        terminal_index = -1
                        if 'maxent' in MGtagType:
                            tagged_sentence_model1 = ""
                            tagged_sentence_model2 = ""
                            tagged_sentence_model3 = ""
                            tagged_sentence_model4 = ""
                        for ccg_terminal in ccg_terminals:
                            terminal_index += 1
                            model2_tag = "_".join(ccg_terminal.mother.name.split("_")[:3])
                            model3_tag = ccg_terminal.mother.name.split("_")[0]
                            model4_tag = "_".join(ccg_terminal.mother.name.split("_")[1:])
                            if ccg_terminal.mother.name not in model1:
                                model1[ccg_terminal.mother.name] = {}
                            if model2_tag not in model2:
                                model2[model2_tag] = {}
                            if model3_tag not in model3:
                                model3[model3_tag] = {}
                            if model4_tag not in model4:
                                model4[model4_tag] = {}
                            if MGtagType == 'atomic':
                                #the unigram atomic tag models
                                if MGcats[terminal_index] not in model1[ccg_terminal.mother.name]:
                                    model1[ccg_terminal.mother.name][MGcats[terminal_index]] = [1, {words[terminal_index]: 1}]
                                else:
                                    model1[ccg_terminal.mother.name][MGcats[terminal_index]][0] += 1
                                    if words[terminal_index] not in model1[ccg_terminal.mother.name][MGcats[terminal_index]][1]:
                                        model1[ccg_terminal.mother.name][MGcats[terminal_index]][1][words[terminal_index]] = 1
                                    else:
                                        model1[ccg_terminal.mother.name][MGcats[terminal_index]][1][words[terminal_index]] += 1
                                if MGcats[terminal_index] not in model2[model2_tag]:
                                    model2[model2_tag][MGcats[terminal_index]] = [1, {words[terminal_index]: 1}]
                                else:
                                    model2[model2_tag][MGcats[terminal_index]][0] += 1
                                    if words[terminal_index] not in model2[model2_tag][MGcats[terminal_index]][1]:
                                        model2[model2_tag][MGcats[terminal_index]][1][words[terminal_index]] = 1
                                    else:
                                        model2[model2_tag][MGcats[terminal_index]][1][words[terminal_index]] += 1
                                if MGcats[terminal_index] not in model3[model3_tag]:
                                    model3[model3_tag][MGcats[terminal_index]] = [1, {words[terminal_index]: 1}]
                                else:
                                    model3[model3_tag][MGcats[terminal_index]][0] += 1
                                    if words[terminal_index] not in model3[model3_tag][MGcats[terminal_index]][1]:
                                        model3[model3_tag][MGcats[terminal_index]][1][words[terminal_index]] = 1
                                    else:
                                        model3[model3_tag][MGcats[terminal_index]][1][words[terminal_index]] += 1
                                if MGcats[terminal_index] not in model4[model4_tag]:
                                    model4[model4_tag][MGcats[terminal_index]] = [1, {words[terminal_index]: 1}]
                                else:
                                    model4[model4_tag][MGcats[terminal_index]][0] += 1
                                    if words[terminal_index] not in model4[model4_tag][MGcats[terminal_index]][1]:
                                        model4[model4_tag][MGcats[terminal_index]][1][words[terminal_index]] = 1
                                    else:
                                        model4[model4_tag][MGcats[terminal_index]][1][words[terminal_index]] += 1
                            elif MGtagType in ['supertag', 'hybrid']:
                                #the unigram supertag/hybrid models
                                if str(supertag_indices[terminal_index]) not in model1[ccg_terminal.mother.name]:
                                    model1[ccg_terminal.mother.name][str(supertag_indices[terminal_index])] = [1, {words[terminal_index]: 1}]
                                else:
                                    model1[ccg_terminal.mother.name][str(supertag_indices[terminal_index])][0] += 1
                                    if words[terminal_index] not in model1[ccg_terminal.mother.name][str(supertag_indices[terminal_index])][1]:
                                        model1[ccg_terminal.mother.name][str(supertag_indices[terminal_index])][1][words[terminal_index]] = 1
                                    else:
                                        model1[ccg_terminal.mother.name][str(supertag_indices[terminal_index])][1][words[terminal_index]] += 1
                                if str(supertag_indices[terminal_index]) not in model2[model2_tag]:
                                    model2[model2_tag][str(supertag_indices[terminal_index])] = [1, {words[terminal_index]: 1}]
                                else:
                                    model2[model2_tag][str(supertag_indices[terminal_index])][0] += 1
                                    if words[terminal_index] not in model2[model2_tag][str(supertag_indices[terminal_index])][1]:
                                        model2[model2_tag][str(supertag_indices[terminal_index])][1][words[terminal_index]] = 1
                                    else:
                                        model2[model2_tag][str(supertag_indices[terminal_index])][1][words[terminal_index]] += 1
                                if str(supertag_indices[terminal_index]) not in model3[model3_tag]:
                                    model3[model3_tag][str(supertag_indices[terminal_index])] = [1, {words[terminal_index]: 1}]
                                else:
                                    model3[model3_tag][str(supertag_indices[terminal_index])][0] += 1
                                    if words[terminal_index] not in model3[model3_tag][str(supertag_indices[terminal_index])][1]:
                                        model3[model3_tag][str(supertag_indices[terminal_index])][1][words[terminal_index]] = 1
                                    else:
                                        model3[model3_tag][str(supertag_indices[terminal_index])][1][words[terminal_index]] += 1
                                if str(supertag_indices[terminal_index]) not in model4[model4_tag]:
                                    model4[model4_tag][str(supertag_indices[terminal_index])] = [1, {words[terminal_index]: 1}]
                                else:
                                    model4[model4_tag][str(supertag_indices[terminal_index])][0] += 1
                                    if words[terminal_index] not in model4[model4_tag][str(supertag_indices[terminal_index])][1]:
                                        model4[model4_tag][str(supertag_indices[terminal_index])][1][words[terminal_index]] = 1
                                    else:
                                        model4[model4_tag][str(supertag_indices[terminal_index])][1][words[terminal_index]] += 1
                            elif MGtagType in ['supertag_maxent', 'hybrid_maxent']:
                                #the maxent supertag/hybrid corpora are created here
                                if MGtagType == 'supertag_maxent':
                                    supertag_ref = MGST_REF_table[str(supertag_indices[terminal_index])]
                                else:
                                    supertag_ref = MGHST_REF_table[str(supertag_indices[terminal_index])]
                                tagged_sentence_model1+=words[terminal_index]+"|"+ccg_terminal.mother.name+"|"+supertag_ref+" "
                                tagged_sentence_model2+=words[terminal_index]+"|"+model2_tag+"|"+supertag_ref+" "
                                tagged_sentence_model3+=words[terminal_index]+"|"+model3_tag+"|"+supertag_ref+" "
                                tagged_sentence_model4+=words[terminal_index]+"|"+model4_tag+"|"+supertag_ref+" "
                            elif MGtagType == 'atomic_maxent':
                                #the atomic maxent corpora are created here
                                tag_ref = MGT_REF_table[MGcats[terminal_index]]
                                tagged_sentence_model1+=words[terminal_index]+"|"+ccg_terminal.mother.name+"|"+tag_ref+" "
                                tagged_sentence_model2+=words[terminal_index]+"|"+model2_tag+"|"+tag_ref+" "
                                tagged_sentence_model3+=words[terminal_index]+"|"+model3_tag+"|"+tag_ref+" "
                                tagged_sentence_model4+=words[terminal_index]+"|"+model4_tag+"|"+tag_ref+" "
            else:
                #the lstm corpora are created here..
                if bracketing_index == 0:
                    print '\nCreating development corpus...'
                if bracketing_index == train_start:
                    print '\nCreating training corpus...'
                    current_lstm_words_file = mg_lstm_words_train_corpus_file
                    current_lstm_supertags_file = mg_lstm_supertags_train_corpus_file
                else:
                    if bracketing_index == test_start:
                        print '\nCreating test corpus...'
                        current_lstm_words_file = mg_lstm_words_test_corpus_file
                        current_lstm_supertags_file = mg_lstm_supertags_test_corpus_file
                sentence = (' ').join([ word for word in words ])
                current_lstm_words_file.write(sentence)
                supertagstring = ''
                word_supertagstring = ''
                for i in range(len(words)):
                    supertagstring += MGST_REF_table[str(supertag_indices[i])] + ' '
                    word_supertagstring += words[i] + '|' + MGST_REF_table[str(supertag_indices[i])] + ' '
                supertagstring = supertagstring[:-1]
                word_supertagstring = word_supertagstring[:-1]
                mg_lstm_corpus_file.write(word_supertagstring)
                current_lstm_supertags_file.write(supertagstring)
                if bracketing_index != len(PARSE_list) - 1 and bracketing_index != train_start - 1 and bracketing_index != test_start - 1:
                    current_lstm_words_file.write('\n')
                    current_lstm_supertags_file.write('\n')
                    mg_lstm_corpus_file.write('\n')
            if MGtagType in ['supertag_maxent', 'hybrid_maxent']:
                ccg_mg_supertag_corpus_model1_file.write(tagged_sentence_model1.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_supertag_corpus_model1_file.write('\n')
                ccg_mg_supertag_corpus_model2_file.write(tagged_sentence_model2.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_supertag_corpus_model2_file.write('\n')
                ccg_mg_supertag_corpus_model3_file.write(tagged_sentence_model3.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_supertag_corpus_model3_file.write('\n')
                ccg_mg_supertag_corpus_model4_file.write(tagged_sentence_model4.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_supertag_corpus_model4_file.write('\n')
            elif MGtagType == 'atomic_maxent':
                ccg_mg_tag_corpus_model1_file.write(tagged_sentence_model1.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_tag_corpus_model1_file.write('\n')
                ccg_mg_tag_corpus_model2_file.write(tagged_sentence_model2.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_tag_corpus_model2_file.write('\n')
                ccg_mg_tag_corpus_model3_file.write(tagged_sentence_model3.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_tag_corpus_model3_file.write('\n')
                ccg_mg_tag_corpus_model4_file.write(tagged_sentence_model4.strip())
                if bracketing_index < len_PARSE_list:
                    ccg_mg_tag_corpus_model4_file.write('\n')
        if buildCorpusOnly:
            mg_lstm_words_train_corpus_file.close()
            mg_lstm_supertags_train_corpus_file.close()
            mg_lstm_words_dev_corpus_file.close()
            mg_lstm_supertags_dev_corpus_file.close()
            mg_lstm_words_test_corpus_file.close()
            mg_lstm_supertags_test_corpus_file.close()
            mg_lstm_corpus_file.close()
        #convert the counts into unigram probabilities
        if MGtagType in ['atomic', 'supertag', 'hybrid']:
            for ccg_tag in model1:
                total_counts = 0
                for MG_tag in model1[ccg_tag]:
                    total_counts += model1[ccg_tag][MG_tag][0]
                for MG_tag in model1[ccg_tag]:
                    model1[ccg_tag][MG_tag][0] = float("{0:.4f}".format(model1[ccg_tag][MG_tag][0]/total_counts))
            for ccg_tag in model2:
                total_counts = 0
                for MG_tag in model2[ccg_tag]:
                    total_counts += model2[ccg_tag][MG_tag][0]
                for MG_tag in model2[ccg_tag]:
                    model2[ccg_tag][MG_tag][0] = float("{0:.4f}".format(model2[ccg_tag][MG_tag][0]/total_counts))
            for ccg_tag in model3:
                total_counts = 0
                for MG_tag in model3[ccg_tag]:
                    total_counts += model3[ccg_tag][MG_tag][0]
                for MG_tag in model3[ccg_tag]:
                    model3[ccg_tag][MG_tag][0] = float("{0:.4f}".format(model3[ccg_tag][MG_tag][0]/total_counts))
            for ccg_tag in model4:
                total_counts = 0
                for MG_tag in model4[ccg_tag]:
                    total_counts += model4[ccg_tag][MG_tag][0]
                for MG_tag in model4[ccg_tag]:
                    model4[ccg_tag][MG_tag][0] = float("{0:.4f}".format(model4[ccg_tag][MG_tag][0]/total_counts))
        if MGtagType == 'atomic':
            with open(self.seed_folder+"/CCG_MG_ATOMIC_taggingModel1", 'w') as taggingModel1File:
                json.dump(model1, taggingModel1File)
            with open(self.seed_folder+"/CCG_MG_ATOMIC_taggingModel2", 'w') as taggingModel2File:
                json.dump(model2, taggingModel2File)
            with open(self.seed_folder+"/CCG_MG_ATOMIC_taggingModel3", 'w') as taggingModel3File:
                json.dump(model3, taggingModel3File)
            with open(self.seed_folder+"/CCG_MG_ATOMIC_taggingModel4", 'w') as taggingModel4File:
                json.dump(model4, taggingModel4File)
        elif MGtagType == 'supertag':
            if ns:
                NS = '_NS'
            else:
                NS = ''
            with open(self.seed_folder+"/CCG_MG_SUPERTAG"+NS+"_taggingModel1", 'w') as taggingModel1File:
                json.dump(model1, taggingModel1File)
            with open(self.seed_folder+"/CCG_MG_SUPERTAG"+NS+"_taggingModel2", 'w') as taggingModel2File:
                json.dump(model2, taggingModel2File)
            with open(self.seed_folder+"/CCG_MG_SUPERTAG"+NS+"_taggingModel3", 'w') as taggingModel3File:
                json.dump(model3, taggingModel3File)
            with open(self.seed_folder+"/CCG_MG_SUPERTAG"+NS+"_taggingModel4", 'w') as taggingModel4File:
                json.dump(model4, taggingModel4File)
        elif MGtagType == 'hybrid':
            if ns:
                NS = '_NS'
            else:
                NS = ''
            with open(self.seed_folder+"/CCG_MG_HYBRID"+NS+"_taggingModel1", 'w') as taggingModel1File:
                json.dump(model1, taggingModel1File)
            with open(self.seed_folder+"/CCG_MG_HYBRID"+NS+"_taggingModel2", 'w') as taggingModel2File:
                json.dump(model2, taggingModel2File)
            with open(self.seed_folder+"/CCG_MG_HYBRID"+NS+"_taggingModel3", 'w') as taggingModel3File:
                json.dump(model3, taggingModel3File)
            with open(self.seed_folder+"/CCG_MG_HYBRID"+NS+"_taggingModel4", 'w') as taggingModel4File:
                json.dump(model4, taggingModel4File)
        if MGtagType == 'supertag_lstm':
            with open('SuperSuperTagger/'+data_dir+'/MGST_REF_table', 'w') as (MGST_REF):
                json.dump(MGST_REF_table, MGST_REF)
            with open('SuperSuperTagger/'+data_dir+'/REF_MGST_table', 'w') as (REF_MGST):
                json.dump(REF_MGST_table, REF_MGST)
            with open('SuperSuperTagger/'+data_dir+'/tag_dict', 'w') as tag_dict_file:
                json.dump(tag_dict, tag_dict_file)
            with open('SuperSuperTagger/'+data_dir+'/seed_tag_dict', 'w') as seed_tag_dict_file:
                json.dump(seed_tag_dict, seed_tag_dict_file)
        elif MGtagType == 'supertag_maxent':
            ccg_mg_supertag_corpus_model1_file.close()
            if ns:
                NS2 = 'ns'
            else:
                NS2 = ''
            if not ns:
                with open(self.seed_folder+'/'+'MGST_REF_table', 'w') as MGST_REF:
                    json.dump(MGST_REF_table, MGST_REF)
                with open(self.seed_folder+'/'+'REF_MGST_table', 'w') as REF_MGST:
                    json.dump(REF_MGST_table, REF_MGST)
            else:
                with open(self.seed_folder+'/'+'MGSTNS_REF_table', 'w') as MGST_REF:
                    json.dump(MGST_REF_table, MGST_REF)
                with open(self.seed_folder+'/'+'REF_MGSTNS_table', 'w') as REF_MGST:
                    json.dump(REF_MGST_table, REF_MGST)
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/st'+NS2+'agmaxent.model1 --input '+self.seed_folder+'/ccg_mg_supertagged'+NS+'_model1_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_supertag_corpus_model2_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/st'+NS2+'agmaxent.model2 --input '+self.seed_folder+'/ccg_mg_supertagged'+NS+'_model2_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_supertag_corpus_model3_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/st'+NS2+'agmaxent.model3 --input '+self.seed_folder+'/ccg_mg_supertagged'+NS+'_model3_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_supertag_corpus_model4_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/st'+NS2+'agmaxent.model4 --input '+self.seed_folder+'/ccg_mg_supertagged'+NS+'_model4_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
        elif MGtagType == 'hybrid_maxent':
            if ns:
                NS2 = 'ns'
            else:
                NS2 = ''
            if not ns:
                with open(self.seed_folder+'/'+'MGHST_REF_table', 'w') as MGHST_REF:
                    json.dump(MGHST_REF_table, MGHST_REF)
                with open(self.seed_folder+'/'+'REF_MGHST_table', 'w') as REF_MGHST:
                    json.dump(REF_MGHST_table, REF_MGHST)
            else:
                with open(self.seed_folder+'/'+'MGHSTNS_REF_table', 'w') as MGHST_REF:
                    json.dump(MGHST_REF_table, MGHST_REF)
                with open(self.seed_folder+'/'+'REF_MGHSTNS_table', 'w') as REF_MGHST:
                    json.dump(REF_MGHST_table, REF_MGHST)
            ccg_mg_supertag_corpus_model1_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/hyb'+NS+'maxent.model1 --input '+self.seed_folder+'/ccg_mg_hybrid'+NS+'_model1_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_supertag_corpus_model2_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/hyb'+NS+'maxent.model2 --input '+self.seed_folder+'/ccg_mg_hybrid'+NS+'_model2_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_supertag_corpus_model3_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/hyb'+NS+'maxent.model3 --input '+self.seed_folder+'/ccg_mg_hybrid'+NS+'_model3_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_supertag_corpus_model4_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/hyb'+NS+'maxent.model4 --input '+self.seed_folder+'/ccg_mg_hybrid'+NS+'_model4_corpus --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
        elif MGtagType == 'atomic_maxent':
            with open(self.seed_folder+'/'+'MGT_REF_table', 'w') as MGT_REF:
                json.dump(MGT_REF_table, MGT_REF)
            with open(self.seed_folder+'/'+'REF_MGT_table', 'w') as REF_MGT:
                json.dump(REF_MGT_table, REF_MGT)
            ccg_mg_tag_corpus_model1_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/tagmaxent.model1 --input '+self.seed_folder+'/ccg_mg_tagged_corpus_model1 --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_tag_corpus_model2_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/tagmaxent.model2 --input '+self.seed_folder+'/ccg_mg_tagged_corpus_model2 --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_tag_corpus_model3_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/tagmaxent.model3 --input '+self.seed_folder+'/ccg_mg_tagged_corpus_model3 --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
            ccg_mg_tag_corpus_model4_file.close()
            os.system('./candc-1.00/bin/train_super --comment "" --model '+self.seed_folder+'/tagmaxent.model4 --input '+self.seed_folder+'/ccg_mg_tagged_corpus_model4 --super-category_cutoff '+str(self.super_category_cutoff)+' --super-rare_cutoff '+str(self.super_rare_cutoff)+' --super-tagdict_min '+str(self.super_tag_dict_min)+' --super-forward_beam_ratio '+str(self.super_forward_beam_ratio)+' --super-beam_width '+str(self.super_beam_width)+' --super-beam_ratio '+str(self.super_beam_ratio))
        if not buildCorpusOnly:
            print '\nSuccessfully extracted models...'
        return

    def get_ccg_tree_score(self, node, tree_score):
        if len(node.daughters) == 1:
            tree_score -= 1
        for daughter in node.daughters:
            tree_score = self.get_ccg_tree_score(daughter, tree_score)
        return tree_score

    def extractOvertLexicon(self, fromAutoGen=False, fromCorporaStats=False, useAutos=False):
        self.seed_word_token_count = 0
        self.auto_word_token_count = 0
        if not useAutos:
            print "\nExtracting overt lexicon from seeds...\n"
        else:
            print "\nExtracting overt lexicon from seeds and autos...\n"
        if self.overwrite_auto == None:
            if self.constructLexiconWindow != None:
                self.destroyWindow(self.constructLexiconWindow, 'constructLexiconWindow')
        try:
            autoTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
        except IOError:
            autoTreeCatMappings = {}
        try:
            TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
        except IOError:
            TreeCatMappings = {}
        del(self.nameCats['male'][:])
        del(self.nameCats['female'][:])
        seed_lexicon = {}
        auto_lexicon = {}
        self.getLexicon(TreeCatMappings, 'seeds', seed_lexicon)
        if useAutos:
            self.getLexicon(autoTreeCatMappings, 'autos', auto_lexicon)
        with open(self.seed_folder+"/"+'OvertLexicon', 'w') as overtLexiconFile:
            json.dump(seed_lexicon, overtLexiconFile)
        with open(self.auto_folder+"/"+'OvertLexicon', 'w') as overtLexiconFile:
            json.dump(auto_lexicon, overtLexiconFile)
        if not fromAutoGen and not fromCorporaStats:
            self.nothing(self.mainWindow, "Overt lexicon successfully constructed.")
        print "Overt lexicon successfully constructed..\n"
        return

    def getLexicon(self, TCM, corpus_type, lexicon):
        for PARSE in TCM:
            parse = ast.literal_eval(PARSE)
            MGFileName = parse[0].encode('ascii')
            seed_line_num = parse[1]
            MGcats = TCM[PARSE]
            parts = MGFileName.split('/')
            if parts[1] != 'new_parses':
                parseStringFile = open('./wsj_strings/'+parts[1]+'/'+parts[2]+"_strings", 'r')
                lines = parseStringFile.readlines()
                line = lines[seed_line_num]
                words = line.split(" ")
                if corpus_type == 'seeds':
                    self.seed_word_token_count += len(words)
                else:
                    self.auto_word_token_count += len(words)
            else:
                parseStringFile = open(self.seed_folder+'/new_parse_strings/'+parts[2], 'r')
                for line in parseStringFile:
                    words = line.split(" ")
                    break
            index = -1
            for word in words:
                index += 1
                words[index] = re.sub("&", "ANDANDAND", words[index], count=5)
                words[index] = tokenize.word_tokenize(words[index].lower())[0]
                words[index] = re.sub("ANDANDAND", "&", words[index], count=5)
            index = -1
            for word in words:
                index += 1
                if word in lexicon:
                    if MGcats[index] not in lexicon[word]:
                        lexicon[word].append(MGcats[index])
                else:
                    lexicon[word] = [MGcats[index]]
                if corpus_type == 'seeds':
                    if word in male_names:
                        self.nameCats['male'].append(MGcats[index])
                    elif word in female_names:
                        self.nameCats['female'].append(MGcats[index])
        if corpus_type == 'seeds':
            with open(self.seed_folder+"/nameCats", 'w') as nameCatsFile:
                json.dump(self.nameCats, nameCatsFile)

    def extractMGsupertags(self, subcat_full_derivation_bracketing, derivation_bracketing, ignore_null_c=False, ns=False):
        sf_derivation_tree = gen_derived_tree.gen_derivation_tree(subcat_full_derivation_bracketing)
        derivation_tree = gen_derived_tree.gen_derivation_tree(derivation_bracketing)
        if ns:
            autobank.strip_derivation_tree(sf_derivation_tree)
        self.set_derivation_tree_heads(sf_derivation_tree)
        merge_links = []
        supertags = []
        anchors = []
        self.get_overt_indices(derivation_tree, sf_derivation_tree)
        self.get_merge_links(sf_derivation_tree, merge_links, ignore_null_c=ignore_null_c)
        self.construct_supertags(merge_links, supertags)
        self.fix_unanchored_tags(supertags, anchors, ignore_null_c=ignore_null_c, link_with='commanding')
        anchors = []
        self.fix_unanchored_tags(supertags, anchors, ignore_null_c=ignore_null_c, link_with='complement')
        anchord = []
        #the third time round this function is executed simply to get the new set of anchors
        self.fix_unanchored_tags(supertags, anchors, ignore_null_c=ignore_null_c)
        self.check_links(supertags)
        self.add_overt_only_supertags(sf_derivation_tree, anchors, supertags)
        return supertags

    def get_overt_indices(self, derivation_tree, sf_derivation_tree):
        tree_copy = copy.deepcopy(derivation_tree)
        self.assign_id_to_overt_terminals(tree_copy, 0)
        new_bracketing = tree_copy.generate_bracketing()
        new_bracketing = new_bracketing.encode('utf8')[1:-1]
        new_bracketing = re.sub(" \(", "(", new_bracketing, count=100000)
        new_bracketing = re.sub(" \)", ")", new_bracketing, count=100000)
        (X, Y, xbar_tree) = gen_derived_tree.main(new_bracketing, show_indices=True, return_xbar_tree=True, allowMoreGoals=True)
        mappings = {}
        self.get_id_indices_mappings(xbar_tree, mappings, 0)
        self.assign_indices_to_overt_terminals(sf_derivation_tree, 0, mappings, xbar_tree, new_bracketing)
        return mappings

    def assign_indices_to_overt_terminals(self, node, derivation_tree_index, id_indices_mappings, xbar_tree, new_bracketing):
        if len(node.daughters) != 0:
            for daughter in node.daughters:
                derivation_tree_index = self.assign_indices_to_overt_terminals(daughter, derivation_tree_index, id_indices_mappings, xbar_tree, new_bracketing)
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

    def get_id_indices_mappings(self, node, mappings, index):
        if len(node.daughters) > 0:
            for daughter in node.daughters:
                index = self.get_id_indices_mappings(daughter, mappings, index)
        else:
            try:
                int(node.name[1:-1])
                mappings[node.name[1:-1]] = str(index)
                index += 1
            except ValueError:
                x=0
        return index

    def assign_id_to_overt_terminals(self, node, ID):
        if len(node.daughters) != 0:
            for daughter in node.daughters:
                ID = self.assign_id_to_overt_terminals(daughter, ID)
        elif node.name[0] != '[':
            parts = node.name.split(" ")
            parts[0] = "*"+str(ID)+"*"
            node.name = " ".join(parts)
            ID += 1
        return ID
            
    def displaySupertags(self, supertags):
        supertagNum = 0
        for supertag in supertags:
            print "*********************************************"
            supertagNum += 1
            print "Supertag "+str(supertagNum)+":\n"
            if type(supertag) != type([]):
                print supertag.name
            else:
                for link in supertag:
                    print (link[0].name, link[1], link[2].name, link[3])

    def add_overt_only_supertags(self, node, anchors, supertags):
        #overt catgories that are not anchors for any null categories are added to the supertag
        #list at this point..
        if len(node.daughters) > 0:
            for daughter in node.daughters:
                self.add_overt_only_supertags(daughter, anchors, supertags)
        elif node.name[0] != '[':
            if node not in anchors:
                supertags.append(node)

    def check_links(self, supertags, fixing_unanchored=False):
        #this function just checks to make sure that every supertag has one and only one overt category in it
        for supertag in supertags:
            overt_cat = None
            for link in supertag:
                if link[0].name[0] != '[':
                    if overt_cat == None:
                        overt_cat = link[0]
                    elif link[0] != overt_cat:
                        raise Exception("Oops!  MG supertag detected with multiple overt anchors!")
                if link[2].name[0] != '[':
                    if overt_cat == None:
                        overt_cat = link[2]
                    elif link[2] != overt_cat:
                        raise Exception("Oops!  MG supertag detected with multiple overt anchors!")
            if overt_cat == None and not fixing_unanchored:
                raise Exception("Oops!  MG supertag detected without overt anchor!")
            elif overt_cat == None:
                return False
            else:
                return True

    def fix_unanchored_tags(self, supertags, anchors, ignore_null_c=False, link_with='commanding'):
        #in principle, we could have extended projections without any overt category, e.g. a complex version of pro
        #we need to link this empty ep with the ep on which it is dependent.  (Main clause empty eps are assumed not to exist here)
        for supertag in supertags:
            found_overt_category = False
            for link in supertag:
                if link[0].name[0] != '[':
                    found_overt_category = True
                    anchors.append(link[0])
                    break
                elif link[2].name[0] != '[':
                    found_overt_category = True
                    anchors.append(link[2])
                    break                                  
            if not found_overt_category:
                SUPERTAG = []
                for link in supertag:
                    SUPERTAG.append(link)
                if link_with == 'commanding':
                    for link in SUPERTAG:
                        if self.link_with_commanding_head(link[0], supertag, ignore_null_c):
                            self.merge_supertags(supertags)
                            break
                        elif self.link_with_commanding_head(link[2], supertag, ignore_null_c):
                            self.merge_supertags(supertags)
                            break
                elif link_with == 'complement':
                    if self.link_with_complement(link[0], supertag, ignore_null_c):
                        self.merge_supertags(supertags)
                        break
                    elif self.link_with_complement(link[2], supertag, ignore_null_c):
                        self.merge_supertags(supertags)
                        break

    def merge_supertags(self, supertags):
        #when fixing unanchored tags it is possible to end up with two tags containing the same
        #atomic category (null or overt).. these tags should be unified.
        unified_tags = True
        while unified_tags:
            unified_tags = False
            for supertag in supertags:
                if unified_tags:
                    break
                tag1_cats = []
                for link in supertag:
                    tag1_cats.append(link[0])
                    tag1_cats.append(link[2])
                for SUPERTAG in supertags:
                    if unified_tags:
                        break
                    if supertag == SUPERTAG:
                        continue
                    for link in SUPERTAG:
                        if link[0] in tag1_cats or link[2] in tag1_cats:
                            for LINK in SUPERTAG:
                                if LINK not in supertag:
                                    supertag.append(LINK)
                            supertags.remove(SUPERTAG)
                            unified_tags = True
                            break
                
    def construct_supertags(self, merge_links, supertags):
        while len(merge_links) > 0:
            supertags.append([merge_links[0]])
            del(merge_links[0])
            added_link = True
            while added_link:
                added_link = False
                for link in merge_links:
                    if added_link:
                        break
                    for LINK in supertags[-1]:
                        if link[0] in [LINK[0], LINK[2]]:
                            supertags[-1].append(link)
                            merge_links.remove(link)
                            added_link = True
                            break
                        elif link[2] in [LINK[0], LINK[2]]:
                            supertags[-1].append(link)
                            merge_links.remove(link)
                            added_link = True
                            break     

    def get_merge_links(self, node, merge_links, ignore_null_c=False):
        if len(node.daughters) != 0:
            for daughter in node.daughters:
                self.get_merge_links(daughter, merge_links, ignore_null_c=ignore_null_c)
        else:
            if node.name[0] == '[':
                null_head_features = self.get_derivation_tree_features(node)
                if ignore_null_c:
                    if node.name.split(" ")[0] in ['[topicalizer]', '[focalizer]', '[relativizer]', '[wh]']:
                        return
                    for feature in null_head_features:
                        try:
                            if '\xe2\x89\x88' not in feature.encode('utf8'):
                                if cat_pattern.search(feature).group(0).lower() == 'c':
                                    return
                        except Exception as e:
                            x=0
                if '=' not in null_head_features[0]:
                    #i.e. this node is the top head in its extended projection, and it does not
                    #have any complement to link with, e.g. pro
                    self.link_with_commanding_head(node, merge_links, ignore_null_c)
                    return
                else:
                    if '[pro-' in node.name:
                        #i.e. this node is the bottom head in its extended projection
                        #and so must link with something that c-commands it, rather than something it governs
                        self.link_with_commanding_head(node, merge_links, ignore_null_c)
                        return
                    self.link_with_complement(node, merge_links, ignore_null_c)
        MERGE_LINKS = []
        for ml in merge_links:
            if ml not in MERGE_LINKS:
                MERGE_LINKS.append(ml)
        del(merge_links[:])
        for ml in MERGE_LINKS:
            merge_links.append(ml)

    def link_with_complement(self, node, merge_links, ignore_null_c=False):
        daughter_index = node.mother.daughters.index(node)
        if daughter_index == 0:
            sister = node.mother.daughters[1]
        else:
            sister = node.mother.daughters[0]
        checked_sister_feature = len(self.get_derivation_tree_features(sister.lex_head)) - len(self.get_derivation_tree_features(sister))
        if ignore_null_c and sister.lex_head.name[0] == '[':
            if sister.lex_head.name.split(" ")[0] in ['[topicalizer]', '[focalizer]', '[relativizer]', '[wh]']:
                return False
            for feature in self.get_derivation_tree_features(sister.lex_head):
                if '\xe2\x89\x88' not in feature.encode('utf8'):
                    if cat_pattern.search(feature).group(0).lower() == 'c':
                        return False
        if [node, 0, sister.lex_head, checked_sister_feature] not in merge_links:
            merge_links.append([node, 0, sister.lex_head, checked_sister_feature])
                
    def link_with_commanding_head(self, node, merge_links, ignore_null_c=False):
        governor_mother = node
        max_projection_governee = node
        while governor_mother.lex_head == node:
            max_projection_governee = governor_mother
            gm = governor_mother
            governor_mother = governor_mother.mother
            try:
                governor_mother.lex_head
            except Exception as e:
                #this can happen in the case of fragments, which may have a [pro-x] as their uppermost head
                return False
        for daughter in governor_mother.daughters:
            if daughter != max_projection_governee:
                governor = daughter
                break
        governor_features = self.get_derivation_tree_features(governor_mother.lex_head)
        governee_features = self.get_derivation_tree_features(node)
        checked_governee_feature = len(self.get_derivation_tree_features(node)) - len(self.get_derivation_tree_features(max_projection_governee))
        checked_governor_feature = len(governor_features) - len(self.get_derivation_tree_features(governor))
        #we don't want to create links via ≈ adjunctizer features
        if ignore_null_c and governor.lex_head.name[0] == '[':
            if governor.lex_head.name.split(" ")[0] in ['[topicalizer]', '[focalizer]', '[relativizer]', '[wh]']:
                return False
            for feature in self.get_derivation_tree_features(governor.lex_head):
                if '\xe2\x89\x88' not in feature.encode('utf8'):
                    if cat_pattern.search(feature).group(0).lower() == 'c':
                        return False
        if '\xe2\x89\x88' in governee_features[checked_governee_feature].encode('utf8') or '\xe2\x89\x88' in governor_features[checked_governor_feature].encode('utf8'):
            return False
        if [governor_mother.lex_head, checked_governor_feature, node, checked_governee_feature] not in merge_links:
            merge_links.append([governor_mother.lex_head, checked_governor_feature, node, checked_governee_feature])
            return True
        else:
            return False

    def set_derivation_tree_heads(self, node):
        if len(node.daughters) != 0:
            for daughter in node.daughters:
                self.set_derivation_tree_heads(daughter)
        else:
            head = node
            head.lex_head = head
            current_features = self.get_derivation_tree_features(head)
            current_stripped_features = [re.sub('{.*?}', '', f) for f in current_features]
            current_node = head
            current_node_adjoinee = self.node_is_adjoinee(current_node)
            #the cut off point at which a chain transitions from being a head to a moving chain
            #is always the merge operation involving either its selectee feature, or, in the case of adjuncts, its selector feature..
            while current_node.mother != None and (len(current_node.mother.daughters) == 1 or (not_selectee_feature.search(current_stripped_features[0]) or current_node_adjoinee) and not u'\u2248' in current_stripped_features[0]):
                current_node = current_node.mother
                current_node.lex_head = head
                current_features = self.get_derivation_tree_features(current_node)
                current_stripped_features = [re.sub('{.*?}', '', f) for f in current_features]
                current_node_adjoinee = self.node_is_adjoinee(current_node)

    def node_is_adjoinee(self, node):
        if node.mother != None and len(node.mother.daughters) == 2:
            for daughter in node.mother.daughters:
                if daughter != node:
                    sister_features = self.get_derivation_tree_features(daughter)
                    if u'\u2248' in sister_features[0]:
                        return True
        return False

    def get_derivation_tree_features(self, node, return_name_and_type=False):
        head_chain = node.name.split(", ")[0]
        char_index = len(head_chain)
        #to find the start position of the typing separator :, :: etc, we go right to left to ensure we don't misidentify an actual colon in the string
        while char_index != -1:
            char_index -= 1
            if head_chain[char_index] == ":":
                if head_chain[char_index+1] != " ":
                    char_index += 1
                char_index += 1
                break
        head_string = head_chain[0:char_index]
        if u':\u0305:\u0305' in head_string:
            parts = head_string.split(u':\u0305:\u0305')
            TYPE = u':\u0305:\u0305'
            name = parts[0].strip()
        elif u':\u0305' in head_string:
            parts = head_string.split(u':\u0305')
            TYPE = u':\u0305'
            name = parts[0].strip()
        elif '::' in head_string:
            parts = head_string.split('::')
            TYPE = '::'
            name = parts[0].strip()
        elif ':' in head_string:
            parts = head_string.split(':')
            TYPE = ':'
            name = parts[0].strip()
        else:
            raise Exception("Oops! Failed to find separator on derivation tree node..")
        features = head_chain[char_index:].strip()
        if not return_name_and_type:
            return features.split(" ")
        else:
            features = features.split(" ")
            features.insert(0, TYPE)
            return (features, name)
        
    def confirm_reparse_all_trees(self, singleTree=None, corpus=None):
        if self.confirmReparseWindow != None:
            self.destroyWindow(self.confirmReparseWindow, 'confirmReparseWindow')
        self.confirmReparseWindow = Toplevel(self.mainWindow)
        self.confirmReparseWindow.title("Confirm Reparse "+corpus)
        self.confirmReparseWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.confirmReparseWindow, 'confirmReparseWindow'))
        w = 600
        h = 200
        (x, y) = self.getCentrePosition(w, h)
        self.confirmReparseWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.confirmReparseWindow, text="Which type of tags do you want the parser to use?")
        label.pack()
        buttonFrame = Frame(self.confirmReparseWindow)
        buttonFrame.pack()
        corpus = corpus.lower()
        supertagButton = Button(buttonFrame, text="MG supertags", command=lambda: self.overwrite_derivation_trees_question(compose='supertags', failed_parses=[], different_parses=[], singleTree=singleTree, corpus=corpus))
        supertagButton.pack()
        supertagButtonNS = Button(buttonFrame, text="MG supertags without subcat features", command=lambda: self.overwrite_derivation_trees_question(compose='supertagsNS', failed_parses=[], different_parses=[], singleTree=singleTree, corpus=corpus))
        supertagButtonNS.pack()
        compButton = Button(buttonFrame, text="atomic (null and overt) MG categories", command=lambda: self.overwrite_derivation_trees_question(compose='fullComp', failed_parses=[], different_parses=[], singleTree=singleTree, corpus=corpus))
        compButton.pack()
        hybButton = Button(buttonFrame, text="hybrid1: A'-movement excluded", command=lambda: self.overwrite_derivation_trees_question(compose='hybrid', failed_parses=[], different_parses=[], singleTree=singleTree, corpus=corpus))
        hybButton.pack()
        #tried to combine hybrid with reduced subcat approach but couldn't because too many cases of infinite recursion came up..
        hybNSButton = Button(buttonFrame, text="hybrid2: hybrid1 without subcat features", command=lambda: self.overwrite_derivation_trees_question(compose='hybridNS', failed_parses=[], different_parses=[], singleTree=singleTree, corpus=corpus))
        hybNSButton.pack()
        cancelButton = Button(buttonFrame, text="cancel", command=lambda: self.destroyWindow(self.confirmReparseWindow, 'confirmReparseWindow'))
        cancelButton.pack()

    def overwrite_derivation_trees_question(self, compose="", failed_parses=None, different_parses=None, singleTree=None, corpus=None):
        if self.confirmReparseWindow != None:
            self.destroyWindow(self.confirmReparseWindow, 'confirmReparseWindow')
        self.overwriteDerivationWindow = Toplevel(self.mainWindow)
        self.overwriteDerivationWindow.title("Overwrite Derivation Trees? "+corpus)
        self.overwriteDerivationWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.overwriteDerivationWindow, 'overwriteDerivationWindow'))
        w = 450
        h = 75
        (x, y) = self.getCentrePosition(w, h)
        self.overwriteDerivationWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.overwriteDerivationWindow, text="Where the correct Xbar tree is generated, do you\nwish to overwrite the currently saved derivation trees?")
        label.pack()
        buttonFrame = Frame(self.overwriteDerivationWindow)
        buttonFrame.pack()
        yesButton = Button(buttonFrame, text="yes", command=lambda: self.reparse_all_trees(compose=compose, failed_parses=failed_parses, different_parses=different_parses, singleTree=singleTree, corpus=corpus, overwrite_derivation=True))
        yesButton.pack(side=LEFT)
        noButtonNS = Button(buttonFrame, text="no", command=lambda: self.reparse_all_trees(compose=compose, failed_parses=failed_parses, different_parses=different_parses, singleTree=singleTree, corpus=corpus, overwrite_derivation=False))
        noButtonNS.pack(side=LEFT)
        cancelButton = Button(buttonFrame, text="cancel", command=lambda: self.destroyWindow(self.overwriteDerivationWindow, 'overwriteDerivationWindow'))
        cancelButton.pack(side=LEFT)

    def reparse_all_trees(self, construct_lexicon=False, compose='fullComp', times=[], start_file=None, start_line_num=None, failed_parses=None, different_parses=None, singleTree=None, continue_parsing=False, total_start_time=None, parse_times=None, break_time_start=None, corpus=None, overwrite_derivation=False):
        if self.overwriteDerivationWindow != None:
            self.destroyWindow(self.overwriteDerivationWindow, 'overwriteDerivationWindow')
        if self.viewParseQuestionWindow != None:
            self.destroyWindow(self.viewParseQuestionWindow, 'viewParseQuestionWindow')
        if corpus == 'seeds':
            try:
                TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
            except IOError:
                TreeCatMappings = {}
        elif corpus == 'autos':
            try:
                TreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
            except IOError:
                TreeCatMappings = {}
        TreeList = []
        for PARSE in TreeCatMappings:
            TreeList.append(PARSE)
        #For line numbers with only one or two digits, e.g. 3, we need to add zeros (e.g. to 003) so that sorting works properly, but then we need to strip it out
        index = -1
        for PARSE in TreeList:
            index += 1
            parts = PARSE.split(" ")
            if len(parts[1]) == 2:
                parts[1] = "00"+parts[1]
            elif len(parts[1]) == 3:
                parts[1] = "0"+parts[1]
            TreeList[index] = " ".join(parts)
        TreeList.sort()
        index = -1
        if len(TreeList) == 0:
            if corpus == 'autos':
                self.nothing(self.mainWindow, "No auto trees to reparse!")
            else:
                self.nothing(self.mainWindow, "No seed trees to reparse!")
            return
        for PARSE in TreeList:
            index += 1
            parts = PARSE.split(" ")
            if parts[1][0] == "0":
                parts[1] = parts[1][1:]
            if parts[1][0] == "0":
                parts[1] = parts[1][1:]
            TreeList[index] = " ".join(parts)
        if times == None:
            times = []
        if total_start_time == None:
            total_start_time = default_timer()
        else:
            #we get here if a parse failed but the user chose to continue parsing the rest of the seeds
            break_time_end = default_timer()
            break_time = break_time_end - break_time_start
            total_start_time=total_start_time+break_time
        found_start_parse = False
        #to start reparsing from a specific file and line number, uncomment and edit the following two lines, inserting the parse you want to start on (if in Autobank the line is 3 then enter 2, i.e. one before the one you want to start on as here lines start from 0):
        #start_file = 'wsj_MGbankAuto/05/wsj_0515.mrg'
        #start_line_num = 2
        if parse_times == None:
            parse_times = []
        if 'supertags' in compose:
            ignore_null_c = False
            null_c_lexicon = None
        elif 'hybrid' in compose:
            ignore_null_c = True
            null_c_lexicon = []
            for lexicon in [self.CovertLexicon, self.ExtraposerLexicon, self.TypeRaiserLexicon, self.ToughOperatorLexicon, self.NullExcorporatorLexicon]:
                autobank.get_null_c_lexicon(lexicon, null_c_lexicon, compose)
        for PARSE in TreeList:
            parse = ast.literal_eval(PARSE)
            miniOvertLexicon = []
            MGFileName = parse[0].encode('ascii')
            seeds = json.load(open(MGFileName))
            seed_line_num = parse[1]
            if singleTree != None:
                if found_start_parse:
                    break
                elif singleTree[0] != parse[0] or singleTree[1] != parse[1]:
                    times = [0]
                    continue
                else:
                    found_start_parse = True
            if start_file != None:
                if not found_start_parse and not (MGFileName == start_file and seed_line_num == start_line_num):
                    if times == []:
                        times=[0]
                        parse_times=[0]
                    continue
                elif not found_start_parse:
                    found_start_parse = True
                    if seed_line_num != start_line_num:
                        continue
            if continue_parsing:
                continue_parsing = False
                continue
            parserSettings = seeds[str(seed_line_num)][6]
            parts = MGFileName.split('/')
            if parts[1] != "new_parses":
                ptbFileName = self.ptb_folder+"/"+parts[1]+"/"+parts[2]
                PTBparseFile = open(ptbFileName, 'r')
                current_line = -1
                words = None
                for line in PTBparseFile:
                    current_line += 1
                    if current_line == seed_line_num:
                        PTB_TREE = autobank.build_tree(line)
                        PTB_tree = PTB_TREE[0]
                        terminals = PTB_TREE[2]
                        words = [t.name.lower() for t in terminals]
                        vp_ellipsis = autobank.contains_vp_ellipsis(PTB_tree)
                        if self.parserSettings["constrainMoveWithPTB"]:
                            moveable_spans = []
                            autobank.get_moveable_spans(PTB_tree, terminals, moveable_spans)
                        else:
                            moveable_spans = None
                        if self.parserSettings["constrainConstWithPTBCCG"]:
                            source_spans = []
                            try:
                                ccg_parses = json.load(open("CCGbank/"+parts[1]+"/"+parts[2].split(".")[0]+".ccg"))
                                ccg_bracketing = ccg_parses[parts[2].split(".")[0]+"."+str(current_line+1)]
                                ccg_tree = autobank.build_tree(ccg_bracketing)
                                ccg_terminals = ccg_tree[2]
                                autobank.set_indices(ccg_terminals)
                                ccg_tree = ccg_tree[0]
                            except Exception as e:
                                ccg_tree = None
                            autobank.get_source_spans(PTB_tree, ccg_tree, source_spans, terminals)
                        else:
                            source_spans = None
                        break
                PTBparseFile.close()
            else:
                #as we have no PTB tree to inspect in this case we just set vp_ellipsis to True as we do
                #not want the parser to block [pro-v] categories..
                vp_ellipsis = True
                source_spans = None
                moveable_spans = None
                newSentFileName = self.seed_folder+"/"+parts[1]+"/"+parts[2]
                new_parse_file = open(self.seed_folder+"/new_parse_strings/"+parts[2])
                for line in new_parse_file:
                    words = line.split(" ")
                    break
            if compose == 'fullComp':
                MGcats = TreeCatMappings[unicode(PARSE)]
                i=-1
                for word in words:
                    i+=1
                    words[i] = words[i].lower()
                i=-1
                for word in words:
                    i+=1
                    features = MGcats[i].split(" ")
                    lexEntry = self.constructMGlexEntry(word, features)
                    if lexEntry != None:
                        miniOvertLexicon.append([lexEntry, i])
            elif 'supertags' in compose or 'hybrid' in compose:
                old_sfd_bracketing = seeds[str(seed_line_num)][3]
                old_d_bracketing = seeds[str(seed_line_num)][4]
                if 'NS' in compose:
                    ns = True
                else:
                    ns = False
                supertags = self.extractMGsupertags(old_sfd_bracketing, old_d_bracketing, ignore_null_c, ns=ns)
                supertag_index = -1
                for supertag in supertags:
                    cat_ids = {}
                    ID = 0
                    supertag_index += 1
                    if type(supertag) != type([]):
                        if supertag not in cat_ids:
                            cat_ids[supertag] = ID
                            ID += 1
                        (features, word) = self.get_derivation_tree_features(supertag, return_name_and_type=True)
                        if int(supertag.index) == -1:
                            raise Exception("Error! Null category detected that is not anchored to an overt category!")
                        else:
                            SUPERTAG = [self.constructMGlexEntry(word, features), cat_ids[supertag], int(supertag.index)]
                        supertags[supertag_index] = SUPERTAG
                    else:
                        for link in supertag:
                            if link[0] not in cat_ids:
                                cat_ids[link[0]] = ID
                                ID += 1
                            (features, word) = self.get_derivation_tree_features(link[0], return_name_and_type=True)
                            link[0] = [self.constructMGlexEntry(word, features), cat_ids[link[0]], int(link[0].index)]
                            if link[2] not in cat_ids:
                                cat_ids[link[2]] = ID
                                ID += 1
                            (features, word) = self.get_derivation_tree_features(link[2], return_name_and_type=True)
                            link[2] = [self.constructMGlexEntry(word, features), cat_ids[link[2]], int(link[2].index)]
                supertag_duplicates = []
                #duplicates arise when the tree has ATB movement so a given lexical item appears twice in the derivation tree
                for st in supertags:
                    if supertags.count(st) > 1:
                        if [st, supertags.count(st)] not in supertag_duplicates:
                            supertag_duplicates.append([st, supertags.count(st)])
                for st in supertag_duplicates:
                    while st[1] > 1:
                        supertags.remove(st[0])
                        st[1] -= 1
            try:
                self.mainWindow.withdraw()
                timeAndDate = {'time':time.strftime("%H:%M:%S"), 'date':time.strftime("%d/%m/%Y")}
                if compose == 'fullComp':
                    if parts[1] != "new_parses":
                        print "\nReparsing PTB sentence:", ptbFileName, "Ln:", str(seed_line_num+1)+',', 'using parser setting:', parserSettings+", and composition strategy: fully compositional, "+"on "+timeAndDate['date']+" at "+timeAndDate['time']
                    else:
                        print "\nReparsing new sentence:", newSentFileName+',', 'using parser setting:', parserSettings+", and composition strategy: fully compositional, "+"on "+timeAndDate['date']+" at "+timeAndDate['time']
                    with timeout(self.parserSettings['timeout_seconds']):
                        self.parseMessage()
                        start_time = default_timer()
                        PARSERSETTINGS = parserSettings
                        if 'UseAllNull' in parserSettings:
                            useAllNull = True
                            PARSERSETTINGS = PARSERSETTINGS[:-10]
                        else:
                            useAllNull = False
                        if 'SkipRel' in parserSettings:
                            skipRel = True
                            PARSERSETTINGS = PARSERSETTINGS[:-7]
                        else:
                            skipRel = False
                        if 'SkipPro' in parserSettings:
                            skipPro = True
                            PARSERSETTINGS = PARSERSETTINGS[:-7]
                        else:
                            skipPro = False
                        if PARSERSETTINGS == 'basicOnly':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                        elif PARSERSETTINGS == 'fullFirst':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                        elif PARSERSETTINGS == 'basicAndRight':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                        elif PARSERSETTINGS == 'basicAndExcorp':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                        elif PARSERSETTINGS == 'basicAndTough':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                        end_time = default_timer() - start_time
                        parse_times.append(parse_time)
                    self.destroyWindow(self.parsing, 'parsing')
                    self.mainWindow.deiconify()
                elif 'supertags' in compose or 'hybrid' in compose:
                    if compose == 'supertags':
                        composition_strategy = 'MG supertags (with subcat features)'
                    elif compose == 'supertagsNS':
                        composition_strategy = 'MG supertags (without subcat features)'
                    elif compose == 'hybridNS':
                        composition_strategy = 'hybrid: null c heads + supertags (without subat features)'
                    else:
                        composition_strategy = 'hybrid: null c heads + supertags (with subcat features)'
                    if parts[1] != "new_parses":
                        print "\nReparsing PTB sentence:", ptbFileName, "Ln:", str(seed_line_num+1)+',', 'using parser setting:', parserSettings+", and composition strategy: "+composition_strategy+", "+"on "+timeAndDate['date']+" at "+timeAndDate['time']
                    else:
                        print "\nReparsing new sentence:", newSentFileName+',', 'using parser setting:', parserSettings+", and composition strategy: "+composition_strategy+", "+"on "+timeAndDate['date']+" at "+timeAndDate['time']
                    with timeout(self.parserSettings['timeout_seconds']):
                        self.parseMessage()
                        start_time = default_timer()
                        PARSERSETTINGS = parserSettings
                        if 'UseAllNull' in parserSettings:
                            PARSERSETTINGS = PARSERSETTINGS[:-10]
                        if 'SkipRel' in parserSettings:
                            PARSERSETTINGS = PARSERSETTINGS[:-7]
                        if 'SkipPro' in parserSettings:
                            PARSERSETTINGS = PARSERSETTINGS[:-7]
                        if PARSERSETTINGS == 'basicOnly':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                        elif PARSERSETTINGS == 'fullFirst':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                        elif PARSERSETTINGS == 'basicAndRight':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                        elif PARSERSETTINGS == 'basicAndExcorp':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                        elif PARSERSETTINGS == 'basicAndTough':
                            (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans)
                        end_time = default_timer() - start_time
                        parse_times.append(parse_time)
                    self.destroyWindow(self.parsing, 'parsing')
                    self.mainWindow.deiconify()
            except IOError:#Exception as e:
                self.mainWindow.deiconify()
                if self.parsing != None:
                    self.destroyWindow(self.parsing, 'parsing')
                print "Error!!!"
                subcat_derivation_bracketings = []
            if len(subcat_derivation_bracketings) == 0:
                success = False
                print "\nThe parser failed to produce any parses for seed tree:\n"+parse[0]+" ln: "+str(parse[1]+1)
                message = "\nThe parser failed to produce any parses for seed tree:\n"+parse[0]+" ln: "+str(parse[1]+1)+"\n(hint: Try increasing the timeout value).."
                failed_parses.append(parse)
                break_time_start = default_timer()
                self.viewParseQuestion(message, parse, reason="failed", times=times, construct_lexicon=construct_lexicon, compose=compose, failed_parses=failed_parses, different_parses=different_parses, singleTree=singleTree, total_start_time=total_start_time, parse_times=parse_times, break_time_start=break_time_start, corpus=corpus, overwrite_derivation=overwrite_derivation)
                return
            old_xbar_bracketing = seeds[str(seed_line_num)][1]
            old_sd_bracketing = seeds[str(seed_line_num)][0]
            with open('Junk', 'w') as JunkFile:
                #making sure the derivation bracketings are in the same format as those in the seed
                #files.. json makes certain changes..
                json.dump(xbar_bracketings, JunkFile)
            new_xbar_bracketings = json.load(open('Junk'))
            with open('Junk', 'w') as JunkFile:
                json.dump(subcat_derivation_bracketings, JunkFile)
            new_subcat_derivation_bracketings = json.load(open('Junk'))
            if old_xbar_bracketing not in new_xbar_bracketings:
                print len(new_xbar_bracketings)
                pdb.set_trace()
                print "\nThe parser produced different\nparses for seed tree:\n"+parse[0]+" ln: "+str(parse[1]+1)
                message = "\nThe parser produced different\nparses for seed tree:\n"+parse[0]+" ln: "+str(parse[1]+1)+"\n(hint: Were the parser settings appropriate?)."
                different_parses.append(parse)
                break_time_start = default_timer()
                self.viewParseQuestion(message, parse, reason="different", times=times, construct_lexicon=construct_lexicon, compose=compose, failed_parses=failed_parses, different_parses=different_parses, singleTree=singleTree, total_start_time=total_start_time, parse_times=parse_times, break_time_start=break_time_start, corpus=corpus, overwrite_derivation=overwrite_derivation)
                return
            elif overwrite_derivation:
                i = new_xbar_bracketings.index(old_xbar_bracketing)
            print "Parsing successful.. Total processing time:", autobank.time_taken(end_time)
            times.append(int(end_time))
            #to also check that the derivation tree is the same, uncomment the following:
            #if old_sd_bracketing not in new_subcat_derivation_bracketings:
                #message = "\nSubcat full derivation bracketings\nare different for a seed tree.. view tree?"
                #self.viewParseQuestion(message, parse, reason='different', times=times, construct_lexicon=construct_lexicon, compose=compose, failed_parses=failed_parses, different_parses=different_parses, singleTree=singleTree, total_start_time=total_start_time, parse_times=parse_times, break_time_start=break_time_start, corpus=corpus, overwrite_derivation=overwrite_derivation)
                #return
            if overwrite_derivation:
                seeds[str(seed_line_num)] = (subcat_derivation_bracketings[i], xbar_bracketings[i], derived_bracketings[i], subcat_full_derivation_bracketings[i], derivation_bracketings[i], full_derivation_bracketings[i], parserSettings)
                with open(MGFileName, 'w') as SEED_FILE:
                    json.dump(seeds, SEED_FILE)
        total_time = default_timer() - total_start_time
        if singleTree == None:
            print "\nParsing of all seeds complete..\n"
            numTrees = str(len(TreeList))
        else:
            numTrees = str(1)
        print "Number of trees processed:  "+numTrees
        print "Number of trees for which parsing failed: "+str(len(failed_parses))
        print "Number of trees for which only incorrect parses were generated: "+str(len(different_parses))
        if singleTree == None:
            print "Total time to process all seeds: ", autobank.time_taken(total_time)
            print "Total time taken just for parsing: ", autobank.time_taken(sum(parse_times))
            print "Quickest time taken to parse a tree: ", autobank.time_taken(min(parse_times))
            print "Slowest time taken to parse a tree: ", autobank.time_taken(max(parse_times))
            print "Average (mean) time taken to parse each tree: ", autobank.time_taken(sum(parse_times)/len(parse_times))
        print ""
        if len(failed_parses) > 0:
            print "List of trees for which parsing failed:\n"
            for parse in failed_parses:
                print parse[0]+" ln: "+str(parse[1]+1)
            print ""
        if len(different_parses) > 0:
            print "List of trees for which only incorrect parses were generated:\n"
            for parse in different_parses:
                print parse[0]+" ln: "+str(parse[1]+1)
            print ""
        if singleTree != None:
            self.nothing(self.mainWindow, "Reparsing of seed sentence was succesful!")
        elif len(failed_parses) == 0 and len(different_parses) == 0:
            self.nothing(self.mainWindow, "Reparsing of all seeds was succesful!")
        else:
            self.nothing(self.mainWindow, "Reparsing of seeds completed.\nThere were "+str(len(failed_parses))+" failed parses and "+str(len(different_parses))+" incorrect parses (see terminal).", height=75, width=500)
        
    def modifyCats(self, catToModify, catType):
        if len(self.catsBox.curselection()) == 0:
            self.nothing(self.viewDelCatsWindow, "You haven't selected any categories to modify!")
            return
        elif len(self.catsBox.curselection()) > 1:
            self.nothing(self.viewDelCatsWindow, "You can only modify one category at a time!")
            return
        else:
            catToModify = self.catsBox.get(self.catsBox.curselection()).split("     ")[0]
        if catType == "overt":
            self.addOvertCat(catToModify)
        elif catType == "covert":
            nullLex = self.catsBox.get(self.catsBox.curselection()).split("     ")[1][1:-1]
            self.addCovertCat(catToModify, nullLex)

    def modify(self, oldCat=None, confirmWhat=None, newFeatures=None, catType=None, cbVars=None, morphemeName=None, nullLex=None, oldMorphemeName=None, oldCatIsCoord=None, entry=None, morphemeNameEntry=None, inLoop=False, MGtagType=None):
        if self.confirmWindow != None:
            self.destroyWindow(self.confirmWindow, 'confirmWindow')
        if 'tempSeed' in os.listdir(os.getcwd()):
            rmtree('tempSeed')
        if 'tempAuto' in os.listdir(os.getcwd()):
            rmtree('tempAuto')
        #chop off any trailing whitespace..
        newFeatures = newFeatures.strip()
        if self.viewDelCatsWindow != None:
            self.destroyWindow(self.viewDelCatsWindow, 'viewDelCatsWindow')
        try:
            CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
        except IOError:
            CatTreeMappings = {}
        CatTreeMappingsCopy = copy.deepcopy(CatTreeMappings)
        try:
            TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
        except IOError:
            TreeCatMappings = {}
        TreeCatMappingsCopy = copy.deepcopy(TreeCatMappings)
        try:
            nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
        except IOError:
            nullCatTreeMappings = {}
        nullCatTreeMappingsCopy = copy.deepcopy(nullCatTreeMappings)
        try:
            nullTreeCatMappings = json.load(open(self.seed_folder+"/"+'nullTreeCatMappings'))
        except IOError:
            nullTreeCatMappings = {}
        nullTreeCatMappingsCopy = copy.deepcopy(nullTreeCatMappings)
        if self.viewDelCatsWindow != None:
            self.destroyWindow(self.viewDelCatsWindow, 'viewDelCatsWindow')
        try:
            autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
        except IOError:
            autoCatTreeMappings = {}
        autoCatTreeMappingsCopy = copy.deepcopy(autoCatTreeMappings)
        try:
            autoTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
        except IOError:
            autoTreeCatMappings = {}
        autoTreeCatMappingsCopy = copy.deepcopy(autoTreeCatMappings)
        try:
            autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
        except IOError:
            autoNullCatTreeMappings = {}
        autoNullCatTreeMappingsCopy = copy.deepcopy(autoNullCatTreeMappings)
        try:
            autoNullTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoNullTreeCatMappings'))
        except IOError:
            autoNullTreeCatMappings = {}
        autoNullTreeCatMappingsCopy = copy.deepcopy(autoNullTreeCatMappings)
        covertCatCommentsCopy = json.load(open(self.seed_folder+"/"+'covertCatComments', 'r'))
        overtCatCommentsCopy = json.load(open(self.seed_folder+"/"+'overtCatComments', 'r'))
        if catType == 'overt':
            newComments = self.ocentry.get().strip()
            if len(newComments) > 100:
                self.nothing(self.addOvertCatWindow, message="Comments exceed 100 characters!")
                return
            ctm = CatTreeMappings
            actm = autoCatTreeMappings
            #self.destroyWindow(self.addOvertCatWindow, 'addOvertCatWindow')
            if cbVars['coordinator'][0].get() == 1:
                newCat = u':\u0305:\u0305 '+newFeatures
            else:
                newCat = ':: '+newFeatures
            ptbCats = []
            for e in cbVars:
                if e != 'coordinator' and cbVars[e][0].get():
                    ptbCats.append(e)
        elif catType == 'covert':
            if morphemeName[0:2] == '[[':
                morphemeName = morphemeName[1:]
            if morphemeName[-2:] == ']]':
                morphemeName = morphemeName[0:-1]
            if self.CovertCoordinator.get() == 1:
                newCatLexEntry = list(self.constructMGlexEntry(morphemeName, (':\u0305:\u0305 '+newFeatures).split(" ")))
                newCat = morphemeName+u' :\u0305:\u0305 '+newFeatures
                with open('Junk', 'w') as junkFile:
                    json.dump(newCatLexEntry, junkFile)
                newCatLexEntry = json.load(open('Junk'))#1
            else:
                newCatLexEntry = list(self.constructMGlexEntry(morphemeName, (':: '+newFeatures).split(" ")))
                newCat = morphemeName+' :: '+newFeatures
                with open('Junk', 'w') as junkFile:
                    json.dump(newCatLexEntry, junkFile)
                newCatLexEntry = json.load(open('Junk'))#2
            newMorphemeName = newCat.split(" ")[0]
            if (self.covertLexUpdated == False or inLoop == True):
                #we need access to both the old and new covertLexicons for this category (which may or
                #may not be the same covert lexicon).
                if self.lexiconVar.get() == 'Covert':
                    targetCovertLexicon = self.CovertLexicon
                    targetCovertLexiconFileName = self.seed_folder+'/CovertLexicon'
                elif self.lexiconVar.get() == 'Extraposer':
                    targetCovertLexicon = self.ExtraposerLexicon
                    targetCovertLexiconFileName = self.seed_folder+'/ExtraposerLexicon'
                elif self.lexiconVar.get() == 'TypeRaiser':
                    targetCovertLexicon = self.TypeRaiserLexicon
                    targetCovertLexiconFileName = self.seed_folder+'/TypeRaiserLexicon'
                elif self.lexiconVar.get() == 'ToughOperator':
                    targetCovertLexicon = self.ToughOperatorLexicon
                    targetCovertLexiconFileName = self.seed_folder+'/ToughOperatorLexicon'
                elif self.lexiconVar.get() == 'NullExcorporator':
                    targetCovertLexicon = self.NullExcorporatorLexicon
                    targetCovertLexiconFileName = self.seed_folder+'/NullExcorporatorLexicon'
                if self.oldCatIsNewCat == False or newMorphemeName != oldMorphemeName:
                    #if there were duplicates then the confirm button would already have lead to the
                    #addCOVERTcat function so we don;t want to do this twice.
                    ctm = nullCatTreeMappings
                    actm = autoNullCatTreeMappings
                    oldFeatures = " ".join(oldCat.split(" ")[2:])
                    #self.destroyWindow(self.addCovertCatWindow, 'addOvertCatWindow')
                    if oldCatIsCoord:
                        oldCatLexEntry = list(self.constructMGlexEntry(oldMorphemeName, (':\u0305:\u0305 '+oldFeatures).split(" ")))
                    else:
                        oldCatLexEntry = list(self.constructMGlexEntry(oldMorphemeName, (':: '+oldFeatures).split(" ")))
                    if inLoop == True:
                        #if we are within the loop, then the moved cateogry has already been
                        #added to the target (new) lexicon and so in order to retrieve the original version of it
                        #we simply remove the category temporarily so we can create a copy and restore
                        #the lexicon if parsing fails..
                        try:
                            targetCovertLexicon.remove(newCatLexEntry)
                        except Exception as e:
                            #after I fixed a bug where in new sentence mode the sourcelexicon ended up
                            #with non json characters in its features, an error was thrown by this bit
                            #of code but only once.. I couldn't get the error to repeat.. so just to be safe
                            #im converting both the target lexicon and the new lex entry itself to json format
                            #if an error is thrown..
                            with open('Junk', 'w') as JunkFile:
                                json.dump(newCatLexEntry, JunkFile)
                                newCatLexEntry = json.load(open('Junk'))#3
                            sclindex = -1
                            for entry in targetCovertLexicon:
                                sclindex+=1
                                #have to convert back to json entry by entry as if we try to do the whole
                                #thing at once it creates a new sourceCovertLexicon
                                with open('Junk', 'w') as JunkFile:
                                    json.dump(entry, JunkFile)
                                targetCovertLexicon[sclindex] = json.load(open('Junk'))
                            targetCovertLexicon.remove(newCatLexEntry)
                    #if we are in the loop and the target lexicon and source lexicon are the same,
                    #then by restoring the oriniginal target lexicon in the previous line, we have also restored the
                    #original source, otherwise it doesnt matter as the source is unchanged at this point..
                    sourceCovertLexicon = self.getCovertLexicon(nullLex)
                    #we need to remove the old category from its lexicon.. we will keep a deep copy
                    #of the lexicon in its state before this item was removed, however, so tat we can simply
                    #reset it if the parsing fails..
                    targetCovertLexiconCopy = copy.deepcopy(targetCovertLexicon)
                    if inLoop == True:
                        #having made the target lex copy, we now insert the moved item back in as we
                        #don't want to go in addCOVERTcat again to do this..
                        targetCovertLexicon.append(newCatLexEntry)
                    sourceCovertLexiconCopy = copy.deepcopy(sourceCovertLexicon)
                    try:
                        sourceCovertLexicon.remove(oldCatLexEntry)
                    except Exception as e:
                        #for some reason, going into test sentence mode results in the covert
                        #lexicons having the unicode rep of adjuncts, rather than the json one
                        sclindex = -1
                        for entry in sourceCovertLexicon:
                            sclindex+=1
                            #have to convert back to json entry by entry as if we try to do the whole
                            #thing at once it creates a new sourceCovertLexicon
                            with open('Junk', 'w') as JunkFile:
                                json.dump(entry, JunkFile)
                            sourceCovertLexicon[sclindex] = json.load(open('Junk'))
                        sourceCovertLexicon.remove(oldCatLexEntry)
                    #now we insert the new category into the relevant lexicon.. again, we have kept a record
                    #of that lexicon (which may or may not be the same lexicon as the one that originally
                    #contained the unmodifed item) so we can reset it if parsing fails..
                    if inLoop == False:
                        #no need to add to the target lexicon if we're in the loop as its already been done
                        self.addCOVERTcat(catType=catType, oldCat=oldCat, MGentry=newCatLexEntry, features=newCat.split(" ")[1:], oldMorphemeName=oldMorphemeName, nullString=morphemeName, oldCatIsCoord=oldCatIsCoord, confirmWhat='modCats', entry=entry, morphemeNameEntry=morphemeNameEntry, nullLex=nullLex, inLoop=True)
                    self.covertLexUpdated = False
            #we need to test whether or not the targetCovertLexicon was actually updated,
            #and if it wasn't (e.g. because the user aborted at confirmation stage), we
            #we print a message to the screen saying no changes were made..
            else:
                self.covertLexUpdated = False
        copytree(self.seed_folder, 'tempSeed')
        copytree(self.auto_folder, 'tempAuto')
        success = True
        if self.addOvertCatWindow != None:
            self.destroyWindow(self.addOvertCatWindow, 'addOvertCatWindow')
        if self.addCovertCatWindow != None:
            self.newComments = self.ccentry.get().strip()
            if len(self.newComments) > 100:
                self.nothing(self.addCovertCatWindow, message="Comments a exceed 100 characters!")
                return
            self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow')
        #if oldCat == newCat at this point it means all that has changed is the comments and so no parsing is necessary..
        if (catType == 'covert' and oldCat in nullCatTreeMappings and newMorphemeName != oldMorphemeName) or ((not (self.oldCatIsNewCat and catType == 'covert') and not (self.oldCatIsNewCat and self.oldAndNewPosMappingsSubset)) and ((catType == 'overt' and oldCat in CatTreeMappings) or (catType == 'covert' and oldCat in nullCatTreeMappings))):
            ctmCopy = copy.deepcopy(ctm)
            actmCopy = copy.deepcopy(actm)
            TreeList = []
            for PARSE in ctmCopy[oldCat]:
                TreeList.append(str(PARSE))
            if oldCat in actm:
                for PARSE in actm[oldCat]:
                    TreeList.append(str(PARSE))
            #For line numbers with only one digit, e.g. 3, we need to add a zero (03) so that sorting works properly, but then we need to strip it out
            TreeListIndex = -1
            for PARSE in TreeList:
                TreeListIndex += 1
                parts = PARSE.split(" ")
                if len(parts[1]) == 2:
                    parts[1] = "00"+parts[1]
                elif len(parts[1]) == 3:
                    parts[1] = "0"+parts[1]
                TreeList[TreeListIndex] = " ".join(parts)
            TreeList.sort()
            TreeListIndex = -1
            for PARSE in TreeList:
                TreeListIndex += 1
                parts = PARSE.split(" ")
                if parts[1][0] == "0":
                    parts[1] = parts[1][1:]
                if parts[1][0] == "0":
                    parts[1] = parts[1][1:]
                TreeList[TreeListIndex] = ast.literal_eval(" ".join(parts))
            for entry in TreeList:
                miniOvertLexicon = []
                fileName = entry[0].encode('ascii')
                ENTRY = str([fileName, entry[1]])
                self.seed_line_num = entry[1]
                #the fileName refers to the seed corpus version.. now we want the original PTB file..
                parts = fileName.split("/")
                fileName = self.ptb_folder+"/"+parts[1]+"/"+parts[2]
                if 'Seed' in ENTRY:
                    tempFileName = 'tempSeed'+"/"+parts[1]+"/"+parts[2]
                elif 'Auto' in ENTRY:
                    tempFileName = 'tempAuto'+"/"+parts[1]+"/"+parts[2]
                seeds = json.load(open(tempFileName))
                lineNum = entry[1]
                #we need to load the parser setting that was used when the original tree was manually annotated
                parserSettings = seeds[str(lineNum)][6]
                if parts[1] != 'new_parses':
                    parseFile = open(fileName)
                    current_line = -1
                    words = None
                    for line in parseFile:
                        current_line += 1
                        if current_line == lineNum:
                            PTB_TREE = autobank.build_tree(line)
                            PTB_tree = PTB_TREE[0]
                            terminals = PTB_TREE[2]
                            words = [t.name.lower() for t in terminals]
                            vp_ellipsis = autobank.contains_vp_ellipsis(PTB_tree)
                            if self.parserSettings["constrainMoveWithPTB"]:
                                moveable_spans = []
                                autobank.get_moveable_spans(PTB_tree, terminals, moveable_spans)
                            else:
                                moveable_spans = None
                            if self.parserSettings["constrainConstWithPTBCCG"]:
                                source_spans = []
                                try:
                                    ccg_parses = json.load(open("CCGbank/"+parts[1]+"/"+parts[2].split(".")[0]+".ccg"))
                                    ccg_bracketing = ccg_parses[parts[2].split(".")[0]+"."+str(current_line+1)]
                                    ccg_tree = autobank.build_tree(ccg_bracketing)
                                    ccg_terminals = ccg_tree[2]
                                    autobank.set_indices(ccg_terminals)
                                    ccg_tree = ccg_tree[0]
                                except Exception as e:
                                    ccg_tree = None
                                autobank.get_source_spans(PTB_tree, ccg_tree, source_spans, terminals)
                            else:
                                source_spans = None
                            break
                else:
                    vp_ellipsis = True
                    moveable_spans = None
                    source_spans = None
                    newSentFileName = self.seed_folder+"/"+parts[1]+"/"+parts[2]
                    new_parse_file = open(self.seed_folder+"/new_parse_strings/"+parts[2])
                    for line in new_parse_file:
                        words = line.split(" ")
                        break
                if 'Seed' in ENTRY:
                    MGcats = TreeCatMappings[unicode(ENTRY)]
                    corpus = 'seeds'
                elif 'Auto' in ENTRY:
                    MGcats = autoTreeCatMappings[unicode(ENTRY)]
                    corpus = 'autos'
                i=-1
                indices = []
                for cat in MGcats:
                    i+=1
                    if cat == oldCat:
                        MGcats[i] = newCat.strip()
                        indices.append(i)
                if 'Seed' in ENTRY:
                    if parts[1] != 'new_parses':
                        for index in indices:
                            if terminals[index].mother.truncated_name not in ptbCats:
                                success = False
                if success == True:
                    if MGtagType == 'atomic':
                        i=-1
                        for word in words:
                            i+=1
                            features = MGcats[i].split(" ")
                            lexEntry = self.constructMGlexEntry(word, features)
                            if lexEntry != None:
                                miniOvertLexicon.append([lexEntry, i])
                    elif MGtagType == 'supertag':
                        old_sfd_bracketing = seeds[str(self.seed_line_num)][3]
                        old_d_bracketing = seeds[str(self.seed_line_num)][4]
                        ns = False
                        ignore_null_c = False
                        null_c_lexicon = None
                        supertags = self.extractMGsupertags(old_sfd_bracketing, old_d_bracketing, ignore_null_c, ns=ns)
                        #now we need to go through all these supertags and replace any instances of the old category with
                        #the updated version of the category..we also need to unpack any supertags with these instances in
                        #as the indices will now be wrong and it is not possible to always predict what they should be
                        #given the newly modified category.. we therefore simply allow all atomic categories inside the supertag
                        #to be free
                        supertags_to_remove = []
                        for st in supertags:
                            if catType == 'overt':
                                if type(st) != type([]):
                                    parts=st.name.split(" ")
                                    if parts[0][0] == '[' and parts[0][-1] == ']':
                                        continue
                                    else:
                                        cat = " ".join(parts[1:])
                                        if cat == oldCat:
                                            st.name = parts[0]+" "+newCat
                                else:
                                    for link in st:
                                        parts = link[0].name.split(" ")
                                        if parts[0][0] != '[' or parts[0][-1] != ']':
                                            cat = " ".join(parts[1:])
                                            if cat == oldCat:
                                                link[0].name = parts[0]+" "+newCat
                                                if st not in supertags_to_remove:
                                                    supertags_to_remove.append(st)
                                        parts = link[2].name.split(" ")
                                        if parts[0][0] != '[' and parts[0][-1] != ']':
                                            cat = " ".join(parts[1:])
                                            if cat == oldCat:
                                                link[2].name = parts[0]+" "+newCat
                                                if st not in supertags_to_remove:
                                                    supertags_to_remove.append(st)
                            elif catType == 'covert':
                                if type(st) != type([]):
                                    parts=st.name.split(" ")
                                    if parts[0][0] != '[' or parts[0][-1] != ']':
                                        continue
                                    else:
                                        cat = st.name
                                        if cat == oldCat:
                                            st.name = newCat          
                                else:
                                    for link in st:
                                        parts = link[0].name.split(" ")
                                        if parts[0][0] == '[' and parts[0][-1] == ']':
                                            cat = link[0].name
                                            if cat == oldCat:
                                                link[0].name = newCat
                                                if st not in supertags_to_remove:
                                                    supertags_to_remove.append(st)
                                        parts = link[2].name.split(" ")
                                        if parts[0][0] == '[' and parts[0][-1] == ']':
                                            cat = link[2].name
                                            if cat == oldCat:
                                                link[2].name = newCat
                                                if st not in supertags_to_remove:
                                                    supertags_to_remove.append(st)
                        modified_null_lexicon = []
                        for st in supertags_to_remove:
                            supertags.remove(st)
                            for link in st:
                                if link[0].name.split(" ")[0][0] == '[' and link[0].name.split(" ")[0][-1] == ']':
                                    if link[0].name not in [node.name for node in modified_null_lexicon]:
                                        modified_null_lexicon.append(link[0])
                                else:
                                    if link[0] not in supertags:
                                        supertags.append(link[0])
                                if link[2].name.split(" ")[0][0] == '[' and link[2].name.split(" ")[0][-1] == ']':
                                    if link[2].name not in [node.name for node in modified_null_lexicon]:
                                        modified_null_lexicon.append(link[2])
                                else:
                                    if link[2] not in supertags:
                                        supertags.append(link[2])
                        ml_index = -1
                        for null_head in modified_null_lexicon:
                            ml_index += 1
                            (features, word) = self.get_derivation_tree_features(null_head, return_name_and_type=True)
                            NULL_HEAD = self.constructMGlexEntry(word, features)
                            modified_null_lexicon[ml_index] = NULL_HEAD
                        supertag_index = -1
                        for supertag in supertags:
                            cat_ids = {}
                            ID = 0
                            supertag_index += 1
                            if type(supertag) != type([]):
                                if supertag not in cat_ids:
                                    cat_ids[supertag] = ID
                                    ID += 1
                                (features, word) = self.get_derivation_tree_features(supertag, return_name_and_type=True)
                                if int(supertag.index) == -1:
                                    raise Exception("Error! Null category detected that is not anchored to an overt category!")
                                else:
                                    SUPERTAG = [self.constructMGlexEntry(word, features), cat_ids[supertag], int(supertag.index)]
                                supertags[supertag_index] = SUPERTAG
                            else:
                                for link in supertag:
                                    if link[0] not in cat_ids:
                                        cat_ids[link[0]] = ID
                                        ID += 1
                                    (features, word) = self.get_derivation_tree_features(link[0], return_name_and_type=True)
                                    link[0] = [self.constructMGlexEntry(word, features), cat_ids[link[0]], int(link[0].index)]
                                    if link[2] not in cat_ids:
                                        cat_ids[link[2]] = ID
                                        ID += 1
                                    (features, word) = self.get_derivation_tree_features(link[2], return_name_and_type=True)
                                    link[2] = [self.constructMGlexEntry(word, features), cat_ids[link[2]], int(link[2].index)]
                        supertag_duplicates = []
                        #duplicates arise when the tree has ATB movement so a given lexical item appears twice in the derivation tree
                        for st in supertags:
                            if supertags.count(st) > 1:
                                if [st, supertags.count(st)] not in supertag_duplicates:
                                    supertag_duplicates.append([st, supertags.count(st)])
                        for st in supertag_duplicates:
                            while st[1] > 1:
                                supertags.remove(st[0])
                                st[1] -= 1
                    try:
                        if self.mode == 'annotation':
                            if self.treeCompareWindow != None:
                                self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')                               
                        self.mainWindow.withdraw()
                        timeAndDate = {'time':time.strftime("%H:%M:%S"), 'date':time.strftime("%d/%m/%Y")}
                        if MGtagType == 'atomic':
                            if parts[1] != "new_parses":
                                print "\nReparsing PTB sentence:", fileName, "Ln:", str(lineNum+1)+',', 'using parser setting:', parserSettings+" on "+timeAndDate['date']+" at "+timeAndDate['time']
                            else:
                                print "\nReparsing new sentence:", newSentFileName+',', 'using parser setting:', parserSettings+" on "+timeAndDate['date']+" at "+timeAndDate['time']
                            with timeout(self.parserSettings['timeout_seconds']):
                                self.parseMessage()
                                start_time = default_timer()
                                PARSERSETTINGS = parserSettings
                                if 'UseAllNull' in parserSettings:
                                    useAllNull = True
                                    PARSERSETTINGS = PARSERSETTINGS[:-10]
                                else:
                                    useAllNull = False
                                if 'SkipRel' in parserSettings:
                                    skipRel = True
                                    PARSERSETTINGS = PARSERSETTINGS[:-7]
                                else:
                                    skipRel = False
                                if 'SkipPro' in parserSettings:
                                    skipPro = True
                                    PARSERSETTINGS = PARSERSETTINGS[:-7]
                                else:
                                    skipPro = False
                                if PARSERSETTINGS == 'basicOnly':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                elif PARSERSETTINGS == 'fullFirst':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                elif PARSERSETTINGS == 'basicAndRight':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                elif PARSERSETTINGS == 'basicAndExcorp':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                elif PARSERSETTINGS == 'basicAndTough':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, LEXICON=miniOvertLexicon, CovertLexicon=self.CovertLexicon, ExtraposerLexicon=self.ExtraposerLexicon, TypeRaiserLexicon=self.TypeRaiserLexicon, ToughOperatorLexicon=self.ToughOperatorLexicon, NullExcorporatorLexicon=self.NullExcorporatorLexicon, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], useAllNull=useAllNull, skipRel=skipRel, skipPro=skipPro, start_time=start_time, MOVEABLE_SPANS=moveable_spans, SOURCE_SPANS=source_spans, vp_ellipsis=vp_ellipsis)
                                end_time = default_timer() - start_time
                            self.destroyWindow(self.parsing, 'parsing')
                            self.mainWindow.deiconify()
                        elif MGtagType == 'supertag':
                            composition_strategy = 'MG supertags (with subcat features)'
                            print "\nReparsing PTB sentence:", fileName, "Ln:", str(lineNum+1)+',', 'using parser setting:', parserSettings+", and composition strategy: "+composition_strategy+", "+"on "+timeAndDate['date']+" at "+timeAndDate['time']
                            with timeout(self.parserSettings['timeout_seconds']):
                                self.parseMessage()
                                start_time = default_timer()
                                PARSERSETTINGS = parserSettings
                                if 'UseAllNull' in parserSettings:
                                    PARSERSETTINGS = PARSERSETTINGS[:-10]
                                if 'SkipRel' in parserSettings:
                                    PARSERSETTINGS = PARSERSETTINGS[:-7]
                                if 'SkipPro' in parserSettings:
                                    PARSERSETTINGS = PARSERSETTINGS[:-7]
                                if PARSERSETTINGS == 'basicOnly':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans, modified_null_lexicon=modified_null_lexicon)
                                elif PARSERSETTINGS == 'fullFirst':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, t_move_on = True, x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans, modified_null_lexicon=modified_null_lexicon)
                                elif PARSERSETTINGS == 'basicAndRight':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), r_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans, modified_null_lexicon=modified_null_lexicon)
                                elif PARSERSETTINGS == 'basicAndExcorp':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), x_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans, modified_null_lexicon=modified_null_lexicon)
                                elif PARSERSETTINGS == 'basicAndTough':
                                    (parse_time, derivation_bracketings, derived_bracketings, xbar_bracketings, XBAR_trees, subcat_derivation_bracketings, subcat_full_derivation_bracketings, full_derivation_bracketings) = cky_mg.main(sentence=" ".join(words), t_move_on = True, show_trees=False, print_expressions = False, return_bracketings = True, return_xbar_trees = True, printPartialAnalyses=self.parserSettings['printPartialAnalyses'], supertags=supertags, start_time=start_time, MOVEABLE_SPANS=moveable_spans, null_c_lexicon=null_c_lexicon, SOURCE_SPANS=source_spans, modified_null_lexicon=modified_null_lexicon)
                                end_time = default_timer() - start_time
                            self.destroyWindow(self.parsing, 'parsing')
                            self.mainWindow.deiconify()                         
                    except IOError:#Exception as e:
                        self.mainWindow.deiconify()
                        if self.parsing != None:
                            self.destroyWindow(self.parsing, 'parsing')
                        print "Error!!!"
                        subcat_derivation_bracketings = []
                    if len(subcat_derivation_bracketings) == 0:
                        success = False
                        message = "\nThe new MG category failed to produce\nany parses for tree:\n"+entry[0]+" ln: "+str(entry[1]+1)+"\n(hint: Were the parser settings appropriate?).."
                        self.viewParseQuestion(message, entry, corpus=corpus)
                        if catType == 'covert':
                            self.restoreOldLexicons(sourceCovertLexicon, sourceCovertLexiconCopy, targetCovertLexicon, targetCovertLexiconCopy)
                        self.restore_mappings(CatTreeMappingsCopy, TreeCatMappingsCopy, nullCatTreeMappingsCopy, nullTreeCatMappingsCopy, covertCatCommentsCopy, overtCatCommentsCopy, autoCatTreeMappingsCopy, autoTreeCatMappingsCopy, autoNullCatTreeMappingsCopy, autoNullTreeCatMappingsCopy)
                        return
                    else:
                        seeds = json.load(open(tempFileName))
                        old_xbar_bracketing = seeds[str(lineNum)][1]
                        old_sfd_bracketing = seeds[str(lineNum)][3]
                        with open('Junk', 'w') as JunkFile:
                            #making sure the xbar bracketings are in the same format as those in the seed
                            #files.. json makes certain changes..
                            json.dump(xbar_bracketings, JunkFile)
                        new_xbar_bracketings = json.load(open('Junk'))
                        with open('Junk', 'w') as JunkFile:
                            #making sure the derivation bracketings are in the same format as those in the seed
                            #files.. json makes certain changes..
                            json.dump(subcat_full_derivation_bracketings, JunkFile)
                        new_subcat_full_derivation_bracketings = json.load(open('Junk'))
                        if old_xbar_bracketing not in new_xbar_bracketings:
                            success = False
                            message = "\nThe new MG category produced different\nparses for tree:\n"+entry[0]+" ln: "+str(entry[1]+1)+"\n(hint: Were the parser settings appropriate?)."
                            self.viewParseQuestion(message, entry, corpus=corpus)
                            if catType == 'covert':
                                self.restoreOldLexicons(sourceCovertLexicon, sourceCovertLexiconCopy, targetCovertLexicon, targetCovertLexiconCopy)
                            self.restore_mappings(CatTreeMappingsCopy, TreeCatMappingsCopy, nullCatTreeMappingsCopy, nullTreeCatMappingsCopy, covertCatCommentsCopy, overtCatCommentsCopy, autoCatTreeMappingsCopy, autoTreeCatMappingsCopy, autoNullCatTreeMappingsCopy, autoNullTreeCatMappingsCopy)
                            return
                        #if old_sfd_bracketing not in new_subcat_full_derivation_bracketings:
                            #success = False
                            #message = "\nSubcat full derivation bracketings\nare different for a tree.. view tree?"
                            #self.viewParseQuestion(message, entry, corpus=corpus)
                            #if catType == 'covert':
                                #self.restoreOldLexicons(sourceCovertLexicon, sourceCovertLexiconCopy, targetCovertLexicon, targetCovertLexiconCopy)
                            #self.restore_mappings(CatTreeMappingsCopy, TreeCatMappingsCopy, nullCatTreeMappingsCopy, nullTreeCatMappingsCopy, covertCatCommentsCopy, overtCatCommentsCopy, autoCatTreeMappingsCopy, autoTreeCatMappingsCopy, autoNullCatTreeMappingsCopy, autoNullTreeCatMappingsCopy)
                            #return
                        else:
                            i = new_xbar_bracketings.index(old_xbar_bracketing)
                            seeds[str(lineNum)] = (subcat_derivation_bracketings[i], xbar_bracketings[i], derived_bracketings[i], subcat_full_derivation_bracketings[i], derivation_bracketings[i], full_derivation_bracketings[i], parserSettings)
                            with open(tempFileName, 'w') as tempFile:
                                json.dump(seeds, tempFile)
                            subcat_derivation_tree = gen_derived_tree.gen_derivation_tree(subcat_derivation_bracketings[i])
                            nullMGcats = autobank.get_null_MGcats(subcat_derivation_tree, [])
                            with open('junk', 'w') as junkFile:
                                json.dump(nullMGcats, junkFile)
                            nullMGcats = json.load(open('junk'))
                            if 'Seed' in ENTRY:
                                NTCM = nullTreeCatMappings
                                NCTM = nullCatTreeMappings
                            elif 'Auto' in ENTRY:
                                NTCM = autoNullTreeCatMappings
                                NCTM = autoNullCatTreeMappings
                            if NTCM[ENTRY] != nullMGcats:
                                #we have to update the null cat-tree mappings as different null categories were used
                                #in order to produce the same xbar tree.. since we updated the derivation tree, we must also update the mappings
                                #we start by removing the tree from all the old null cat entries in nullCatTreeMappings
                                for nullMGcat in NTCM[ENTRY]:
                                    if nullMGcat in NCTM:#we need this line because if the same category appears twice in one tree, it may a;ready have been deleted from nullCatTreeMappings if this tree was the only entry for it
                                        if entry in NCTM[nullMGcat]:
                                            NCTM[nullMGcat].remove(entry)
                                            if len(NCTM[nullMGcat]) == 0:
                                                del(NCTM[nullMGcat])
                            for nullMGcat in nullMGcats:
                                if nullMGcat in NCTM:
                                    if entry not in NCTM[nullMGcat]:
                                        NCTM[nullMGcat].append(entry)
                                else:
                                    NCTM[nullMGcat] = [entry]
                            NTCM[ENTRY] = nullMGcats
                    if success:
                        print "Parsing successful.. Total processing time: ", autobank.time_taken(end_time)
                else:
                    message = "\nThe new MG category was not associated with\nthe correct PTB category for one of the seed trees."
                    self.viewParseQuestion(message, entry, corpus=corpus)
                    return
        if success == True and ((self.oldCatIsNewCat == False or (catType=='covert' and oldMorphemeName != newMorphemeName) or (catType=='overt' and self.oldAndNewPosMappingsMatch == False))):
            if self.derivationWindow != None:
                self.destroyWindow(self.derivationWindow, 'derivationWindow')
            #if newCat == oldCat and self.oldAndNewPosMappingsMatch == True then all that's changed is the comments..
            #load TreeCatMappings again as it was altered when MGcats were altered above and although this gave the right effect
            #if we ever change that bit of code I want the effect on TreeCatMappings to remain constant
            try:
                TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
            except IOError:
                TreeCatMappings = {}
            try:
                CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
            except IOError:
                CatTreeMappings = {}
            try:
                autoTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
            except IOError:
                autoTreeCatMappings = {}
            try:
                autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
            except IOError:
                autoCatTreeMappings = {}
            rmtree(self.seed_folder)
            rmtree(self.auto_folder)
            os.rename('tempSeed', self.seed_folder)
            os.rename('tempAuto', self.auto_folder)
            if catType == 'overt':
                del(self.overtCatComments[oldCat])
                self.overtCatComments[newCat] = newComments
                if oldCat != newCat:
                    if oldCat in CatTreeMappings:
                        for entry in CatTreeMappings[oldCat]:
                            fileName = entry[0].encode('ascii')
                            ENTRY = str([fileName, entry[1]])
                            while oldCat in TreeCatMappings[unicode(ENTRY)]:
                                index = TreeCatMappings[unicode(ENTRY)].index(oldCat)
                                TreeCatMappings[unicode(ENTRY)][index] = newCat.strip()
                        CatTreeMappings[newCat.strip()] = CatTreeMappings[oldCat]
                        del(CatTreeMappings[oldCat])
                    if oldCat in autoCatTreeMappings:
                        for entry in autoCatTreeMappings[oldCat]:
                            fileName = entry[0].encode('ascii')
                            ENTRY = str([fileName, entry[1]])
                            while oldCat in autoTreeCatMappings[unicode(ENTRY)]:
                                index = autoTreeCatMappings[unicode(ENTRY)].index(oldCat)
                                autoTreeCatMappings[unicode(ENTRY)][index] = newCat.strip()
                        autoCatTreeMappings[newCat.strip()] = autoCatTreeMappings[oldCat]
                        del(autoCatTreeMappings[oldCat])
                for ptbCat in self.PosMappings:
                    if oldCat in self.PosMappings[ptbCat]:
                        if ptbCat in ptbCats:
                            index = self.PosMappings[ptbCat].index(oldCat)
                            self.PosMappings[ptbCat][index] = newCat
                        else:
                            self.PosMappings[ptbCat].remove(oldCat)
                            if len(self.PosMappings[ptbCat]) == 1:
                                self.PosMappings[ptbCat] = ["No categories available", ""]
                for ptbCat in ptbCats:
                    if newCat not in self.PosMappings[ptbCat]:
                        if self.PosMappings[ptbCat][0] == "No categories available":
                            del(self.PosMappings[ptbCat][0])
                        self.PosMappings[ptbCat].append(newCat)
                with open(self.seed_folder+"/"+'CatTreeMappings', 'w') as CatTreeMappingsFile:
                    json.dump(CatTreeMappings, CatTreeMappingsFile)
                with open(self.seed_folder+"/"+'TreeCatMappings', 'w') as TreeCatMappingsFile:
                    json.dump(TreeCatMappings, TreeCatMappingsFile)
                with open(self.seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
                    json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
                with open(self.seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
                    json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
                with open(self.auto_folder+"/"+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
                    json.dump(autoCatTreeMappings, autoCatTreeMappingsFile)
                with open(self.auto_folder+"/"+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
                    json.dump(autoTreeCatMappings, autoTreeCatMappingsFile)
                with open(self.auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
                    json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
                with open(self.auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
                    json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)
                with open(self.seed_folder+"/"+'PosMappings', 'w') as PosMappingsFile:
                    json.dump(self.PosMappings, PosMappingsFile)
                with open(self.seed_folder+"/"+'overtCatComments', 'w') as overtCatCommentsFile:
                    json.dump(self.overtCatComments, overtCatCommentsFile)
            elif catType == 'covert':
                lexiconsToRemove = []
                try:
                    for e in self.covertCatComments[newCat]:
                        if e[0] == targetCovertLexiconFileName.split("/")[1]:
                            lexiconsToRemove.append(e)
                except KeyError:
                    x=0
                while len(lexiconsToRemove) > 0:
                    self.covertCatComments[newCat].remove(lexiconsToRemove[0])
                    del(lexiconsToRemove[0])
                for e in self.covertCatComments[oldCat]:
                    if e[0] == nullLex:
                        self.covertCatComments[oldCat].remove(e)
                        if len(self.covertCatComments[oldCat]) == 0:
                            del(self.covertCatComments[oldCat])
                        try:
                            newComments = self.ccentry.get().strip()
                        except TclError:
                            newComments = self.newComments
                        except AttributeError:
                            newComments = self.newComments
                        if newCat in self.covertCatComments:
                            self.covertCatComments[newCat].append([targetCovertLexiconFileName.split("/")[1], newComments])
                        else:
                            self.covertCatComments[newCat] = [[targetCovertLexiconFileName.split("/")[1], newComments]]
                if self.oldCatIsNewCat == False or newMorphemeName != oldMorphemeName:
                    with open(self.seed_folder+"/"+nullLex, 'w') as oldLexFile:
                        json.dump(sourceCovertLexicon, oldLexFile)
                    with open(targetCovertLexiconFileName, 'w') as newLexFile:
                        json.dump(targetCovertLexicon, newLexFile)
                    with open(self.seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
                        json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
                    with open(self.seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
                        json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
                    with open(self.auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
                        json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
                    with open(self.auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
                        json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)
                with open(self.seed_folder+"/"+'covertCatComments', 'w') as covertCatCommentsFile:
                    json.dump(self.covertCatComments, covertCatCommentsFile)
                self.checkedForDuplicates = False
                self.covertLexUpdated = False
            if self.mode == 'annotation':
                if self.treeCompareWindow != None:
                    self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')
                self.xbar_bracketings = []
                self.freshXbarWindow()
                self.destroySpins()
                self.destroyButtons()
                self.refreshTerminals()
        elif success == True:
            if self.derivationWindow != None:
                self.destroyWindow(self.derivationWindow, 'derivationWindow')
            i=-1
            if catType == 'covert':
                #this is the case where the cats are the same and so its just the comments that need updating..
                for e in self.covertCatComments[oldCat]:
                    i+=1
                    if e[0] == nullLex:
                        try:
                            self.covertCatComments[newCat][i] = [e[0], self.ccentry.get().strip()]
                        except TclError:
                            self.covertCatComments[newCat][i] = [e[0], self.newComments]
                        except AttributeError:
                            self.covertCatComments[newCat][i] = [e[0], self.newComments]
                with open(self.seed_folder+"/"+'covertCatComments', 'w') as covertCatCommentsFile:
                    json.dump(self.covertCatComments, covertCatCommentsFile)
            elif catType == 'overt':
                del(self.overtCatComments[oldCat])
                self.overtCatComments[newCat] = newComments
                with open(self.seed_folder+"/"+'overtCatComments', 'w') as overtCatCommentsFile:
                    json.dump(self.overtCatComments, overtCatCommentsFile)
            if self.mode == 'annotation':
                if self.treeCompareWindow != None:
                    self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')
                self.xbar_bracketings = []
                self.freshXbarWindow()
                self.destroySpins()
                self.destroyButtons()
                self.refreshTerminals()
        else:
            #if this was a covert category we attempted to modify, we need to reinstate the original
            #null lexicons..
            if catType == 'covert':
                self.restoreOldLexicons(sourceCovertLexicon, sourceCovertLexiconCopy, targetCovertLexicon, targetCovertLexiconCopy)
            #as the modification failed, we simply delete the temp file holding the
            #(perhaps partially modified) seed set..
            rmtree('tempSeed')
            rmtree('tempAuto')
            self.restore_mappings(CatTreeMappingsCopy, TreeCatMappingsCopy, nullCatTreeMappingsCopy, nullTreeCatMappingsCopy, covertCatCommentsCopy, overtCatCommentsCopy, autoCatTreeMappingsCopy, autoTreeCatMappingsCopy, autoNullCatTreeMappingsCopy, autoNullTreeCatMappingsCopy)
        self.viewDelCats(catType)
        if self.save_button != None:
            self.save_button.config(state=DISABLED)
        self.nothing(self.mainWindow, "Modification successful!")
        return

    def restoreOldLexicons(self, sourceCovertLexicon, sourceCovertLexiconCopy, targetCovertLexicon, targetCovertLexiconCopy):
        if targetCovertLexicon == self.CovertLexicon:
            self.CovertLexicon = targetCovertLexiconCopy
        elif targetCovertLexicon == self.ExtraposerLexicon:
            self.ExtraposerLexicon = targetCovertLexiconCopy
        elif targetCovertLexicon == self.TypeRaiserLexicon:
            self.TypeRaiserLexicon = targetCovertLexiconCopy
        elif targetCovertLexicon == self.ToughOperatorLexicon:
            self.ToughOperatorLexicon = targetCovertLexiconCopy
        elif targetCovertLexicon == self.NullExcorporatorLexicon:
            self.NullExcorporatorLexicon = targetCovertLexiconCopy
        if sourceCovertLexicon == self.CovertLexicon:
            self.CovertLexicon = sourceCovertLexiconCopy
        elif sourceCovertLexicon == self.ExtraposerLexicon:
            self.ExtraposerLexicon = sourceCovertLexiconCopy
        elif sourceCovertLexicon == self.TypeRaiserLexicon:
            self.TypeRaiserLexicon = sourceCovertLexiconCopy
        elif sourceCovertLexicon == self.ToughOperatorLexicon:
            self.ToughOperatorLexicon = sourceCovertLexiconCopy
        elif sourceCovertLexicon == self.NullExcorporatorLexicon:
            self.NullExcorporatorLexicon = sourceCovertLexiconCopy
        self.covertLexicons = [(self.CovertLexicon, 'CovertLexicon'),
        (self.ExtraposerLexicon, 'ExtraposerLexicon'),
        (self.TypeRaiserLexicon, 'TypeRaiserLexicon'),
        (self.ToughOperatorLexicon, 'ToughOperatorLexicon'),
        (self.NullExcorporatorLexicon, 'NullExcorporatorLexicon')]
        
    def restore_mappings(self, CatTreeMappingsCopy, TreeCatMappingsCopy, nullCatTreeMappingsCopy, nullTreeCatMappingsCopy, covertCatCommentsCopy, overtCatCommentsCopy, autoCatTreeMappingsCopy, autoTreeCatMappingsCopy, autoNullCatTreeMappingsCopy, autoNullTreeCatMappingsCopy):
        #seems to be a bug where if modification fails, sometimes a tree goes missing from
        #CatTreeMappings.. so I'm saving all these files from the copies I took before modification began.
        with open(self.seed_folder+"/"+'CatTreeMappings', 'w') as CatTreeMappingsFile:
            json.dump(CatTreeMappingsCopy, CatTreeMappingsFile)
        with open(self.seed_folder+"/"+'TreeCatMappings', 'w') as TreeCatMappingsFile:
            json.dump(TreeCatMappingsCopy, TreeCatMappingsFile)
        with open(self.seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
            json.dump(nullCatTreeMappingsCopy, nullCatTreeMappingsFile)
        with open(self.seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
            json.dump(nullTreeCatMappingsCopy, nullTreeCatMappingsFile)
        with open(self.auto_folder+"/"+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
            json.dump(autoCatTreeMappingsCopy, autoCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
            json.dump(autoTreeCatMappingsCopy, autoTreeCatMappingsFile)
        with open(self.auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
            json.dump(autoNullCatTreeMappingsCopy, autoNullCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
            json.dump(autoNullTreeCatMappingsCopy, autoNullTreeCatMappingsFile)
        with open(self.seed_folder+"/"+'covertCatComments', 'w') as covertCatCommentsFile:
            json.dump(covertCatCommentsCopy, covertCatCommentsFile)
        with open(self.seed_folder+"/"+'overtCatComments', 'w') as overtCatCommentsFile:
            json.dump(overtCatCommentsCopy, overtCatCommentsFile)
        
    def viewParseQuestion(self, message, seed, reason=None, times=None, construct_lexicon=None, compose=None, failed_parses=None, different_parses=None, singleTree=None, total_start_time=None, parse_times=None, break_time_start=None, corpus=None, overwrite_derivation=False):
        if total_start_time != None:
            break_time = default_timer()
        if corpus == 'seeds':
            self.corpusType = 'seed'
        elif corpus == 'autos':
            self.corpusType = 'auto'
        if self.viewParseQuestionWindow != None:
            self.destroyWindow(self.viewParseQuestionWindow, 'viewParseQuestionWindow')
        self.viewParseQuestionWindow = Toplevel(self.mainWindow)
        self.viewParseQuestionWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.viewParseQuestionWindow, 'viewParseQuestionWindow'))
        w=1000
        h=190
        (x, y) = self.getCentrePosition(w, h)
        self.viewParseQuestionWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        if reason == None:
            self.viewParseQuestionWindow.title("Aborting Modification!!!")
        elif reason == 'failed':
            self.viewParseQuestionWindow.title("Parsing Failed!!!")
        elif reason == 'different':
            self.viewParseQuestionWindow.title("Wrong Parse Generated!!!")           
        label = Label(self.viewParseQuestionWindow, text=message)
        label.pack()
        if reason == None or singleTree != None:
            label2 = Label(self.viewParseQuestionWindow, text="\nWould you like to view this tree?\n")
        else:
            label2 = Label(self.viewParseQuestionWindow, text="\nWhat would you like to do?\n")
        label2.pack()
        buttonFrame = Frame(self.viewParseQuestionWindow)
        buttonFrame.pack()
        if reason == None or singleTree != None:
            yesButton = Button(buttonFrame, text="Yes", command=lambda: self.viewParse(seed))
            yesButton.pack(side=LEFT)
            noButton = Button(buttonFrame, text="No", command=lambda: self.destroyWindow(self.viewParseQuestionWindow, 'viewParseQuestionWindow'))
            noButton.pack(side=LEFT)
        else:
            viewParseButton = Button(buttonFrame, text="open parse", command=lambda: self.viewParse(seed))
            viewParseButton.pack(side=LEFT)
            continueParsingButton = Button(buttonFrame, text="continue reparsing", command=lambda: self.reparse_all_trees(construct_lexicon=False, compose=compose, times=times, start_file=seed[0], start_line_num=seed[1], failed_parses=failed_parses, different_parses=different_parses, continue_parsing=True, total_start_time=total_start_time, parse_times=parse_times, break_time_start=break_time_start, corpus=corpus, overwrite_derivation=overwrite_derivation))
            continueParsingButton.pack(side=LEFT)
            stopButton = Button(buttonFrame, text="close", command=lambda: self.destroyWindow(self.viewParseQuestionWindow, 'viewParseQuestionWindow'))
            stopButton.pack(side=LEFT)

    def viewParse(self, seed):
        self.seed = seed
        self.fileManagerFolder = seed[0]
        self.section_folder = seed[0].split("/")[1]
        if self.section_folder != 'new_parses':
            self.ptb_file = seed[0].split("/")[2]
            ptbFile = self.ptb_folder+"/"+"/".join(seed[0].split("/")[1:])
            ptbBracketing = self.getPTBbracketing(ptbFile, seed)
            t = autobank.build_tree(ptbBracketing)
            tree = t[0]
            self.terminals = t[2]
            newPtbBracketing = tree.generate_bracketing()
            self.PTBbracketing = newPtbBracketing
            try:
                trees = [Tree.parse(newPtbBracketing[1:-1], remove_empty_top_bracketing=True)]
            except AttributeError:
                trees = [Tree.fromstring(newPtbBracketing[1:-1], remove_empty_top_bracketing=True)]
        self.newSentMode = False
        self.test_words = None
        self.untokenizedTestSentence = None
        self.mainWindow.destroy()
        seed_bracketings = json.load(open(seed[0]))[str(seed[1])]
        self.subcat_derivation_bracketings = [seed_bracketings[0]]
        self.xbar_bracketings = [seed_bracketings[1]]
        self.derivation_bracketings = [seed_bracketings[4]]
        self.derived_bracketings = [seed_bracketings[2]]
        self.subcat_full_derivation_bracketings = [seed_bracketings[3]]
        self.full_derivation_bracketings = [seed_bracketings[5]]
        db = cky_mg.fix_coord_annotation(self.derivation_bracketings[0])
        sdb = cky_mg.fix_coord_annotation(self.subcat_derivation_bracketings[0])
        sfdb = cky_mg.fix_coord_annotation(self.subcat_full_derivation_bracketings[0])
        fdb = cky_mg.fix_coord_annotation(self.full_derivation_bracketings[0])
        while "  " in sfdb:
            sfdb = re.sub("  ", " ", sfdb, count=10000)
        while "  " in sfdb:
            fdb = re.sub("  ", " ", sfdb, count=10000)
        try:
            self.xbar_trees = [Tree.parse(self.xbar_bracketings[0], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.subcat_derivation_trees = [Tree.parse(sdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.full_derivation_trees = [Tree.parse(fdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.subcat_full_derivation_trees = [Tree.parse(sfdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.derivation_trees = [Tree.parse(db, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.derived_trees = [Tree.parse(self.derived_bracketings[0], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
        except AttributeError:
            self.xbar_trees = [Tree.fromstring(self.xbar_bracketings[0], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.subcat_derivation_trees = [Tree.fromstring(sdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.full_derivation_trees = [Tree.fromstring(fdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.subcat_full_derivation_trees = [Tree.fromstring(sfdb, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.derivation_trees = [Tree.fromstring(db, remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
            self.derived_trees = [Tree.fromstring(self.derived_bracketings[0], remove_empty_top_bracketing=True, node_pattern=node_pattern, leaf_pattern=leaf_pattern)]
        self.trees = self.xbar_trees
        if self.section_folder != 'new_parses':
            self.ptb_file_line_number = seed[1]
        if self.corpusType == 'seed':
            self.removingFromAutos = False
        elif self.corpusType == 'auto':
            self.removingFromAutos = True
        self.genMainWindow(mode='viewer')
        if self.section_folder != 'new_parses':
            self.PTBtree = trees[0]
            self.showtrees(treeType='PTB', trees=trees)
        self.showtrees(treeType='MG')
        self.refreshSpins()

    def getPTBbracketing(self, ptbFile, seed):
        l=-1
        for line in open(ptbFile):
            l+= 1
            if l == seed[1]:
                if line[-1:] == '\n':
                    line = line[0:-1]
                return line
        
    def destroyViewDelCatsWindow(self, window, windowName):
        self.destroyWindow(window, windowName)

    def confirm(self, catType=None, confirmWhat=None, oldCat=None, entry=None, morphemeNameEntry=None, nullLex=None, oldMorphemeName=None,  oldCatIsCoord=None, message=None):
        noReparse = False
        if self.confirmWindow != None:
            self.destroyWindow(self.confirmWindow, 'confirmWindow')
        if confirmWhat == 'delCats':
            if len(self.catsBox.curselection()) == 0:
                self.nothing(self.viewDelCatsWindow, "You haven't selected any categories to delete!")
                return
            catToDelete = self.catsBox.get(self.catsBox.curselection()[0])[1:].split("     ")[0]
            if catType == 'overt':
                try:
                    CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
                except IOError:
                    CatTreeMappings = {}
                if catToDelete in CatTreeMappings:
                    numParsesWithCat = len(CatTreeMappings[catToDelete])
                else:
                    numParsesWithCat = 0
                try:
                    autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
                except IOError:
                    autoCatTreeMappings = {}
                if catToDelete in autoCatTreeMappings:
                    numAutoParsesWithCat = len(autoCatTreeMappings[catToDelete])
                else:
                    numAutoParsesWithCat = 0
            elif catType == 'covert':
                try:
                    nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
                except IOError:
                    nullCatTreeMappings = {}
                if catToDelete in nullCatTreeMappings:
                    numParsesWithCat = len(nullCatTreeMappings[catToDelete])
                else:
                    numParsesWithCat = 0
                try:
                    autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
                except IOError:
                    autoNullCatTreeMappings = {}
                if catToDelete in autoNullCatTreeMappings:
                    numAutoParsesWithCat = len(autoNullCatTreeMappings[catToDelete])
                else:
                    numAutoParsesWithCat = 0
            if numParsesWithCat == 1:
                seedParsesText = '1 parse'
                containSeeds = 'contains'
            else:
                seedParsesText = str(numParsesWithCat)+' parses'
                containSeeds = 'contain'
            if numAutoParsesWithCat == 1:
                autoParsesText = '1 parse'
                containAutos = 'contains'
            else:
                autoParsesText = str(numAutoParsesWithCat)+' parses'
                containAutos = 'contain'
            if numAutoParsesWithCat == numParsesWithCat == 0:
                message = "Are you sure you wish to delete the selected category?\nThere are currently no parses in either the seed set\nor the auto set which contain this category.."
            else:
                if numAutoParsesWithCat > 0 and numParsesWithCat > 0:
                    this = 'These'
                    message = "Are you sure you wish to delete the selected category?\n"+seedParsesText+" in the seed set currently "+containSeeds+" this category and "+autoParsesText+"\nin the auto set currently "+containAutos+" it. "+this+" will be removed!"
                elif numParsesWithCat > 0:
                    if numParsesWithCat > 1:
                        this = 'These'
                    else:
                        this = 'This'
                    message = "Are you sure you wish to delete the selected category?\n"+seedParsesText+" in the seed set currently "+containSeeds+" this category.\n"+this+" will be removed!"
                elif numAutoParsesWithCat > 0:
                    if numAutoParsesWithCat > 1:
                        this = 'These'
                    else:
                        this = 'This'
                    message = "Are you sure you wish to delete the selected category?\n"+autoParsesText+" in the auto set currently "+containAutos+" this category.\n"+this+" will be removed!"
            h=90
            command=lambda: self.deleteCats(catType)
            self.confirmWindow = Toplevel(self.viewDelCatsWindow)
            confirmButtonText = 'Delete'
            self.confirmWindow.title('Confirm Deletion')
        elif confirmWhat == 'modCats':
            features = self.sortSubCat(entry.get().strip().split(" "))
            if message == 'This overt MG category is currently\nnot used in any seed parses.\nNo reparsing is therefore required.\n\nDo you still wish to continue?' or message == 'This covert MG category is currently\nnot used in any seed parses.\nNo reparsing is therefore required.\n\nDo you still wish to continue?':
                h=120
                noReparse = True
            else:
                h=90
            if catType == 'overt':
                if len(features) == 0:
                    self.nothing(self.addOvertCatWindow, message="You haven't entered any MG features!")
                    return
                if len(self.ocentry.get().strip()) > 100:
                    self.nothing(self.addOvertCatWindow, message="Comments exceed 100 characters!")
                    return
                mapping = []
                #if this category is a coordinator, we indicate this with an overline on the
                #double colon type indicator
                if self.checkboxVars['coordinator'][0].get() == 1:
                    features = u':\u0305:\u0305 '+features
                else:
                    features = ':: '+features
                features = features.split(" ")
                #we need to sort the sel_features (subcat) since the parser does this
                #and this ensures that the order in which the user types in subcat features
                #doesn't matter..
                new_features = []
                new_features.append(features[0])
                new_features += self.sortSubCat(features[1:], return_list=True)
                features = " ".join(new_features)
                for cat in PTBcats:
                    if cat not in mapping:
                        if self.checkboxVars[cat][0].get() == 1:
                            mapping.append(cat)
                if len(mapping) == 0:
                    self.nothing(self.addOvertCatWindow, message="You haven't selected any PTB categories!")
                    return
                alreadyExistsIn = []
                with open('Junk', 'w') as JunkFile:
                    json.dump(features, JunkFile)
                json_features = json.load(open('Junk'))
                for cat in PTBcats:
                    if features in self.PosMappings[cat] or json_features in self.PosMappings[cat]:
                        alreadyExistsIn.append(cat)
                if len(alreadyExistsIn) != 0:
                    if json_features != oldCat:
                        self.alreadyExists(currentWindow=self.addOvertCatWindow, alreadyExistsIn=alreadyExistsIn, mapping=mapping, features=features, catType='overt')
                        return
                features = " ".join(features.split(" ")[1:])
                if self.checkboxVars['coordinator'][0].get() == 1:
                    newCat = u':\u0305:\u0305 '+features
                else:
                    newCat = ':: '+features
                oldCatPTBmappings = self.getOldCatPTBMappings(oldCat)
                newCatPTBmappings = self.getNewCatPTBMappings()
                self.oldAndNewPosMappingsSubset = True
                for oldCatMapping in oldCatPTBmappings:
                    if oldCatMapping not in newCatPTBmappings:
                        self.oldAndNewPosMappingsSubset = False
                        break
                if oldCatPTBmappings == newCatPTBmappings:
                    self.oldAndNewPosMappingsMatch = True
                if oldCat == newCat and self.ocentry.get().strip() == self.overtCatComments[oldCat] and self.oldAndNewPosMappingsMatch:
                    self.nothing(self.addCovertCatWindow, "You haven't modified the category!")
                    self.oldAndNewPosMappingsMatch = False
                    self.oldAndNewPosMappingsSubset = False
                    return
                elif newCatPTBmappings == []:
                    self.nothing(self.addCovertCatWindow, "You haven't selected any PTB categories!")
                    self.oldAndNewPosMappingsMatch = False
                    self.oldAndNewPosMappingsSubset = False
                    return
                elif oldCat == newCat and self.oldAndNewPosMappingsSubset:
                    #we just need to update the message to say that only the comments have been changed so no parsing is necessary
                    message = "No reparsing is required to update this category.\n\nDo you wish to commit the change?"
                    h=90
                    noReparse = True
                    self.oldCatIsNewCat = True
                if not noReparse:
                    commandAtomic=lambda: self.modify(oldCat = oldCat, newFeatures=features, cbVars=self.checkboxVars, catType='overt', MGtagType='atomic')
                    commandSupertag=lambda: self.modify(oldCat = oldCat, newFeatures=features, cbVars=self.checkboxVars, catType='overt', MGtagType='supertag')
                else:
                    command=lambda: self.modify(oldCat = oldCat, newFeatures=features, cbVars=self.checkboxVars, catType='overt')
            elif catType == 'covert':
                if self.addCovertCatWindow != None:
                    self.newComments = self.ccentry.get().strip()
                    if len(self.newComments) > 100:
                        self.nothing(self.addCovertCatWindow, message="Comments exceed 100 characters!")
                        return
                try:
                    morphemeName = morphemeNameEntry.get()
                    MGfeatures = features
                except TclError:
                    if self.morphemeName != None and self.MGfeatures != None:
                        morphemeName = self.morphemeName
                        MGfeatures = self.MGfeatures
                    else:
                        raise Exception("Oops, unable to retrieve info from destroyed addCovertCatWindow!")
                except AttributeError:
                    if self.morphemeName != None and self.MGfeatures != None:
                        morphemeName = self.morphemeName
                        MGfeatures = self.MGfeatures
                    else:
                        raise Exception("Oops, unable to retrieve info from destroyed addCovertCatWindow!")
                if morphemeName[0] != "[":
                    morphemeName = "["+morphemeName
                if morphemeName[-1] != "]":
                    morphemeName = morphemeName+"]"
                for char in morphemeName[1:-1]:
                    if char in [")", "(", " ", ":", ";", ",", "{", "}"]:
                        self.nothing(self.addCovertCatWindow, message="Illicit characters detected in morpheme label!")
                        return
                oldFeatures = oldCat.split(" ")
                #quick check to make sure the user has actually modified the category
                self.oldCatIsNewCat = False
                self.oldAndNewPosMappingsMatch = False
                self.oldAndNewPosMappingsSubset = False
                if (oldFeatures[1] == "::" and self.CovertCoordinator.get() == 0) or (oldFeatures[1] in [':\u0305:\u0305', ':\\u0305:\\u0305'] and self.CovertCoordinator.get() == 1):
                    if " ".join(oldFeatures[2:]) == MGfeatures:
                        if self.lexiconVar.get()+"Lexicon" == nullLex and oldMorphemeName == morphemeName:
                            self.oldCatIsNewCat = True
                            for e in self.covertCatComments[oldCat]:#5
                                if e[0] == nullLex:
                                    if morphemeName == oldFeatures[0]:
                                        try:
                                            if e[1] == self.ccentry.get().strip():
                                                self.nothing(self.addCovertCatWindow, "You haven't modified the category!")
                                                self.oldAndNewPosMappingsMatch = False
                                                self.oldAndNewPosMappingsSubset = False
                                                self.checkedForDuplicates = False
                                                return
                                        except TclError:
                                            if e[1] == self.newComments:
                                                self.nothing(self.addCovertCatWindow, "You haven't modified the category!")
                                                self.oldAndNewPosMappingsMatch = False
                                                self.oldAndNewPosMappingsSubset = False
                                                self.checkedForDuplicates = False
                                                return
                if self.CovertCoordinator.get() == 1:
                    newCatLexEntry = list(self.constructMGlexEntry(morphemeName, (':\u0305:\u0305 '+MGfeatures).split(" ")))
                    with open('Junk', 'w') as junkFile:
                        json.dump(newCatLexEntry, junkFile)
                    newCatLexEntry = json.load(open('Junk'))#4
                    newCat = morphemeName+u' :\u0305:\u0305 '+(MGfeatures.strip())
                else:
                    newCatLexEntry = list(self.constructMGlexEntry(morphemeName, (':: '+MGfeatures).split(" ")))
                    with open('Junk', 'w') as junkFile:
                        json.dump(newCatLexEntry, junkFile)
                    newCatLexEntry = json.load(open('Junk'))#5
                    newCat = morphemeName+u' :: '+(MGfeatures.strip())
                if self.checkedForDuplicates == False:
                    #unfortuantely, we have to send the code on a loop here to avoid the situation
                    #where too many message windows pop up at the same time..this is because even though you
                    #generate a confirmation window, any code after that generation is executed before the user
                    #has had a chance to press a button.. so we come back to confirm() after the appropriate checks
                    #of duplicate copies of this category in other lexicons and keep an indicator variable to tell us
                    #that the checks have already been done.  THIS WAS BADLY CODED BEFORE I KNEW WHAT I WAS DOING..
                    #the whole modify() function could do with rewriting from scratch.
                    self.checkedForDuplicates = True
                    if self.lexiconVar.get() == 'None':
                        self.nothing(self.addCovertCatWindow, 'You must select a null lexicon!')
                        self.checkedForDuplicates = False
                        return
                    elif self.lexiconVar.get() == 'Covert':
                        LEXICON = self.CovertLexicon
                        fileName = self.seed_folder+'/CovertLexicon'
                    elif self.lexiconVar.get() == 'Extraposer':
                        LEXICON = self.ExtraposerLexicon
                        fileName = self.seed_folder+'/ExtraposerLexicon'
                    elif self.lexiconVar.get() == 'TypeRaiser':
                        LEXICON = self.TypeRaiserLexicon
                        fileName = self.seed_folder+'/TypeRaiserLexicon'
                    elif self.lexiconVar.get() == 'ToughOperator':
                        LEXICON = self.ToughOperatorLexicon
                        fileName = self.seed_folder+'/ToughOperatorLexicon'
                    elif self.lexiconVar.get() == 'NullExcorporator':
                        LEXICON = self.NullExcorporatorLexicon
                        fileName = self.seed_folder+'/NullExcorporatorLexicon'
                    alreadyInLexicons = []
                    for covertLexicon in self.covertLexicons:
                        name_set = set([entry[0] for entry in covertLexicon[0]])
                        if self.lexiconVar.get()+"Lexicon" != covertLexicon[1] and newCatLexEntry[0] in name_set and not (covertLexicon[1] == nullLex and len([entry[0] for entry in covertLexicon[0] if entry[0] == newCatLexEntry[0]]) == 1):
                            if self.addCovertCatWindow != None:
                                self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow')
                            #had to destory the addCovertCat window at this point and force user to start again
                            #because there were problems otherwise..
                            self.nothing(self.mainWindow, "The name "+newCatLexEntry[0]+" is already used in the "+covertLexicon[1]+"!", width=500)
                            return
                        elif nullLex == covertLexicon[1] and self.lexiconVar.get()+"Lexicon" != nullLex:
                            #we will not alert the user that the category we are adding is already in its
                            #current lexicon, unless the source and target lexicons are the same
                            continue
                        elif newCatLexEntry in covertLexicon[0]:
                            alreadyInLexicons.append(covertLexicon[1])
                    if len(alreadyInLexicons) > 0 and newCat != oldCat:
                        MGentry = self.constructMGlexEntry(oldMorphemeName, newCat.split(" ")[1:])
                        self.alreadyExists(currentWindow = self.addCovertCatWindow, features=newCat.split(" ")[1:], nullString = oldMorphemeName, alreadyInLexicons=alreadyInLexicons, catType='covert', LEXICON=LEXICON, entry=entry, fileName=fileName, confirmWhat=confirmWhat, oldCat=oldCat, morphemeNameEntry=morphemeNameEntry, nullLex=nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, messageConfirm=message, MGentry=MGentry)
                        self.checkedForDuplicates = False
                        return
                else:
                    self.modify(oldCat = oldCat, confirmWhat=confirmWhat, newFeatures=MGfeatures, catType='covert', morphemeName = "["+morphemeName+"]", nullLex = nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, entry=entry, morphemeNameEntry=morphemeNameEntry, inLoop=True)
                    return
                if self.oldCatIsNewCat:
                    message = "No reparsing is required to update this category.\n\nDo you wish to commit the change?"
                    h = 90
                    noReparse = True
                if not noReparse:
                    commandAtomic=lambda: self.modify(oldCat = oldCat, confirmWhat=confirmWhat, newFeatures=MGfeatures, catType='covert', morphemeName = "["+morphemeName+"]", nullLex = nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, entry=entry, morphemeNameEntry=morphemeNameEntry, MGtagType='atomic')
                    commandSupertag=lambda: self.modify(oldCat = oldCat, confirmWhat=confirmWhat, newFeatures=MGfeatures, catType='covert', morphemeName = "["+morphemeName+"]", nullLex = nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, entry=entry, morphemeNameEntry=morphemeNameEntry, MGtagType='supertag')
                else:
                    command=lambda: self.modify(oldCat = oldCat, confirmWhat=confirmWhat, newFeatures=MGfeatures, catType='covert', morphemeName = "["+morphemeName+"]", nullLex = nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, entry=entry, morphemeNameEntry=morphemeNameEntry)
            self.confirmWindow = Toplevel(self.addOvertCatWindow)
            confirmButtonText = 'Continue'
            self.confirmWindow.title('Confirm Modification')
        elif confirmWhat == 'removeParse':
            h=70
            command = self.removeTree
            if self.removingFromFilebox == True:
                self.confirmWindow = Toplevel(self.fileManagerWindow)
                self.removingFromFilebox = False
            else:
                self.confirmWindow = Toplevel(self.mainWindow)
            confirmButtonText = 'Continue'
            self.confirmWindow.title('Confirm Remove Seed Parse')
        elif confirmWhat == 'moveParse':
            h=90
            command = self.moveParse2
            if self.movingFromFilebox == True:
                self.confirmWindow = Toplevel(self.fileManagerWindow)
                self.movingFromFilebox = False
            else:
                self.confirmWindow = Toplevel(self.mainWindow)
            confirmButtonText = 'Continue'
            self.confirmWindow.title('Confirm move auto parse to seeds')
        #to avoid a bug where if the user types in modifications while the confirm
        #modifiction screen is displayed the system crashes, we withdraw (but importantly do not
        #destroy the entry windows
        if catType == 'covert':
            if self.addCovertCatWindow != None:
                self.addCovertCatWindow.withdraw()
        elif catType == 'overt':
            if self.addOvertCatWindow != None:
                self.addOvertCatWindow.withdraw()
        self.confirmWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.confirmWindow, 'confirmWindow'))
        w=520
        (x, y) = self.getCentrePosition(w, h)
        self.confirmWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.confirmWindow, text=message)
        label.pack()
        buttonFrame = Frame(self.confirmWindow)
        buttonFrame.pack()
        if confirmWhat == 'modCats' and not noReparse:
            confirmAtomicButton = Button(buttonFrame, text='use atomic tags', command=commandAtomic)
            confirmAtomicButton.pack(side=LEFT)
            confirmSupertagButton = Button(buttonFrame, text='use supertags', command=commandSupertag)
            confirmSupertagButton.pack(side=LEFT)
        else:
            confirmButton = Button(buttonFrame, text=confirmButtonText, command=command)
            confirmButton.pack(side=LEFT)
        cancelButton = Button(buttonFrame, text="Cancel", command=self.destroyConfirmWindow)
        cancelButton.pack(side=LEFT)

    def destroyConfirmWindow(self):
        self.checkedForDuplicates = False
        self.destroyWindow(self.confirmWindow, 'confirmWindow')

    def getOldCatPTBMappings(self, old_cat):
        cats = []
        for entry in self.PosMappings:
            if old_cat in self.PosMappings[entry]:
                cats.append(entry)
        cats.sort()
        return cats

    def getNewCatPTBMappings(self):
        cats = []
        for entry in self.checkboxVars:
            if self.checkboxVars[entry][0].get() == 1:
                if entry != 'coordinator':
                    cats.append(entry)
        cats.sort()
        return cats

    def sortSubCat(self, features, return_list=False):
        new_features = []
        for feature in features:
            if "^" in feature and "^" != feature[-1]:
                feature = re.sub("\^", "", feature)
                feature = feature+"^"
            feature = re.sub("\(", "[", feature, count=1000)
            feature = re.sub("\)", "]", feature, count=1000)
            if feature.count("[") != feature.count("]"):
                self.nothing(self.mainWindow, message="Invalid feature sequence detected (check brackets)!")
                raise Exception("Error: Invalid feature sequence detected (check brackets)!")
            while ".." in feature:
                feature = re.sub("\.\.", ".", feature)
            while "||" in feature:
                feature = re.sub("\|\|", "|", feature)
            sel_features = []
            sf = re.search('{.*?}', feature)
            if sf:
                sf = sf.group(0)
                #in case the user entered in the optional brackets as round brackets
                #we convert them to square brackets here (so they don't get misinterpreted
                #when converting bracketings to trees..
                sf = sf[1:-1]
                #now we will correct any instances where | was incorrectly entered as . and vice versa
                inside_or = False
                new_string = ""
                for char in sf:
                    if char == '[':
                        inside_or = True
                    elif char == ']':
                        inside_or = False
                    elif char == ".":
                        if inside_or:
                            char = "|"
                    elif char == "|":
                        if not inside_or:
                            char = "."
                    new_string += char
                sf = new_string
                sf = sf.split(".")
                while '' in sf:
                    sf.remove('')
                f_index = -1
                for f in sf:
                    f_index += 1
                    if re.search('^[+-]\[.*?\]', f):
                        #first we sort the features inside any optional bracketing
                        sign = f[0]
                        bracketed_features = f[2:-1]
                        bracketed_features = re.sub("\.", "|", bracketed_features, count = 1000)
                        bracketed_features=list(set(bracketed_features.split("|")))
                        while '' in bracketed_features:
                            bracketed_features.remove('')
                        bracketed_features.sort()
                        bracketed_features=sign+"["+"|".join(bracketed_features)+"]"
                        sf[f_index] = bracketed_features
                    elif re.search('^\[.*\]', f):
                        #the case where the +/- are inside not outside the OR brackets
                        bracketed_features = f[1:-1]
                        bracketed_features = re.sub("\.", "|", bracketed_features, count = 1000)
                        bracketed_features=list(set(bracketed_features.split("|")))
                        while '' in bracketed_features:
                            bracketed_features.remove('')
                        bracketed_features.sort()
                        bracketed_features="["+"|".join(bracketed_features)+"]"
                        sf[f_index] = bracketed_features
                sf = ".".join(sf)
                feature = re.sub('{.*?}', '{'+sf+'}', feature)
                sf = re.search('{.*?}', feature).group(0)
                #find the index positions of { and }
                sf=sf[1:-1]
                sf = list(set(sf.split(".")))
                sf.sort()
                subcat = '{'+".".join(sf)+'}'
                feature = re.sub("{.*?}", "", feature)
                featurePrefix = ""
                featureSuffix = ""
                featureCat = ""
                in_prefix = True
                for char in feature:
                    if char not in string.letters:
                        if in_prefix:
                            featurePrefix+=char
                        else:
                            featureSuffix+=char
                    else:
                        featureCat+=char
                        in_prefix = False
                feature = featurePrefix+featureCat+subcat+featureSuffix
            if feature != '':
                new_features.append(feature)
        if return_list == True:
            return new_features
        else:
            new_features = " ".join(new_features)
            return new_features

    def deleteCats(self, catType):
        self.lengthCountsOfRemovedTrees = {}
        self.destroyWindow(self.confirmWindow, 'confirmWindow')
        try:
            CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
        except IOError:
            CatTreeMappings = {}
        try:
            TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
        except IOError:
            TreeCatMappings = {}
        try:
            nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
        except IOError:
            nullCatTreeMappings = {}
        try:
            nullTreeCatMappings = json.load(open(self.seed_folder+"/"+'nullTreeCatMappings'))
        except IOError:
            nullTreeCatMappings = {}
        try:
            autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
        except IOError:
            autoCatTreeMappings = {}
        try:
            autoTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoTreeCatMappings'))
        except IOError:
            autoTreeCatMappings = {}
        try:
            autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
        except IOError:
            autoNullCatTreeMappings = {}
        try:
            autoNullTreeCatMappings = json.load(open(self.auto_folder+"/"+'autoNullTreeCatMappings'))
        except IOError:
            autoNullTreeCatMappings = {}
        currentFile = ""
        catsToDelete = []
        if catType == 'overt':
            CTM = CatTreeMappings
            ACTM = autoCatTreeMappings
        elif catType == 'covert':
            CTM = nullCatTreeMappings
            ACTM = autoNullCatTreeMappings
        for i in self.catsBox.curselection():
            #there will only be one selection possible as I removed the option to select
            #multiple categories in listbox as this loop didn't work for deleting multiple categories..
            #the five spaces split is to get rid of the name of the lexicon which is
            #attached to the category names for null categories..
            catToDelete = self.catsBox.get(i)[1:].split("     ")[0]
            #If overt, we now delete the category from PosMappings and also from PosDepMappings..
            #we can't delete its effects from DepMappings, however, and for this reason these are only
            #constructed once the user clicks auto generate corpus using the entire seed set
            #in its current state..
            if catType == 'overt':
                self.overtComments.set('')
                del(self.overtCatComments[catToDelete])
                with open(self.seed_folder+"/"+'overtCatComments', 'w') as overtCatCommentsFile:
                    json.dump(self.overtCatComments, overtCatCommentsFile)
                for PTBentry in self.PosMappings:
                    if catToDelete in self.PosMappings[PTBentry]:
                        self.PosMappings[PTBentry].remove(catToDelete)
                        if len(self.PosMappings[PTBentry]) == 1:
                            self.PosMappings[PTBentry] = ["No categories available", ""]
                self.catsBox.delete(i)
            elif catType == 'covert':
                self.covertComments.set('')
                covLex = self.catsBox.get(i)[1:].split("     ")[1][1:-1]
                for e in self.covertCatComments[catToDelete]:
                    if e[0] == covLex:
                        self.covertCatComments[catToDelete].remove(e)
                        break
                if len(self.covertCatComments[catToDelete]) == 0:
                    del(self.covertCatComments[catToDelete])
                with open(self.seed_folder+"/"+'covertCatComments', 'w') as covertCatCommentsFile:
                    json.dump(self.covertCatComments, covertCatCommentsFile)
                for entry in self.covertRefs[catToDelete]:
                    if entry[0] == self.catsBox.get(i)[1:].split("     ")[1][1:-1]:
                        covertLex = self.getCovertLexicon(entry[0])
                        del(covertLex[entry[1]])
                        with open(self.seed_folder+"/"+entry[0], 'w') as covertLexFile:
                            json.dump(covertLex, covertLexFile)
                        self.covertRefs = self.updateCovertRefs()[1]
                        self.catsBox.delete(i)
                        break
            #we also need to remove all the trees that this MG cat was involved in generating
            #from the seed set..
            for MAPPINGS in [CTM, ACTM]:
                if catToDelete in MAPPINGS:
                    mappingIndex = 0
                    for entry in MAPPINGS[catToDelete]:
                        mappingIndex+=1
                        if entry[0].split("/")[1] != "new_parses" and MAPPINGS == CTM:
                            #if deleting a seed, we need to increment the counts for this line length
                            ptbFile = self.ptb_folder+"/"+"/".join(entry[0].split("/")[1:])
                            ptbBracketing = self.getPTBbracketing(ptbFile, entry)
                            terminals = autobank.build_tree(ptbBracketing)[2]
                            self.terminals = terminals
                            if len(terminals) in self.lengthCountsOfRemovedTrees:
                                self.lengthCountsOfRemovedTrees[len(terminals)] += 1
                            else:
                                self.lengthCountsOfRemovedTrees[len(terminals)] = 1
                        else:
                            self.terminals = None
                        if MAPPINGS == CTM:
                            ctm = CatTreeMappings
                            tcm = TreeCatMappings
                            ntcm = nullTreeCatMappings
                            nctm = nullCatTreeMappings
                        elif MAPPINGS == ACTM:
                            ctm = autoCatTreeMappings
                            tcm = autoTreeCatMappings
                            ntcm = autoNullTreeCatMappings
                            nctm = autoNullCatTreeMappings
                        if str([entry[0].encode('ascii'), entry[1]]) in tcm:
                            del(tcm[str([entry[0].encode('ascii'), entry[1]])])
                        if str([entry[0].encode('ascii'), entry[1]]) in ntcm:
                            del(ntcm[str([entry[0].encode('ascii'), entry[1]])])
                        if currentFile != entry[0]:
                            if currentFile != "":
                                if seedSet == {}:
                                    try:
                                        os.remove(currentFile)
                                        if currentFile.split("/")[1] == "new_parses":
                                            os.remove(self.seed_folder+'/new_parse_strings/'+currentFile.split("/")[2])
                                    except Exception:
                                        pass
                                else:
                                    with open(currentFile, 'w') as seedFile:
                                        json.dump(seedSet, seedFile)
                            currentFile = entry[0]
                            seedSet = json.load(open(currentFile))
                        del(seedSet[str(entry[1])])
                        with open(currentFile, 'w') as seedFile:
                            json.dump(seedSet, seedFile)
                        for mappings in [ctm, nctm]:
                            for ENTRY in mappings:
                                #now we need to go through CatTreeMappings and nullCatTreeMappings and remove all instances
                                #of the parses which we have just removed from the corpus..
                                #we want to miss out the parses inside the MG category itself however,
                                #as these will be deleted when the whole MG category is deleted, otherwise
                                #this loop will not be able to finish..
                                if ENTRY == catToDelete:
                                    continue
                                parses_to_delete = []
                                for file_line in mappings[ENTRY]:
                                    if (file_line[0], file_line[1]) == (entry[0], entry[1]):
                                        parses_to_delete.append(file_line)
                                for file_line in parses_to_delete:
                                    mappings[ENTRY].remove(file_line)
                        if mappingIndex == len(MAPPINGS[catToDelete]):
                            #for the final file in MAPPINGS we must save the new seeds at this point before exiting this loop for the final time
                            if seedSet == {}:
                                try:
                                    os.remove(currentFile)
                                    if currentFile.split("/")[1] == "new_parses":
                                        os.remove(self.seed_folder+'/new_parse_strings/'+currentFile.split("/")[2])
                                except Exception:
                                    pass
                            else:
                                with open(currentFile, 'w') as seedFile:
                                    json.dump(seedSet, seedFile)
                    #finally, we also delete the MG category entry from CatTreeMappings or nullCatTreeMappings..
                    if catType == 'overt':
                        del(ctm[catToDelete])
                    elif catType == 'covert':
                        del(nctm[catToDelete])
                    entries_to_delete = []
                    for mappings in [ctm, nctm]:
                        for entry in mappings:
                            if len(mappings[entry]) == 0:
                                entries_to_delete.append(entry)
                        for entry in entries_to_delete:
                            del(mappings[entry])
                        entries_to_delete = []
                        if len(mappings) == 1:
                            for entry in mappings:
                                if mappings[entry] == []:
                                    mappings = {}
        with open(self.seed_folder+"/"+'PosMappings', 'w') as PosMappingsFile:
            json.dump(self.PosMappings, PosMappingsFile)
        with open(self.seed_folder+"/"+'CatTreeMappings', 'w') as CatTreeMappingsFile:
            json.dump(CatTreeMappings, CatTreeMappingsFile)
        with open(self.seed_folder+"/"+'TreeCatMappings', 'w') as TreeCatMappingsFile:
            json.dump(TreeCatMappings, TreeCatMappingsFile)
        with open(self.seed_folder+"/"+'nullCatTreeMappings', 'w') as nullCatTreeMappingsFile:
            json.dump(nullCatTreeMappings, nullCatTreeMappingsFile)
        with open(self.seed_folder+"/"+'nullTreeCatMappings', 'w') as nullTreeCatMappingsFile:
            json.dump(nullTreeCatMappings, nullTreeCatMappingsFile)
        with open(self.auto_folder+"/"+'autoCatTreeMappings', 'w') as autoCatTreeMappingsFile:
            json.dump(autoCatTreeMappings, autoCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoTreeCatMappings', 'w') as autoTreeCatMappingsFile:
            json.dump(autoTreeCatMappings, autoTreeCatMappingsFile)
        with open(self.auto_folder+"/"+'autoNullCatTreeMappings', 'w') as autoNullCatTreeMappingsFile:
            json.dump(autoNullCatTreeMappings, autoNullCatTreeMappingsFile)
        with open(self.auto_folder+"/"+'autoNullTreeCatMappings', 'w') as autoNullTreeCatMappingsFile:
            json.dump(autoNullTreeCatMappings, autoNullTreeCatMappingsFile)
        self.updateSelfDotCats(catType)
        if self.derivationWindow != None:
            self.destroyWindow(self.derivationWindow, 'derivationWindow')
        if self.treeCompareWindow != None:
            self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')
        if self.lengthCountsOfRemovedTrees != {}:
            self.viewCurrentPTBtree()
        elif self.mode == 'annotation':
            if self.treeCompareWindow != None:
                self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')
            self.xbar_bracketings = []
            self.freshXbarWindow()
            self.destroySpins()
            self.destroyButtons()
            self.refreshTerminals()
            

    def getCovertLexicon(self, lexiconName):
        #takes as input the string name of a lexicon and just returns the relevant
        #null lexicon data structure
        if lexiconName == 'CovertLexicon':
            return self.CovertLexicon
        if lexiconName == 'ExtraposerLexicon':
            return self.ExtraposerLexicon
        if lexiconName == 'TypeRaiserLexicon':
            return self.TypeRaiserLexicon
        if lexiconName == 'ToughOperatorLexicon':
            return self.ToughOperatorLexicon
        if lexiconName == 'NullExcorporatorLexicon':
            return self.NullExcorporatorLexicon

    def addCovertCat(self, catToModify=None, nullLex=None):
        coordCheckBoxVal = 0
        self.lexiconVar = Variable()
        if catToModify != None:
            self.modifying = True
            #we are modifying an existing category rather than adding a new one..
            title = "Modify Covert MG Category"
            if catToModify.split(" ")[2] == u':\u0305:\u0305':
                coordCheckBoxVal = 1
            if nullLex == 'CovertLexicon':
                radioOn = 'Covert'
            elif nullLex == 'ExtraposerLexicon':
                radioOn = 'Extraposer'
            elif nullLex == 'ToughOperatorLexicon':
                radioOn = 'ToughOperator'
            elif nullLex == 'TypeRaiserLexicon':
                radioOn = 'TypeRaiser'
            elif nullLex == 'NullExcorporatorLexicon':
                radioOn = 'NullExcorporator'
        else:
            self.modifying = False
            title = "Add New Covert MG Category"
        self.CovertCoordinator = IntVar(value=coordCheckBoxVal)
        if self.addCovertCatWindow != None:
            self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow')
        if self.addOvertCatWindow != None:
            self.destroyWindow(self.addOvertCatWindow, 'addOvertCatWindow')
        self.addCovertCatWindow = Toplevel(self.mainWindow)
        w=475
        h=235
        (x, y) = self.getCentrePosition(w, h)
        self.addCovertCatWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.addCovertCatWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow'))
        self.addCovertCatWindow.title(title)
        if catToModify == None:
            labelFeatureSeq = Label(self.addCovertCatWindow, text="New MG Feature Sequence: ")
        else:
            labelFeatureSeq = Label(self.addCovertCatWindow, text="MG Feature Sequence: ")
        labelFeatureSeq.grid(row=0,column=0)
        fentry = Entry(self.addCovertCatWindow, width = 30)
        fentry.grid(row=0,column=1)
        fentry.focus_set()
        labelNullString = Label(self.addCovertCatWindow, text="Morpheme label (e.g. decl): ")
        labelNullString.grid(row=1)
        morphemeNameEntry = Entry(self.addCovertCatWindow, width = 30)
        morphemeNameEntry.grid(row=1,column=1)
        self.ccentry = Entry(self.addCovertCatWindow, width = 40)#creating this here but packing later to get the traversal order correct for tabbing
        coordinator = Checkbutton(self.addCovertCatWindow, text='Check this box if this is a coordinator', variable=self.CovertCoordinator)
        coordinator.grid(row=2, column=0, columnspan=2)
        globalCheckButtonFrame = Frame(self.addCovertCatWindow)
        globalCheckButtonFrame.grid(row=3, column=0, columnspan=2)
        checkButtonLabel = Label(globalCheckButtonFrame, text='Select which null lexicon to add this category to:')
        checkButtonLabel.grid(row=0)
        checkButtonFrame1 = Frame(globalCheckButtonFrame)
        checkButtonFrame1.grid(row=1, column=0, columnspan=3)
        MainCB = Radiobutton(checkButtonFrame1, text='Covert', variable=self.lexiconVar, value='Covert')
        MainCB.pack(side=LEFT)
        extraposerCB = Radiobutton(checkButtonFrame1, text='Extraposer', variable=self.lexiconVar, value='Extraposer')
        extraposerCB.pack(side=LEFT)
        checkButtonFrame2 = Frame(globalCheckButtonFrame)
        checkButtonFrame2.grid(row=2, column=0, columnspan=3)
        typeRaiserCB = Radiobutton(checkButtonFrame2, text='Type Raiser', variable=self.lexiconVar, value='TypeRaiser')
        typeRaiserCB.pack(side=LEFT)
        toughOperatorCB = Radiobutton(checkButtonFrame2, text='Tough Operator', variable=self.lexiconVar, value='ToughOperator')
        toughOperatorCB.pack(side=LEFT)
        if catToModify != None:
            self.lexiconVar.set(radioOn)
        else:
            self.lexiconVar.set(None)
        labelComments = Label(self.addCovertCatWindow, text="Comments (max 100 chars): ")
        labelComments.grid(row=5, column=0, columnspan=2)
        self.ccentry.grid(row=6,column=0, columnspan=2)
        buttonFrame = Frame(self.addCovertCatWindow)
        buttonFrame.grid(row=7, column=0, columnspan=2)
        if catToModify == None:
            addButton = Button(buttonFrame, width=10, text="Add", command=lambda: self.addCovertCAT(features=fentry.get(), nullString='['+morphemeNameEntry.get()+']'))
        else:
            oldCat = " ".join(catToModify.split(" ")[1:])
            for e in self.covertCatComments[oldCat]:
                if e[0] == nullLex:
                    oldComment = e[1]
                    break
            self.ccentry.insert(END, oldComment)
            try:
                nullCatTreeMappings = json.load(open(self.seed_folder+"/"+'nullCatTreeMappings'))
            except IOError:
                nullCatTreeMappings = {}
            try:
                autoNullCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoNullCatTreeMappings'))
            except IOError:
                autoNullCatTreeMappings = {}
            if oldCat in nullCatTreeMappings:
                numCatTreeMappings = str(len(nullCatTreeMappings[oldCat]))
            else:
                numCatTreeMappings = str(0)
            if oldCat in autoNullCatTreeMappings:
                numAutoCatTreeMappings = str(len(autoNullCatTreeMappings[oldCat]))
            else:
                numAutoCatTreeMappings = str(0)
            fentry.insert(END, " ".join(oldCat.split(" ")[2:]))
            oldMorphemeName = oldCat.split(" ")[0]
            if oldCat.split(" ")[1] == u':\u0305:\u0305':
                oldCatIsCoord = True
            elif oldCat.split(" ")[1] == u'::':
                oldCatIsCoord = False
            morphemeNameEntry.insert(END, oldMorphemeName)
            if oldCat in nullCatTreeMappings or oldCat in autoNullCatTreeMappings:
                addButton = Button(buttonFrame, width=10, text="Apply Changes", command=lambda: self.confirm(catType='covert', confirmWhat='modCats', oldCat=oldCat, entry=fentry, morphemeNameEntry=morphemeNameEntry, nullLex = nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, message='This covert MG category is used in '+numCatTreeMappings+' seed parses and '+numAutoCatTreeMappings+' auto parses. \nAll of these must now be reparsed.\nDo you wish to reparse with atomic tags or supertags?'))
            else:
                addButton = Button(buttonFrame, width=10, text="Apply Changes", command=lambda: self.confirm(catType='covert', confirmWhat='modCats', oldCat=oldCat, entry=fentry, morphemeNameEntry=morphemeNameEntry, nullLex = nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, message='This covert MG category is currently\nnot used in any seed parses.\nNo reparsing is therefore required.\n\nDo you still wish to continue?'))
        addButton.pack(side=LEFT)
        closeButton = Button(buttonFrame, width=10, text="Close", command=lambda: self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow'))
        closeButton.pack(side=LEFT)

    def addCovertCAT(self, features, nullString):
        features = features.strip()
        for char in nullString[1:-1]:
            if char in [")", "(", " ", ":", ";", ",", "{", "}"]:
                self.nothing(self.addCovertCatWindow, message="Illicit characters detected in morpheme label!")
                return
        if len(nullString) > 1:
            if nullString[0:2] == '[[':
                nullString = nullString[1:]
            if nullString[-2:] == ']]':
                nullString = nullString[:-1]
        if len(features) == 0:
            self.nothing(self.addCovertCatWindow, message="You haven't entered any MG features!")
            return
        if nullString == '[]':
            self.nothing(self.addCovertCatWindow, message="You must enter a null string for this category!")
            return
        if len(self.ccentry.get().strip()) > 100:
            self.nothing(self.addOvertCatWindow, message="Comments exceed 100 characters!")
            return
        #if this new category is a coordinator, we indicate this with an overline on the
        #double colon type indicator..
        if self.CovertCoordinator.get() == 1:
            features = u':\u0305:\u0305 '+features
        else:
            features = ':: '+features
        features = features.split(" ")
        f_index = 0
        for feature in features[1:]:
            f_index += 1
            strippedFeature = re.sub('{.*?}', '', feature)
            if not re.search('\w', strippedFeature):
                self.nothing(self.addOvertCatWindow, message="Invalid features entered!")
                return
            elif '~' in strippedFeature and f_index != len(features[1:]):
                self.nothing(self.addOvertCatWindow, message="Rightward movement features must appear in final position!", width = 450)
                return
        #we need to sort the sel_features (subcat) since the parser does this
        #and this ensures that the order in which the user types in subcat features
        #doesn't matter..
        new_features = []
        new_features.append(features[0])
        new_features += self.sortSubCat(features[1:], return_list=True)
        features = new_features
        MGentry = self.constructMGlexEntry(nullString, features)
        self.addCOVERTcat(MGentry=MGentry, features=features, nullString=nullString)

    def addCOVERTcat(self, catType=None, MGentry=None, features=None, nullString=None, entry=None, morphemeNameEntry=None, confirmWhat=None, oldCat=None, nullLex=None, oldCatIsCoord=None, oldMorphemeName=None, inLoop=False):
        if self.lexiconVar.get() == 'None':
            self.nothing(self.addCovertCatWindow, 'You must select a null lexicon!')
            return
        elif self.lexiconVar.get() == 'Covert':
            LEXICON = self.CovertLexicon
            fileName = self.seed_folder+'/CovertLexicon'
        elif self.lexiconVar.get() == 'Extraposer':
            LEXICON = self.ExtraposerLexicon
            fileName = self.seed_folder+'/ExtraposerLexicon'
        elif self.lexiconVar.get() == 'TypeRaiser':
            LEXICON = self.TypeRaiserLexicon
            fileName = self.seed_folder+'/TypeRaiserLexicon'
        elif self.lexiconVar.get() == 'ToughOperator':
            LEXICON = self.ToughOperatorLexicon
            fileName = self.seed_folder+'/ToughOperatorLexicon'
        elif self.lexiconVar.get() == 'NullExcorporator':
            LEXICON = self.NullExcorporatorLexicon
            fileName = self.seed_folder+'/NullExcorporatorLexicon'
        with open('Junk', 'w') as JunkFile:
            json.dump(MGentry, JunkFile)
        MGentry = json.load(open('Junk'))
        if MGentry in LEXICON:
            self.nothing(self.addCovertCatWindow, message="That category is already in "+self.lexiconVar.get()+"Lexicon!")
            return
        else:
            alreadyInLexicons = []
            for covertLexicon in self.covertLexicons:
                name_set = set([entry[0] for entry in covertLexicon[0]])
                if covertLexicon[1] == nullLex and self.lexiconVar.get()+"Lexicon" != nullLex:
                    continue
                elif MGentry in covertLexicon[0]:
                    alreadyInLexicons.append(covertLexicon[1])
                elif self.lexiconVar.get()+"Lexicon" != covertLexicon[1] and MGentry[0] in name_set:
                    self.nothing(self.addCovertCatWindow, "The name "+MGentry[0]+" is already used in the "+covertLexicon[1]+"!", width=500)
                    return
            if len(alreadyInLexicons) > 0:
                self.alreadyExists(currentWindow=self.addCovertCatWindow, features=features, nullString = nullString, alreadyInLexicons=alreadyInLexicons, catType='covert', LEXICON=LEXICON, MGentry=MGentry, fileName=fileName, confirmWhat=confirmWhat)
            else:
                self.confirmedAddCovertCat(catType='covert', oldCat=oldCat, LEXICON=LEXICON, MGentry=MGentry, nullLex=nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, fileName=fileName, confirmWhat=confirmWhat, entry=entry, morphemeNameEntry=morphemeNameEntry)

    def confirmedAddCovertCat(self, LEXICON=None, fileName=None, MGentry=None, confirmWhat=None, catType=None, oldCat=None, morphemeNameEntry=None, nullLex=None, oldMorphemeName=None, oldCatIsCoord=None, messageConfirm=None, entry=None):
        if self.alreadyExistsWindow != None:
            self.destroyWindow(self.alreadyExistsWindow, 'alreadyExistsWindow')
        if list(MGentry) not in LEXICON:
            LEXICON.append(list(MGentry))
        self.covertLexUpdated = True
        if self.modifying == False:
            #if the lexicon is being modified then it will be saved if necessary in modify()
            with open(fileName, 'w') as LexiconFile:
                json.dump(LEXICON, LexiconFile)
            if 'conj' in MGentry[2]:
                covCat = MGentry[0]+" "+u':\u0305:\u0305'+" "+" ".join(MGentry[1])
            else:
                covCat = MGentry[0]+" "+u'::'+" "+" ".join(MGentry[1])
            if covCat in self.covertCatComments:
                self.covertCatComments[covCat].append([fileName.split("/")[1], self.ccentry.get().strip()])
            else:
                self.covertCatComments[covCat] = [[fileName.split("/")[1], self.ccentry.get().strip()]]
            with open(self.seed_folder+'/covertCatComments', 'w') as commentsFile:
                json.dump(self.covertCatComments, commentsFile)
        #as we are destroying the addCovert.CatWindow, we must record some stuff now owing to the loop that we may have entered.
        try:
            #if we are entering the loop then entry (the field where feaures are entered) will have a value otherwise it will not, same for morphemeNameEntry
            self.MGfeatures = entry.get()
            self.morphemeName = morphemeNameEntry.get()
            self.newComments = self.ccentry.get().strip()
        except AttributeError:
            x=0
        if self.addCovertCatWindow != None:
            self.newComments = self.ccentry.get().strip()
            #had to comment the following out as it was cosing the window too early
            #when modifying comments in covert cats
            #self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow')
        if self.modifying == False:
            self.addCovertCat()
        else:
            self.modifying = False
            if self.checkedForDuplicates == False:
                self.confirm(confirmWhat=confirmWhat, oldCat=oldCat, catType=catType, entry=entry, morphemeNameEntry=morphemeNameEntry, nullLex=nullLex, oldMorphemeName=oldMorphemeName, oldCatIsCoord=oldCatIsCoord, message=messageConfirm)
            else:
                #we now exit the loop taking us right back to modify()
                self.checkedForDuplicates = False
                return
        
    def addOvertCat(self, catToModify=None):
        coordCheckBoxVal = 0
        if self.addOvertCatWindow != None:
            self.destroyWindow(self.addOvertCatWindow, 'addOvertCatWindow')
        if self.addCovertCatWindow != None:
            self.destroyWindow(self.addCovertCatWindow, 'addCovertCatWindow')
        if catToModify != None:
            #we are modifying an existing category rather than adding a new one..
            title = "Modify Overt MG Category"
            if catToModify.split(" ")[1] == u':\u0305:\u0305':
                coordCheckBoxVal = 1
        else:
            title = "Add New Overt MG Category"
        message = "Include this MG category in the dropdown lists of words with the following PTB categories:"
        self.addOvertCatWindow = Toplevel(self.mainWindow)
        self.addOvertCatWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.addOvertCatWindow,'addOvertCatWindow'))
        w=650
        h=500
        (x, y) = self.getCentrePosition(w, h)
        self.addOvertCatWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.addOvertCatWindow.title(title)
        self.checkboxVars = {}
        if catToModify == None:
            for cat in PTBcats:
                var = BooleanVar()
                var.set(0)
                self.checkboxVars[cat] = [var]
        else:
            for cat in PTBcats:
                var = BooleanVar()
                if " ".join(catToModify.split(" ")[1:]) in self.PosMappings[cat]:
                    var.set(1)
                else:
                    var.set(0)
                self.checkboxVars[cat] = [var]
        labelFeatureSeq = Label(self.addOvertCatWindow, text="New MG Feature Sequence: ")
        labelFeatureSeq.grid(row=0)
        entry = Entry(self.addOvertCatWindow, width = 60)
        entry.grid(row=1,column=0)
        entry.focus_set()
        self.ocentry = Entry(self.addOvertCatWindow, width = 50)#creating this here and packing later to get the traversal order correct for tabbing
        self.checkboxVars['coordinator'] = [IntVar(value=coordCheckBoxVal)]
        coordinator = Checkbutton(self.addOvertCatWindow, text='Check this box if this is a coordinator', variable=self.checkboxVars['coordinator'][0]) 
        coordinator.grid(row=2)
        labelMappings = Label(self.addOvertCatWindow, text=message)
        labelMappings.grid(row=3)
        self.checkFrame = Frame(self.addOvertCatWindow)
        self.checkFrame.grid(row=4)
        self.checkboxVars['coordinator'].append(coordinator)
        row = 4
        column = -1
        for cat in PTBcats:
            column += 1
            cb = Checkbutton(self.checkFrame, text=cat, variable = self.checkboxVars[cat][0])
            cb.grid(row=row, column=column)
            self.checkboxVars[cat].append(cb)
            if column == 5:
                row+=1
                column = -1
        labelComments = Label(self.addOvertCatWindow, text="Comments (max 100 chars): ")
        labelComments.grid(row=11)
        self.ocentry.grid(row=12,column=0)
        buttonFrame = Frame(self.addOvertCatWindow)
        buttonFrame.grid(row=13, column=0)
        if catToModify == None:
            addButton = Button(buttonFrame, width=10, text="Add", command=lambda: self.addOvertCAT(features=entry.get(), cbVars=self.checkboxVars))
        else:
            oldCat = " ".join(catToModify.split(" ")[1:])
            try:
                CatTreeMappings = json.load(open(self.seed_folder+"/"+'CatTreeMappings'))
            except IOError:
                CatTreeMappings = {}
            if oldCat in CatTreeMappings:
                numCatTreeMappings = str(len(CatTreeMappings[oldCat]))
            else:
                numCatTreeMappings = str(0)
            try:
                autoCatTreeMappings = json.load(open(self.auto_folder+"/"+'autoCatTreeMappings'))
            except IOError:
                autoCatTreeMappings = {}
            if oldCat in autoCatTreeMappings:
                numAutoCatTreeMappings = str(len(autoCatTreeMappings[oldCat]))
            else:
                numAutoCatTreeMappings = str(0)
            entry.insert(END, " ".join(oldCat.split(" ")[1:]))
            self.ocentry.insert(END, self.overtCatComments[oldCat])
            if numCatTreeMappings == "0":
                addButton = Button(buttonFrame, width=10, text="Apply Changes", command=lambda: self.confirm(catType='overt', confirmWhat='modCats', oldCat=oldCat, entry=entry, message='This overt MG category is currently\nnot used in any seed parses.\nNo reparsing is therefore required.\n\nDo you still wish to continue?'))
            else:
                addButton = Button(buttonFrame, width=10, text="Apply Changes", command=lambda: self.confirm(catType='overt', confirmWhat='modCats', oldCat=oldCat, entry=entry, message='This overt MG category is used in '+numCatTreeMappings+' seed parses and '+numAutoCatTreeMappings+' auto parses.\nAll of these must now be reparsed.\nWhich parsing method do you wish to use?'))
        addButton.pack(side=LEFT)
        closeButton = Button(buttonFrame, width=10, text="Close", command=lambda: self.destroyWindow(self.addOvertCatWindow, 'addOvertCatWindow'))
        closeButton.pack(side=LEFT)
            
    def addOvertCAT(self, features, cbVars):
        #first we chop any trailing whitespace off of the features (in case the user accidentally left a space at the beginning or end)
        features = features.strip()
        if len(features) == 0:
            self.nothing(self.addOvertCatWindow, message="You haven't entered any MG features!")
            return
        if len(self.ocentry.get().strip()) > 100:
            self.nothing(self.addOvertCatWindow, message="Comments exceed 100 characters!")
            return
        mapping = []
        #if this new category is a coordinator, we indicate this with an overline on the
        #double colon type indicator
        if cbVars['coordinator'][0].get() == 1:
            features = u':\u0305:\u0305 '+features
        else:
            features = ':: '+features
        features = features.split(" ")
        f_index = 0
        for feature in features[1:]:
            f_index += 1
            strippedFeature = re.sub('{.*?}', '', feature)
            if not re.search('\w', strippedFeature):
                self.nothing(self.addOvertCatWindow, message="Invalid features entered!")
                return
            elif '~' in strippedFeature and f_index != len(features[1:]):
                self.nothing(self.addOvertCatWindow, message="Rightward movement features must appear in final position!", width = 450)
                return
        #we need to sort the sel_features (subcat) since the parser does this
        #and this ensures that the order in which the user types in subcat features
        #doesn't matter..
        new_features = []
        new_features.append(features[0])
        new_features += self.sortSubCat(features[1:], return_list=True)
        features = " ".join(new_features)
        for cat in PTBcats:
            if cat not in mapping:
                if cbVars[cat][0].get() == 1:
                    mapping.append(cat)
        if len(mapping) == 0:
            self.nothing(self.addOvertCatWindow, message="You haven't selected any PTB categories!")
            return
        alreadyExistsIn = []
        for cat in PTBcats:
            if features in self.PosMappings[cat]:
                alreadyExistsIn.append(cat)
        if len(alreadyExistsIn) != 0:
            self.alreadyExists(currentWindow=self.addOvertCatWindow, alreadyExistsIn=alreadyExistsIn, mapping=mapping, features=features, catType='overt')
        else:
            self.addPosMapping(mapping, features)
            self.destroyWindow(self.addOvertCatWindow, 'addCovertCatWindow')
            self.addOvertCat()

    def getCentrePosition(self, w, h):
        #a method which returns the coordinates for placing a given window of
        #a given size in the centre of the screen..
        #w = width of the window
        #h = height of window
        # get screen width and height
        ws = self.mainWindow.winfo_screenwidth() # width of the screen
        hs = self.mainWindow.winfo_screenheight() # height of the screen
        # calculate x and y coordinates for the window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        # set the dimensions of the screen 
        # and where it is placed
        return (x, y)

    def noParse(self):
        if self.noParse2accept != None:
            self.destroyWindow(self.noParse2accept, 'noParse2accept')
        #this is called if the user clicks 'accept parse' but there's no MG parse in the window
        self.noParse2accept = Toplevel(self.mainWindow)
        self.noParse2accept.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.noParse2accept, 'noParse2accept'))
        w=300
        h=50
        (x, y) = self.getCentrePosition(w, h)
        self.noParse2accept.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.noParse2accept, text="There's no MG parse to accept!")
        label.pack()
        okButton = Button(self.noParse2accept, text="Ok", command=lambda: self.destroyWindow(self.noParse2accept, 'noParse2accept'))
        okButton.pack()

    def noParses(self):
        self.xbar_trees = []
        self.derivation_trees = []
        self.subcat_derivation_trees = []
        self.subcat_full_derivation_trees = []
        self.full_derivation_trees = []
        self.derived_trees = []
        self.XBAR_trees = []
        self.xbar_bracketings = []
        self.derived_bracketings = []
        self.derivation_bracketings = []
        self.subcat_derivation_bracketings = []
        self.subcat_full_derivation_bracketings = []
        self.full_derivation_bracketings = []
        self.freshXbarWindow()
        if self.noParsesFound != None:
            self.destroyWindow(self.noParsesFound, 'noParsesFound')
        #this is called if the user clicks 'parse' but no parses can be generated
        self.noParsesFound = Toplevel(self.mainWindow)
        self.noParsesFound.title("")
        self.noParsesFound.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.noParsesFound, 'noParsesFound'))
        w=300
        h=75
        (x, y) = self.getCentrePosition(w, h)
        self.noParsesFound.geometry('%dx%d+%d+%d' % (w, h, x, y))
        label = Label(self.noParsesFound, text="No parses found with these MG categories!\n(hint: try adjusting parser settings.)")
        label.pack()
        okButton = Button(self.noParsesFound, text="Ok", command=lambda: self.destroyWindow(self.noParsesFound, 'noParsesFound'))
        okButton.pack()

    def freshXbarWindow(self):
        if self.xbar_tframe != None:
            self.xbar_tframe.destroy()
        if self.task == 'parse':
            if self.TERMINALS_FRAME != None:
                self.TERMINALS_FRAME.destroy()
        self.xbar_tframe = Frame(self.outer_xbar_tframe, relief=SUNKEN)
        self.xbar_tframe.pack(side='left', fill=BOTH, expand=True)
        self.XBARwindow = CanvasFrame(self.xbar_tframe, highlightthickness=2, highlightbackground='black', bg='white', height=10)
        self.XBARwindow.pack(fill=BOTH, expand=True)
        labelFrame = Frame(self.xbar_tframe)
        labelFrame.pack()
        if self.newSentMode:
            self.xbar_label = Label(labelFrame, text='MG Tree')
        else:
            self.xbar_label = Label(labelFrame, text='Target')
        self.xbar_label.pack(side='left')
        self.viewMGbracketButton = Button(labelFrame, text='[...]', command=lambda: self.viewBracketing('main'))
        self.viewMGbracketButton.pack(side='left')
        self.fullMGTreeButton = Button(labelFrame, text='FS', command=lambda: self.newFullTreeWindow(treeType='MG', spin=self.spin))
        self.fullMGTreeButton.pack(side='left')

    def nothing(self, parentWindow, message, width = None, height = None, failure_messages=None, extraButton=None):
        if self.nothingEntered != None:
            self.destroyWindow(self.nothingEntered, 'nothingEntered')
        self.nothingEntered = Toplevel(parentWindow)
        self.nothingEntered.title("")
        self.nothingEntered.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.nothingEntered, 'nothingEntered'))
        if width != None:
            w = width
        else:
            w=375
        if height != None:
            h = height
        else:
            h=52
        (x, y) = self.getCentrePosition(w, h)
        self.nothingEntered.geometry('%dx%d+%d+%d' % (w, h, x, y))
        if failure_messages != None and len(failure_messages) > 0:
            empty_label = Label(self.nothingEntered, text = " ")
            empty_label.pack()
        label = Label(self.nothingEntered, text=message)
        label.pack()
        if failure_messages != None and len(failure_messages) > 0:
            empty_label = Label(self.nothingEntered, text = " ")
            empty_label.pack()
            if len(failure_messages) == 1:
                reason_label = Label(self.nothingEntered, text = failure_messages[0])
                reason_label.pack()
                empty_label = Label(self.nothingEntered, text = " ")
                empty_label.pack()
            else:
                index = 0
                while len(failure_messages) > 0:
                    index += 1
                    reason_label = Label(self.nothingEntered, text = str(index)+".  "+failure_messages[0])
                    reason_label.pack()
                    del(failure_messages[0])
                empty_label = Label(self.nothingEntered, text = " ")
                empty_label.pack()
            empty_label = Label(self.nothingEntered, text = " ")
            empty_label.pack()
        if extraButton == 'Deps':
            buttonFrame = Frame(self.nothingEntered)
            buttonFrame.pack()
            okButton = Button(buttonFrame, text="Ok", command=lambda: self.destroyWindow(self.nothingEntered, 'nothingEntered'))
            okButton.pack(side=LEFT)
            viewDepsButton = Button(buttonFrame, text="View Reified Dep Mappings", command=lambda: self.viewDepMappings(self.reifiedSentDepMappings, matched=False))
            viewDepsButton.pack(side=LEFT)
            viewMatchedDepsButton = Button(buttonFrame, text="View Matched Dep Mappings", command=lambda: self.viewDepMappings(self.reifiedSentDepMappings, matched=True))
            viewMatchedDepsButton.pack(side=LEFT)
        else:
            okButton = Button(self.nothingEntered, text="Ok", command=lambda: self.destroyWindow(self.nothingEntered, 'nothingEntered'))
            okButton.pack()
        self.nothingEntered.attributes("-topmost", True)
                        
                                        
    def addPosMapping(self, mapping, features):
        self.overtCatComments[features] = self.ocentry.get().strip()
        for cat in mapping:
            if features not in self.PosMappings[cat]:
                if 'No categories available' == self.PosMappings[cat][0]:
                    del(self.PosMappings[cat][0])
                self.PosMappings[cat].append(features)
                self.PosMappings[cat].sort()
        if self.mode == 'annotation':
            if self.treeCompareWindow != None:
                self.destroyWindow(self.treeCompareWindow, 'treeCompareWindow')
            self.xbar_bracketings = []
            self.destroySpins()
            self.destroyButtons()
            self.freshXbarWindow()
            self.refreshTerminals()
        self.destroyWindow(self.addOvertCatWindow, 'addOvertCatWindow')
        self.alreadyExistsWindow=None
        with open(self.seed_folder+'/PosMappings', 'w') as PosMappingsFile:
            json.dump(self.PosMappings, PosMappingsFile)
        with open(self.seed_folder+'/overtCatComments', 'w') as overtCatCommentsFile:
            json.dump(self.overtCatComments, overtCatCommentsFile)
        self.addOvertCat()

    def alreadyExists(self, currentWindow=None, alreadyExistsIn=None, mapping=None, features=None, nullString=None, alreadyInLexicons=None, catType=None, LEXICON=None, entry=None, fileName=None, confirmWhat=None, oldCat=None, morphemeNameEntry=None, nullLex=None, oldMorphemeName=None, oldCatIsCoord=None, messageConfirm=None, MGentry=None):
        if self.alreadyExistsWindow != None:
            self.destroyWindow(self.alreadyExistsWindow, 'alreadyExistsWindow')
        self.alreadyExistsWindow = Toplevel(currentWindow)
        self.alreadyExistsWindow.protocol('WM_DELETE_WINDOW', lambda: self.destroyWindow(self.alreadyExistsWindow,'alreadyExistsWindow'))
        self.alreadyExistsWindow.title('Warning!')
        w=600             
        h=120
        (x, y) = self.getCentrePosition(w, h)
        self.alreadyExistsWindow.geometry('%dx%d+%d+%d' % (w, h, x, y))
        if catType == 'overt':
            cats = ""
            for cat in alreadyExistsIn:
                cats+=cat+" "
            if len(cats)>0:
                cats=cats[:-1]
            message = Label(self.alreadyExistsWindow, text=("The overt MG category:\n"+features+"\nis already associated with PTB categories:\n"+cats+"\nCannot create duplicates!"))
        elif catType == 'covert':
            lexiconsList = ", ".join(alreadyInLexicons)
            covertMGcat = nullString+" "+" ".join(features) 
            message = Label(self.alreadyExistsWindow, text=("The covert MG category:\n"+covertMGcat+"\nis already present in the following lexicon:\n"+lexiconsList+"\nCannot create duplicates!"))
        message.pack()
        buttonBox = Frame(self.alreadyExistsWindow)
        buttonBox.pack()
        okButton = Button(buttonBox, text="Cancel", command=lambda: self.destroyWindow(self.alreadyExistsWindow, 'alreadyExistsWindow'))
        okButton.pack(side=LEFT)

    def autoGenerateCorpus(self, updateDepMappingsOnly=False):
        if self.overwrite_auto == None:
            if self.overwriteWindow != None:
                self.destroyWindow(self.overwriteWindow, 'overwriteWindow')
            self.mainWindow.withdraw()
        elif self.overwrite_auto:
            self.overwriteAuto = 'overwrite'
        else:
            self.overwriteAuto = 'add'
        if self.supertagStrategy == None:
            self.supertagStrategy = self.parserSettings['supertaggingStrategy']
        if not updateDepMappingsOnly:
            if self.train_tagger:
                if "compressed_parses" in os.listdir(os.getcwd()+"/CCGbank"):
                    shutil.rmtree("CCGbank/compressed_parses/")
                if self.supertagStrategy == 'CCG_OVERT_MG':
                    self.extractTaggingModels(MGtagType='atomic', gen_new_ccg_trees=True)
                elif self.supertagStrategy == 'CCG_OVERT_MG_MAXENT':
                    self.extractTaggingModels(MGtagType='atomic_maxent', gen_new_ccg_trees=True)
                    self.extractTaggingModels(MGtagType='atomic')
                elif self.supertagStrategy == 'CCG_MG_SUPERTAG':
                    self.extractTaggingModels(MGtagType='supertag', gen_new_ccg_trees=True)
                elif self.supertagStrategy == 'CCG_MG_SUPERTAG_MAXENT':
                    self.extractTaggingModels(MGtagType='supertag_maxent', gen_new_ccg_trees=True)
                    self.extractTaggingModels(MGtagType='supertag')
                elif self.supertagStrategy == 'CCG_MG_SUPERTAG_NS':
                    self.extractTaggingModels(MGtagType='supertag', gen_new_ccg_trees=True, ns=True)
                elif self.supertagStrategy == 'CCG_MG_SUPERTAG_NS_MAXENT':
                    self.extractTaggingModels(MGtagType='supertag_maxent', gen_new_ccg_trees=True, ns=True)
                    self.extractTaggingModels(MGtagType='supertag', ns=True)
                elif self.supertagStrategy == 'CCG_MG_HYBRID':
                    self.extractTaggingModels(MGtagType='hybrid', gen_new_ccg_trees=True)
                elif self.supertagStrategy == 'CCG_MG_HYBRID_MAXENT':
                    self.extractTaggingModels(MGtagType='hybrid_maxent', gen_new_ccg_trees=True)
                    self.extractTaggingModels(MGtagType='hybrid')
                elif self.supertagStrategy == 'CCG_MG_HYBRID_NS_MAXENT':
                    self.extractTaggingModels(MGtagType='hybrid_maxent', gen_new_ccg_trees=True, ns=True)
                    self.extractTaggingModels(MGtagType='hybrid', ns=True)
        autocorpus_folder = self.ptb_folder+"_MGbankAuto"
        if self.overwriteAuto == 'overwrite' and autocorpus_folder in os.listdir(os.getcwd()):
            shutil.rmtree(autocorpus_folder)
            os.mkdir(autocorpus_folder)
        self.DepMappings.clear()
        self.RevDepMappings.clear()
        self.PosDepsMappings.clear()
        for cat in PTBcats:
            self.PosDepsMappings[cat] = {}
        with open(self.seed_folder+"/"+'PosDepsMappings', 'w') as PosDepsMappingsFile:
            json.dump(self.PosDepsMappings, PosDepsMappingsFile)
        TreeCatMappings = json.load(open(self.seed_folder+"/"+'TreeCatMappings'))
        for section_folder in sorted(os.listdir(self.ptb_folder)):
            if section_folder == '.DS_Store':
                continue
            if section_folder not in os.listdir(autocorpus_folder):
                os.mkdir(autocorpus_folder+"/"+section_folder)
        if self.extract_dep_mappings or updateDepMappingsOnly:
            print "\nExtracting Penn tree to MG tree dependency mappings from seeds... (This may take a few minutes.)\n"
            for section_folder in sorted(os.listdir(self.ptb_folder)):
                if section_folder == '.DS_Store':
                    continue
                if section_folder not in os.listdir(self.seed_folder):
                    continue
                print "Extracting dependency mappings from section folder: ", section_folder
                for seed_file in sorted(os.listdir(self.seed_folder+"/"+section_folder)):
                    if seed_file == '.DS_Store':
                        continue
                    seeds = json.load(open(self.seed_folder+"/"+section_folder+"/"+seed_file))
                    ptbbracketings = []
                    for line in open(self.ptb_folder+"/"+section_folder+"/"+seed_file):
                        ptbbracketings.append(line)
                    for parse in seeds:
                        subcat_derivation_bracketing = seeds[parse][0]
                        try:
                            subcat_derivation_bracketing = subcat_derivation_bracketing.encode('utf8')#.decode('unicode_escape')
                        except UnicodeDecodeError:
                            x=0
                        ptb_bracketing = ptbbracketings[int(parse)]
                        ptb_bracketing = ptb_bracketing.encode('ascii')
                        (X, Y, xbar_tree) = gen_derived_tree.main(subcat_derivation_bracketing, show_indices=True, return_xbar_tree=True, allowMoreGoals=True)
                        (PTB_deps, MG_deps, terminals, PTB_tree) = autobank.get_deps_terminals(ptb_bracketing, True, xbar_tree)
                        (dep_mappings, reverse_dep_mappings) = autobank.get_dep_mappings(PTB_deps, MG_deps)
                        dep_mappings_head_head_word = copy.deepcopy(dep_mappings)
                        dep_mappings_both_head_words = copy.deepcopy(dep_mappings)
                        dep_mappings = autobank.remove_word_info_from_mapping(dep_mappings, False)
                        dep_mappings_head_head_word = autobank.remove_word_info_from_mapping(dep_mappings_head_head_word, False, True)
                        dep_mappings_both_head_words = autobank.remove_word_info_from_mapping(dep_mappings_both_head_words, False, True, True)
                        reverse_dep_mappings_head_head_word = copy.deepcopy(reverse_dep_mappings)
                        reverse_dep_mappings_both_head_words = copy.deepcopy(reverse_dep_mappings)
                        reverse_dep_mappings = autobank.remove_word_info_from_mapping(reverse_dep_mappings, False)
                        reverse_dep_mappings_head_head_word = autobank.remove_word_info_from_mapping(reverse_dep_mappings_head_head_word, False, True)
                        reverse_dep_mappings_both_head_words = autobank.remove_word_info_from_mapping(reverse_dep_mappings_both_head_words, False, True, True)
                        dep_mappings = autobank.merge_dicts(dep_mappings, dep_mappings_head_head_word, dep_mappings_both_head_words)
                        reverse_dep_mappings = autobank.merge_dicts(reverse_dep_mappings, reverse_dep_mappings_head_head_word, reverse_dep_mappings_both_head_words)
                        #now we add all these mappings to the main data structures and save these into dummy files and retrieve them,
                        #and remove duplicates.. we have to do it this way because json makes certain changes to the
                        #data structures meaning that idenity will fail (e.g. it makes things unicode, changes tuples to lists etc)
                        #we can't just save them into the original files
                        #because now the keys themselves will be changed, not just the values and so things
                        #may get overwritten after duplicates are enterd into the dictionary and then altered..
                        with open('Junk', 'w') as JunkFile:
                            json.dump(dep_mappings, JunkFile)
                        dep_mappings = json.load(open("Junk"))
                        with open('Junk', 'w') as JunkFile:
                            json.dump(reverse_dep_mappings, JunkFile)
                        reverse_dep_mappings = json.load(open("Junk"))
                        for entry in dep_mappings:
                            if entry not in self.DepMappings:
                                self.DepMappings[entry] = copy.deepcopy(dep_mappings[entry])
                            else:
                                for MGdepChain in dep_mappings[entry]:
                                    if MGdepChain not in self.DepMappings[entry]:
                                        self.DepMappings[entry].append(copy.deepcopy(MGdepChain))
                        for entry in reverse_dep_mappings:
                            if entry not in self.RevDepMappings:
                                self.RevDepMappings[entry] = copy.deepcopy(reverse_dep_mappings[entry])
                            else:
                                for MGdepChain in reverse_dep_mappings[entry]:
                                    if MGdepChain not in self.RevDepMappings[entry]:
                                        self.RevDepMappings[entry].append(copy.deepcopy(MGdepChain))
                        INDEX = -1
                        for MGCAT in TreeCatMappings[str([self.seed_folder+"/"+section_folder+"/"+seed_file, int(parse)])]:
                            INDEX+=1
                            #To strip off the subcat features to see if this helps the system
                            #to generalize uncomment the following line..
                            #BUT! You also need to remove all subcats from the null elements lexicons!!!
                            #MGCAT = re.sub('{.*?}', '', MGCAT, count = 1000)
                            if MGCAT not in self.PosDepsMappings[terminals[INDEX].mother.truncated_name]:
                                self.PosDepsMappings[terminals[INDEX].mother.truncated_name][MGCAT] = []
                            count = 0
                            for dep in PTB_deps:
                                ignore_dep = False
                                #for relation in dep['relations']:
                                    #if 'ARGM' in relation:
                                        #ignore_dep = True
                                        #break
                                if ignore_dep == False:
                                    #we want to keep the ARGM adjunct deps out of the subcat frames as these
                                    #are unpredicatable and will lead to data sparsity...
                                    if dep['head_word_span'][0] == INDEX:
                                        count += 1
                                        dep_copy = copy.deepcopy(dep)
                                        del(dep_copy['head_word'])
                                        del(dep_copy['head_word_span'])
                                        del(dep_copy['non_head_word'])
                                        del(dep_copy['non_head_word_span'])
                                        if count == 1:
                                            self.PosDepsMappings[terminals[INDEX].mother.truncated_name][MGCAT].append([dep_copy])
                                        else:
                                            self.PosDepsMappings[terminals[INDEX].mother.truncated_name][MGCAT][-1].append(dep_copy)
                            #remove duplicate entries of sets of dependencies.. we have to save it and open it again
                            #because json makes certain changes to the data structure which mean identity fails the first
                            #time around checking if not..
                            succeeded = False
                            while not succeeded:
                                #for some reason, on certain machines, a json error is thrown at this point
                                #sometimes, but not always.. so we keep trying until it succeeds.. probably something
                                #to do with the network connection
                                try:
                                    with open(self.seed_folder+"/"+'PosDepsMappings', 'w') as PosDepsMappingsFile:
                                        json.dump(self.PosDepsMappings, PosDepsMappingsFile)
                                    self.PosDepsMappings = json.load(open(self.seed_folder+"/"+"PosDepsMappings"))
                                    succeeded = True
                                except Exception as e:
                                    x=0
                            new_set = []
                            for entry in self.PosDepsMappings[terminals[INDEX].mother.truncated_name][MGCAT]:
                                if entry not in new_set:
                                    new_set.append(entry)
                            self.PosDepsMappings[terminals[INDEX].mother.truncated_name][MGCAT] = new_set
                            with open(self.seed_folder+"/"+'PosDepsMappings', 'w') as PosDepsMappingsFile:
                                json.dump(self.PosDepsMappings, PosDepsMappingsFile)
            print "\nSuccessfully extracted MG-PTB tree dependency mappings..."
            with open(self.seed_folder+"/"+'DepMappings', 'w') as depMapFile:
                json.dump(self.DepMappings, depMapFile)
            with open(self.seed_folder+"/"+'RevDepMappings', 'w') as revDepMapFile:
                json.dump(self.RevDepMappings, revDepMapFile)
        else:
            print "\nLoading MG-PTB tree dependency mappings..."
            self.PosDepsMappings = json.load(open(self.seed_folder+"/"+'PosDepsMappings'))
            self.DepMappings = json.load(open(self.seed_folder+"/"+'DepMappings'))
            self.RevDepMappings = json.load(open(self.seed_folder+"/"+'RevDepMappings'))
        if updateDepMappingsOnly == False:
            self.quit = True
            self.start_auto_gen = True
            self.newSentMode = False
            self.test_words = None
            self.untokenizedTestSentence = None
            if self.overwrite_auto == None:
                self.mainWindow.destroy()
        else:
            self.mainWindow.deiconify()
            self.nothing(self.mainWindow, "Dependency mappings successfully updated!")

