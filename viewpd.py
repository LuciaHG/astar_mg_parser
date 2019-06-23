#!/usr/bin/env python
"""code for displaying trees in graphical form, extracting dependencies from Xbar trees, searching trees etc.
    Author: John torr (john.torr@cantab.net)"""

import nltk
import re
import argparse
import json
import sys
import gen_derived_tree
import pdb
import autobank

brackets='()'
open_b, close_b = brackets
open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
node_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
leaf_pattern = '[^%s%s]+' % (open_pattern, close_pattern)

if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='command line arguments.')
    cmd_parser.add_argument('-vp', dest='tree_line_num', metavar='tree_line_num', type=str, nargs=1, default=['NONE'], help='The line number of the parse you want displayed.')
    cmd_parser.add_argument('-full_derivation', dest='full_derivation', default=False, action='store_true',
                            help='If this flag is included along with -vp, the full MG derivation tree will be displayed.')
    cmd_parser.add_argument('-derivation', dest='derivation', default=False, action='store_true',
                            help='If this flag is included along with -vp, the abbreviated MG derivation tree will be displayed.')
    cmd_parser.add_argument('-derived', dest='derived', default=False, action='store_true',
                            help='If this flag is included along with -vp, the MG derived tree will be displayed.')
    cmd_parser.add_argument('-vs', dest='string_line_num', metavar='string_line_num', type=str, nargs=1, default=['NONE'], help='The line number of the parse whose terminals you want to view.')
    cmd_parser.add_argument('-vt', dest='time_line_num', metavar='time_line_num', type=str, nargs=1, default=['NONE'], help='The line number of the parse whose time you wish to view.')
    cmd_parser.add_argument('-vtg', dest='tags_line_num', metavar='tags_line_num', type=str, nargs=1, default=['NONE'], help='The line number of the parse whose tags you wish to view.')
    cmd_parser.add_argument('-f', dest='parse_file', metavar='parse_file', type=str, nargs=1, default=['NONE'], help='The name of the json file containing the parses.')
    cmd_parser.add_argument('-s', dest='string', metavar='string', type=str, nargs=1, default=['NONE'], help='The string you want to search for (can be a regex) will also search derivation bracketing.')
    cmd_parser.add_argument('-exact_match', dest='exact_match', default=False, action='store_true', help='If this flag is included, only an exact string match is used, not regex.')
    cmd_parser.add_argument('-vmgd', dest='deps_line_num', metavar='deps_line_num', type=str, nargs=1, default=['NONE'], help='The line number of the parse whose syntactic and semantic dependencies you want displayed.')
    cmd_parser.add_argument('-vsemd', dest='sem_deps_line_num', metavar='sem_deps_line_num', type=str, nargs=1, default=['NONE'], help='The line number of the parse whose semantic dependencies you want displayed.')
    cmd_parser.add_argument('-vsynd', dest='syn_deps_line_num', metavar='syn_deps_line_num', type=str, nargs=1, default=['NONE'], help='The line number of the parse whose syntactic dependencies you want displayed.')
    args = cmd_parser.parse_args()
    if args.parse_file[0] == 'NONE':
        raise Exception("You must indicate a file to process using -f")
    parses = json.load(open(args.parse_file[0]))
    if args.tree_line_num[0] != 'NONE' or args.deps_line_num[0] != 'NONE' or args.string_line_num[0] != 'NONE' or args.sem_deps_line_num[0] != 'NONE' or args.syn_deps_line_num[0] != 'NONE' or args.time_line_num[0] != 'NONE' or args.tags_line_num[0] != 'NONE':
        if args.tree_line_num[0] != 'NONE':
            line_num = args.tree_line_num[0]
        if args.deps_line_num[0] != 'NONE':
            line_num = args.deps_line_num[0]
        if args.string_line_num[0] != 'NONE':
            line_num = args.string_line_num[0]
        if args.sem_deps_line_num[0] != 'NONE':
            line_num = args.sem_deps_line_num[0]
        if args.syn_deps_line_num[0] != 'NONE':
            line_num = args.syn_deps_line_num[0]
        if args.time_line_num[0] != 'NONE':
            line_num = args.time_line_num[0]
        if args.tags_line_num[0] != 'NONE':
            line_num = args.tags_line_num[0]
        try:
            if int(line_num) < 0:
                raise Exception("argument to -vp/-vd/-vsemd/-vsynd must be a positive integer indicating the line number of the sentence!")
        except Exception as e:
            raise Exception("argument to -vp/-vd/-vsemd/-vsynd must be a positive integer indicating the line number of the sentence!")
        index = -1
        for parse in parses:
            index += 1
            if parse["parse_num"] == line_num:
                xbar_bracketing = parses[index]["trees"][1]
                try:
                    full_derivation_bracketing = parses[index]["trees"][0].encode('utf8')
                    derivation_bracketing = parses[index]["trees"][4].encode('utf8')
                    derived_bracketing = parses[index]["trees"][2].encode('utf8')
                except Exception as e:
                    full_derivation_bracketing = None
                    derivation_bracketing = None
                    derived_bracketing = None
                string = parses[index]["sentence"]
                try:
                    time = str(parses[index]["end_time"])
                except Exception as e:
                    time = 'None'
                tags = parses[index]['best_k']
        if args.tree_line_num[0] != 'NONE':
            try:
                try:
                    xbar_tree = nltk.Tree.parse(xbar_bracketing, remove_empty_top_bracketing=True, leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                    full_derivation_tree = nltk.Tree.parse(full_derivation_bracketing, remove_empty_top_bracketing=True, leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                    derivation_tree = nltk.Tree.parse(derivation_bracketing, remove_empty_top_bracketing=True, leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                    derived_tree = nltk.Tree.parse(derived_bracketing, remove_empty_top_bracketing=True,
                                                      leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                except AttributeError:
                    xbar_tree = nltk.Tree.fromstring(xbar_bracketing, remove_empty_top_bracketing=True,
                                                     leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                    full_derivation_tree = nltk.Tree.fromstring(full_derivation_bracketing, remove_empty_top_bracketing=True,
                                                     leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                    derivation_tree = nltk.Tree.fromstring(derivation_bracketing, remove_empty_top_bracketing=True,
                                                    leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                    derived_tree = nltk.Tree.fromstring(derived_bracketing, remove_empty_top_bracketing=True,
                                                        leaf_pattern=leaf_pattern, node_pattern=node_pattern)
                if args.full_derivation:
                    full_derivation_tree.draw()
                elif args.derivation:
                    derivation_tree.draw()
                elif args.derived:
                    derived_tree.draw()
                else:
                    xbar_tree.draw()
            except Exception as e:
                sys.stderr.write("\nNo parse found for that sentence!\n")
        elif args.deps_line_num[0] != 'NONE' or args.sem_deps_line_num[0] != 'NONE' or args.syn_deps_line_num[0] != 'NONE':
            (X, Y, xbar_tree) = gen_derived_tree.main(full_derivation_bracketing, show_indices=True, return_xbar_tree=True, allowMoreGoals=True)
            MG_terminals = autobank.get_MG_terminals(xbar_tree, terminals = [])
            autobank.add_truncated_names(xbar_tree)
            if args.deps_line_num[0] != 'NONE':
                MG_deps = autobank.get_MGdeps(xbar_tree, MG_terminals, deps=[])
            elif args.sem_deps_line_num[0] != 'NONE':
                MG_deps = autobank.get_MGdeps(xbar_tree, MG_terminals, deps=[], returnSynDeps=False)
            elif args.syn_deps_line_num[0] != 'NONE':
                MG_deps = autobank.get_MGdeps(xbar_tree, MG_terminals, deps=[], returnSemDeps=False)
            with open('deps', 'w') as deps_file:
                json.dump(MG_deps, deps_file)
            sys.stderr.write("\n"+str(MG_deps)+"\n")
        elif args.string_line_num[0] != 'NONE':
            sys.stderr.write("\n"+string+"\n")
            with open('string', 'w') as string_file:
                string_file.write(string)
        elif args.time_line_num[0] != 'NONE':
            sys.stderr.write("\n"+str(time)+"\n")
            with open('time', 'w') as time_file:
                time_file.write(time)
        elif args.tags_line_num[0] != 'NONE':
            sys.stderr.write(str(tags))
    elif args.string[0] != 'NONE':
        if not args.exact_match:
            reg_ex = re.compile(args.string[0])
        matched_sents = []
        for parse in parses:
            try:
                full_derivation_bracketing = parse["trees"][0].encode('utf8')
            except Exception as e:
                full_derivation_bracketing = ""
            if not args.exact_match:
                if reg_ex.search(parse['sentence']) or args.string[0] in full_derivation_bracketing or args.string[0].lower() in parse['sentence'].lower():
                    matched_sents.append(int(parse["parse_num"]))
            else:
                if args.string[0] in full_derivation_bracketing or args.string[0].lower() in parse['sentence'].lower():
                    matched_sents.append(int(parse["parse_num"]))
        if len(matched_sents) == 0:
            sys.stderr.write("\nNo parses in that file contain that string!\n")
        else:
            sys.stderr.write("\nThe following "+str(len(matched_sents))+" parses in that file contain that string:\n\n")
            sys.stderr.write(str(matched_sents))
            sys.stderr.write("\n\n")






                
