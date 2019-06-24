A* MG Parser
=========

This folder contains the code for the A* MG parser described in the ACL 2019 paper "Wide-Coverage Neural A* Parsing for Minimalist Grammars", authored by John Torr, Miloš Stanojević, Mark Steedman and Shay Cohen.  It also contains the code for the supertagger created by Miloš Stanojević.  Please direct any questions about the parser itself to John at john.torr@cantab.net and any questions about the supertagger to Miloš at milosh.stanojevic@gmail.com.

To use the parser, you will first need to install the supertagger (see below).  You will not need to do any training or testing, however; pretrained models are included and the parser script will run the supertagger in test mode automatically. Note that the parser uses Python2.7 whereas the Supertagger requires Python3.  If this is an issue you can try using the 2to3 command from the terminal to convert all the python2.7 scripts to python3 scripts.  E.g.

2to3 astar.py -w

Which will modify the script in place and also create a .bak backup file of the original.

Installation of the supertagger
---------------

Basic requirements for the supertagger:
- g++ with c++11 support (gcc4.8.1+) and pthreads
- cmake
- JDK 8 (Scala requires JDK >=8 but DyNet at the moment has problem with JDK >=9) you MUST set JAVA_HOME variable
- git
- mercurial

OPTIONAL
- Intel MKL -- speeds up computation on Intel CPU (you would need to modify relevant variables in install_dependencies.sh)
- CUDA and cuDNN -- speeds up computation on NVidia GPU (you would need to modify relevant variables in install_dependencies.sh)
- pip3 install tensorflow      -- required if you want to use BERT embeddings
- pip3 install allennlp        -- required if you want to use ELMo embeddings
- pip2 install jnius           -- required if you want to use supertagger from python2

If all the basic requirements (including seeting the JAVA_HOME shell variable) are installed the rest
of the requirements will be installed automatically by first going to the root directory of the project
(the one that contains "src" as subdirectory) and run the following command:

     ./scripts/install_dependencies.sh

This command will take some time to install all the other dependencies (Scala, SBT, SWIG, Jep, Eigen and DyNet) and store them in directories `dependencies` and `lib`.

Installation of the parser
---------------

Basic requirements for the parser:

- pip2 install nltk
- python2 -c "import nltk ; nltk.download('punkt')"
- pip2 install fibonacci_heap_mod


Usage
----------

To parse sentences you must first create a text file which contains one sentence per line.  Make sure the file is in the folder astar_mg_parser.  Then, cd into this folder from a terminal.  If the name of the file you created is 'sentences.txt' you must execute the following to parse the sentences using the Abstract version of the parser (see the paper for the difference between the Abstract and Reified parsers):

./astar.sh abstract_model abstract_data sentences.txt True

Or, to parse with the Reified parser:

./astar.sh reified_model reified_data sentences.txt False

The elmo embeddings and supertagger will first be loaded, after which parsing will commence.  When parsing is complete, you can find the trees inside abstract_model/parses or reified_model/parses.  You can use the script viewpd.py to visualise the trees in graph form.  To view the Xbar tree for the sentence in line 5, for example, you would execute:

python2 viewpd.py -f abstract_model/parses -vp 4

You can also view the full MG derivation tree, with MG expressions at non-terminals:

python2 viewpd.py -f abstract_model/parses -vp 4 -derivation_full

Or an abbreviated MG derivation tree with fine-grained selectional restrictions and agreement features removed, and with operation names at non-terminals:

python2 viewpd.py -f abstract_model/parses -vp 4 -derivation

Or the MG derived tree:

python2 viewpd.py -f abstract_model/parses -vp 4 -derived

This script can also be used for other purposes, such as to extract all syntactic or semantic dependencies from the Xbar trees (used for evaluation in the paper) or to search the trees for a given string/regex.  To see a full list of the functions and usages of this script, just execute:

python2 viewpd.py --help

