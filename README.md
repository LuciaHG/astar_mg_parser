A* MG Parser
=========

This folder contains the code for the A* MG parser described in the ACL 2019 paper "Wide-Coverage Neural A* Parsing for Minimalist Grammars", authored by John Torr, Miloš Stanojević, Mark Steedman and Shay Cohen.  It also contains the code for the supertagger created by Miloš Stanojević.  Please direction any questions about the parser itself to John at john.torr@cantab.net and any questions about the supertagger to Miloš at milosh.stanojevic@gmail.com.

To use the parser, you will first need to install the supertagger (see below).  You will not need to do any training or testing, however; the models are included and the parser script will run the supertagger in test mode automatically.  Note that the parser uses Python2.7 whereas the Supertagger requires Python3.  If this is an issue you can try using the 2to3 command from the terminal to convert all the the python2 scripts to python3 scripts.  E.g.

2to3 astar.py -w

Which will modify the script in place and also create a .bak backup file of the original.

Installation
---------------

Basic requirements for the parser:

- pip install nltk

Usage
----------

To parse sentences you must first create a text file which contains one sentence per line.  Make sure the file is in the folder astar_mg_parser.  Then, cd into this folder from a terminal.  If the name of the file you created is 'sentences.txt' you would execute the following to parse the sentences using the Abstract version of the parser (see the paper for the difference between the Abstract and Reified parsers).

./astar.sh abstract_model abstract_data sentences.txt True

Or, to parse with the Reified parser:

./astar.sh reified_model reified_data sentences.txt False

The elmo embeddings and supertagger will first be loaded and then parsing will commence.  When parsing has finished, you can find the trees inside abstract_model/parses or reified_model/parses.  You can use the script viewpd.py to visualise the trees in graph form using nltk's tree displaying tool.  To view the Xbar tree for sentence in line 5, for example, you would execute:

python viewpd.py -f abstract_model/parses -vp 4

You can also view the full MG derivation tree, with MG expressions at non-terminals:

python viewpd.py -f abstract_model/parses -vp 4 -derivation_full

Or an abbreviated MG derivation tree with fine-grained selectional restrictions and agreement features removed, and with operation names at non-terminals:

python viewpd.py -f abstract_model/parses -vp 4 -derivation

Or the MG derived tree:

python viewpd.py -f abstract_model/parses -vp 4 -derived

This script can also be used for other purposes, such as to extract all syntactic or semantic dependencies from the Xbar trees (used for evaluation in the paper) or to search the trees for a given regex.  To see a full list of the functions and usages of this script, just execute:

python viewpd.py --help


SuperSuperTagger
=========

This is a general supertagger that can be used for MG, CCG or whatever tags you like.

If you have any problems using it shout at:

Miloš Stanojević        \
m.stanojevic@ed.ac.uk   \
University of Edinburgh 

Installation
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

BERT
----

If you want to use BERT embedder you need to download the uncased
version of BERT model, extract it into some folder and then set
environment variable BERT_DIR to point to that location.

You also need to specify the right `--hyper_params_file` in the
training command. Additionally, you need to set `--embeddings_dim 100`.
The exact number doesn't matter but you must specify that command.

In the training phase, before doing any actual training, the program
will precompute all BERT embeddings and that may take some time. In case
it crashes (let's say you didn't specify BERT_DIR) you will need to delete
temporary files it has created in the same directory as your data.
These files are easy to recognize because they have BERT in their name.

All of the things said above about BERT also hold for ELMo except that you
don't need to point to the location of ELMo.


Training
----------

To train the tagger you first need dev and train set in the right format.
Unlike some other supertaggers this one keeps words and tags in two separate files.
In the following training example files dev.words and dev.tags contains words and
tags of the development set and train.words and train.tags for the train set.
       
    $CODE_DIR/scripts/run.sh edin.supertagger.MainTrain \
    --model_dir             $MODEL_DIR \
    --hyper_params_file     $CONFIG_FILE \
    --embeddings_dim        100 \
    --train_file            $DATA_DIR/train \
    --dev_file              $DATA_DIR/dev \
    --epochs                200 \
    --all_in_memory         true \
    --dynet-mem             8000

Among other relevant parameters is `--model_dir` which says where should the trained
 model be stored. `--embedding_file` points to the location of the Glove embeddings.
 `--embeddings_lowercased` should always be true if used with Glove embeddings.
 The Glove embeddings I used are located in `/afs/inf.ed.ac.uk/user/m/mstanoje/Data/Embeddings/glove.6B.100d.txt`.
 `--epochs` specifies the number of desired epochs.
 I usually set it to a very high number and then just kill the process when I think that the training is finished.
 `--dynet-mem` says how many megabytes should be used for neural computation.
 8000MB turned out to be enough for the experiments I've ran but if you are running the tagger on a weaker machine
 you might consider lowering this number.

`--hyper_params_file` specifies the file that describes the structure of the model (number and size of the neural layers etc.). The one that should work the best is probably `configs/MG_tagger_BERT.yaml`.

The informative logging information will be printed out on standard error so I recommend redirecting standard error to some file in order to track the progress of training.

Testing
---------

To apply the already trained model to the test data you need to run the following command:

    $CODE_DIR/scripts/run.sh edin.supertagger.MainApply \
    --model_dirs           $MODEL_DIR \
    --input_file   $DATA_DIR/test.words \
    --output_file  $MODEL_DIR/single_best \
    --output_file_best_k  $MODEL_DIR/best_40 \
    --top_K 40 \
    --dynet-mem           8000

Parameter `--output_file` specifies where to output 1-best tags. `--output_file_best_k` specifies where to output k-best tags per word in the format of C&C supertagger. `--top_K` specifies how many best tags to output. `--top_beta` is similar to beta parameter from C&C tagger.

Evaluation
-------------

The following line produces some evaluation statistics about k-best tags.

    $CODE_DIR/scripts/run.sh edin.supertagger.MainEvaluateNBest \
    --gold_file    $DATA_DIR/test.tags
    --predicted_file_nbest $MODEL_DIR/best_40
    --maxK 40

