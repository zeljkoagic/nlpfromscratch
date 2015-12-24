#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

> $HOME/run/commands_create_osl_files.txt

# merge lists of sources and targets into a list of all languages
# if [ ! -f $HOME/data/lists/all_languages.txt ]; then
#     cat $HOME/data/lists/sources.txt $HOME/data/lists/targets.txt > $HOME/data/lists/all_languages.txt
# fi

# iterate through all language pairs
for lang in `cat $HOME/data/lists/sources.txt`; do
    echo "python $HOME/src/helper/conll2osl.py $HOME/data/conll/$lang.bible.svm_mira.conll 1> $HOME/data/raw/$lang.bible.osl 2> $HOME/data/logs/$lang.bible.osl.log"
    echo "python $HOME/src/helper/conll2osl.py $HOME/data/conll/$lang.watchtower.svm_mira.conll 1> $HOME/data/raw/$lang.watchtower.osl 2> $HOME/data/logs/$lang.watchtower.osl.log"
done

# remove a temporary file
# rm translate.txt
