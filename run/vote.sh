#!/usr/bin/env bash

#HOME=/home/zagic/nlpfromscratch
HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

touch $HOME/run/commands_vote.txt

for language in `cat $HOME/data/lists/all_languages.txt`; do
    echo "python $HOME/src/projection/vote_pos_and_deps.py --target $HOME/data/conll/${language}.conll --votes $HOME/data/projections/${language}.all-proj > $HOME/data/votes/${language}.vote" >> commands_vote.txt
done

# parallel -j $1 < $HOME/run/commands_vote.txt