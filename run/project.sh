#!/usr/bin/env bash

#HOME=/home/zagic/nlpfromscratch
HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

touch $HOME/run/commands_project.txt

# currently just the sources
# and onehot-binary setup
for target in `cat $HOME/data/lists/all_languages.txt`; do
    for source in `cat $HOME/data/lists/sources.txt`; do
        if [ "$source" != "$target" ]; then
            echo "python $HOME/src/projection/project.py --source $HOME/data/conll/${source}.tts.conll
            --target $HOME/data/conll/${target}.conll
            --word_alignment $HOME/data/walign/${source}-${target}.wal
            --sentence_alignment $HOME/data/salign/${source}-${target}.sal
            --trees --binary --norm_before softmax --norm_after identity --with_pp --use_similarity
            1> $HOME/data/projections/${source}-${target}.proj 2> $HOME/data/logs/${source}-${target}.proj.log" >> $HOME/run/commands_project.txt
        fi
    done

    #paste -d "#" $HOME/data/projections/*-${target}.proj \
    #    | awk '{if(if(substr($1, 0, 1)=="#"){print ""}else{print}}' > $HOME/data/projections/${target}.all-proj

done
