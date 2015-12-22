#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

> $HOME/run/commands_project.txt

# for target in `cat $HOME/data/lists/sources.txt`; do
for target in en; do
    for source in `cat $HOME/data/lists/sources.txt`; do
        if [ "$source" != "$target" ]; then
            echo "python $HOME/src/projection/project.py --source $HOME/data/conll/${source}.tt.conll.out --target $HOME/data/conll/${target}.tt.conll.out --word_alignment $HOME/data/walign/${source}-${target}.wal --sentence_alignment $HOME/data/salign/${source}-${target}.sal --norm_before identity --norm_after identity --binary --temperature 0.8 --trees 1> $HOME/data/projections/${source}-${target}.proj 2> $HOME/data/logs/${source}-${target}.proj.log" >> $HOME/run/commands_project.txt
        fi
    done

    # paste -d "#" $HOME/data/projections/*-${target}.proj | awk '{if(substr($1, 0, 1)=="#"){print ""}else{print}}' > $HOME/data/projections/${target}.all-proj

done
