#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

> $HOME/run/commands_project_bible.txt

# for target in `cat $HOME/data/lists/sources.txt`; do
for target in en; do
    for source in `cat $HOME/data/lists/sources.txt`; do
        for corpus in bible; do
            for algorithm in svm_mira; do
                if [ "$source" != "$target" ]; then
                    echo "python $HOME/src/projection/project.py --source $HOME/data/conll/${source}.$corpus.$algorithm.conll --target $HOME/data/conll/${target}.$corpus.$algorithm.conll --word_alignment $HOME/data/walign/${source}-${target}.$corpus.wal --sentence_alignment $HOME/data/salign/${source}-${target}.$corpus.sal --norm_before standardize --norm_after identity --binary --stop_after 5000 1> $HOME/data/projections/${source}-${target}.$corpus.$algorithm.proj 2> $HOME/data/logs/${source}-${target}.$corpus.$algorithm.proj.log" >> $HOME/run/commands_project_bible.txt
                fi
            done
        done
    done
done

# paste -d "#" $HOME/data/projections/*-${target}.proj | awk '{if(substr($1, 0, 1)=="#"){print ""}else{print}}' > $HOME/data/projections/${target}.all-proj
