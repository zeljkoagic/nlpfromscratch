#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

> $HOME/run/commands_project_bible.txt

# for target in `cat $HOME/data/lists/sources.txt`; do
for target in en; do
    for source in `cat $HOME/data/lists/sources.txt`; do
        for source in bible; do
            for algorithm in crf_sgd svm_mira; do
                if [ "$source" != "$target" ]; then
                    echo "python $HOME/src/projection/project.py --source $HOME/data/conll/${source}.$source.$algorithm.conll --target $HOME/data/conll/${target}.$source.$algorithm.conll --word_alignment $HOME/data/walign/${source}-${target}.$source.wal --sentence_alignment $HOME/data/salign/${source}-${target}.$source.sal --norm_before identity --norm_after identity --trees --binary 1> $HOME/data/projections/${source}-${target}.$source.$algorithm.proj 2> $HOME/data/logs/${source}-${target}.$source.$algorithm.proj.log" >> $HOME/run/commands_project_bible.txt
                fi
            done
        done
    done
done

# paste -d "#" $HOME/data/projections/*-${target}.proj | awk '{if(substr($1, 0, 1)=="#"){print ""}else{print}}' > $HOME/data/projections/${target}.all-proj