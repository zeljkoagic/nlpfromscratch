#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

# for target in `cat $HOME/data/lists/sources.txt`; do
for corpus in bible; do
    > $HOME/run/commands_project_${corpus}.txt
    for target in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
        for source in `cat /home/bplank/preprocess-holy-data/languages/src-train-languages.txt`; do
            for parser in svm_mira; do
                for trees in 0 1; do
                    for binary in 1; do
                        for with_pp in 0; do
                            if [ "$source" != "$target" ]; then
                                echo "python $HOME/src/projection/project.py
                                --source /home/bplank/multilingparse/data/unlab/tinytok/bible2project/$source.2proj.conll
                                --target /home/bplank/multilingparse/data/unlab/tinytok/bible2project/$target.2proj.conll
                                --word_alignment $HOME/data/walign/${source}-${target}.$corpus.wal
                                --sentence_alignment /home/bplank/preprocess-holy-data/data/salign/${source}-${target}.$corpus.sal
                                --norm_before standardize --norm_after softmax --binary $binary --trees $trees --with_pp $with_pp 1> $HOME/data/projections/${source}-${target}.$corpus.$parser.trees=${trees}.binary=${binary}.with_pp=${with_pp}.proj 2> $HOME/data/logs/${source}-${target}.$corpus.$parser.trees=${trees}.binary=${binary}.with_pp=${with_pp}.proj.log" >> $HOME/run/commands_project_${corpus}.txt
                            fi
                        done
                    done
                done
            done
        done
    done
done

# paste -d "#" $HOME/data/projections/*-${target}.proj | awk '{if(substr($1, 0, 1)=="#"){print ""}else{print}}' > $HOME/data/projections/${target}.all-proj
