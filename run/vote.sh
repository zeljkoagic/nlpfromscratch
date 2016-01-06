#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

for corpus in bible; do
    > $HOME/run/commands_vote_${corpus}.txt
    for trees in 0 1; do
        for binary in 1; do
            for with_pp in 0; do
                for parser in svm_mira; do
                    for target in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
                        echo "python vote_pos_and_deps.py" \
                        "--target /home/bplank/multilingparse/data/unlab/tinytok/bible2project/en.2proj.conll" \
                        "--projections $HOME/data/projections/*-${target}.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.proj" \
                        "--decode --skip_untagged --stop_after $1" \
                        "1> $HOME/data/votes/$target.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.vote" \
                        "2> $HOME/data/logs/$target.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.vote.log" >> $HOME/run/commands_vote_${corpus}.txt
                    done
                done
            done
        done
    done
done
