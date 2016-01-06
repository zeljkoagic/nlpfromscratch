#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

stop=${1:-100000}

for corpus in bible; do
    > $HOME/run/commands_vote_${corpus}.txt
    for trees in 0 1; do
        for binary in 1; do
            for with_pp in 0; do
                for aligner in ibm1 ibm2; do
                    for parser in svm_mira; do
                        for target in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
                            echo "python $HOME/src/projection/vote_pos_and_deps.py" \
                            "--target /home/bplank/multilingparse/data/unlab/tinytok/bible2project/$target.2proj.conll" \
                            "--projections $HOME/data/projections/*-${target}.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.$aligner.proj" \
                            "--decode --skip_untagged --stop_after $stop" \
                            "1> $HOME/data/votes/$target.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.$aligner.vote" \
                            "2> $HOME/data/logs/$target.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.$aligner.vote.log" \
                            >> $HOME/run/commands_vote_${corpus}_${aligner}.txt
                        done
                    done
                done
            done
        done
    done
done
