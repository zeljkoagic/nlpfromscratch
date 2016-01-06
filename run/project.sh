#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

stop=${1:-100000}

# for target in `cat $HOME/data/lists/sources.txt`; do
for corpus in bible; do
    > $HOME/run/commands_project_${corpus}_ibm1.txt
    for trees in 0 1; do
        for binary in 1; do
            for with_pp in 0; do
                for aligner in ibm1 ibm2; do
                    for target in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
                        for source in `cat /home/bplank/preprocess-holy-data/languages/src-${corpus}.txt`; do
                            for parser in svm_mira; do
                                if [ "$source" != "$target" ]; then
                                    echo "python $HOME/src/projection/project.py" \
                                    "--source /home/bplank/multilingparse/data/unlab/tinytok/bible2project/$source.2proj.conll" \
                                    "--target /home/bplank/multilingparse/data/unlab/tinytok/bible2project/$target.2proj.conll" \
                                    "--word_alignment /home/bplank/preprocess-holy-data/data/walign/${source}-${target}.$corpus.reverse.wal" \
                                    "--sentence_alignment /home/bplank/preprocess-holy-data/data/salign/${source}-${target}.$corpus.sal" \
                                    "--norm_before standardize --norm_after softmax --binary $binary --trees $trees --with_pp $with_pp" \
                                    "--stop_after $stop" \
                                    "1> $HOME/data/projections/${source}-${target}.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.$aligner.proj" \
                                    "2> $HOME/data/logs/${source}-${target}.$corpus.$parser.trees_${trees}.binary_${binary}.with_pp_${with_pp}.$aligner.proj.log" \
                                    >> $HOME/run/commands_project_${corpus}_${aligner}.txt
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done
