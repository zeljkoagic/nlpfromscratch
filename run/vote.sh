#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch

for corpus in bible watchtower; do
    for trees in 0 1; do
        for binary in 0 1; do
            for similarity in 0; do
                for aligner in ibm1 ibm2; do
                    for unitvote in 0 1; do
                        for target in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
                            echo "python $HOME/src/projection/vote_pos_and_deps.py" \
                            "--target /home/bplank/parse-holy-data/data/2project/${corpus}/$target.2proj.conll" \
                            "--projections $HOME/data/projections/*-${target}.corpus_${corpus}.aligner_${aligner}.trees_${trees}.binary_${binary}.similarity_${similarity}.proj" \
                            "unit_vote_pos $unitvote --skip_untagged --decode" \
                            "1> $HOME/data/votes/${target}.corpus_${corpus}.aligner_${aligner}.trees_${trees}.binary_${binary}.similarity_${similarity}.unitvote_${unitvote}.vote" \
                            "2> $HOME/data/logs/${target}.corpus_${corpus}.aligner_${aligner}.trees_${trees}.binary_${binary}.similarity_${similarity}.unitvote_${unitvote}.vote.log" \
                            >> $HOME/run/commands_vote_${corpus}_${aligner}.txt
                        done
                    done
                done
            done
        done
    done
done
