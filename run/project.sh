#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

stop=${1:-100000}

# for target in `cat $HOME/data/lists/sources.txt`; do
for corpus in bible watchtower; do
    for trees in 0 1; do
        for binary in 0 1; do
            for similarity in 0; do
                for aligner in ibm1 ibm2; do
                    for target in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
                        for source in `cat /home/bplank/preprocess-holy-data/languages/src-${corpus}.txt`; do

                                if [ "$source" != "$target" ]; then
                                    echo "python $HOME/src/projection/project.py" \
                                    "--source /home/bplank/parse-holy-data/data/2project/${corpus}/$source.2proj.conll" \
                                    "--target /home/bplank/parse-holy-data/data/2project/${corpus}/$target.2proj.conll" \
                                    "--word_alignment /home/bplank/preprocess-holy-data/data/walign/${source}-${target}.$corpus.$aligner.reverse.wal" \
                                    "--sentence_alignment /home/bplank/preprocess-holy-data/data/salign/${source}-${target}.$corpus.sal" \
                                    "--norm_before standardize --norm_after identity --binary $binary --trees $trees" \
                                    "--use_similarity $similarity" \
                                    "1> $HOME/data/projections/${source}-${target}.corpus_${corpus}.aligner_${aligner}.trees_${trees}.binary_${binary}.similarity_${similarity}.proj" \
                                    "2> $HOME/data/logs/${source}-${target}.corpus_${corpus}.aligner_${aligner}.trees_${trees}.binary_${binary}.similarity_${similarity}.proj.log" \
                                    >> $HOME/run/commands_project_${corpus}_${aligner}.txt
                                fi

                        done
                    done
                done
            done
        done
    done
done
