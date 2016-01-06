#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

# merge lists of sources and targets into a list of all languages
# if [ ! -f $HOME/data/lists/all_languages.txt ]; then
#     cat $HOME/data/lists/sources.txt $HOME/data/lists/targets.txt > $HOME/data/lists/all_languages.txt
# fi

# iterate through all language pairs
for corpus in bible; do
    > $HOME/run/commands_walign_${corpus}.txt
    for l1 in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
        for l2 in `cat /home/bplank/preprocess-holy-data/languages/trg-src-${corpus}.txt`; do
            if [ "$l1" != "$l2" ]; then
                # extract sentence pairs (does lowercasing too)
                # currently supports only fast_align!
                # echo "python $HOME/src/helper/extract_sentence_pairs.py $HOME/data/raw/${l1}.bible.osl $HOME/data/raw/${l2}.bible.osl $HOME/data/salign/${l1}-${l2}.bible.sal 1> $HOME/data/walign/${l1}-${l2}.bible.spairs 2> $HOME/data/logs/${l1}-${l2}.bible.spairs.log" >> $HOME/run/commands_spair_bible.txt
                # echo "python $HOME/src/helper/extract_sentence_pairs.py $HOME/data/raw/${l1}.watchtower.osl $HOME/data/raw/${l2}.watchtower.osl $HOME/data/salign/${l1}-${l2}.watchtower.sal 1> $HOME/data/walign/${l1}-${l2}.watchtower.spairs 2> $HOME/data/logs/${l1}-${l2}.watchtower.spairs.log" >> $HOME/run/commands_spair_watchtower.txt
                # perform word alignment
                echo "$HOME/tools/walign/fast_align/fast_align -d -o -v -r -i
                /home/bplank/preprocess-holy-data/data/walign/${l1}-${l2}.${corpus}.spairs 1> $HOME/data/walign/${l1}-${l2}.${corpus}.wal 2> $HOME/data/logs/${l1}-${l2}.${corpus}.log" >> $HOME/run/commands_walign_${corpus}.txt
            fi
        done
    done
done
