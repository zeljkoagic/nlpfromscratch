#!/usr/bin/env bash

HOME=/home/zagic/nlpfromscratch
# HOME=/Users/zagic/Work/cph/pycharm_projects/nlpfromscratch

> $HOME/run/commands_salign_bible.txt
> $HOME/run/commands_salign_watchtower.txt

# merge lists of sources and targets into a list of all languages
# if [ ! -f $HOME/data/lists/all_languages.txt ]; then
#     cat $HOME/data/lists/sources.txt $HOME/data/lists/targets.txt > $HOME/data/lists/all_languages.txt
# fi

# iterate through all language pairs
for l1 in `cat $HOME/data/lists/sources.txt`; do
    for l2 in `cat $HOME/data/lists/sources.txt`; do
        if [ "$l1" != "$l2" ]; then
            # perform sentence alignment
            echo "$HOME/tools/salign/hunalign-1.1/src/hunalign/hunalign -utf -bisent -cautious -realign /dev/null $HOME/data/raw/${l1}.bible.osl $HOME/data/raw/${l2}.bible.osl 1> $HOME/data/salign/${l1}-${l2}.bible.sal 2> $HOME/data/logs/${l1}-${l2}.bible.sal.log" >> $HOME/run/commands_salign_bible.txt
            echo "$HOME/tools/salign/hunalign-1.1/src/hunalign/hunalign -utf -bisent -cautious -realign /dev/null $HOME/data/raw/${l1}.watchtower.osl $HOME/data/raw/${l2}.watchtower.osl 1> $HOME/data/salign/${l1}-${l2}.watchtower.sal 2> $HOME/data/logs/${l1}-${l2}.watchtower.sal.log" >> $HOME/run/commands_salign_watchtower.txt
        fi
    done
done

# remove a temporary file
# rm translate.txt
