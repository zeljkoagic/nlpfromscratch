# NLP for almost all languages, almost from scratch

Featuring multi-source annotation projection!

## Directory structure

data/

    walign/L1-L2.{wal, spairs}          word alignments, sentence pairs (currently fast_align format)

    logs/

    salign/L1-L2.sal                    sentence alignments (currently hunalign format)

    lists/sources.txt, targets.txt      lists of source & target languages, one entry per line

    raw/LANG.osl                        raw text, format: <sentence_id>[\t]<word-split sentence>[\n]

    resources/

        train/SOURCE.{conll, tt}        training sets for taggers and parsers

        test/LANG.{conll, tt}           test sets

tools/

    salign/                             sentence aligners

        hunalign-1.1/

    walign/                             word aligners

src/

    util/

        lowercase.py                    lowercase any text file