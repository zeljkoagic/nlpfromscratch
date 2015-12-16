# currently only supports fast_align format!

import sys

src = open(sys.argv[1]).readlines()
trg = open(sys.argv[2]).readlines()

for line in open(sys.argv[3]):
    line = line.strip()
    if line:
        line = line.split()
        if len(line) == 3:
            src_id = int(line[0])
            trg_id = int(line[1])
            print("%s ||| %s" % (src[src_id].strip().lower(), trg[trg_id].strip().lower()))
