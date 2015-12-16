import sys


def read_file(filename):
    lines = []
    for line in open(filename):
        line = line.strip()
        if line:
            line = line.split()  # presume conll 2006
            lines.append((line[3], line[6], line[7]))  # pos, head, deprel
    return lines

gold = read_file(sys.argv[1])
system = read_file(sys.argv[2])

pos_correct = 0
uas_correct = 0

for g, s in zip(gold, system):
    if g[0] == s[0]:
        pos_correct += 1
    if g[1] == s[1]:
        uas_correct += 1

print(pos_correct / len(gold), uas_correct / len(gold))
