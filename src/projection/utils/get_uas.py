import sys

gold = open(sys.argv[1]).readlines()
system = open(sys.argv[2]).readlines()

correct = 0
total = 0

for g, s in zip(gold, system):
    g = g.strip()
    s = s.strip()
    if g and s and len(g.split()) > 1:
        g = g.split()[6]
        s = s.split()[6]
        if g == s:
            correct += 1
        total += 1

print(correct / total)
