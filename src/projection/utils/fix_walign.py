import sys

for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        line = line.split()
        pairs = line[::2]
        probabilities = line[1::2]
        new_pairs = []
        it = 0
        for pair in pairs:
            pair = pair.split("-")
            new_pairs.append(str(pair[1] + "-" + pair[0]))
            new_pairs.append(str(probabilities[it]))
            it += 1
        print(" ".join(new_pairs))
