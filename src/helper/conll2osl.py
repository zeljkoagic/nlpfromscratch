import sys

sentence = []

for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        line = line.split()
        sentence.append(line[1])
    else:
        print(" ".join(sentence))
        sentence = []
