import sys

avg = 0

for line in open(sys.argv[1]):
    line = line.strip().split()
    if line:
        line = line[1::2]
        for item in line:
            avg = (avg + float(item)) / 2

print(avg)
