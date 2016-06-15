def is_projective(heads):
    proj = True
    spans = set()
    for token, head in enumerate(heads):
        s = tuple(sorted([token, head]))
        spans.add(s)
    for l, h in sorted(spans):
        for l1, h1 in sorted(spans):
            if (l, h) == (l1, h1): continue
            if l < l1 < h and h1 > h:
                # print "non proj:",(l,h),(l1,h1)
                proj = False
    return proj
