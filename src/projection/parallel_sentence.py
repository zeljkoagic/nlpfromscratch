class ParallelSentence:
    def __init__(self, target, sources, gold=None):
        self.sources = sources
        self.target = target
        self.gold = gold


class SourceSentence:
    def __init__(self, weights, pos, tokens, language, alignments):
        self.weights = weights
        self.pos = pos
        self.tokens = tokens
        self.language = language
        self.alignments = alignments
