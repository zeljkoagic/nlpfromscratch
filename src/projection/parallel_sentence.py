class ParallelSentence:
    def __init__(self, target, sources, gold=None):
        self.sources = sources
        self.target = target
        self.gold = gold


class SourceSentence:
    def __init__(self, weights, pos, forms, language, alignments):
        self.weights = weights
        self.pos = pos
        self.forms = forms
        self.language = language
        self.alignments = alignments
