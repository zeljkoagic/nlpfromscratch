import numpy as np

class ConllToken:
    """todo"""
    def __init__(self, idx, form, lemma, cpos, fpos, feats, head, deprel):
        self.idx = int(idx)
        self.form = form
        self.lemma = lemma
        self.cpos = cpos
        self.fpos = fpos
        self.feats = feats
        self.head = int(head)
        self.deprel = deprel

    @classmethod
    def null_token(cls):
        return ConllToken(0, "_", "_", "_", "_", "_", -1, "_")

    def is_null(self):
        """TODO
        """
        return not bool(self.idx)

    @classmethod
    def from_list(cls, items):
        """Initializes CoNLL token from list."""
        if len(items) == 8:
            if not items[6].isdigit():
                items[6] = -1
            return ConllToken(items[0], items[1], items[2], items[3], items[4], items[5], items[6], items[7])
        else:
            raise Exception("Token init requires %s items, %s provided." % (8, len(items)))

    @classmethod
    def from_line(cls, line):
        """Initializes CoNLL token from line."""
        return ConllToken.from_list(line.split("\t"))

    def __str__(self):
        """Prints CoNLL token in CoNLL 2006 format."""
        return str("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t_\t_" % (self.idx, self.form, self.lemma, self.cpos, self.fpos,
                                                             self.feats, self.head, self.deprel))


def get_next_sentence(conll_file_handle):
    """Reads next sentence from standard CoNLL file.

    :param conll_file_handle: file handle for CoNLL-formatted file
    :return: list of tokens
    """
    next_sentence = []

    line = conll_file_handle.readline().strip().split()

    while line:
        current_token = ConllToken.from_list(line[:8])
        next_sentence.append(current_token)
        line = conll_file_handle.readline().strip().split()

    return next_sentence


def sentences(conll_file_handle, sentence_getter):
    """TODO
    :param conll_file_handle:
    :param sentence_getter:
    :return:
    """
    while True:
        next_sentence = sentence_getter(conll_file_handle)
        if not next_sentence:
            break
        yield next_sentence


def print_conll(sentences):
    """Prints CoNLL sentences to stdout."""
    for sentence in sentences:
        for token in sentence:
            print(token)
        print


def get_next_sentence_and_graph(conll_file_handle):
    """Reads next sentence, graph, and POS-tags from augmented CoNLL file.

    :param conll_file_handle: file handle for augmented CoNLL-formatted file (lines 9-end contain graph data)
    :return: <list of tokens, sentence graph, parts of speech, dependency labels> 4-tuple
    """
    next_sentence = []
    graph_parts = []
    parts_of_speech = []
    dependency_labels = []

    line = conll_file_handle.readline().strip().split()

    while line:
        next_sentence.append(ConllToken.from_list(line[:8]))
        graph_parts.append(line[8:])
        parts_of_speech.append(line[3])
        dependency_labels.append(line[7])
        line = conll_file_handle.readline().strip().split()

    # the graph has to be stored after loading the sentence
    # as we don't know the sentence length before
    next_graph = np.ones((len(next_sentence) + 1, len(next_sentence) + 1)) * np.nan

    it = 0
    for part in graph_parts:
        it += 1
        for item in part:
            head, confidence = item.split(":")
            next_graph[it][int(head)] = float(confidence)

    return next_sentence, next_graph, parts_of_speech, dependency_labels


def get_next_sentence_and_tree(conll_file_handle):
    """Reads next sentence, graph, and POS-tags from augmented CoNLL file. The graph is a one-hot representation of the
    parse tree provided by the parser decoder, i.e., it is instantiated from the parse, and not from the weight matrix.

    :param conll_file_handle: file handle for augmented CoNLL-formatted file (lines 9-end contain graph data)
    :return: <list of tokens, sentence dependency tree, parts of speech, dependency labels> 4-tuple
    """
    next_sentence = []
    next_heads = []
    parts_of_speech = []
    dependency_labels = []

    line = conll_file_handle.readline().strip().split()

    while line:
        next_sentence.append(ConllToken.from_list(line[:8]))
        next_heads.append(int(line[6]))
        parts_of_speech.append(line[3])
        dependency_labels.append(line[7])
        line = conll_file_handle.readline().strip().split()

    # create graph (n+1 x n+1)
    next_graph = np.ones((len(next_sentence) + 1, len(next_sentence) + 1)) * np.nan

    # assign the collected heads to the graph
    it = 0
    for head in next_heads:
        next_graph[it+1][head] = 1.0
        it += 1

    return next_sentence, next_graph, parts_of_speech, dependency_labels


def write(sentences, filename):
    """Writes CoNLL sentences into a file."""
    with open(filename) as conll_file:
        for sentence in sentences:
            for token in sentence:
                conll_file.write(str(token) + "\n")
            conll_file.write("\n")
