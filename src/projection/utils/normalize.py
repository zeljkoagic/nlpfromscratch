import sys
import pandas as pd
import numpy as np

def softmax(sentence_matrix, temperature=1.0):
    """Softmax normalization.

    :param sentence_matrix: (n x n+1) weight matrix from the parser
    :param temperature: softmax temperature
    :return: softmaxed weight matrix
    """
    softmaxed = np.zeros_like(sentence_matrix)
    np.exp(sentence_matrix / temperature)

    for it in range(softmaxed.shape[0]):
        exped = np.exp(sentence_matrix[it, ] / temperature)  # We normalize per row
        softmaxed[it, ] = exped / np.sum(exped)

    return softmaxed


def rank(sentence_matrix, use_integers=False):
    """
    TODO
    :param sentence_matrix: (n x n+1) weight matrix from the parser
    :return: ranked weight matrix
    """
    ranked = np.zeros_like(sentence_matrix)

    # create rankings
    if use_integers:
        ranks = range(1, ranked.shape[1] + 1)
    else:
        ranks = [(x+1)/ranked.shape[1] for x in range(ranked.shape[1])]

    for it in range(ranked.shape[0]):
        mappings = dict(zip(sorted(sentence_matrix[it, ]), ranks))
        for jt in range(ranked.shape[1]):
            ranked[it, jt] = mappings[sentence_matrix[it, jt]]

    return ranked


def stdev_norm(sentence_matrix):
    """
    TODO
    :param sentence_matrix:
    :return:
    """
    normalized = np.zeros_like(sentence_matrix)

    for it in range(normalized.shape[0]):
        stdev = np.std(sentence_matrix[it, ])
        mean = np.mean(sentence_matrix[it, ])
        for jt in range(normalized.shape[1]):
            normalized[it, jt] = clip(0.5 + sentence_matrix[it, jt] - float(mean) / float(2 * stdev + 0.0000001))

    return normalized


def quantilize(sentence_matrix):
    pass


def clip(x):  # helper shit
    if x < 0:
        return 0.0
    elif x > 1:
        return 1.0
    else:
        return x

# softmax normalization by AJ
if __name__ == "__main__":

    for line in open(sys.argv[1]):
        if line != "\n":
            parts = line.strip().split("\t")

            kv_pairs = [kv_pair.split(":") for kv_pair in parts[-1].split()]
            score_dict = {int(k): float(v) for k, v in kv_pairs}
            scores = pd.Series(score_dict)

            softmax = scores.apply(np.exp) / scores.apply(np.exp).sum()
            transformed_pairs = ["{}:{}".format(idx, prob) for idx, prob in softmax.iteritems()]

            print("\t".join(parts[:-1]), end='\t')
            print(" ".join(transformed_pairs))
        else:
            print()
