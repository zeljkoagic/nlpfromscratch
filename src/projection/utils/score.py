class TokenScorer:

    def __init__(self):
        """

        :return:
        """
        self.token_count = 0
        self.correct_pos_tags = 0
        self.correct_deprels = 0
        self.correct_heads = 0
        self.correct_heads_and_deprels = 0

    def reset(self):
        self.__init__()

    def update(self, gold_token, system_token):
        """

        :param gold_token:
        :param system_token:
        :return:
        """
        self.token_count += 1

        if system_token.cpos == gold_token.cpos:  # POS accuracy
            self.correct_pos_tags += 1
        if system_token.head == gold_token.head and system_token.deprel == gold_token.deprel:  # LAS
            self.correct_heads_and_deprels += 1
        if system_token.head == gold_token.head:  # UAS
            self.correct_heads += 1
        if system_token.deprel == gold_token.deprel:  # LA
            self.correct_deprels += 1

    def get_score_list(self):
        if self.token_count == 0:
            return []
        return [(self.correct_pos_tags / self.token_count) * 100,
                (self.correct_heads_and_deprels / self.token_count) * 100,
                (self.correct_heads / self.token_count) * 100,
                (self.correct_deprels / self.token_count) * 100]
