from torchtext.data import Field


class SentPosiField(Field):
    def pad(self, minibatch):
        """
        Pad a batch of examples using this field.

        :param data: [[[b1_s1_fp, b1_s1_bp], [b1_s2_fp, b1_s2_bp], ...], ...], list of examples' sent positions
        :return: a padded list and the sent numbers, we use [0, 0] to pad the sentence position of each example
        """
        assert isinstance(minibatch, list)
        assert self.fix_length is None

        max_sent_num = max([len(ex) for ex in minibatch])
        sent_nums = [len(ex) for ex in minibatch]
        padded_data = [ex + [[0, 0]] * (max_sent_num - len(ex)) for ex in minibatch]
        return (padded_data, sent_nums)


class WordSentIdField(Field):
    def pad(self, minibatch):
        """
        Pad a batch of examples using this field.

        :param data: [[[0, 0, 0, 1, 1, 1, 1, 2, ...], [0, 0, 0, 0, 1, 1, 1, 2, ...], ...], list of examples' word sent ids
        :return: a padded list, we use 0 to pad the sentence ids of each example
        """
        assert isinstance(minibatch, list)
        assert self.fix_length is None

        src_lengths = [len(ex) for ex in minibatch]
        max_src_len = max(src_lengths)
        padded_data = [ex + [0] * (max_src_len - len(ex)) for ex in minibatch]
        return (padded_data, src_lengths)