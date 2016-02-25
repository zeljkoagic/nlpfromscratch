from itertools import count


class Vocab:
    def __init__(self, next_fn=None):
        if next_fn is None:
            self.id_counter = count()
        else:
            self.id_counter = next_fn

        self.item_to_id = {}
        self.id_to_item = {}

    def set(self, item, value):
        self.item_to_id[item] = value
        self.id_to_item[value] = item

    def map(self, item):
        return self[item]

    def __getitem__(self, item):
        id_for_item = self.item_to_id.get(item)
        if id_for_item is None:
            id_for_item = next(self.id_counter)
            self.item_to_id[item] = id_for_item
            self.id_to_item[id_for_item] = item

        return id_for_item

    def rev_map(self, id_):
        return self.id_to_item[id_]
