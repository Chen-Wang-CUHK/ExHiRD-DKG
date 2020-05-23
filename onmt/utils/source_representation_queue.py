"""
The code is from ken's github
"""
import numpy as np


class SourceReprentationQueue:
    # A FIFO memory for storing the encoder representations
    def __init__(self, capacity, seed=3435):
        self.capacity = capacity
        self.queue = []
        self.position = 0
        np.random.seed(seed)

    def put(self, tensor):
        if len(self.queue) < self.capacity:
            self.queue.append(None)
        self.queue[self.position] = tensor
        self.position = (self.position + 1) % self.capacity

    def sample(self, sample_size):
        if len(self.queue) < sample_size:
            return None
        # return random.choince(self.queue, sample_size)
        idxs = np.random.choice(len(self.queue), sample_size, replace=False)
        src_states_samples_list = [self.queue[i] for i in idxs]
        # insert a place-holder for the ground-truth source represenation to a random index
        place_holder_idx = np.random.randint(0, sample_size + 1)
        src_states_samples_list.insert(place_holder_idx, None)  # len=N+1
        return src_states_samples_list, place_holder_idx

    def __len__(self):
        return len(self.queue)