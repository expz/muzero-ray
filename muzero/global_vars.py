from collections import defaultdict

import ray


@ray.remote
class GlobalVars:
    def __init__(self):
        self._counters = defaultdict(int)
        print('Global variables initialized.')

    def increment(self, counter: str):
        self.add(counter, 1)

    def decrement(self, counter: str):
        self.add(counter, -1)

    def add(self, counter: str, n: int):
        self._counters[counter] += n

    def get_count(self, counter: str):
        return self._counters[counter]
    
    def get_and_increment_count(self, counter: str):
        count = self._counters[counter]
        self.increment(counter)
        return count