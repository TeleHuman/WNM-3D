import time


class ContextTimer:
    def __init__(self, trainer):
        self.last_key = None
        self.trainer = trainer
        self.start_times = {}
        self.key_stack = []

    def with_label(self, key):
        self.last_key = key
        return self

    def __enter__(self):
        self.key_stack.append(self.last_key)  # Push key to stack
        self.start_times[self.last_key] = time.time()  # Start timing for this key
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        key = self.key_stack.pop()  # Pop key from stack
        diff = time.time() - self.start_times[key]
        self.trainer.log({f"{key}_time": diff})
