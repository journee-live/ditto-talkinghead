from threading import Lock


class AtomicCounter:
    def __init__(self, initial=0):
        self.value = initial
        self.lock = Lock()

    def increment(self, amount=1):
        with self.lock:
            self.value += amount
            return self.value

    def decrement(self, amount=1):
        with self.lock:
            self.value -= amount
            return self.value

    def get(self):
        with self.lock:
            return self.value

    def set(self, amount):
        with self.lock:
            self.value = amount
