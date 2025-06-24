import random

from vmen.metrics.BaseMetric import BaseMetric


class RandomScore(BaseMetric):
    def __init__(self):
        pass
        
    def __call__(self, caption, path):
        if type(caption) == str:
            return random.random()
        
        elif type(caption) == list:
            return [random.random() for _ in range(len(caption))]
    
    def get_state(self):
        return f"rand"
    
if __name__ == "__main__":
    metric = RandomScore()
    scores = metric(["Text"], ["Path to images"])
    print("Random ", scores)
