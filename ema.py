from utils import clamp
import torch
class EMA():
    def __init__(self, decay,epsilon,lower_limit,upper_limit):
        self.decay = decay
        self.shadow = None
        self.backup = None
        self.epsilon = epsilon
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def register(self,delta):
        if delta.requires_grad:
            self.shadow = delta.data.clone()

    def update(self,delta):
        if delta.requires_grad:
            new_average = (1.0 - self.decay) * delta.data + self.decay * self.shadow
            self.shadow = new_average.clone()

            self.shadow = clamp(torch.clamp(self.shadow,-self.epsilon,self.epsilon),self.lower_limit,self.upper_limit)

    def apply_shadow(self,delta):
        if delta.requires_grad:
            self.backup = delta.data
            delta.data = self.shadow

    def restore(self,delta):
        if delta.requires_grad:
            delta.data= self.backup
        self.backup = None