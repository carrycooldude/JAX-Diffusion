import pickle
from unet import UNet
from diffusion import Diffusion

class TrainState:
    def __init__(self, params, model, diffusion, optimizer, opt_state):
        self.params = params
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.opt_state = opt_state

    def replace(self, **kwargs):
        return TrainState(**{**self.__dict__, **kwargs})

def inspect_checkpoint():
    with open("checkpoint.pkl", "rb") as f:
        state = pickle.load(f)
    print("State attributes:", state.__dict__.keys())
    if hasattr(state, 'model'):
        print("Model type:", type(state.model))
    if hasattr(state, 'diffusion'):
        print("Diffusion type:", type(state.diffusion))
    
if __name__ == "__main__":
    inspect_checkpoint()
