#################### Define the Class for the development of the model
class Module(object):

    def __init__(self):
        pass

    def forward(self , * input):
        raise NotImplementedError
    
    def backward(self , * gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []

    def param_grad(self):
        return []

    def update_param(self, lr):
        pass

    def init_weights(self):
        pass

    def set_zero_grad(self):
        pass
