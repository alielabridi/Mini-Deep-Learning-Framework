## This is the framework for a Mini Deep Learning Arch.
import torch
from torch import Tensor
import math
from sequential import Sequential
from activations import Relu, Tanh, LeakyRelu, Sigmoid
from mse import MSE
from linear import Linear
from helpers import *
import matplotlib
import matplotlib.pyplot as plt


### Generation of the DATA
train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)
#Standarize Data
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)
#Convert to Labels so that we can train
train_target_hot = conv_to_one_hot(train_target)
test_target_hot = conv_to_one_hot(test_target)


### Build the Network
hidden_layers = 3

layers = []
linear = Linear(2, 25, bias_init= True)
layers.append(linear)
layers.append(Relu())
for i in range(hidden_layers-1):
    layers.append(Linear(25, 25, bias_init= True))
    layers.append(Relu())
layers.append(Tanh())
layers.append(Linear(25, 2, bias_init= True))
model = Sequential(layers)

#print model summary
print("Model Summary:")
print(model)

### Select Parameters to train the model
criterion = MSE()
lr = 0.05
nb_epochs = 250
print_step = 25
mini_batch_size = 100

loss_at_print = []
test_accuracy = []
training_accuracy = []

### Training of the model
for e in range(nb_epochs+1):
    for b in range(0, train_input.size(0), mini_batch_size):
        
        ##Forward propagation
        output = model.forward(train_input.narrow(0, b, mini_batch_size).t())

        #Calculate loss
        loss = criterion.forward(output,train_target_hot.narrow(0,b,mini_batch_size).t())
        
        # put to zero weights and bias
        model.zero_grad()
        
        ##Backpropagation
        #Calculate grad of loss
        loss_grad = criterion.backward()

        #Grad of the model
        model.backward(loss_grad)

        #Update parameters
        model.grad_step(lr=lr)
        
        
    if e % print_step ==0 :
        print(f'Epoc : {e}, Loss: {loss}')
        #Save loss in vector
        loss_at_print.append(float(loss))
        test_prediction = model.forward(test_input.t())
        test_accuracy.append(float(compute_nb_errors(model,test_input, test_target)) / test_target.size(0) * 100)
        print(f'\tTest Accuracy : {100-test_accuracy[-1]}%')
        training_prediction = model.forward(train_input.t())
        training_accuracy.append(float(compute_nb_errors(model,train_input, train_target)) / train_target.size(0) * 100)
        print(f'\tTraining Accuracy : {100-training_accuracy[-1]}%')

### Printing and plotting Errors and Accuracy
print("\n\n *********************************** \n\n")
print('Number of training errors: ', float(compute_nb_errors(model,train_input, train_target)))
print(f"Accuracy in training: { 100.0 - float(compute_nb_errors(model,train_input, train_target)) / train_target.size(0) * 100} %")
print('Number of test errors: ', float(compute_nb_errors(model,test_input, test_target)))
print(f"Accuracy in test: { 100.0 - float(compute_nb_errors(model,test_input, test_target)) / test_target.size(0) * 100} %")

nb_epocs_array = list(range(0, nb_epochs+1, print_step))

test_accuracy_ =  [100 - int(x) for x in test_accuracy] 

fig, ax1 = plt.subplots()

ax1.plot(nb_epocs_array, loss_at_print, 'g-')
ax1.set_xlabel('Number of Epocs')
ax1.set_ylabel('Loss', color='g')

ax2 = ax1.twinx()
ax2.plot(nb_epocs_array, test_accuracy_, 'b-')
ax2.set_ylabel('Accuracy Test Set', color='b')

plt.grid()
plt.title("MiniDeepLearningFramework- 2*25 input unit - 25*2 output")
plt.savefig('figures/loss-numberepochs.png')
plt.savefig('figures/loss-numberepochs.pdf')
# plt.show()
