import numpy as np

input_data = np.array( [[0,0,3],
                        [1,2,1],
                        [0,0,1],
                        [0,1,1]])
output_labels = np.array([  [1,0,1],
                            [0,1,0],
                            [0,0,1],
                            [0,1,1]])

def activate(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def activateelu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x

def binaryStep(x):
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0.5:
                x[i][k] = 1
            else:
                x[i][k] = 0
    return x


M0 = len(input_data[0])
N0 = 30
synaptic_weight0 = 2 * np.random.rand(M0,N0) - 1
M1 = N0
N1 = 20
synaptic_weight1 = 2 * np.random.rand(M1,N1) - 1
M2 = N1
N2 = 10
synaptic_weight2 = 2 * np.random.rand(M2,N2) - 1
M3 = N2
N3 = len(output_labels[0])
synaptic_weight3 = 2 * np.random.rand(M3,N3) - 1

# print(synaptic_weight0)
# print(synaptic_weight1)
# print(synaptic_weight2)
# print(synaptic_weight3)

epoch = 1000

for i in range(1, epoch):
    if( i % 10000 == 0):
        print(i)
    #foward Propagation
    layer0 = input_data
    layer1 = activate(np.dot(layer0, synaptic_weight0))
    layer2 = activate(np.dot(layer1, synaptic_weight1))
    layer3 = activate(np.dot(layer2, synaptic_weight2))
    layer4 = activate(np.dot(layer3, synaptic_weight3))

    
    layer4_error = output_labels - layer4                       #calculate layer4 error
    layer4_gradient = layer4_error * activate(layer4, True)     #compute gradiant
    synaptic_weight3 += np.dot(layer3.T, layer4_gradient)       #update weight

    layer3_error = np.dot(layer4_gradient, synaptic_weight3.T)  #calculate layer3 error
    layer3_gradient = layer3_error * activate(layer3, True)     #compute gradiant
    synaptic_weight2 += np.dot(layer2.T, layer3_gradient)       #update weight

    layer2_error = np.dot(layer3_gradient, synaptic_weight2.T)  #calculate layer2 error
    layer2_gradient = layer2_error * activate(layer2, True)     #compute gradiant
    synaptic_weight1 += np.dot(layer1.T, layer2_gradient)       #update weight

    layer1_error = np.dot(layer2_gradient, synaptic_weight1.T)  #calculate layer1 error
    layer1_gradient = layer1_error * activate(layer1, True)     #compute gradiant
    synaptic_weight0 += np.dot(layer0.T, layer1_gradient)       #update weight



# print("\ninput:\n" + str(input_data))
# print("expected output:\n" + str(output_labels))

layer0 = input_data
layer1 = activate(np.dot(layer0, synaptic_weight0))
#print(layer1)
layer2 = activate(np.dot(layer1, synaptic_weight1))
#print(layer2)
layer3 = activate(np.dot(layer2, synaptic_weight2))
#print(layer3)
output = activate(np.dot(layer3, synaptic_weight3))
print("\n" + str(np.round(output, 3)) + "\n")
print("\n" + str(output_labels) + "\n")
print("\n" + str(binaryStep(output)) + "\n")