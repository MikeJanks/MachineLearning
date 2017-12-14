import numpy as np
f = '../../DataSet/Youtube/USvideos-adjusted-edited.csv'
data = np.genfromtxt(f, dtype=None, skip_header=True, delimiter= ",")

test_input_data = data[2000:2010]
test_output_labels = np.zeros((len(test_input_data), 1))
print(test_output_labels)
test_output_labels.fill(1)
print(test_output_labels)

input_data = np.array( [[0,0,3],
                        [1,2,1],
                        [0,0,1],
                        [0,1,1]])
                        
output_labels = np.zeros((len(input_data),1))

print(output_labels)
