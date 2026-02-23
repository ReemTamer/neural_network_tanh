import numpy as np

# Define tanh activation function
def tanh(x):
    return np.tanh(x)
# Initialize Input Values
i1, i2 = 0.05, 0.10

print("Input values:")
print(f"i1 = {i1}, i2 = {i2}")

# Initialize Random Weights from [-0.5, 0.5]
np.random.seed(42)
# Generate random weights
w1 = np.random.uniform(-0.5, 0.5)
w2 = np.random.uniform(-0.5, 0.5)
w3 = np.random.uniform(-0.5, 0.5)
w4 = np.random.uniform(-0.5, 0.5)
w5 = np.random.uniform(-0.5, 0.5)
w6 = np.random.uniform(-0.5, 0.5)
w7 = np.random.uniform(-0.5, 0.5)
w8 = np.random.uniform(-0.5, 0.5)
# Bias values as specified in the question
b1, b2 = 0.5, 0.7
print("Random weights (from [-0.5, 0.5]):")
print(f"w1 = {w1:.6f}")
print(f"w2 = {w2:.6f}")
print(f"w3 = {w3:.6f}")
print(f"w4 = {w4:.6f}")
print(f"w5 = {w5:.6f}")
print(f"w6 = {w6:.6f}")
print(f"w7 = {w7:.6f}")
print(f"w8 = {w8:.6f}")
print(f"\nBias values:")
print(f"b1 = {b1}")
print(f"b2 = {b2}")
# Hidden Layer Computation
net_h1 = w1 * i1 + w2 * i2 + b1 * 1
out_h1 = tanh(net_h1)

# Neuron h2
net_h2 = w3 * i1 + w4 * i2 + b1 * 1
out_h2 = tanh(net_h2)

print("Hidden Layer Calculations:")
print(f"Neuron h1:")
print(f"  net_h1 = ({w1:.6f} × {i1}) + ({w2:.6f} × {i2}) + ({b1} × 1) = {net_h1:.6f}")
print(f"  out_h1 = tanh({net_h1:.6f}) = {out_h1:.6f}")
print(f"\nNeuron h2:")
print(f"  net_h2 = ({w3:.6f} × {i1}) + ({w4:.6f} × {i2}) + ({b1} × 1) = {net_h2:.6f}")
print(f"  out_h2 = tanh({net_h2:.6f}) = {out_h2:.6f}")
#Output Layer Computation
net_o1 = w5 * out_h1 + w6 * out_h2 + b2 * 1
out_o1 = tanh(net_o1)

# Neuron o2
net_o2 = w7 * out_h1 + w8 * out_h2 + b2 * 1
out_o2 = tanh(net_o2)

print("Output Layer Calculations:")
print(f"Neuron o1:")
print(f"  net_o1 = ({w5:.6f} × {out_h1:.6f}) + ({w6:.6f} × {out_h2:.6f}) + ({b2} × 1) = {net_o1:.6f}")
print(f"  out_o1 = tanh({net_o1:.6f}) = {out_o1:.6f}")
print(f"\nNeuron o2:")
print(f"  net_o2 = ({w7:.6f} × {out_h1:.6f}) + ({w8:.6f} × {out_h2:.6f}) + ({b2} × 1) = {net_o2:.6f}")
print(f"  out_o2 = tanh({net_o2:.6f}) = {out_o2:.6f}")
# Display Final Results
print("FINAL NETWORK OUTPUTS:")
print(f"Output o1 = {out_o1:.8f}")
print(f"Output o2 = {out_o2:.8f}")

print("Network architecture: 2 inputs → 2 hidden neurons → 2 outputs")
print(f"Activation function: tanh")
print(f"Weights range: [-0.5, 0.5]")
print(f"Bias values: b1={b1}, b2={b2}")
# Target values 
target_o1, target_o2 = 0.1, 0.99

# Calculate errors 
error_o1 = 0.5 * (target_o1 - out_o1)**2
error_o2 = 0.5 * (target_o2 - out_o2)**2
total_error = error_o1 + error_o2

print("ERROR CALCULATION :")
print(f"Target values: o1_target = {target_o1}, o2_target = {target_o2}")
print(f"Error o1 = 0.5 × ({target_o1} - {out_o1:.6f})² = {error_o1:.8f}")
print(f"Error o2 = 0.5 × ({target_o2} - {out_o2:.6f})² = {error_o2:.8f}")
print(f"Total Error = {total_error:.8f}")
