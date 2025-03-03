import random
i1, i2 = 0.05, 0.10 
b1, b2 = 0.5, 0.7  
Tar_o1, Tar_o2 = 0.01, 0.99 
learnRate = 0.5  
w1, w2 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
w3, w4 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
w5, w6 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
w7, w8 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)

def tanh(x):
    exp_x = (2.718281828459045) ** x
    exp_neg_x = (2.718281828459045) ** (-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

def tanhDer(x):
    return 1 - x**2  

def equation(x, w, b):
    return sum(i * z for i, z in zip(x, w)) + b

netH1 = equation([i1, i2], [w1, w2], b1)
h1 = tanh(netH1)
netH2 = equation([i1, i2], [w3, w4], b1)
h2 = tanh(netH2)
netO1 = equation([h1, h2], [w5, w6], b2)
o1 = tanh(netO1)
netO2 = equation([h1, h2], [w7, w8], b2)
o2 = tanh(netO2)

print('forward')
print(f'h1: {h1} and h2: {h2}')
print(f'o1: {o1} and o2: {o2}')

Error_o1 = 0.5 * (Tar_o1 - o1) ** 2
Error_o2 = 0.5 * (Tar_o2 - o2) ** 2
Error_total = Error_o1 + Error_o2

delta_o1 = (o1 - Tar_o1) * tanhDer(o1)
delta_o2 = (o2 - Tar_o2) * tanhDer(o2)
deltaH1 = (delta_o1 * w5 + delta_o2 * w7) * tanhDer(h1)
deltaH2 = (delta_o1 * w6 + delta_o2 * w8) * tanhDer(h2)

w1 -= learnRate * deltaH1 * i1
w2 -= learnRate * deltaH1 * i2
w3 -= learnRate * deltaH2 * i1
w4 -= learnRate * deltaH2 * i2
w5 -= learnRate * delta_o1 * h1
w6 -= learnRate * delta_o1 * h2
w7 -= learnRate * delta_o2 * h1
w8 -= learnRate * delta_o2 * h2

print('\nBackpropagation')
print("Updated Weights:")
print(f"W1: {w1}, W2: {w2}, W3: {w3}, W4: {w4}")
print(f"W5: {w5}, W6: {w6}, W7: {w7}, W8: {w8}")
