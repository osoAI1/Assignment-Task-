import random

def tanh(x):
    exp_x = (2.718281828459045) ** x
    exp_neg_x = (2.718281828459045) ** (-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

def equation(x, w, b):
    return sum(i * z for i, z in zip(x, w)) + b

i1, i2 = 0.05, 0.10 
b1, b2 = 0.5, 0.7  

w1, w2 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
w3, w4 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
w5, w6 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
w7, w8 = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)

net_h1 = equation([i1, i2], [w1, w2], b1)
h1 = tanh(net_h1)

net_h2 = equation([i1, i2], [w3, w4], b1)
h2 = tanh(net_h2)

net_o1 = equation([h1, h2], [w5, w6], b2)
o1 = tanh(net_o1)

net_o2 = equation([h1, h2], [w7, w8], b2)
o2 = tanh(net_o2)

print(f'h1: {h1} and h2: {h2}')
print(f'o1: {o1} and o2: {o2}')