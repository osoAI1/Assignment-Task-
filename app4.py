i1, i2 = 0.05, 0.10  
b1, b2 = 0.35, 0.60
W1, W2, W3, W4 = 0.15, 0.20, 0.25, 0.30 
W5, W6, W7, W8 = 0.40, 0.45, 0.50, 0.55 
Tar_o1, Tar_o2 = 0.01, 0.99
learnRate = 0.5

def sigmoid(x):
    return 1 / (1 + (2.718281828459045) ** (-x))

def sigmDer(x):
    return x * (1 - x)

netH1 = W1 * i1 + W2 * i2 + b1
netH2 = W3 * i1 + W4 * i2 + b1

outH1 = sigmoid(netH1)
outH2 = sigmoid(netH2)

netO1 = W5 * outH1 + W6 * outH2 + b2
netO2 = W7 * outH1 + W8 * outH2 + b2

outO1 = sigmoid(netO1)
outO2 = sigmoid(netO2)

Error_o1 = 0.5 * (Tar_o1 - outO1) ** 2
Error_o2 = 0.5 * (Tar_o2 - outO2) ** 2
Error_total = Error_o1 + Error_o2

deltaO1 = (outO1 - Tar_o1) * sigmDer(outO1)
deltaO2 = (outO2 - Tar_o2) * sigmDer(outO2)

deltaH1 = (deltaO1 * W5 + deltaO2 * W7) * sigmDer(outH1)
deltaH2 = (deltaO1 * W6 + deltaO2 * W8) * sigmDer(outH2)

W1New = W1 - learnRate * deltaH1 * i1
W2New = W2 - learnRate * deltaH1 * i2
W3New = W3 - learnRate * deltaH2 * i1
W4New = W4 - learnRate * deltaH2 * i2
W5New = W5 - learnRate * deltaO1 * outH1
W6New = W6 - learnRate * deltaO1 * outH2
W7New = W7 - learnRate * deltaO2 * outH1
W8New = W8 - learnRate * deltaO2 * outH2

print("Backpropagation")
print(f"W1==> {W1New:.6f}, W2==> {W2New:.6f}, W3==> {W3New:.6f}, W4==> {W4New:.6f}")
print(f"W5==> {W5New:.6f}, W6==> {W6New:.6f}, W7==> {W7New:.6f}, W8==> {W8New:.6f}")
