import numpy as np
# Encoding for characters
char_to_vec = {
    'd': [1, 0, 0, 0],
    'o': [0, 1, 0, 0],
    'g': [0, 0, 1, 0],
    's': [0, 0, 0, 1]
}
vocab = list(char_to_vec.keys())

#wights
Wx = np.random.randn(3, 4) * 0.1 
Wh = np.random.randn(3, 3) * 0.1   
Wy = np.random.randn(4, 3) * 0.1   

h0 = np.zeros(3)   

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def forward(seq, h):
    hs, ys = [], []
    for ch in seq:
        x = np.array(char_to_vec[ch])
        h = np.tanh(Wx @ x + Wh @ h)
        y = softmax(Wy @ h)
        hs.append(h)
        ys.append(y)
    return hs, ys

def BPTT(seq, target, learinRate=0.1):
    global Wx, Wh, Wy

    hs, ys = forward(seq, h0)

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)

    dh_next = np.zeros_like(h0)
    loss = 0

    for t in reversed(range(len(seq))):

        x = np.array(char_to_vec[seq[t]])

        y_true = np.array(char_to_vec[target[t]])
        
        dy = ys[t] - y_true
        loss += -np.sum(y_true * np.log(ys[t] + 1e-8))

        dWy += np.outer(dy, hs[t])

        dh = Wy.T @ dy + dh_next
        da = dh * (1 - hs[t]**2)

        dWx += np.outer(da, x)
        dWh += np.outer(da, hs[t-1] if t > 0 else h0)
        dh_next = Wh.T @ da

    Wx -= learinRate * dWx
    Wh -= learinRate * dWh
    Wy -= learinRate * dWy
    return loss

# Train
print("forward:")
for epoch in range(50):
    loss = BPTT("dog", "ogs")
    if epoch % 10 == 0:#0-->10-->15-->20-->epoch+10
       
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# Test
_, preds = forward("dog", h0)
print("\npredictions:")
for i, y in enumerate(preds):
    pred = vocab[np.argmax(y)]

    print(f"Step {i+1}/3: {pred} --> {np.round(y, 2)}")
