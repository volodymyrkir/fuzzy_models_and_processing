import numpy as np
import matplotlib.pyplot as plt

N = 8
W = np.array([f'a{i+1}' for i in range(N)])
uX = np.array([1.0, 1.0, 0.6, 0.0, 0.7, 0.4, 0.1, 0.0])
uY = np.array([0.6, 0.9, 0.5, 0.3, 0.0, 0.5, 1.0, 0.7])

plt.figure(figsize=(10, 5))
plt.plot(W, uX, 'bo-', label='X (u1)')
plt.plot(W, uY, 'go-', label='Y (u2)')
plt.title("Fuzzy Sets X and Y")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()
plt.show()

X_complement = 1 - uX
Y_complement = 1 - uY
X_intersection_Y = np.minimum(uX, uY)
X_union_Y = np.maximum(uX, uY)
X_xor_Y = np.abs(uX - uY)


plt.figure(figsize=(10, 5))
plt.plot(W, X_complement, 'ro-', label='X complement')
plt.plot(W, Y_complement, 'mo-', label='Y complement')
plt.title("Fuzzy Complements of X and Y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(W, X_intersection_Y, 'co-', label='X ∩ Y')
plt.plot(W, X_union_Y, 'yo-', label='X ∪ Y')
plt.plot(W, X_xor_Y, 'ko-', label='X ⊖ Y')
plt.title("Fuzzy Intersection, Union, and Symmetric Difference")
plt.legend()
plt.grid()
plt.show()

# SUM(ABS(U1-U2)
hamming_abs = np.sum(np.abs(uX - uY))
# abs * 1/n
hamming_rel = hamming_abs / N
# SQRT(SUM(u1-u2)^2)
euclidean_abs = np.sqrt(np.sum((uX - uY) ** 2))
# abs * 1 / sqrt(n)
euclidean_rel = euclidean_abs / np.sqrt(N)

print("Hamming Distance (Absolute):", hamming_abs)
print("Hamming Distance (Relative):", hamming_rel)
print("Euclidean Distance (Absolute):", euclidean_abs)
print("Euclidean Distance (Relative):", euclidean_rel)

X_crisp = (uX > 0.5).astype(int)
Y_crisp = (uY > 0.5).astype(int)

print("Crisp Subset Closest to X: {" + ', '.join(W[np.where(X_crisp == 1)]) + "}")
print("Crisp Subset Closest to Y: {" + ', '.join(W[np.where(Y_crisp == 1)]) + "}")

def hamming_distance(u, u_complement):
    return np.sum(np.abs(u - u_complement))

def euclidean_distance(u, u_complement):
    return np.sqrt(np.sum((u - u_complement)**2))

v_X = (2 / N) * hamming_distance(uX, X_crisp)
v_Y = (2 / N) * hamming_distance(uY, Y_crisp)

eta_X = (2 / np.sqrt(N)) * euclidean_distance(uX, X_crisp)
eta_Y = (2 / np.sqrt(N)) * euclidean_distance(uY, Y_crisp)

print(f"Fuzzy Hamming for X: {v_X:.5f}")
print(f"Fuzzy Hamming for Y: {v_Y:.5f}")
print(f"Fuzzy Euclidean for X: {eta_X:.5f}")
print(f"Fuzzy Euclidean for Y: {eta_Y:.5f}")
