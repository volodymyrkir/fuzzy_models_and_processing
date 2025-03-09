import numpy as np
import matplotlib.pyplot as plt


W = np.array([f'a{i+1}' for i in range(8)])
u1 = np.array([1, 1, 0.6, 0, 0.7, 0.4, 0.1, 0])
u2 = np.array([0.6, 0.9, 0.5, 0.3, 0, 0.5, 1, 0.7])
n = len(u1)

plt.figure(figsize=(10, 5))
plt.plot(W, u1, 'bo-', label='X (u1)')
plt.plot(W, u2, 'go-', label='Y (u2)')
plt.title("Fuzzy Sets X and Y")
plt.xlabel("Elements of W")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()
plt.show()

X_complement = 1 - u1
Y_complement = 1 - u2
X_intersection_Y = np.minimum(u1, u2)
X_union_Y = np.maximum(u1, u2)
X_XOR_Y = np.maximum(u1, u2) - np.minimum(u1, u2)


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
plt.plot(W, X_XOR_Y, 'ko-', label='X ⊕ Y (XOR)')
plt.title("Fuzzy Intersection, Union, and Difference")
plt.legend()
plt.grid()
plt.show()

# SUM(ABS(U1-U2)
hamming_abs = np.sum(np.abs(u1 - u2))
# abs * 1/n
hamming_rel = hamming_abs / len(W)
# SQRT(SUM(u1-u2)^2)
euclidean_abs = np.sqrt(np.sum((u1 - u2) ** 2))
# abs * 1 / sqrt(n)
euclidean_rel = euclidean_abs / np.sqrt(len(W))

print("Hamming Distance (Absolute):", hamming_abs)
print("Hamming Distance (Relative):", hamming_rel)
print("Euclidean Distance (Absolute):", euclidean_abs)
print("Euclidean Distance (Relative):", euclidean_rel)

X_crisp = (u1 > 0.5).astype(int)
Y_crisp = (u2 > 0.5).astype(int)

print("Crisp Subset Closest to X: {" + ', '.join(W[np.where(X_crisp == 1)]) + "}")
print("Crisp Subset Closest to Y: {" + ', '.join(W[np.where(Y_crisp == 1)]) + "}")

def hamming_distance(u, u_complement):
    return np.sum(np.abs(u - u_complement))

def euclidean_distance(u, u_complement):
    return np.sqrt(np.sum((u - u_complement)**2))

v_X = (2 / n) * hamming_distance(u1, X_complement)
v_Y = (2 / n) * hamming_distance(u2, Y_complement)

eta_X = (2 / np.sqrt(n)) * euclidean_distance(u1, X_complement)
eta_Y = (2 / np.sqrt(n)) * euclidean_distance(u2, Y_complement)

print(f"Fuzzy Hamming for X: {v_X:.5f}")
print(f"Fuzzy Hamming for Y: {v_Y:.5f}")
print(f"Fuzzy Euclidean for X: {eta_X:.5f}")
print(f"Fuzzy Euclidean for Y: {eta_Y:.5f}")
