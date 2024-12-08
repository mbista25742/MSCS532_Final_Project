import numpy as np
import time

def traditional_matrix_multiply_manual(A, B):
    M, N = A.shape
    N, P = B.shape
    C = np.zeros((M, P))

    for i in range(M):
        for j in range(P):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

    return C

def blocked_matrix_multiply(A, B, block_size):
    M, N = A.shape
    N, P = B.shape
    C = np.zeros((M, P))

    for i in range(0, M, block_size):
        for j in range(0, P, block_size):
            for k in range(0, N, block_size):
                A_block = A[i:i+block_size, k:k+block_size]
                B_block = B[k:k+block_size, j:j+block_size]
                C[i:i+block_size, j:j+block_size] += np.dot(A_block, B_block)

    return C

def numpy_dot_multiply(A, B):
    return np.dot(A, B)

def measure_time(func, *args):
    start_time = time.perf_counter()
    func(*args)
    end_time = time.perf_counter()
    return end_time - start_time

M, N, P = 500, 500, 500  
A = np.random.rand(M, N)
B = np.random.rand(N, P)

# Traditional matrix multiplication (manual)
print("Benchmarking Traditional Matrix Multiplication (Manual)...")
traditional_time = measure_time(traditional_matrix_multiply_manual, A, B)
print(f"Traditional matrix multiplication time (manual): {traditional_time:.6f} seconds")

# Blocked matrix multiplication for various block sizes
block_sizes = [32, 64, 128]  
blocked_times = {}

print("\nBenchmarking Blocked Matrix Multiplication for various block sizes...")
for block_size in block_sizes:
    blocked_time = measure_time(blocked_matrix_multiply, A, B, block_size)
    blocked_times[block_size] = blocked_time
    print(f"Blocked matrix multiplication time (block size {block_size}): {blocked_time:.6f} seconds")

# Numpy optimized matrix multiplication using np.dot
print("\nBenchmarking Optimized Matrix Multiplication (numpy.dot)...")
numpy_dot_time = measure_time(numpy_dot_multiply, A, B)
print(f"Optimized numpy.dot matrix multiplication time: {numpy_dot_time:.6f} seconds")

# Comparison of results
print("\nComparison of Traditional, Blocked, and Optimized Matrix Multiplication:")
print(f"Traditional matrix multiplication time (manual): {traditional_time:.6f} seconds")

for block_size, blocked_time in blocked_times.items():
    print(f"Blocked matrix multiplication time (block size {block_size}): {blocked_time:.6f} seconds")

print(f"Optimized numpy.dot matrix multiplication time: {numpy_dot_time:.6f} seconds")
