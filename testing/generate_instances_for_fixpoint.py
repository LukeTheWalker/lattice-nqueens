import random
import numpy as np

def generate_instance(n: int, seed: int):
    random.seed(seed)
    # U
    U:list = [random.randint(0,3)for _ in range(n)]
    # C
    C = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i, n, 1):
            C[i][j] = random.randint(0,1)
            C[j][i] = C[i][j]
            if i == j:
                C[i][j] = 0

    with open(f"test_cases/pco_{seed}.txt", "+w") as f:
        f.write("N\n")
        f.write(f"{n}\n")
        f.write("U\n")
        for i in range(n):
            f.write(f"{i};{U[i]}\n")
        
        f.write("C\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{i},{j};{C[i][j]}\n")
    return 


if __name__ == '__main__':
    generate_instance(4, 43)