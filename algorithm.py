import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
N = 8
M = 2
L = 4
C = 20
BASE = 0.02
RATE = 1

value_matrix = np.zeros((N,M,L+1))
policy_matrix = np.zeros((N,M,L+1))

def phi(i,j):
    turn = (j)*N + i - 1
    return 1 - (1-BASE)*np.exp(-RATE*turn)

def s_prime(i,j,l):
    s = 1

    if i != N-1:
        for next_i in range(i+1, N):
            s *= (1-phi(next_i,j)*policy_matrix[next_i][j][l])
    if i != 0 and j < M-1:
        for prev_i in range(i-1,-1,-1):
            s *= (1-phi(prev_i,j+1)*policy_matrix[prev_i][j+1][l])
    return s

#base case of backward induction
for n in range(N-1,-1,-1):
    for l in range(0,L+1):
        if l == 0:
            value_matrix[n][M-1][l] = 0
            policy_matrix[n][M-1][l] = 0
        else:
            value_matrix[n][M-1][l] = C
            policy_matrix[n][M-1][l] = 1

#backward induction
for m in range(M - 2, -1, -1):
    for n in range(N - 1, -1, -1):
        for l in range(1, L + 1):
            current_phi = phi(n, m)
            s_val = s_prime(n, m, L) 
            value_ans = C + (1 - current_phi) * s_val * value_matrix[n][m + 1][l - 1]
            value_pass = s_val * value_matrix[n][m + 1][l]
            if value_ans >= value_pass:
                value_matrix[n][m][l] = value_ans
                policy_matrix[n][m][l] = 1
            else:
                value_matrix[n][m][l] = value_pass
                policy_matrix[n][m][l] = 0

def plot_policy_heatmap(policy_matrix, remaining_guesses):
    N, M, _ = policy_matrix.shape
    policy_slice = policy_matrix[:, :, remaining_guesses]
    
    colors = ["#8a0101", '#86efac'] 
    cmap = ListedColormap(colors)

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(policy_slice, 
                     cmap=cmap, 
                     linewidths=0.5, 
                     linecolor='white',
                     cbar=False,            
                     annot=True,            
                     fmt='g', 
                     annot_kws={"size": 10, "weight": "bold"})
    
    ax.set_title(f"SPNE Optimal Policy (Remaining Guesses: {remaining_guesses})\n0 = PASS, 1 = GUESS", 
                 fontsize=16, pad=20, weight='bold')
    ax.set_xlabel("Cycle (M)", fontsize=14, labelpad=10)
    ax.set_ylabel("Player (N)", fontsize=14, labelpad=10)
    
    # 3. ROTATE X-LABELS: Added rotation=45 and horizontalalignment='right'
    ax.set_xticklabels([f"Cycle {m+1}" for m in range(M)], rotation=45, ha='right')
    ax.set_yticklabels([f"Player {n+1}" for n in range(N)], rotation=0)
    
    plt.tight_layout()
    plt.show()


is_optimal_always_ans = (policy_matrix[:,:,1:] == 1).all()
print(is_optimal_always_ans)

for l in range(1,L+1):
    plot_policy_heatmap(policy_matrix, l)