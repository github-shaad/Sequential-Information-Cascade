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
base_intelligence = np.zeros((N))

def phi(prev_phi):
    return prev_phi + np.random.normal()