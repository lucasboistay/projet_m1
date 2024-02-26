
# MODÈLE D'ISING EN 2D 

# Importation des modules 
import numpy as np 
import matplotlib.pyplot as plt 
import random as rd 


# Définition des constantes du problème 
J = 1         # Terme d'énergie (J)
T = 300       # Température absolue (K)
k = 1.38e-23  # Constante de Boltzmann (J/K)
𝛽 = 1/(k*T)   # Température réduite 

n = 20
m = 20

def grille(n, m, spin): # Génère une grille de taille n x m de spin up ou down 
    if spin == "up":
        x = 1           # 1 pour un spin up  
    else:
        x = -1          # -1 pour un spin down 
    mat = np.zeros((n + 1, m + 1), dtype = int)
    for i in range(n+1):
        for j in range(m+1):
            mat[i, j] = x
    return mat
    
    
grid = grille(n, m, "up")
print(grid)



# On fixe des conditions aux limites périodiques et on génère un spin 
def flip(grille, i, j):
    if (i > len(grille[0]) or j > len(grille)):
        raise Exception("Dimensions impossibles")
    else: 
        grille[i, j] *= -1
    n = len(grille)         # Nombre de ligne 
    m = len(grille[0])      # Nombre de colonne 
    for i in range(n):      # On fixe la dernière colonne à la première colonne
        grille[i][m-1] = grille[i][0]
    for j in range(m):      # On fixe la dernière ligne à la première ligne 
        grille[n-1][j] = grille[0][j]
        
        
        
# Calcul de la variation d'énergie 
def deltaE(grille,i,j):
    delta = 2 * grille[i, j]*(grille[i-1, j] + grille[i+1, j] + grille[i, j-1] + grille[i, j+1])
    return delta


# Applique la méthode de Monte Carlo 
def applique_monte_carlo(grille, i,j):
    flip(grille, i, j)                # On effectue le flip 
    energie = deltaE(grille, i, j)    # On calcul l'énergie 
    flip(grille, i, j)                # On annule le flip avant de faire une quelconque procédure 
    
    if energie < 0: # On accepte le changement et on le réalise vraiment 
        flip(grille, i, j)
    else:           # On accepte ou non le changement 
        proba = np.exp(-𝛽*energie)    # On définit une probabilité 
        test = rd.random()
        if test < proba:              # On accepte le flip 
            flip(grille, i, j)
        else:
            pass 

        

def magnetisation(grille):
    n = len(grille)
    m = len(grille[0])
    N = 0
    for i in range(n):
        for j in range(m):
            N += grille[i][j] 
    return N/(n*m) 


t_ini = 0  # temps initial


les_T = [1000*i for i in range(1, 6)]
les_M = []

for T in les_T:
    grid = grille(n, m, "up")
    les_m = []
    for t in range(t_ini, T):
        i = rd.choice(list(range(n)))
        j = rd.choice(list(range(m)))
        applique_monte_carlo(grid, i, j) 
        les_m.append(magnetisation(grid))
    les_M.append(les_m)

print()
print(grid)
print()

plt.figure(figsize = (14, 11))
for i in range(len(les_M)):
    plt.plot(les_M[i], label = "${}T_mc$".format(i + 1)) 
    plt.legend()
plt.ylim(-0.1, 1.1)
plt.title("Évolution de la magnétisation en fonction du temps de Monte Carlo à température fixée", size = 20)
plt.xlabel("Temps de Monte Carlo [s]", size = 20)
plt.ylabel("Magnétisation [A/m]", size = 20)
plt.axhline(c = "black", linewidth = 1)
plt.axvline(c = "black", linewidth = 1)
plt.show()


