# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:27:40 2024

@author: roron
"""
import random
import matplotlib.pyplot as plt
import numpy as np
# Number of iterations
n_iterations = 200
       
def map_rw(n,x0):
    positionL=[x0]
    position=x0
    for i in range(n):
        step = random.choice([-1, 1])
        #print(step)
        positionL.append(positionL[i]+step)
        position+=step
    #print("Final position after", n, "iterations:", position)
    #print("The path chosen with", n, "step is", positionL)
    return positionL

Lx=[]
for r in range(n_iterations+1):
    Lx.append(r)

nb_run=2000
List_df=[] #Liste de destination finale
for k in range(nb_run):
    y=map_rw(n_iterations,0)
    plt.plot(Lx,y)
    plt.xlabel('Nombre de pas')
    plt.ylabel('Position')
    List_df.append(y[n_iterations])
plt.show()
print(np.mean(List_df))

#Moyenne de position finale = position initiale aprrès nombre d'itérations
L_mean=[]
comptage=[]
for p in range(20):
    L=[]
    comptage.append(p)
    for x in range(nb_run):
        y=map_rw(n_iterations,p)
        L.append(y[n_iterations])
    L_mean.append(np.mean(L))

plt.plot(comptage,L_mean)
plt.xlabel('Position intial')
plt.ylabel('Moyenne de position finale')
plt.show()
    
#HISTOGRAMME

plt.hist(List_df, range = (-60,60), bins = 120)
plt.xlabel('''Nombre d'occurences''')
plt.ylabel('Positions finales')
plt.title('Exemple d\' histogramme simple')
plt.show()
#Variance = 2(t-t0)
Var=[]  
comptage=[]
for t in range(n_iterations):
    L=[]
    comptage.append(t)
    for x in range(nb_run):
        y=map_rw(t,0)
        L.append((y[t]-p)**2)
    Var.append(np.mean(L))
    

plt.plot(comptage,Var)
plt.xlabel('temps écoulé')
plt.ylabel('Variance')
plt.show()
    
    
    
    
    
    
    
    

