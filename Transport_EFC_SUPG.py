# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:57:59 2020

@author: MOUHCINE/MEHDI
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
a=-5#float(input("donner la valeur de a ="))
b=5#float(input("donner la valeur de b ="))
N=500#int(input("donner la valeur de N ="))
c=1#float(input("donner la vitesse c ="))
ht=0.01#float(input("donner le pas de temps ht ="))
Tmax=1#float(input("donner le temps maximal Tmax ="))
h=(b-a)/(N+1)
def U0(x):
    if x>=-1 and x<=1 :
        return 5
    else:
        return 0

def Uexact(t,x,c) :
    return(U0(x-c*t))
x=np.linspace(a,b,N+1)
f=[U0(e) for e in x]
M=np.zeros([N+1,N+1])
for i in range(N):
    M[i][i]=2*h/3
for i in range(N):    
    M[i][i+1]=h/6
    M[i+1][i]=h/6
M[0][0]=h/3  
M[N][N]=h/3   
L=np.zeros([N+1,N+1])
for i in range(N):    
    L[i][i+1]=1/2
    L[i+1][i]=-1/2
L[0][0]=-1/2  
L[N][N]=1/2  
R=np.zeros([N+1,N+1])
for i in range(N):    
    R[i][i+1]=-1/2
    R[i+1][i]=1/2
R[0][0]=-1/2  
R[N][N]=1/2 
S=np.zeros([N+1,N+1])
for i in range(N):
    S[i][i]=2/h
for i in range(N):    
    S[i][i+1]=-1/h
    S[i+1][i]=-1/h 
S[0][0]=1/h      
S[N][N]=1/h
z=0 
i=1
T=h/c
while z<=Tmax:
    G=M+c*T*R+np.dot(ht,c*c*T*S+L)
    G[0,:]=0
    G[0,0]=1
    S1=np.dot(M+c*T*R,f[0:N+1])
    S1[0]=0
    fnew=list(np.linalg.solve(G,S1))
    o=[Uexact(z,e,c) for e in x]
    plt.plot(x,fnew,label='U approchée')
    plt.plot(x,o,label='U Exacte')
    #plt.title(' dt=%1.2f ' %ht  )
    plt.legend()
    plt.grid()
    plt.pause(0.005)
    f=fnew
    z=z+ht

    
    
    
    
    
    
    