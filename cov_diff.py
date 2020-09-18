
import numpy as np
import matplotlib.pyplot as plt
from math import cos,pi,exp
a=0#float(input("donner la valeur de a ="))
b=1#float(input("donner la valeur de b ="))
N=10#int(input("donner la valeur de N ="))
h=(b-a)/(N+1)
c=1
T=h/c
ep=0.005
r=1/(exp(1/ep)-1)
exact = lambda x: -r+r*exp(x/ep)
x=np.linspace(a,b,N+1)
f=[exact(e) for e in x]
L=np.zeros([N+1,N+1])
for i in range(N):    
    L[i][i+1]=1/2
    L[i+1][i]=-1/2
L[0][0]=-1/2  
L[N][N]=1/2  
S=np.zeros([N+1,N+1])
for i in range(N):
    S[i][i]=2/h
for i in range(N):    
    S[i][i+1]=-1/h
    S[i+1][i]=-1/h 
S[0][0]=1/h      
S[N][N]=1/h
S1=np.zeros((N+1,1))
S2=np.zeros((N+1,1))
G=np.dot(ep,S)+c*L
G[0,:]=0
G[0,0]=1
G[N,:]=0
G[N,N]=1
S1[0]=0
S1[N]=1
fnew=list(np.linalg.solve(G,S1))
G2=np.dot(ep+T*c*c,S)+c*L
G2[0,:]=0
G2[0,0]=1
G2[N,:]=0
G2[N,N]=1
S2[0]=0
S2[N]=1
fnew2=list(np.linalg.solve(G2,S2))
plt.plot(x,fnew,label='G_standar')
plt.plot(x,fnew2,label='SUPG')
plt.plot(x,f,label=' Exacte')
plt.legend()
plt.grid()
  
    
    
    
    
    
    
    