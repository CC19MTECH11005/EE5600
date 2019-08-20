import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#Matrix for V
n1 = np.array([9,0])
n2 = np.array([0,-16])
V  = np.vstack((n1,n2))
print(V)

#Entering the point P on hyperbola
P = np.array ([8,3*np.sqrt(3)])
print(P)

#Multiplying P and V

B = P@V
print(B)

n = np.array([48*np.sqrt(3), 72])
print(n)

M = np.array([48*np.sqrt(3), 72])
G = np.array([8, 3*np.sqrt(3)])
g = G.T
L = M@g
print(L)



#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
theta = np.linspace(-5,5,len)

#Given hyperbola parameters
#Eqn : x.T@V@x = F
V = np.array(([9,0],[0,-16]))
F = 144

eigval,eigvec = LA.eig(V)
print(eigval)
print(eigvec)

D = np.diag(eigval)
W = eigvec
print("D=\n",D)
print("W=\n",W)

#Standard hyperbola : y.T@D@y=1
y1 = np.linspace(-1,1,len)
y2 = np.sqrt((1-D[0,0]*np.power(y1,2))/(D[1,1]))
y3 = -1*np.sqrt((1-D[0,0]*np.power(y1,2))/(D[1,1]))
y = np.hstack((np.vstack((y1,y2)),np.vstack((y1,y3))))

#Plotting standard hyperbola
#plt.plot(y[0,:len],y[1,:len],color='b',label='Std hyperbola')
#plt.plot(y[0,len+1:],y[1,len+1:],color='b')

#Affine Transformation
#Equation : y = W.T@(x-c)/(K**0.5)
x = (W @ (y)) * F**0.5

#Plotting required hyperbola
plt.plot(x[0,:len],x[1,:len],color='r',label='Hyperbola')
plt.plot(x[0,len+1:],x[1,len+1:],color='r')

def line_gen(A,B):
  len =10
  x_AB = np.zeros((2,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
  
E = np.array ([10,5/np.sqrt(3)])
F = np.array ([5,5*np.sqrt(3)])
x_EF = line_gen(E,F)

plt.plot(x_EF[0,:],x_EF[1,:],label='$NORMAL$')

print('[48√3  72]x=600√3')

plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
