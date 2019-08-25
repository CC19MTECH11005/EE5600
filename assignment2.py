from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#if using termux
import subprocess
import shlex
#end if

#creating x,y for 3D plotting
len = 10
xx, yy = np.meshgrid(range(len), range(len))

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='equal')

#defining line : x(k) = A + k*l
O = np.array([0,0,0]).reshape((3,1))
l = np.array([1,1,0]).reshape((3,1))
A = np.array([1,1,0]).reshape((3,1))
B = np.array([0,3,4]).reshape((3,1))
B2 = np.array([-3/2,3/2,4]).reshape((3,1))

#generating points in Line 
#l1_p = line_dir_pt(l,O)
x_OA = line_gen(O,A)
x_BB2 = line_gen(B,B2)

#plotting line
#plt.plot(l1_p[0,:],l1_p[1,:],l1_p[2,:],label="Line $L_1$")
plt.plot(x_OA[0,:],x_OA[1,:],x_OA[2,:],label='$line1$')
plt.plot(x_BB2[0,:],x_BB2[1,:],x_BB2[2,:],label='$line2$')

ax.scatter(O[0],O[1],O[2],'o',label="Point O")
ax.scatter(A[0],A[1],A[2],'o',label="Point A")
ax.scatter(B[0],B[1],B[2],'o',label="Point B")
ax.scatter(B2[0],B2[1],B2[2],'o',label="Point B2")

plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.axis('equal')

plt.show()
