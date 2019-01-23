#! /usr/bin/env python
#
from timestamp import timestamp
#
timestamp()
def fem1d_ex321 ( ):
#*****************************************************************************80
#
#    - (au')' + cu = f  for 0 < x < 1
#       u(0) = u(1) = 0
#       a=1 ; c=-1 ; f=-x^2
#    exact  = (sin(x) +2 sin(1-x)) / sin(1) +x^2 - 2
  import matplotlib.pyplot as plt
  import numpy as np
  import platform
  from fem1d_bar3 import fem1d_bar3
  print('test3')
#
# input parameters
#
  e_num = 4 # four elements
  npe = 2 # number of nodes in each element (2: linear ; 3:quadratic)
  quad_num = 3  # number of Gauss points
#
#  Geometry definitions.
#
  x_lo = 0.0
  x_hi = 1.0
#
# Input boundary conditions
  ebcdof = np.array([0, 4])  # both edges specified
  ebcval = np.zeros((ebcdof.shape[0], 1)) # specified as 0
  nbcdof = np.array([])
  nbcval = np.array([1])
#
  ndf=1 # no of dof is 1
  nn = e_num*(npe-1)+1   # total number of nodes in finite elements
  n=ndf*nn  # total number of degrees of freedom
# automatic mesh generarion
  a1=np.reshape(np.arange(0,nn-1,npe-1),(e_num,1))
  a11=np.tile(a1,(1,npe))
  if npe==2:
    b11=np.tile(np.array([0,1]),(e_num,1))
  elif npe==3:
    b11=np.tile(np.array([0,1,2]),(e_num,1))
  elemNodes=a11+b11
  print('elemnodes',elemNodes) # nodal connectivity for each element
#  print('c11',elemNodes)

#
  print ( '' )
  print ( 'FEM1D_example 3.2.1' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Solve -( A(x) U\'(x) )\' + C(x) U(x) = F(x)' )
  print ( '  for 0 < x < 1, with U(0) = U(1) = 0.' )
  print ( '  A(X)  = 1.0' )
  print ( '  C(X)  =-1.0' )
  print ( '  F(X)  = X^2' )
  print ( '  U(X)  = (sin(x) +2 sin(1-x)) / sin(1) +x^2 - 2' )
  print ( '' )
  print ( '  Number of d.o.fs   = %d' % ( n ) )
  print ( '  Number of nodes    = %d' % ( nn ) )
  print ( '  Number of elements = %d' % ( e_num ) )
#
  x = np.linspace ( x_lo, x_hi, nn ) # mesh generation

#  u = fem1d_bvp_linear ( n, a91, c91, f91, x )
  u,react = fem1d_bar3 (e_num, n, npe, quad_num, elemNodes, \
                  ebcdof,ebcval,nbcdof,nbcval, a91, c91, f91, x)

  g = np.zeros ( nn )
  nne=40
  xe= np.linspace (x_lo, x_hi, nne)
  ge= np.zeros (nne)
  for i in range (0,nne):
    ge[i]=exact91(xe[i])
  for i in range ( 0, nn ):
    g[i] = exact91 ( x[i] )
#
#  Print a table.
#
  print ( '' )
  print ( '     I    X         U         Uexact   reaction   Error' )
  print ( '' )

  for i in range ( 0, n ):
    print ( '  %4d  %8.5f  %8.5f  %8.5f  %8.5f  %8e' \
      % ( i, x[i], u[i], g[i], react[i], abs ( u[i] - g[i] ) ) )
#
  print ( '' )
#
#  Plot the computed solution.
#
  fig = plt.figure ( )
  plt.plot ( x, u )
  plt.plot ( xe, ge )
  plt.xlabel ( '<---X--->' )
  plt.ylabel ( '<---U(X)--->' )
  plt.title ( 'FEM1D_ex321' )
  plt.legend(['FEM', 'Exact'], loc='lower right')
#  plt.savefig ( 'fem1d_ex321.png' )
  plt.show ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'FEM1D_ex321' )
  print ( '  Normal end of execution.' )
  return

def a91 ( x ):
  value = 1.0
  return value

def c91 ( x ):
  value = -1.0
  return value

def exact91 ( x ):
  from math import sin
  value = ( sin(x) + 2*sin ( 1.0-x )) / sin ( 1.0 ) +x**2 - 2
  return value

def f91 ( x ):
  value = -x**2
  return value

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  fem1d_ex321 ( )
  timestamp ( )
timestamp()

