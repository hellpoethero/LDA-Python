#! /usr/bin/env python
#
# from timestamp import timestamp
#
# timestamp()

def fem1d_ex451 ( ):

#*****************************************************************************80
#
## FEM1D_BVP_LINEAR_TEST00 tests FEM1D_BVP_LINEAR.
#
#  Discussion:
#
#    - (au')' + cu = f  for 0 < x < 1
#       u(0) = u(1) = 0
#       a=1 ; c=-1 ; f=-x^2
#    exact  = (sin(x) +2 sin(1-x)) / sin(1) +x^2 - 2
#
  import matplotlib.pyplot as plt
  import numpy as np
  import platform
  from fem1d_bar3 import fem1d_bar3
#
# input parameters
#
  e_num=2
  npe = 2 # number of nodes in each element (2: linear ; 3:quadratic)
  quad_num = 3  # number of Gauss points
#
#
  x_lo = 0.0
  x_hi = 2.0
#
# Input boundary conditions
  ebcdof = np.array([2]) # PV at node 2 is specified
  ebcval = np.zeros((ebcdof.shape[0], 1)) # as 0
  nbcdof = np.array([0]) # SV at node 0 is specified
  nbcval = np.array([5.]) # as 5
  ndf=1
  nn = e_num*(npe-1)+1   # total number of nodes in finite elements
  n=ndf*nn  # total number of degrees of freedom
# automatic mesh generarion
  x = np.linspace(x_lo, x_hi, nn)

  a1=np.reshape(np.arange(0,nn-1,npe-1),(e_num,1))
  a11=np.tile(a1,(1,npe))
  if npe==2:
    b11=np.tile(np.array([0,1]),(e_num,1))
  elif npe==3:
    b11=np.tile(np.array([0,1,2]),(e_num,1))
  elemNodes=a11+b11
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
#  Geometry definitions.
  u,react = fem1d_bar3 (e_num, n, npe, quad_num, elemNodes, \
                  ebcdof,ebcval,nbcdof,nbcval, a91, c91, f91, x)

  g = np.zeros ( n )
  for i in range ( 0, n ):
    g[i] = exact91 ( x[i] )
#
#  Print a table.
#
  print ( '' )
  print ( '     I    X         U          Uexact       reaction      Error' )
  print ( '' )

  for i in range ( 0, n ):
    print ( '  %4d  %8.5f  %8.5e  %8.5e  %8.5e  %8e' \
      % ( i, x[i], u[i], g[i], react[i], abs ( u[i] - g[i] ) ) )
#
  print ( '' )
#
#  Plot the computed solution.
#
  fig = plt.figure ( )
  plt.plot ( x, u, 'bo-' )
  plt.xlabel ( '<---X--->' )
  plt.ylabel ( '<---U(X)--->' )
  plt.title ( 'FEM1D_ex34' )
#  plt.savefig ( 'fem1d_ex34.png' )
  plt.show ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'FEM1D_ex451' )
  print ( '  Normal end of execution.' )
  return

def a91 ( x ):
  value = 28.e6*(1.+x)/4.
  return value

def c91 ( x ):
  value = 0.
  return value

def exact91 ( x ):
  from math import log
  value = (56.25-6.25*(1+x)**2-7.5*log((1+x)/3) )/28.e6
  return value

def f91 ( x ):
  value = 6.25*(1.+x)
  return value

if ( __name__ == '__main__' ):
  # from timestamp import timestamp
  # timestamp ( )
  fem1d_ex451 ( )
  # timestamp ( )
# timestamp()

