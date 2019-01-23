
def fem1d_bar3 ( e_num, n, npe, quad_num, elemNodes, ebcdof,ebcval,nbcdof,nbcval, a, c, f, x ):
# load necessary libraries
    import numpy as np
    import math
    import scipy.linalg as la
#
#  Define a 2- or 3- point Gauss-Legendre quadrature rule on [-1,+1].
#
    ndf=1
    if quad_num == 2:
        gp = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)])
        weight = np.array([1.0, 1.0])
    elif quad_num == 3:
        gp = np.array( [-math.sqrt(3)/math.sqrt(5),0,math.sqrt(3)/math.sqrt(5)])
        weight = np.array ( [ 5/9, 8/9 , 5/9 ] )
#
#  Make room for the matrix stiff and right hand side force
#
    stiff = np.zeros ( [ n, n ] )
    mass  = np.zeros ( [ n, n ] )
    force = np.zeros ( n )
#
#  We assemble the finite element matrix by looking at each element,
    def shape(npe,xi,h):
        if npe == 2:
            sf= np.array([ (1-xi)/2,(1+xi)/2 ])
            dsf=np.array([-0.5,0.5])
        elif npe == 3:
            sf = np.array([-xi*(1 - xi) / 2, 1-xi*xi, xi*(1 + xi) / 2])
            dsf = np.array([-(1.-2.*xi)/2.,-2.*xi , (1.+2.*xi)/2.])
        J = h / 2
        gdsf=dsf/J
        return sf,gdsf

    for e in range ( 0, e_num ):
        A = np.zeros([npe*ndf, npe*ndf])
        M = np.zeros([npe*ndf, npe*ndf])
        b = np.zeros(npe*ndf)
        exy = np.zeros(npe)
        indice = elemNodes[e, :] # element connectivity
        if npe==2:
            elemDof = np.array([indice[0],indice[1]])
        elif npe==3:
            elemDof = np.array([indice[0], indice[1],indice[2]])
        for j in range (0,npe):
            exy[j]=x[(npe-1)*e+j]
        h=exy[npe-1]-exy[0]
        for q in range ( 0, quad_num ): # gaussian quadrature begins here
            sf,gdsf = shape(npe,gp[q],h)
            xq=0
            for i in range (0,npe):
                xq = xq + exy[i]*sf[i]
            wq = weight[q] * h/2
            axq = a ( xq )
            cxq = c ( xq )
#           rxq = r ( xq )
            fxq = f ( xq )
            for i in range (0,npe):
                ii=i
                b[ii]=b[ii]+wq*fxq*sf[i]
                for j in range (0,npe):
                    jj=j
                    A[ii,jj]=A[ii,jj] + wq * (axq*gdsf[i]*gdsf[j]+cxq*sf[i]*sf[j])
                    M[ii,jj]=M[ii,jj] + wq * (cxq*sf[i]*sf[j])
   #    print('A',A)
   #    print('b',b)
        stiff[np.ix_(elemDof,elemDof)] += A # assembly performed
        force[np.ix_(elemDof)] += b
# Apply boundary condition force and displacement
    n_nbc = nbcdof.shape[0]
    for i in range(n_nbc):
        c = nbcdof[i]
        force[c] = force[c] + nbcval[i]
    n_ebc = ebcdof.shape[0]
    for i in range(n_ebc):
        c = ebcdof[i]
        stiff[c,:] = 0
        stiff[c,c] = 1
        force[c] = ebcval[i]
#
#  Solve the linear system for the finite element coefficients U.
#
    u = la.solve ( stiff, force )
    react = np.dot(stiff, u)
#  evals,evecs=la.eigh(A,M)
    return u,react

# if ( __name__ == '__main__' ):
    # from timestamp import timestamp
    # timestamp ( )

