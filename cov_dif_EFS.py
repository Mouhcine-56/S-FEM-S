# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:36:57 2020

@author: MOUHCINE
"""
# needed imports
from numpy import  zeros, ones, linspace, zeros_like, asarray
import numpy as np
from math import cos,pi,exp
from matplotlib import pyplot as plt
from Functions1    import elements_spans  # computes the span for each element
from Functions1    import make_knots      # create a knot sequence from a grid
from Functions1   import quadrature_grid # create a quadrature rule over the whole 1d grid
from Functions1    import basis_ders_on_quad_grid # evaluates all bsplines and their derivatives on the quad grid
from Gauss_Legendre import gauss_legendre
from Functions1   import plot_field_1d # plot a solution for 1d problems
# ... assembling the stiffness matrix using stencil forms
def assemble_stiffnessM(nelements, degree, spans, basis, weights, points, matrix):

    # ... sizes
    ne1       = nelements
    p1        = degree
    spans_1   = spans
    basis_1   = basis
    weights_1 = weights
    points_1  = points
    
    k1 = weights.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                j1 = i_span_1 - p1 + jl_1

                v = 0.0
                for g1 in range(0, k1):
                    bi_0 = basis_1[ie1, il_1, 0, g1]
                    bi_x = basis_1[ie1, il_1, 1, g1]                    

                    bj_0 = basis_1[ie1, jl_1, 0, g1]
                    bj_x = basis_1[ie1, jl_1, 1, g1]                    

                    wvol = weights_1[ie1, g1]

                    v += (bi_x * bj_x) * wvol

                matrix[i1, j1]  += v
    # ...

    return matrix 
def assemble_stiffnessN(nelements, degree, spans, basis, weights, points, matrix):

    # ... sizes
    ne1       = nelements
    p1        = degree
    spans_1   = spans
    basis_1   = basis
    weights_1 = weights
    points_1  = points
    
    k1 = weights.shape[1]

    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                j1 = i_span_1 - p1 + jl_1

                v = 0.0
                for g1 in range(0, k1):
                    bi_0 = basis_1[ie1, il_1, 0, g1]
                    bi_x = basis_1[ie1, il_1, 1, g1]                    

                    bj_0 = basis_1[ie1, jl_1, 0, g1]
                    bj_x = basis_1[ie1, jl_1, 1, g1]                    

                    wvol = weights_1[ie1, g1]

                    v += (bi_0 * bj_x) * wvol

                matrix[i1, j1]  += v
    # ...

    return matrix      
# ...
    # ... Assembly procedure for the rhs


p=1 
ep=0.005
c=1
ne = 10  # number of elements
r=1/(exp(1/ep)-1)
exact = lambda x: -r+r*exp(x/ep)
x=np.linspace(0,1,200)
f=[exact(e) for e in x]
grid  = linspace(0., 1., ne+1)
knots = make_knots(grid, p, periodic=False)
spans = elements_spans(knots, p)    
nelements = len(grid) - 1
nbasis    = len(knots) - p - 1

# we need the value a B-Spline and its first derivative
nderiv = 1

# create the gauss-legendre rule, on [-1, 1]
u, w = gauss_legendre( p )

# for each element on the grid, we create a local quadrature grid
points, weights = quadrature_grid( grid, u, w )

# for each element and a quadrature points, 
# we compute the non-vanishing B-Splines
basis = basis_ders_on_quad_grid( knots, p, points, nderiv )
stiffnessM = zeros((nbasis, nbasis))
stiffnessM = assemble_stiffnessM(nelements, p, spans, basis, weights, points, matrix=stiffnessM)
stiffnessN = zeros((nbasis, nbasis))
stiffnessN = assemble_stiffnessN(nelements, p, spans, basis, weights, points, matrix=stiffnessN)
exact = lambda x: 1/(16*pi*pi)*cos(4*pi*x)-1/(16*pi*pi)
#exact = lambda x: -exp(x)+(exp(1)-1)*x+1  
rhs = zeros(nbasis)
#rhs = assemble_rhs(f, nelements, p, spans, basis, weights, points, rhs=rhs)
# apply homogeneous dirichlet boundary conditions
rhs = rhs[1:-1]
stiffnessM = stiffnessM[1:-1, 1:-1]
stiffnessN = stiffnessN[1:-1, 1:-1]
from scipy.sparse.linalg import cg
u, info = cg( ep*stiffnessM+c*stiffnessN, rhs, tol=1e-6, maxiter=5000 )
u = [0.] + list(u) + [1.]
u = asarray(u)
plot_field_1d(knots, p, u, nx=10)
plt.plot(x,f,label=' Exacte')
plt.legend()

