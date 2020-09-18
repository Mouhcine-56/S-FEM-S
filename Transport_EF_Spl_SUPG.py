
# needed imports
from numpy import  zeros, ones, linspace, zeros_like, asarray
import numpy as np
from matplotlib import pyplot as plt
from Functions1    import elements_spans  # computes the span for each element
from Functions1    import make_knots      # create a knot sequence from a grid
from Functions1   import quadrature_grid # create a quadrature rule over the whole 1d grid
from Functions1    import basis_ders_on_quad_grid # evaluates all bsplines and their derivatives on the quad grid
from Gauss_Legendre import gauss_legendre
from Functions1   import plot_field_1d # plot a solution for 1d problems
from Functions1    import normH1
from Functions1    import ErrQuad
from Functions1    import newgrid
# =======================Data===================================
#ne =100# number of elements
ne=100
#T=0.045
#T=0.1
T=0.035
c=1    
p=2
ht=0.01
Tmax=1.5
# ... assembling the stiffness matrix using stencil forms
# ===========================matrix M===============================
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

                    v += (bi_0 * bj_0) * wvol

                matrix[i1, j1]  += v
    # ...

    return matrix    

# ===========================matrix N===============================
    
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
# ===========================matrix R=============================== 
def assemble_stiffnessR(nelements, degree, spans, basis, weights, points, matrix):

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

                    v += (bi_x * bj_0) * wvol

                matrix[i1, j1]  += T*c* v
    # ...

    return matrix
# ===========================matrix S===============================  
def assemble_stiffnessS(nelements, degree, spans, basis, weights, points, matrix):

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

                matrix[i1, j1]  +=T*c*c*v
    # ...

    return matrix     
# ======================Exact Solution====================================

def U0(x):
    if x>=-1 and x<=1 :
        return 5
    else:
        return 0 
def Uexact(t,x,c) :
        return(U0(x-c*t)) 
    
def plot_exact(knots,degree,t,c,nx=101,color='r'):
    xmin = knots[degree]
    xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)
    u=[Uexact(t,e,c) for e in xs]
    plt.plot(xs, u,label='U Exacte')
    plt.legend()
    
# ===========================non-uniform subdivision===============================
#---------------------------------------------
#grid =np.zeros([ne+1,1])
#for i in range(ne+1):
 #   grid[i]=-5+10*np.power((i/(ne)),(1/4))
#grid=np.squeeze(grid)    
#X =np.zeros([ne+p,1])
#for i in range(ne+p):
   # X[i]=-5+10*np.power((i/(ne+p-1)),(1/4))
#X=np.squeeze(X)
# ===========================uniform subdivision===============================    
#----------------------------------------------
grid  = linspace(-5., 5., ne+1)
X  = linspace(-5., 5., ne+p)
#-----------------------------------------------
# ==========================================================

f=[U0(e) for e in X]  
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
stiffnessN = zeros((nbasis, nbasis))
stiffnessR = zeros((nbasis, nbasis))
stiffnessS = zeros((nbasis, nbasis))
stiffnessM = assemble_stiffnessM(nelements, p, spans, basis, weights, points, matrix=stiffnessM)
stiffnessN = assemble_stiffnessN(nelements, p, spans, basis, weights, points, matrix=stiffnessN)
stiffnessR = assemble_stiffnessR(nelements, p, spans, basis, weights, points, matrix=stiffnessR)
stiffnessS = assemble_stiffnessS(nelements, p, spans, basis, weights, points, matrix=stiffnessS)
#------------------------------------------------------------------------------------------
z=0
teta=0.6
temps=[]
Err=[]
while z<=Tmax:
    G=stiffnessM+stiffnessR+np.dot(ht*teta,stiffnessS+stiffnessN)
    # apply homogeneous dirichlet boundary conditions
    G[0,:]=0
    G[0,0]=1
    S=np.dot(stiffnessM+stiffnessR+ht*(teta-1)*(stiffnessS+stiffnessN),f[:])
    S[0]=0
    from scipy.sparse.linalg import gmres
    unew, info = gmres( G, S, tol=1e-6, maxiter=5000 )
    unew = list(unew) 
    unew = asarray(unew)
    o=[Uexact(z,e,c) for e in X]
    Err1=ErrQuad (unew,o)
    temps.append(z)
    Err.append(Err1)
    #plot_field_1d(knots, p, unew, nx=1000)
    #plot_exact(knots,p,z,c,nx=200,color='r')
    #plt.ylim([-1,6])
    #plt.pause(0.005)
    f=unew
    z=z+ht
# ===========================Plot Error=============================== 
Max=max(Err)
plt.plot(temps,Err,label='MaxErr=%1.4f'%Max)
#plt.plot(temps,Err,label='Error')   
#plt.title('Teta=%1.2f'%teta )
plt.legend()
plt.grid()    
#print(Max)


