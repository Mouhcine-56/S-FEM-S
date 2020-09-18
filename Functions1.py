
from numpy import empty
import numpy as np
from matplotlib import pyplot as plt
from math import cos,pi,exp
# ==========================================================
def find_span( knots, degree, x ):
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low+high)//2
        returnVal = span

    return returnVal

# ==========================================================
def all_bsplines( knots, degree, x, span ):
    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )
    values = empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values

# ==========================================================
def point_on_bspline_curve(knots, P, x):
    degree = len(knots) - len(P) - 1
    d = P.shape[-1]

    span = find_span( knots, degree, x )
    b    = all_bsplines( knots, degree, x, span )

    c = np.zeros(d)
    for k in range(0, degree+1):
        c[:] += b[k]*P[span-degree+k,:]
    return c

# ==========================================================
def normH1(u,v,h):
    n=np.size(u)
    err=np.zeros((n,1))
    for j in range(n):
        err[j]=u[j]-v[j]
    V=np.squeeze(err)      
    K=[]
    M=np.zeros((n,1))
    for i in range(n):
        if i <= n-3:
            l=(V[i+2]-V[i])/(2*h)
            K.append(l)
    m0=(V[1]-V[0])/h
    m1=(V[n-1]-V[n-2])/h     
    K.insert(0,m0)            
    K.append(m1) 
    for i in range(n):
        M[i]=V[i]*V[i]+K[i]*K[i]
        
    I=h/3*(M[0]+2*sum(M[2:n-2:2])+4*sum(M[1:n:2])+M[n-1] )
    U=np.sqrt(I)    
    return(U)
# ==========================================================  
def ErrQuad (u,v):
    n=np.size(u)
    i=0
    b=0
    while i<n:
        a=np.power(u[i]-v[i],2)
        b=b+a
        i=i+1
    E=np.sqrt(b/n)
    return E
# ==========================================================  
def Cheby(a,b,N):
    C=np.zeros((N,1))
    for i in range(N):
        C[i]=-0.5*(b-a)*np.cos((i+20)*pi/(N))
    return(C)
# ==========================================================
def newgrid(grid,alpha):
    n=np.size(grid)
    k=np.size(alpha)
    a=grid[0]
    b=grid[n-1]
    newgrid=[]
    for i in range(n-1):
        for j in range(k):
            N=0.5*(grid[i+1]+grid[i]+alpha[j]*(grid[i+1]-grid[i]))
            newgrid.append(N)
    newgrid.insert(0,a)
    newgrid=newgrid[0:-1]
    newgrid.append(b)
    return newgrid 
# ==========================================================  
def plot_field_1d(knots, degree, u, nx, color='b'):
    n = len(knots) - degree - 1

    xmin = knots[degree]
    xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(u), 1))
    P[:,0] = u[:]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    plt.plot(xs, Q[:,0],label='U approchée')
    plt.legend()
    plt.grid()
    
 #==============================================================================
def breakpoints( knots, degree ):
    
    return np.unique( knots[degree:-degree] )

#============================================================================== 
    
def elements_spans( knots, degree ):
   
    breaks = breakpoints( knots, degree )
    nk     = len(knots)
    ne     = len(breaks)-1
    spans  = np.zeros( ne, dtype=int )

    ie = 0
    for ik in range( degree, nk-degree ):
        if knots[ik] != knots[ik+1]:
            spans[ie] = ik
            ie += 1
        if ie == ne:
            break

    return spans

#===============================================================================
def make_knots( breaks, degree, periodic ):
    # Type checking
    assert isinstance( degree  , int  )
    assert isinstance( periodic, bool )

    # Consistency checks
    assert len(breaks) > 1
    assert all( np.diff(breaks) > 0 )
    assert degree > 0
    if periodic:
        assert len(breaks) > degree

    p = degree
    T = np.zeros( len(breaks)+2*p )
    T[p:-p] = breaks

    if periodic:
        period = breaks[-1]-breaks[0]
        T[0:p] = [xi-period for xi in breaks[-p-1:-1 ]]
        T[-p:] = [xi+period for xi in breaks[   1:p+1]]
    else:
        T[0:p] = breaks[ 0]
        T[-p:] = breaks[-1]

    return T

#==============================================================================
def quadrature_grid( breaks, quad_rule_x, quad_rule_w ):
    
    # Check that input arrays have correct size
    assert len(breaks)      >= 2
    assert len(quad_rule_x) == len(quad_rule_w)

    # Check that provided quadrature rule is defined on interval [-1,1]
    assert min(quad_rule_x) >= -1
    assert max(quad_rule_x) <= +1

    quad_rule_x = np.asarray( quad_rule_x )
    quad_rule_w = np.asarray( quad_rule_w )

    ne     = len(breaks)-1
    nq     = len(quad_rule_x)
    quad_x = np.zeros( (ne,nq) )
    quad_w = np.zeros( (ne,nq) )

    # Compute location and weight of quadrature points from basic rule
    for ie,(a,b) in enumerate(zip(breaks[:-1],breaks[1:])):
        c0 = 0.5*(a+b)
        c1 = 0.5*(b-a)
        quad_x[ie,:] = c1*quad_rule_x[:] + c0
        quad_w[ie,:] = c1*quad_rule_w[:]

    return quad_x, quad_w

#==============================================================================
def basis_ders_on_quad_grid( knots, degree, quad_grid, nders, normalize=False ):
    # TODO: add example to docstring
    # TODO: check if it is safe to compute span only once for each element

    ne,nq = quad_grid.shape
    basis = np.zeros( (ne,degree+1,nders+1,nq) )

    for ie in range(ne):
        xx = quad_grid[ie,:]
        for iq,xq in enumerate(xx):
            span = find_span( knots, degree, xq )
            ders = basis_funs_all_ders( knots, degree, xq, span, nders )
            basis[ie,:,:,iq] = ders.transpose()

    if normalize:
        x = scaling_matrix(degree, ne+degree, knots)
        basis *= x[0]

    return basis

#==============================================================================
def scaling_matrix(p, n, T):

    x = np.zeros(n)
    for i in range(0, n):
        x[i] = (p+1)/(T[i+p+1]-T[i])
    return x
#==============================================================================
def basis_funs( knots, degree, x, span ):
    left   = np.empty( degree  , dtype=float )
    right  = np.empty( degree  , dtype=float )
    values = np.empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values


#==============================================================================
def basis_funs_all_ders( knots, degree, x, span, n ):
    left  = np.empty( degree )
    right = np.empty( degree )
    ndu   = np.empty( (degree+1, degree+1) )
    a     = np.empty( (       2, degree+1) )
    ders  = np.zeros( (     n+1, degree+1) ) # output array

    # Number of derivatives that need to be effectively computed
    # Derivatives higher than degree are = 0.
    ne = min( n, degree )

    # Compute nonzero basis functions and knot differences for splines
    # up to degree, which are needed to compute derivatives.
    # Store values in 2D temporary array 'ndu' (square matrix).
    ndu[0,0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
            # compute basis functions and save them into upper triangular part of ndu
            temp       = ndu[r,j] * ndu[j+1,r]
            ndu[r,j+1] = saved + right[r] * temp
            saved      = left[j-r] * temp
        ndu[j+1,j+1] = saved

    # Compute derivatives in 2D output array 'ders'
    ders[0,:] = ndu[:,degree]
    for r in range(0,degree+1):
        s1 = 0
        s2 = 1
        a[0,0] = 1.0
        for k in range(1,ne+1):
            d  = 0.0
            rk = r-k
            pk = degree-k
            if r >= k:
               a[s2,0] = a[s1,0] * ndu[pk+1,rk]
               d = a[s2,0] * ndu[rk,pk]
            j1 = 1   if (rk  > -1 ) else -rk
            j2 = k-1 if (r-1 <= pk) else degree-r
            a[s2,j1:j2+1] = (a[s1,j1:j2+1] - a[s1,j1-1:j2]) * ndu[pk+1,rk+j1:rk+j2+1]
            d += np.dot( a[s2,j1:j2+1], ndu[rk+j1:rk+j2+1,pk] )
            if r <= pk:
               a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
               d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j  = s1
            s1 = s2
            s2 = j

    # Multiply derivatives by correct factors
    r = degree
    for k in range(1,ne+1):
        ders[k,:] = ders[k,:] * r
        r = r * (degree-k)

    return ders

#==============================================================================
    
    