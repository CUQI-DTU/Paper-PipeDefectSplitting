################################################
# By Silja L. Christensen
# August 2022 - May 2023

# This script contains additional functions used by main script.
# Several functions are originally from: https://github.com/CUQI-DTU/Paper-SGP

# Requires installation of CUQIpy: This code runs on a pre-released version of CUQIpy. Install via command:
# pip install git+https://github.com/CUQI-DTU/CUQIpy.git@sprint16_add_JointModel
# Requires installation of astra: https://github.com/astra-toolbox/astra-toolbox
################################################

import numpy as np
import scipy as sp
import sys

import astra

sys.path.append('../CUQIpy/')
import cuqi

#%%
#=======================================================================
# Extra CUQI functionality
#=========================================================================

class myJointGaussianSqrtPrec(cuqi.distribution.JointGaussianSqrtPrec):
    def _sample(self,N):                
        samples = np.empty((np.shape(self.sqrtprec)[1], N+1))
        # initial state   
        samples[:, 0] = np.zeros(np.shape(self.sqrtprec)[1])
        for s in range(N):
            sim = cuqi.solver.CGLS(self.sqrtprec, self.sqrtprecTimesMean + np.random.randn(np.shape(self.sqrtprec)[0]), x0 = np.zeros(np.shape(self.sqrtprec)[1]), maxit = 50, tol=1e-6, shift=0)
            samples[:, s+1], _ = sim.solve()
        
        return samples[:,1:]

    def logpdf(self, x):
        return 0

class myIGConjugate:
    def __init__(self, target: cuqi.distribution.JointDistribution, s_minval = 1e-7):
        self.target = target
        self.s_minval = s_minval

    def step(self, x=None):
        # Extract variables
        b = self.target.get_density("d").data   #get_density    #d
        alpha = self.target.get_density("s").shape                                 #alpha
        beta = self.target.get_density("s").scale                                  #beta

        # Conjugate Inverse Gamma distribution and sample it
        samp = cuqi.distribution.InverseGamma(shape=1/2+alpha,location=0,scale=.5*b**2+beta).sample()
        return np.maximum(self.s_minval,samp)

class myGammaSampler:
    def __init__(self, target: cuqi.distribution.Posterior):
        self.target = target

    def step(self, x=None):
        # Extract variables
        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                  #beta
        
        # Conjugate Inverse Gamma distribution and sample it
        one_sample = np.random.gamma(shape=alpha,scale=1/beta)
        return one_sample


#%%
#=======================================================================
# Astra model
#=========================================================================

class ASTRAModel(cuqi.model.LinearModel):
    def __init__(self, proj_type, proj_geom, vol_geom):

        # Define image (domain) geometry
        domain_geometry = cuqi.geometry.Image2D((vol_geom["GridRowCount"], vol_geom["GridColCount"]), order = "F")

        # Define sinogram (range) geometry
        num_angles = proj_geom["Vectors"].shape[0] if "Vectors" in proj_geom else proj_geom["ProjectionAngles"].shape[0]
        range_geometry = cuqi.geometry.Image2D((num_angles, proj_geom["DetectorCount"]), order = "F")
        
        # Define linear model
        super().__init__(self._forward_func, self._adjoint_func, range_geometry=range_geometry, domain_geometry=domain_geometry)

        # Create ASTRA projector
        self._proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)

        # Store other ASTRA related variables privately
        self._proj_geom = proj_geom
        self._vol_geom = vol_geom

    @property
    def proj_geom(self):
        """ ASTRA projection geometry. """
        return self._proj_geom

    @property
    def vol_geom(self):
        """ ASTRA volume geometry. """
        return self._vol_geom

    @property
    def proj_id(self):
        """ ASTRA projector ID. """
        return self._proj_id

    # CT forward projection
    def _forward_func(self, x: np.ndarray) -> np.ndarray:
        id, sinogram =  astra.create_sino(x, self.proj_id)
        astra.data2d.delete(id)
        return sinogram

    # CT back projection
    def _adjoint_func(self, y: np.ndarray) -> np.ndarray:
        id, volume = astra.create_backprojection(y, self.proj_id)
        astra.data2d.delete(id)
        return volume

class myShiftedFanBeam2DModel(ASTRAModel):
    """ 2D CT model with fanbeam and source+detector shift, assuming object is at position (0,0).

    Parameters
    ------------    
    im_size : tuple of ints
        Dimensions of image in pixels.
    
    det_count : int
        Number of detector elements.
       
    angles : ndarray
        Angles of projections, in radians.

    source_y : scalar
        Source position on y-axis.

    detector_y : scalar
        Detector position on y-axis.

    beamshift_x : scalar
        Source and detector position on x-axis.

    det_length : scalar
        Detector length.

    domain : tuple
        Size of image domain, default (550,550).

    proj_type : string
        String indication projection type.
        Can be "line_fanflat", "strip_fanflat", "cuda" etc.
    
    beam_type : string
        String indication beam type.
        Must be of _vec type.
        For example "fanflat_vec".
    """

    def __init__(
        self,
        im_size=(45,45),
        det_count=50,
        angles=np.linspace(0, 2*np.pi, 60),
        source_y=-600,
        detector_y=500,
        beamshift_x=-125.3,
        det_length=411,
        domain=(550,550),
        proj_type='line_fanflat',
        beam_type="fanflat_vec"
        ):
        
        # Detector spacing
        det_spacing = det_length/det_count

        #Define scan vectors
        s0 = np.array([beamshift_x, source_y])
        d0 = np.array([beamshift_x , detector_y])
        u0 = np.array([det_spacing, 0])
        vectors = np.empty([np.size(angles), 6])
        for i, val in enumerate(angles):
            R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
            s = R @ s0
            d = R @ d0
            u = R @ u0
            vectors[i, 0:2] = s
            vectors[i, 2:4] = d
            vectors[i, 4:6] = u

        # Astra geometries
        proj_geom = astra.create_proj_geom(beam_type, det_count, vectors)
        vol_geom = astra.create_vol_geom(im_size[0], im_size[1], -domain[0]/2, domain[0]/2, -domain[1]/2, domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)  

#%%
#=======================================================================
# Structural Gaussian Prior
#=========================================================================
def SGP(N, domain, maskradii, maskcenterinner, maskcenterouter, maskid, mu, precWGauss, bndcond, flat_order = "F", rot_k = 0):
    
    D1, D2 = DifferenceMatrix2D(N, bndcond)

    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    
    mask = np.zeros((N,N))
    for i in range(len(maskid)):
        #mask = mask + maskid[i]*drawPipe(N,domain,x,y,maskcenterinner[i,:],maskcenterouter[i,:],maskradii[i,0],maskradii[i,1]) 
        mask[drawPipe(N,domain,x,y,maskcenterinner[i,:],maskcenterouter[i,:],maskradii[i,0],maskradii[i,1])] = maskid[i] 

    mask = np.rot90(mask, k = rot_k).flatten(order=flat_order)

    x_prior = np.zeros(N**2)
    w = np.zeros(N**2)
    for i in range(max(maskid)):
        x_prior[mask == i+1] = mu[i]
        w[mask == i+1] = precWGauss[i]

    Wsq = sp.sparse.diags(np.sqrt(w), 0, format='csc')

    return mask, x_prior, w, Wsq, D1, D2

def DifferenceMatrix2D(N, bndcond):
    I = sp.sparse.identity(N, format='csc')

    # 1D finite difference matrix 
    one_vec = np.ones(N)
    diags = np.vstack([-one_vec, one_vec])
    if (bndcond == 'zero'):
        locs = [-1, 0]
        D = sp.sparse.spdiags(diags, locs, N+1, N).tocsc()
    elif (bndcond == 'periodic'):
        locs = [-1, 0]
        D = sp.sparse.spdiags(diags, locs, N+1, N).tocsc()
        D[-1, 0] = 1
        D[0, -1] = -1
    elif (bndcond == 'neumann'):
        locs = [0, 1]
        D = sp.sparse.spdiags(diags, locs, N, N).tocsc()
        D[-1, -1] = 0
    elif (bndcond == 'centered'):
        locs = [-1, 0, 1]
        diags = np.vstack([-one_vec, 2*one_vec, -one_vec])
        D = sp.sparse.spdiags(diags, locs, N, N).tocsc()
        D[-1, -1] = 1
        D[0, 0] = 1

    # 2D finite differences in each direction
    D1 = sp.sparse.kron(I, D).tocsc()
    D2 = sp.sparse.kron(D, I).tocsc()

    return D1, D2

#%%
#=======================================================================
# Phantom
#=========================================================================

def DeepSeaOilPipe8(N,defects):

    radii  = np.array([9,11,16,17.5,23])

    domain = 55
    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    center = np.array([0,0])
    phantom = 2e-2*7.9*drawPipe(N,domain,x,y,center,center,radii[0],radii[1])      # Steel (8.05g/cm^3)
    phantom = phantom+5.1e-2*0.15*drawPipe(N,domain,x,y,center,center,radii[1],radii[2])      # PE-foam
    phantom = phantom+5.1e-2*0.94*drawPipe(N,domain,x,y,center,center,radii[2],radii[3])     # PU rubber      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    phantom = phantom+4.56e-2*2.3*drawPipe(N,domain,x,y,center,center,radii[3],radii[4])    # Concrete 2.3 g/cm^3

    # radial cracks
    if defects == True:

        defectmask = []
        vertices = []

        # radial and angular cracks
        no = 12
        ang = np.array([-3*np.pi/9, -2*np.pi/9, -np.pi/9, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2, 2*np.pi/3, 5*np.pi/4-np.pi/9, 5*np.pi/4, 5*np.pi/4+np.pi/9])-60/180*np.pi
        dist = np.array([20.25, 20.25, 20.25, 20.25, 20.25, 16.75, 13.5, 10, 20.25, 16.75+2, 16.75, 16.75-2])/domain*N
        w = np.array([0.5, 0.4, 0.3, 0.2, 4, 4, 4, 4, 0.4, 0.4, 0.4, 0.4])/domain*N
        l = np.array([4, 4, 4, 4, 0.4, 0.4, 0.4, 0.4, 4, 4, 4, 4])/domain*N
        vals = np.zeros(no)
        vals[8] = 2e-2*7.9
        for i in range(no):
            # coordinates in (x,y), -1 to 1 system
            coordinates0 = np.array([
                [c[0]+w[i]/2, c[1]+dist[i] + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist[i] + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist[i] - l[i]/2],
                [c[0]+w[i]/2, c[1]+dist[i] - l[i]/2]
            ])
            R = np.array([
                [np.cos(ang[i]), -np.sin(ang[i])],
                [np.sin(ang[i]), np.cos(ang[i])]
                ])
            # Rotate around image center
            coordinates = R @ (coordinates0.T - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices.append(np.ceil(np.fliplr(coordinates)))
            # create mask
            tmpmask = create_polygon([N,N], vertices[i])
            defectmask.append(np.array(tmpmask, dtype=bool))
            phantom[defectmask[i]] = vals[i]

        # Cross
        c_cross_ang = -np.pi/2
        c_cross_dist = 20.25/domain*N
        c_cross = c_cross_dist*np.array([np.cos(c_cross_ang), np.sin(c_cross_ang)])+N/2
        #np.array([c[1]-20.25/domain*N, c[0]])
        a = (2/np.sqrt(2))/domain*N
        b = (0.2/np.sqrt(2))/domain*N
        coordinates_cross1 = np.array([
            [c_cross[0]-a+b, c_cross[1]-a],
            [c_cross[0]+a, c_cross[1]+a-b],
            [c_cross[0]+a-b, c_cross[1]+a],
            [c_cross[0]-a, c_cross[1]-a+b]])
        coordinates_cross2 = np.array([
            [c_cross[0]+a-b, c_cross[1]-a],
            [c_cross[0]+a, c_cross[1]-a+b],
            [c_cross[0]-a+b, c_cross[1]+a],
            [c_cross[0]-a, c_cross[1]+a-b]])
        # transform into (row, column) indicies
        vertices.append(np.ceil(np.flipud(coordinates_cross1)))
        # create mask
        tmpmask = create_polygon([N,N], vertices[12])
        defectmask.append(np.array(tmpmask, dtype=bool))
        phantom[defectmask[12]] = 0
        # transform into (row, column) indicies
        vertices.append(np.ceil(np.flipud(coordinates_cross2)))
        # create mask
        tmpmask = create_polygon([N,N], vertices[13])
        defectmask.append(np.array(tmpmask, dtype=bool))
        phantom[defectmask[13]] = 0

        # Circles
        ang_circ = np.array([3*np.pi/4+np.pi/9, 3*np.pi/4+np.pi/9, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4-np.pi/9])-60/180*np.pi
        dist_circ = 20.25/domain*N
        siz = np.array([1, 0.3, 1, 0.3, 0.3])/domain*N
        val = np.array([0, 2e-2*7.9, 0, 4.56e-2*2.3, 2e-2*7.9])

        for i in range(len(ang_circ)):
            tmpmask = ((x-np.cos(ang_circ[i])*dist_circ)**2 + (y-np.sin(ang_circ[i])*dist_circ)**2 <= siz[i]**2)
            defectmask.append(np.array(tmpmask, dtype=bool))
            phantom[defectmask[14+i]] = val[i]

        center_dists = np.hstack([dist, c_cross_dist, dist_circ*np.ones(3)])
        center_x = center_dists*np.hstack([np.sin(-ang), np.sin(np.array([c_cross_ang])), np.cos(ang_circ[np.array([0,2,4])])])+N/2
        center_y = center_dists*np.hstack([np.cos(-ang), np.cos(np.array([c_cross_ang])), np.sin(ang_circ[np.array([0,2,4])])])+N/2
        centers = np.vstack([center_x, center_y])
        
        
        return phantom, radii, defectmask, vertices, centers
    else:
        return phantom, radii
    
def drawPipe(N, domain, x,y ,c1,c2, r1, r2):
    # N is number of pixels on one axis
    # domain is true size of one axis
    # x and y is a meshgrid of the domain
    # r1 and r2 are the inner and outer radii of the pipe layer
    R1 = r1/domain*N
    R2 = r2/domain*N

    M1 = (x-c1[0]/domain*N)**2+(y-c1[1]/domain*N)**2>=R1**2
    M2 = (x-c2[0]/domain*N)**2+(y-c2[1]/domain*N)**2<=R2**2

    return np.logical_and(M1, M2)

def check(p1, p2, base_array):
    """
    Source: https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array

#%%
#=======================================================================
# Acquisition geometry
#=========================================================================

def geom_Data20180911(size):
    
    offset      = 0             # angular offset
    shift       = -12.5           # source offset from center
    stc         = 60               # source to center distance
    ctd         = 50               # center to detector distance
    det_full    = 512
    startAngle  = 0
    if size == "sparseangles":
        p   = 510               # p: number of detector pixels
        q   = 36                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    if size == "sparseangles20percent":
        p   = 510               # p: number of detector pixels
        q   = 72                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    if size == "sparseangles50percent":
        p   = 510               # p: number of detector pixels
        q   = 180                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    elif size == "full":
        p   = 510               # p: number of detector pixels
        q   = 360               # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    elif size == "overfull":
        p   = 510               # p: number of detector pixels
        q   = 720               # q: number of projection angles
        maxAngle    = 364               # measurement max angle
    elif size == "limited90":
        p   = 510               # p: number of detector pixels
        q   = 90               # q: number of projection angles
        startAngle = 15
        maxAngle = 105
    elif size == "limited120":
        p   = 510               # p: number of detector pixels
        q   = 120               # q: number of projection angles
        maxAngle = 120
    elif size == "limited180":
        p   = 510               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        startAngle = 15
        maxAngle = 195
    elif size == "limited180_2":
        p   = 510               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        startAngle = 180
        maxAngle = 360

    dlA         = 41.1*(p/det_full)              # full detector length
    dl          = dlA/p   # length of detector element

    # view angles in rad
    theta = np.linspace(startAngle, maxAngle, q, endpoint=False) 
    theta = theta/180*np.pi
    
    s0 = np.array([shift, -stc])
    d0 = np.array([shift, ctd])
    u0 = np.array([dl, 0])

    vectors = np.empty([q, 6])
    for i, val in enumerate(theta):
        R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
        s = R @ s0
        d = R @ d0
        u = R @ u0
        vectors[i, 0:2] = s
        vectors[i, 2:4] = d
        vectors[i, 4:6] = u

    return p, theta, stc, ctd, shift, vectors, dl, dlA
