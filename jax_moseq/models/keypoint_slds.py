import jax, jax.numpy as jnp, jax.random as jr
from jax.tree_util import tree_map
import numpy as np
from textwrap import fill
from sklearn.decomposition import PCA
from jax_moseq.utils.kalman import kalman_sample
from jax_moseq.utils.transitions import resample_hdp_transitions, init_hdp_transitions
from jax_moseq.utils.autoregression import init_ar_params, resample_ar_params, resample_discrete_stateseqs, pad_affine
from jax_moseq.utils.distributions import sample_vonmises, sample_scaled_inv_chi2
na = jnp.newaxis


#======================================================================#
#                                Utils                                 #
#======================================================================#

def jax_io(fn): 
    """
    Converts a function involving numpy arrays to one that inputs and
    outputs jax arrays.
    """
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs)))


def estimate_coordinates(*, Y, v, h, x, Cd, **kwargs):
    """
    Estimate keypoint coordinates obtained from projecting the 
    latent state ``x`` into keypoint-space (via ``Cd``) and then
    rotating and translating by ``h`` and `v`` respectively

    Parameters
    ----------
    v: jax array, shape (...,t,d), Centroid locations
    h: jax array, shape (...,t), Heading
    x: jax array, shape (...,t,D), Continuous latent state
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation

    Returns
    -------
    Yest: jax array, shape (...,t,k,d), Estimated coordinates
        
    """    
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    Yest = affine_transform(Ybar,v,h)
    return Yest


def center_embedding(k):
    """
    Generates a matrix ``Gamma`` that maps from a (k-1)-dimensional 
    vector space  to the space of k-tuples with zero mean

    Parameters
    ----------
    k: int, Number of keypoints

    Returns
    -------
    Gamma: jax array, shape (k, k-1)

    """  
    # using numpy.linalg.svd because jax version crashes on windows
    return jnp.array(np.linalg.svd(np.eye(k) - np.ones((k,k))/k)[0][:,:-1])


def vector_to_angle(V):
    """Convert 2D vectors to angles in [-pi, pi]. The vector (1,0)
    corresponds to angle of 0. If V is n-dinmensional, the first
    n-1 dimensions are treated as batch dims.     
    """
    return jnp.arctan2(V[...,1],V[...,0])

    
def angle_to_rotation_matrix(h, d=3):
    """Create rotation matrices from an array of angles. If ``d > 2`` 
    then rotation is performed in the first two dims.

    Parameters
    ----------
    h: jax array, shape (*dims)
        Angles (in radians)

    d: int, default=3
        Dimension of each rotation matrix

    Returns
    -------
    m: ndarray, shape (*dims, d, d)
        Stacked rotation matrices 
        
    """
    m = jnp.tile(jnp.eye(d), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m


@jax.jit
def affine_transform(Y, v, h):
    """
    Apply the following affine transform
    
    .. math::
        Y \mapsto R(h) @ Y + v

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    v: jax array, shape (*dims, d), Translations
    h: jax array, shape (*dims), Heading angles
          
    Returns
    -------
    Ytransformed: jax array, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(h, d=Y.shape[-1])
    Ytransformed = (Y[...,na,:]*rot_matrix[...,na,:,:]).sum(-1) + v[...,na,:]
    return Ytransformed


@jax.jit
def inverse_affine_transform(Y, v, h):
    """
    Apply the following affine transform
    
    .. math::
        Y \mapsto R(-h) @ (Y - v)

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    v: jax array, shape (*dims, d), Translations
    h: jax array, shape (*dims), Heading angles
          
    Returns
    -------
    Ytransformed: jax array, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(-h, d=Y.shape[-1])
    Y_transformed = ((Y-v[...,na,:])[...,na,:]*rot_matrix[...,na,:,:]).sum(-1)
    return Y_transformed


def ar_to_lds(As, bs, Qs, Cs):
    """
    Reformat a linear dynamical system with L'th-order autoregressive 
    (AR) dynamics in R^D as a system with 1st-order dynamics in R^(D*L)
    
    Parameters
    ----------  
    As: jax array, shape (*dims, D, D*L),   AR transformation
    bs: jax array, shape (*dims, D),        AR affine term
    Qs: jax array, shape (*dims, D, D),     AR noise covariance
    Cs: jax array, shape (*dims, D_obs, D), obs transformation
    
    Returns
    -------
    As_: jax array, shape (*dims, D*L, D*L)
    bs_: jax array, shape (*dims, D*L)    
    Qs_: jax array, shape (*dims, D*L, D*L)  
    Cs_: jax array, shape (*dims, D_obs, D*L)

    """    
    D,L = As.shape[-2],As.shape[-1]//As.shape[-2]

    As_ = jnp.zeros((*As.shape[:-2],D*L,D*L))
    As_ = As_.at[...,:-D,D:].set(jnp.eye(D*(L-1)))
    As_ = As_.at[...,-D:,:].set(As)
    
    Qs_ = jnp.zeros((*Qs.shape[:-2],D*L,D*L))
    Qs_ = Qs_.at[...,:-D,:-D].set(jnp.eye(D*(L-1))*1e-2)
    Qs_ = Qs_.at[...,-D:,-D:].set(Qs)
    
    bs_ = jnp.zeros((*bs.shape[:-1],D*L))
    bs_ = bs_.at[...,-D:].set(bs)
    
    Cs_ = jnp.zeros((*Cs.shape[:-1],D*L))
    Cs_ = Cs_.at[...,-D:].set(Cs)
    return As_, bs_, Qs_, Cs_


def align_egocentric(Y, *, anterior_idxs, posterior_idxs, **kwargs):
    """
    Perform egocentric alignment of keypoints by translating the 
    centroid to the origin and rotatating so that the vector pointing
    from the posterior bodyparts toward the anterior bodyparts is 
    proportional to (1,0) 

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d)
        Keypoint coordinates

    anterior_idxs: jax array (int)
        Indexes of anterior bodyparts

    posterior_idxs: jax array (int)
        Indexes of posterior bodyparts

    Returns
    -------
    Y_aligned: jax array, shape (*dims, k, d)
        Aligned keypoint coordinates

    v: jax array, shape (*dims, d)
        Centroid positions that were used for alignment

    h: jax array, shape (*dims)
        Heading angles that were used for alignment

    """
    posterior_loc = Y[..., posterior_idxs,:2].mean(-2) 
    anterior_loc = Y[..., anterior_idxs,:2].mean(-2) 
    h = vector_to_angle(anterior_loc-posterior_loc)
    v = Y.mean(-2).at[...,2:].set(0)
    Y_aligned = inverse_affine_transform(Y,v,h)
    return Y_aligned,v,h


def transform_data_for_pca(Y, *, anterior_idxs, posterior_idxs, **kwargs):
    """
    Prepare keypoint coordinates for PCA by performing egocentric 
    alignment, changing basis using ``center_embedding(k)``, and 
    reshaping to a single flat vector per frame. 

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    anterior_idxs: jax array (int), Indexes of anterior bodyparts
    posterior_idxs: jax array (int), Indexes of posterior bodyparts
          
    Returns
    -------
    Y_flat: jax array, shape (*dims, (k-1)*d)
        
    """
    Y_aligned = align_egocentric(Y, 
        anterior_idxs=anterior_idxs, 
        posterior_idxs=posterior_idxs)[0]

    k,d = Y.shape[-2:]
    Y_embedded = center_embedding(k).T @ Y_aligned
    Y_flat = Y_embedded.reshape(*Y.shape[:-2],(k-1)*d)

    return Y_flat


def fit_pca(*, Y, conf, mask, conf_threshold=0.5, verbose=False,
            PCA_fitting_num_frames=1000000, **kwargs):
    """
    Fit a PCA model to transformed keypoint coordinates. If ``conf`` is
    not None, perform linear interpolation over outliers defined by
    ``conf < conf_threshold``.

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d)
        Keypoint coordinates

    conf: None or jax array, shape (*dims,k)
        Confidence value for each keypoint

    mask: jax array, shape (*dims, k)
        Binary indicator for which elements of ``Y`` are valid

    anterior_idxs: jax array (int)
        Indexes of anterior bodyparts

    posterior_idxs: jax array (int)
        Indexes of posterior bodyparts

    conf_threshold: float, default=0.5
        Confidence threshold below which keypoints will be interpolated

    PCA_fitting_num_frames: int, default=1000000
        Maximum number of frames to use for PCA. Frames will be sampled
        randomly if the input data exceed this size. 

    Returns
    -------
    pca, sklearn.decomposition._pca.PCA
        A fit sklearn PCA model

    """
    if conf is not None: 
        if verbose: print('PCA: Interpolating low-confidence detections')
        Y = interpolate(Y, conf<conf_threshold)
       
    if verbose: print('PCA: Performing egocentric alignment')
    Y_flat = transform_data_for_pca(Y, **kwargs)[mask>0]
    PCA_fitting_num_frames = min(PCA_fitting_num_frames, Y_flat.shape[0])
    Y_sample = np.array(Y_flat)[np.random.choice(
        Y_flat.shape[0],PCA_fitting_num_frames,replace=False)]
    if verbose: print(f'PCA: Fitting PCA model on {Y_sample.shape[0]} sample poses')
    return PCA(n_components=Y_flat.shape[-1]).fit(Y_sample)



#======================================================================#
#                            Initialization                            #
#======================================================================#

@jax_io
def interpolate(keypoints, outliers, axis=1):
    """
    Use linear interpolation to impute the coordinates of outliers

    Parameters
    ----------
    keypoints: jax array, shape (...,t,k,d)
        Keypoint coordinates 

    outliers: jax array, shape (...,t,k)
        Binary indicator array where 1 implies that the corresponding 
        keypoint is an outlier.

    Returns
    -------
    interpolated_keypoints: jax array, shape (...,t,k,d)
        Copy of ``keypoints`` where outliers have been replaced by
        linearly interpolated values.

    """   
    keypoints = np.moveaxis(keypoints, axis, 0)
    init_shape = keypoints.shape
    keypoints = keypoints.reshape(init_shape[0],-1)
    
    outliers = np.moveaxis(outliers, axis, 0)
    outliers = np.repeat(outliers[...,None],init_shape[-1],axis=-1)
    outliers = outliers.reshape(init_shape[0],-1)
    
    interp = lambda x,xp,fp: (
        np.ones_like(x)*x.mean() if len(xp)==0 else np.interp(x,xp,fp))
    
    keypoints = np.stack([interp(
        np.arange(init_shape[0]), 
        np.nonzero(~outliers[:,i])[0],
        keypoints[:,i][~outliers[:,i]]
    ) for i in range(keypoints.shape[1])], axis=1)     
    return np.moveaxis(keypoints.reshape(init_shape),0,axis)


def init_obs_params(pca, *, Y, mask, latent_dimension, whiten, **kwargs):
    if whiten:
        Y_flat = transform_data_for_pca(Y, **kwargs)[mask>0]
        latents_flat = jax_io(pca.transform)(Y_flat)[:,:latent_dimension]
        cov = jnp.cov(latents_flat.T)
        W = jnp.linalg.cholesky(cov)
    else:
        W = jnp.eye(latent_dimension)
        
    Cd = jnp.array(jnp.hstack([
        pca.components_[:latent_dimension].T @ W, 
        pca.mean_[:,na]]))
    return Cd
                                 

def init_continuous_stateseqs(pca, *, Y, Cd, **kwargs):

    Y_flat = transform_data_for_pca(Y, **kwargs)
    # using numpy.linalg.pinv because jax version crashes on windows
    return (Y_flat - Cd[:,-1]) @ jnp.array(np.linalg.pinv(Cd[:,:-1]).T)


def init_states(seed, pca, Y, mask, conf, conf_threshold, params, 
                noise_prior, *, obs_hypparams, verbose=False, **kwargs):

    if verbose: print('Initializing states')
    
    if conf is None: 
        Y_interp = Y
    else: 
        Y_interp = interpolate(Y, conf<conf_threshold)
    
    states = {}
    
    states['v'],states['h'] = align_egocentric(
        Y_interp, **kwargs)[1:]
    
    states['x'] = init_continuous_stateseqs(
        pca, Y=Y_interp, **params, **kwargs)
    
    states['s'] = resample_scales(
        seed, Y=Y, conf=conf, **states, **params, 
        s_0=noise_prior, **obs_hypparams)
    
    states['z'] = resample_discrete_stateseqs(
        seed, Y=Y, mask=mask, **states, **params)[0]
    
    return states  

    
def init_params(seed, pca, Y, mask, conf, conf_threshold, *, 
                ar_hypparams, trans_hypparams, verbose=False, **kwargs):
    
    if verbose: print('Initializing parameters')
    
    if conf is None: Y_interp = Y
    else: Y_interp = interpolate(Y, conf<conf_threshold)
        
    params = {'sigmasq':jnp.ones(Y.shape[-2])}
    params['Ab'],params['Q'] = init_ar_params(seed, **ar_hypparams)
    params['betas'],params['pi'] = init_hdp_transitions(seed, **trans_hypparams)
    params['Cd'] = init_obs_params(pca, Y=Y_interp, mask=mask, **kwargs)
    return params


def init_hyperparams(*, conf, error_estimator, latent_dimension, 
                     trans_hypparams, ar_hypparams, obs_hypparams, 
                     cen_hypparams, verbose=False, **kwargs):
    
    if verbose: print('Initializing hyper-parameters')
    
    trans_hypparams = trans_hypparams.copy()
    obs_hypparams = obs_hypparams.copy()
    cen_hypparams = cen_hypparams.copy()
    ar_hypparams = ar_hypparams.copy()
    
    d,nlags = latent_dimension, ar_hypparams['nlags']
    ar_hypparams['S_0'] = ar_hypparams['S_0_scale']*jnp.eye(d)
    ar_hypparams['K_0'] = ar_hypparams['K_0_scale']*jnp.eye(d*nlags+1)
    ar_hypparams['M_0'] = jnp.pad(jnp.eye(d), ((0,0),((nlags-1)*d,1)))
    ar_hypparams['num_states'] = trans_hypparams['num_states']
    ar_hypparams['nu_0'] = d+2
    
    return {
        'ar_hypparams':ar_hypparams, 
        'obs_hypparams':obs_hypparams, 
        'cen_hypparams':cen_hypparams,
        'trans_hypparams':trans_hypparams}


def init_model(states=None, 
               params=None, 
               hypparams=None, 
               noise_prior=None,
               seed=None, 
               pca=None, 
               Y=None, 
               mask=None, 
               conf=None, 
               conf_threshold=0.5, 
               random_seed=0, 
               **kwargs):

    if None in (states,params): assert not None in (pca,Y,mask), fill(
        'Either provide ``states`` and ``params`` or provide a pca model and data')
        
    if seed is not None: seed = jnp.array(seed, dtype='uint32')
    else: seed = jr.PRNGKey(random_seed)

    if hypparams is not None:
        as_scalar = lambda arr: arr.item() if arr.shape==() else arr
        hypparams = tree_map(as_scalar, jax.device_put(hypparams))
    else: hypparams = init_hyperparams(conf=conf, **kwargs)
    kwargs.update(hypparams)
    
    if noise_prior is None: noise_prior = get_noise_prior(conf, **kwargs)
        
    if params is not None: params = jax.device_put(params)
    else: params = init_params(
        seed, pca, Y, mask, conf, conf_threshold, **kwargs)
        
    if states is not None: states = jax.device_put(states)
    else: states = init_states(
        seed, pca, Y, mask, conf, conf_threshold, params, noise_prior, **kwargs)

    return {
        'seed': seed,
        'states': states, 
        'params': params, 
        'hypparams': hypparams, 
        'noise_prior': noise_prior}




#======================================================================#
#                            Gibbs sampling                            #
#======================================================================#

@jax.jit
def resample_continuous_stateseqs(seed, *, Y, mask, v, h, z, s, Cd, sigmasq, Ab, Q, **kwargs):
    d,nlags,n = Ab.shape[1],Ab.shape[2]//Ab.shape[1],Y.shape[0]
    Gamma = center_embedding(Y.shape[-2])
    Cd = jnp.kron(Gamma, jnp.eye(Y.shape[-1])) @ Cd
    ys = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    A, B, Q, C, D = *ar_to_lds(Ab[...,:-1],Ab[...,-1],Q,Cd[...,:-1]),Cd[...,-1]
    R = jnp.repeat(s*sigmasq,Y.shape[-1],axis=-1)[:,nlags-1:]
    mu0,S0 = jnp.zeros(d*nlags),jnp.eye(d*nlags)*10
    xs = jax.vmap(kalman_sample, in_axes=(0,0,0,0,na,na,na,na,na,na,na,0))(
        jr.split(seed, ys.shape[0]), ys[:,nlags-1:], mask[:,nlags-1:-1], 
        z, mu0, S0, A, B, Q, C, D, R)
    xs = jnp.concatenate([xs[:,0,:-d].reshape(-1,nlags-1,d), xs[:,:,-d:]],axis=1)
    return xs


@jax.jit
def resample_heading(seed, *, Y, v, x, s, Cd, sigmasq, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    S = (Ybar[...,na]*(Y - v[...,na,:])[...,na,:]/(s*sigmasq)[...,na,na]).sum(-3)
    kappa_cos = S[...,0,0]+S[...,1,1]
    kappa_sin = S[...,0,1]-S[...,1,0]
    theta = vector_to_angle(jnp.stack([kappa_cos,kappa_sin],axis=-1))
    kappa = jnp.sqrt(kappa_cos**2 + kappa_sin**2)
    return sample_vonmises(seed, theta, kappa)

    
@jax.jit 
def resample_location(seed, *, mask, Y, h, x, s, Cd, sigmasq, sigmasq_loc, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    gammasq = 1/(1/(s*sigmasq)).sum(-1)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    rot_matrix = angle_to_rotation_matrix(h, d=d)
    mu = ((Y - (rot_matrix[...,na,:,:]*Ybar[...,na,:]).sum(-1)) \
          *(gammasq[...,na]/(s*sigmasq))[...,na]).sum(-2)

    m0,S0 = jnp.zeros(d), jnp.eye(d)*1e6
    A,B,Q = jnp.eye(d)[na],jnp.zeros(d)[na],jnp.eye(d)[na]*sigmasq_loc
    C,D,R = jnp.eye(d),jnp.zeros(d),gammasq[...,na]*jnp.ones(d)
    zz = jnp.zeros_like(mask[:,1:], dtype=int)

    return jax.vmap(kalman_sample, in_axes=(0,0,0,0,na,na,na,na,na,na,na,0))(
        jr.split(seed, mask.shape[0]), mu, mask[:,:-1], zz, m0, S0, A, B, Q, C, D, R)


@jax.jit
def resample_obs_params(seed, *, Y, mask, sigmasq, v, h, x, s, sigmasq_C, **kwargs):
    k,d,D = *Y.shape[-2:],x.shape[-1]
    Gamma = center_embedding(k)
    mask = mask.flatten()
    s = s.reshape(-1,k)
    x = x.reshape(-1,D)
    xt = pad_affine(x)

    Sinv = jnp.eye(k)[na,:,:]/s[:,:,na]/sigmasq[na,:,na]
    xx_flat = (xt[:,:,na]*xt[:,na,:]).reshape(xt.shape[0],-1).T
    # serialize this step because of memory constraints
    mGSG = mask[:,na,na] * Gamma.T@Sinv@Gamma
    S_xx_flat = jax.lax.map(lambda xx_ij: (xx_ij[:,na,na]*mGSG).sum(0), xx_flat)
    S_xx = S_xx_flat.reshape(D+1,D+1,k-1,k-1)
    S_xx = jnp.kron(jnp.concatenate(jnp.concatenate(S_xx,axis=-2),axis=-1),jnp.eye(d))
    Sigma_n = jnp.linalg.inv(jnp.eye(d*(D+1)*(k-1))/sigmasq_C + S_xx)

    vecY = inverse_affine_transform(Y, v, h).reshape(-1,k*d)
    S_yx = (mask[:,na,na]*vecY[:,:,na]*jnp.kron(
        jax.vmap(jnp.kron)(xt[:,na,:],Sinv@Gamma), 
        jnp.eye(d))).sum((0,1))         
    mu_n = Sigma_n@S_yx
                         
    return jr.multivariate_normal(seed, mu_n, Sigma_n).reshape(D+1,d*(k-1)).T


@jax.jit
def resample_obs_variance(seed, *, Y, mask, Cd, v, h, x, s, nu_sigma, sigmasq_0, **kwargs):
    k,d = Y.shape[-2:]
    s = s.reshape(-1,k)
    mask = mask.flatten()
    x = x.reshape(-1,x.shape[-1])
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(-1,k-1,d)
    Ytild = inverse_affine_transform(Y,v,h).reshape(-1,k,d)
    S_y = (mask[:,na]*((Ytild - Ybar)**2).sum(-1)/s).sum(0)
    variance = (nu_sigma*sigmasq_0 + S_y)/(nu_sigma+3*mask.sum())
    degs = (nu_sigma+3*mask.sum())*jnp.ones_like(variance)
    return sample_scaled_inv_chi2(seed, degs, variance)


@jax.jit
def resample_scales(seed, *, x, v, h, Y, Cd, sigmasq, nu_s, s_0, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    Ytild = inverse_affine_transform(Y,v,h)
    variance = (((Ytild - Ybar)**2).sum(-1)/sigmasq + s_0*nu_s)/(nu_s+3)
    degs = (nu_s+3)*jnp.ones_like(variance)
    return sample_scaled_inv_chi2(seed, degs, variance)


def resample_model(data, *, states, params, hypparams, seed, 
                   noise_prior, ar_only=False, states_only=False):
    
    seed = jr.split(seed)[1]

    if not states_only: 
        params['betas'],params['pi'] = resample_hdp_transitions(
            seed, **data, **states, **params, 
            **hypparams['trans_hypparams'])
        
    if not states_only: 
        params['Ab'],params['Q']= resample_ar_params(
            seed, **data, **states, **params, 
            **hypparams['ar_hypparams'])
    
    states['z'] = resample_discrete_stateseqs(
        seed, **data, **states, **params)[0]
    
    if not ar_only:     

        if not states_only: 
            params['sigmasq'] = resample_obs_variance(
                seed, **data, **states, **params, 
                **hypparams['obs_hypparams'])
        
        states['x'] = resample_continuous_stateseqs(
            seed, **data, **states, **params)
        
        states['h'] = resample_heading(
            seed, **data, **states, **params)
        
        states['v'] = resample_location(
            seed, **data, **states, **params, 
            **hypparams['cen_hypparams'])
        
        states['s'] = resample_scales(
            seed, **data, **states, **params, 
            s_0=noise_prior, **hypparams['obs_hypparams'])
        
    return {
        'seed': seed,
        'states': states, 
        'params': params, 
        'hypparams': hypparams,
        'noise_prior': noise_prior}



#======================================================================#
#                           Log probabilities                          #
#======================================================================#


def continuous_statseq_log_prob(*, x, z, Ab, Q, **kwargs):
    """
    Calculate the log probability of the trajectory ``x`` at each time 
    step, given switching autoregressive (AR) parameters

    Parameters
    ----------  
    x: jax array, shape (*dims,t,D)
        Continuous latent trajectories in R^D of length t

    z: jax array, shape (*dims,t)
        Discrete state sequences of length t

    Ab: jax array, shape (N,D*L+1) 
        AR transforms (including affine term) for each of N discrete
        states, where D is the dimension of the latent states and 
        L is the the order of the AR process

    Q: jax array, shape (N,D,D) 
        AR noise covariance for each of N discrete states

    Returns
    -------
    log_probability: jax array, shape (*dims,t-L)

    """

    Qinv = jnp.linalg.inv(Q)
    Qdet = jnp.linalg.det(Q)
    
    L = Ab.shape[-1]//Ab.shape[-2]
    x_lagged = get_lags(x, N)
    x_pred = (Ab[z] @ pad_affine(x_lagged)[...,na])[...,0]
    
    d = x_pred - x[:,nlags:]
    return (-(d[...,na,:]*Qinv[z]*d[...,:,na]).sum((2,3))/2
            -jnp.log(Qdet[z])/2  -jnp.log(2*jnp.pi)*Q.shape[-1]/2)


def discrete_statseq_log_prob(*, z, pi, **kwargs):
    """
    Calculate the log probability of a discrete state sequence at each
    time-step given a matrix of transition probabilities

    Parameters
    ----------  
    z: jax array, shape (*dims,t)
        Discrete state sequences of length t

    pi: jax array, shape (N,N)
        Transition probabilities

    Returns
    -------
    log_probability: jax array, shape (*dims,t-1)

    """
    return jnp.log(pi[z[...,:-1],z[...,1:]])


def scale_log_prob(*, s, s_0, nu_s, **kwargs):
    """
    Calculate the log probability of the noise scale for each keypoint
    given the noise prior, which is a scaled inverse chi-square 

    Parameters
    ----------  
    s: jax array, shape (*dims)
        Noise scale for each keypoint at each time-step

    s_0: float or jax array, shape (*dims)
        Prior on noise scale - either a single universal value or a 
        separate prior for each keypoint at each time-step

    nu_s: int
        Degrees of freedom

    Returns
    -------
    log_probability: jax array, shape (*dims)

    """
    return -nu_s*s_0 / s / 2 - (1+nu_s/2)*jnp.log(s)

    
def location_log_prob(*, v, sigmasq_loc):
    """
    Calculate the log probability of the centroid location at each 
    time-step, given the prior on centroid movement

    Parameters
    ----------  
    v: jax array, shape (*dims,t,d)
        Location trajectories in R^d of length t

    sigmasq_loc: float
        Assumed variance in centroid displacements

    Returns
    -------
    log_probability: jax array, shape (*dims,t-1)

    """
    d = v[:,:-1]-v[:,1:]
    return (-(d**2).sum(-1)/sigmasq_loc/2 
            -v.shape[-1]/2*jnp.log(sigmasq_loc*2*jnp.pi))


def obs_log_prob(*, Y, x, s, v, h, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of keypoint coordinates at each
    time-step, given continuous latent trajectories, centroids, heading
    angles, noise scales, and observation parameters

    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    v: jax array, shape (*dims,d), Centroids
    h: jax array, shape (*dims), Heading angles
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint

    Returns
    -------
    log_probability: jax array, shape (*dims,k)

    """
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    sqerr = ((Y - affine_transform(Ybar,v,h))**2).sum(-1)
    return (-1/2 * sqerr/s/sigmasq - d/2 * jnp.log(2*s*sigmasq*jnp.pi))


@jax.jit
def log_joint_likelihood(*, Y, mask, x, s, v, h, z, pi, Ab, Q, Cd, sigmasq, sigmasq_loc, s_0, nu_s, **kwargs):
    """
    Calculate the total log probability for each latent state

    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    mask: jax array, shape (*dims), Binary indicator for valid frames
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    v: jax array, shape (*dims,d), Centroids
    h: jax array, shape (*dims), Heading angles
    z: jax array, shape (*dims), Discrete state sequences
    pi: jax array, shape (N,N), Transition probabilities
    Ab: jax array, shape (N,D*L+1), Autoregressive transforms
    Q: jax array, shape (D,D), Autoregressive noise covariances
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint
    sigmasq_loc: float, Assumed variance in centroid displacements
    s_0: float or jax array, shape (*dims,k), Prior on noise scale
    nu_s: int, Degrees of freedom in noise prior

    Returns
    -------
    log_probabilities: dict
        Dictionary mapping the name of each latent state variables to
        its total log probability

    """
    nlags = Ab.shape[-1]//Ab.shape[-2]
    return {
        'Y': (obs_log_prob(Y=Y, x=x, s=s, v=v, h=h, Cd=Cd, sigmasq=sigmasq)*mask[...,na]).sum(),
        'x': (continuous_stateseq_log_prob(x=x, z=z, Ab=Ab, Q=Q)*mask[...,nlags:]).sum(),
        'z': (discrete_stateseq_log_prob(z=z, pi=pi)*mask[...,nlags+1:]).sum(),
        'v': (location_log_prob(v=v, sigmasq_loc=sigmasq_loc)*mask[...,1:]).sum(),
        's': (scale_log_prob(s=s, nu_s=nu_s, s_0=s_0)*mask[...,na]).sum()}



#############################################################################
def estimate_error(conf, *, slope, intercept):
    return (10**(jnp.log10(conf+1e-6)*slope+intercept))**2

def get_noise_prior(conf, *, error_estimator, use_bodyparts, **kwargs):
    if conf is None: return jnp.ones(len(use_bodyparts))
    else: return estimate_error(conf, **error_estimator)



