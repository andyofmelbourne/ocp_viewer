import numpy as np

def radial_profile(data, centre = None):
    if centre == None :
        centre = [data.shape[0]/2-1, data.shape[1]/2-1]
    
    i, j = np.indices((data.shape))
    r = np.sqrt((j - center[1])**2 + (i - center[0])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def radial_profile_partial(data, ):
    if centre == None :
        centre = [data.shape[0]/2-1, data.shape[1]/2-1]
    
    i, j = np.indices((data.shape))
    r = np.sqrt((j - center[1])**2 + (i - center[0])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def radial_profile_integrate(cspad, rs, rmin, rmax):
    radProf  = np.bincount(rs.ravel(), cspad.astype(np.int64).ravel())    #cspad contains the photon count for each pixel, rs contains the radial value of each pixel
                                    #the bincount() method then creates an array, 'radProf' such that:
                                    #radProf[i] represents the number of photons detected by all pixels that have a radius of i pixels

    if not (((rmin == 0) and (rmax == 0)) or (rmin > rmax)) :
        radProf[:rmin] = 0    #kill the radial profile outside the
        radProf[rmax+1:] = 0  #region of interest
    else :
        pass

    rcount = np.bincount(rs.ravel())    #creates an array of length equal to the maximum radius value, such that rcount[i] countains the number of pixels
                    #that have a radial value of i
    rcount[np.where(rcount == 0)] = 1    #avoid dividing by zero, set rcount = 1 where there are no pixels.  this does no harm, since radProf must equal zero at those values
    r_int = sum(radProf)#count the total number of photons within the region of interest
    radProf[rmin:rmax+1] = radProf[rmin:rmax+1]/rcount[rmin:rmax+1].astype(np.float64)    #normalize the radial profile by the number of pixels that have the respective radius
                                    #the indexing argument is so we don't divide by zero
    return radProf.astype(cspad.dtype), r_int.astype(cspad.dtype)

if __name__ == '__main__':
    array = np.random.random((400, 100)).reshape((400, 100))
    r_profile, nr, r = radial_profile(array, (49, 199))


