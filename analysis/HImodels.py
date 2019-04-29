import numpy as np
from nbodykit.utils import DistributedArray
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog




class ModelA():
    
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1/aa-1

        self.alp = (1+2*self.zz)/(2+2*self.zz)
        self.mcut = 3e9*( 1 + 10*(3*self.aa)**8)
        self.normhalo = 8e5*(1+(3.5/self.zz)**6) 
        self.normsat = self.normhalo*(1.75 + 0.25*self.zz)


    def assignHI(self, halocat, cencat, satcat):
        '''Assign HI in the halo, satellite and centrals
        '''
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(mHIhalo, mHIsat, satcat['GlobalID'].compute(), 
                                cencat.csize, cencat.comm)
        
        return mHIhalo, mHIcen, mHIsat
        
        
    def assignhalo(self, mhalo):
        '''Assign HI in the halo
        '''
        xx  = mhalo/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normhalo
        return mHI

    def assignsat(self, msat):
        '''Assign HI in the satellites of the halo
        '''
        xx  = msat/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normsat
        return mHI
        

    def getinsat(self, mHIsat, satid, totalsize, localsize, comm):
        '''Get the total HI content in satellites of a given halo
        '''
        da = DistributedArray(satid, comm)
        mHI = da.bincount(mHIsat, shared_edges=False)
        
        zerosize = totalsize - mHI.cshape[0]
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
        zeros.local[...] = 0
        mHItotal = DistributedArray.concat(mHI, zeros, localsize=localsize)
        return mHItotal
        
    def assigncen(self, mHIhalo, mHIsat, satid, censize, comm):
        '''Assign HI in the central of the halo, given HI in the total halo and satellite
        '''
        #Assumes every halo has a central...which it does...usually
        mHItotal = self.getinsat(mHIsat, satid, censize, mHIhalo.size, comm)
        return mHIhalo - mHItotal.local
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass'):
        '''creates a mesh of HI given a halo, central and sattelite catalog in 
        real space if position='Position'  or redshift space if position='RSDpos'
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh
        
    


class ModelA2(ModelA):
    '''Same as model A with a different RSD for satellites
    '''
    def __init__(self, aa):

        super().__init__(aa)
        
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity_HI']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos

        

        
    
class ModelB():
    
    def __init__(self, aa, h=0.6776):

        self.aa = aa
        self.zz = 1/aa-1
        self.h = h
        self.mcut = 3e9*( 1 + 10*(3*self.aa)**8) 
        self.normhalo = 1


    def assignHI(self, halocat, cencat, satcat):
        '''Assign HI in the halo, satellite and centrals
        '''
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(cencat['Mass'].compute())
        
        mHIhalo = self.assignhalo(mHIcen, mHIsat, satcat['GlobalID'].compute(), 
                                halocat.csize, halocat.comm)
        return mHIhalo, mHIcen, mHIsat

    def assignhalo(self, mHIcen, mHIsat, satid, hsize, comm):
        '''Assign HI in the halo
        '''
        #Assumes every halo has a central...which it does...usually
        mHItotal = self.getinsat(mHIsat, satid, hsize, mHIcen.size, comm)
        return mHIcen + mHItotal.local


    def getinsat(self, mHIsat, satid, totalsize, localsize, comm):
        '''Get the total HI content in satellites of a given halo
        '''
        da = DistributedArray(satid, comm)
        mHI = da.bincount(mHIsat, shared_edges=False)
        zerosize = totalsize - mHI.cshape[0]
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
        zeros.local[...] = 0
        mHItotal = DistributedArray.concat(mHI, zeros, localsize=localsize)
        return mHItotal

    def _assign(self, mstellar):
        '''Takes in M_stellar and gives M_HI in M_solar
        '''
        mm = 3e8 #5e7
        f = 0.18 #0.35
        alpha = 0.4 #0.35
        mfrac = f*(mm/(mstellar + mm))**alpha
        mh1 = mstellar * mfrac
        return mh1
        
        

    def assignsat(self, msat, scatter=None):
        '''Assign HI in the satellite
        '''
        mstellar = self.moster(msat, scatter=scatter)/self.h
        mh1 = self._assign(mstellar)
        mh1 = mh1*self.h #* np.exp(-self.mcut/msat)
        return mh1


    def assigncen(self, mcen, scatter=None):
        '''Assign HI in the centrals
        '''
        mstellar = self.moster(mcen, scatter=scatter)/self.h
        mh1 = self._assign(mstellar)
        mh1 = mh1*self.h #* np.exp(-self.mcut/mcen)
        return mh1


    def moster(self, Mhalo, scatter=None):
        """ 
        moster(Minf,z): 
        Returns the stellar mass (M*/h) given Minf and z from Table 1 and                                                                  
        Eq. (2,11-14) of Moster++13 [1205.5807]. 
        This version now works in terms of Msun/h units,
        convert to Msun units in the function
        To get "true" stellar mass, add 0.15 dex of lognormal scatter.                    
        To get "observed" stellar mass, add between 0.1-0.45 dex extra scatter.         

        """
        z = self.zz
        Minf = Mhalo/self.h
        zzp1  = z/(1+z)
        M1    = 10.0**(11.590+1.195*zzp1)
        mM    = 0.0351 - 0.0247*zzp1
        beta  = 1.376  - 0.826*zzp1
        gamma = 0.608  + 0.329*zzp1
        Mstar = 2*mM/( (Minf/M1)**(-beta) + (Minf/M1)**gamma )
        Mstar*= Minf
        if scatter is not None: 
            Mstar = 10**(np.log10(Mstar) + np.random.normal(0, scatter, Mstar.size))
        return Mstar*self.h
        #                                                                                                                                          

    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass'):
        '''creates a mesh of HI given a halo, central and sattelite catalog in 
        real space if position='Position'  or redshift space if position='RSDpos'
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh







class ModelC(ModelA):
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity but do not have any dispersion over it
    '''
    def __init__(self, aa):

        super().__init__(aa)
        self.normsat = 0

        self.alp = 0.9
        self.mcut = 1e10
        self.normhalo = 3.5e6*(1+1/self.zz) 


    def derivate(self, param, delta):
        '''Change the value of the parameter 'param' by fraction 'delta'
        '''
        if param == 'alpha':
            self.alp = (1+delta)*self.alp
        elif param == 'mcut':
            self.mcut = 10**( (1+delta)*np.log10(self.mcut))
        elif param == 'norm':
            self.mcut = 10**( (1+delta)*np.log10(self.normhalo))
        else:
            print('Parameter to vary not recongnized. Should be "alpha", "mcut" or "norm"')
            


    def assignHI(self, halocat, cencat, satcat):
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(cencat['Mass'].compute())
        
        return mHIhalo, mHIcen, mHIsat
        
    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        

    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='halos', position='RSDpos', weight='HImass'):
        '''creates a mesh of HI given a halo, central and sattelite catalog in 
        real space if position='Position'  or redshift space if position='RSDpos'
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh



