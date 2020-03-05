import numpy as np
import re, os, sys, yaml
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower, FieldMesh
from nbodykit     import setup_logging
from mpi4py       import MPI
sys.path.append('../../utils/')
import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')

#Get parameter file
cfname = sys.argv[1]

with open(cfname, 'r') as ymlfile:
    args = yaml.load(ymlfile, Loader=yaml.FullLoader)

#
nc = args['nc']
bs = args['bs']
alist = args['alist']
#
#
#Global, fixed things
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
if rank == 0: print(args)




def read_conversions(db):
    """Read the conversion factors we need and check we have the right time."""
    mpart,Lbox,rsdfac,acheck = None,None,None,None
    with open(db+"/attr-v2","r") as ff:
        for line in ff.readlines():
            mm = re.search("MassTable.*\#HUMANE\s+\[\s*0\s+(\d*\.\d*)\s*0+\s+0\s+0\s+0\s+\]",line)
            if mm != None:
                mpart = float(mm.group(1)) * 1e10
            mm = re.search("BoxSize.*\#HUMANE\s+\[\s*(\d+)\s*\]",line)
            if mm != None:
                Lbox = float(mm.group(1))
            mm = re.search("RSDFactor.*\#HUMANE\s+\[\s*(\d*\.\d*)\s*\]",line)
            if mm != None:
                rsdfac = float(mm.group(1))
            mm = re.search("ScalingFactor.*\#HUMANE\s+\[\s*(\d*\.\d*)\s*\]",line)
            if mm != None:
                acheck = float(mm.group(1))
    if (mpart is None)|(Lbox is None)|(rsdfac is None)|(acheck is None):
        print(mpart,Lbox,rsdfac,acheck)
        raise RuntimeError("Unable to get conversions from attr-v2.")
    return mpart, Lbox, rsdfac, acheck
    #



    
    

if __name__=="__main__":
    if rank==0: print('Starting')

    for aa in alist:
        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))
        mpart, Lbox, rsdfac, acheck = read_conversions(args['headerfilez']%aa)
        if np.abs(acheck-aa)>1e-4:
            raise RuntimeError("Read a={:f}, expecting {:f}.".format(acheck,aa))
        if np.abs(Lbox-bs)>1e-4:
            raise RuntimeError("Read L={:f}, expecting {:f}.".format(Lbox,bs))
        if rank == 0: print('Mass of the particle : %0.2e'%mpart)

        halocat = BigFileCatalog(args['halofilez']%aa, dataset=args['halodataset'])
        halocat['Mass'] = halocat['Length'].compute() * mpart
        cencat = BigFileCatalog(args['cenfilez']%aa, dataset=args['cendataset'])
        cencat['Mass'] = cencat['Length'] * mpart
        satcat = BigFileCatalog(args['satfilez']%aa, dataset=args['satdataset'])
        #

        modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
        modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 
        for model in {'ModelA':HImodels.ModelA}:
            HImodel = modeldict[model]
            modelname = model
            mode = modedict[model]

            for satsuppress in [1., 0.5, 0.1, 0.05]:
                if rank == 0: print('satsuppress = %0.2f'%satsuppress)
                HImodelz = HImodel(aa)
                HImodelz.normsat *= satsuppress
                halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)
                if cencat['HImass'].compute().min()  < 0:
                    print(rank, 'cencat for satsuppress = %0.2f'%satsuppress, cencat['HImass'].compute().min(), cencat['HImass'].compute().max() )
