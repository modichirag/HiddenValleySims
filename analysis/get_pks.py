import numpy as np
import re, os, sys, yaml
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower
from nbodykit     import setup_logging
from mpi4py       import MPI
sys.path.append('../utils/')
import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')

#Get model as parameter
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


#
#Which model & configuration to use to assign HI
modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC, 'ModelD':HImodels.ModelD, 'ModelD2':HImodels.ModelD2}
modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos', 'ModelD':'galaxies', 'ModelD2':'galaxies'} 

#




def read_conversions(db):
    """Read the conversion factors we need and check we have the right time."""
    mpart,Lbox,rsdfac,acheck = None,None,None,None
    with open(db+"Header/attr-v2","r") as ff:
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




def calc_pk1d(aa, h1mesh, outfolder):
    '''Compute the 1D redshift-space P(k) for the HI'''

    if rank==0: print('Calculating pk1d')
    pkh1h1   = FFTPower(h1mesh,mode='1d',kmin=0.025,dk=0.0125).power
    # Extract the quantities we want and write the file.
    kk   = pkh1h1['k']
    sn   = pkh1h1.attrs['shotnoise']
    pk   = np.abs(pkh1h1['power'])
    if rank==0:
        fout = open(outfolder + "HI_pks_1d_{:6.4f}.txt".format(aa),"w")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        fout.write("# {:>8s} {:>15s}\n".format("k","Pk_0_HI"))
        for i in range(kk.size):
            fout.write("{:10.5f} {:15.5e}\n".format(kk[i],pk[i]-sn))
        fout.close()
    #




def calc_pkmu(aa, h1mesh, outfolder, los=[0,0,1], Nmu=int(4)):
    '''Compute the redshift-space P(k) for the HI in mu bins'''

    if rank==0: print('Calculating pkmu')
    pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=Nmu,los=los).power
    # Extract what we want.
    kk = pkh1h1.coords['k']
    sn = pkh1h1.attrs['shotnoise']
    pk = pkh1h1['power']
    if rank==0: print('For mu-bins', pkh1h1.coords['mu'])
    # Write the results to a file.
    if rank==0:
        fout = open(outfolder + "HI_pks_mu_{:02d}_{:06.4f}.txt".format(Nmu, aa),"w")
        fout.write("# Redshift space power spectrum in mu bins.\n")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        ss = "# {:>8s}".format(r'k\mu')
        for i in range(pkh1h1.shape[1]):
            ss += " {:15.5f}".format(pkh1h1.coords['mu'][i])
        fout.write(ss+"\n")
        for i in range(1,pk.shape[0]):
            ss = "{:10.5f}".format(kk[i])
            for j in range(pk.shape[1]):
                ss += " {:15.5e}".format(np.abs(pk[i,j]-sn))
            fout.write(ss+"\n")
        fout.close()
    #






def calc_pkll(aa, h1mesh, outfolder, los=[0,0,1]):
    '''Compute the redshift-space P_ell(k) for the HI'''

    if rank==0: print('Calculating pkll')
    pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=8,los=los,\
                      kmin=0.02,dk=0.02,poles=[0,2,4]).poles
    # Extract the quantities of interest.
    kk = pkh1h1.coords['k']
    sn = pkh1h1.attrs['shotnoise']
    P0 = pkh1h1['power_0'].real - sn
    P2 = pkh1h1['power_2'].real
    P4 = pkh1h1['power_4'].real
    # Write the results to a file.
    if rank==0:
        fout = open(outfolder + "HI_pks_ll_{:06.4f}.txt".format(aa),"w")
        fout.write("# Redshift space power spectrum multipoles.\n")
        fout.write("# Subtracting SN={:15.5e} from monopole.\n".format(sn))
        fout.write("# {:>8s} {:>15s} {:>15s} {:>15s}\n".\
                   format("k","P0","P2","P4"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:15.5e} {:15.5e} {:15.5e}\n".\
                       format(kk[i],P0[i],P2[i],P4[i]))
        fout.close()
    #



def calc_bias(aa,h1mesh,suff):
    '''Compute the bias(es) for the HI'''

    if rank==0: print('Calculating bias')
    if rank==0:
        print("Processing a={:.4f}...".format(aa))
        print('Reading DM mesh...')
    dm    = BigFileMesh(args['matterfile']%(aa),'N1024').paint()
    dm   /= dm.cmean()
    if rank==0: print('Computing DM P(k)...')
    pkmm  = FFTPower(dm,mode='1d').power
    k,pkmm= pkmm['k'],pkmm['power']  # Ignore shotnoise.
    if rank==0: print('Done.')
    #

    pkh1h1 = FFTPower(h1mesh,mode='1d').power
    kk = pkh1h1.coords['k']

    pkh1h1 = pkh1h1['power']-pkh1h1.attrs['shotnoise']
    pkh1mm = FFTPower(h1mesh,second=dm,mode='1d').power['power']
    if rank==0: print('Done.')
    # Compute the biases.
    b1x = np.abs(pkh1mm/(pkmm+1e-10))
    b1a = np.abs(pkh1h1/(pkmm+1e-10))**0.5
    if rank==0: print("Finishing processing a={:.4f}.".format(aa))

    #
    if rank==0:
        fout = open(outfolder + "HI_bias_{:6.4f}.txt".format(aa),"w")
        fout.write("# {:>8s} {:>10s} {:>10s} {:>15s}\n".\
                   format("k","b1_x","b1_a","Pkmm"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:10.5f} {:10.5f} {:15.5e}\n".\
                       format(kk[i],b1x[i],b1a[i],pkmm[i].real))
        fout.close()


        

if __name__=="__main__":

    if rank==0: print('Starting Analysis')

    for aa in alist:

        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))

        for model in args['models']:
            HImodel = modeldict[model] 
            mode = modedict[model]
            #Path to save the output here
            outfolder = args['outfolder']%aa + '/pks/%s_N%04d/'%(model, nc)
            if rank == 0: print(outfolder)
            for folder in [args['outfolder']%aa + '/pks/', outfolder]:
                try:  os.makedirs(folder)
                except : pass

            los = [0,0,1]
            try:
                h1meshz = BigFileMesh(args['h1meshz']%(aa, nc), model)
                h1mesh = BigFileMesh(args['h1mesh']%(aa, nc), model)
            except Exception as e:
                print('\nException occured : ', e)

        
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
           
                HImodelz = HImodel(aa)
                #halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)
                #halocat['RSDpos'], cencat['RSDpos'], satcat['RSDpos'] = HImodelz.assignrsd(rsdfac, halocat, cencat, satcat, los=los)
                #h1mesh = HImodelz.createmesh_catalog(bs, nc, halocat, cencat, satcat, mode=mode, position='RSDpos', weight='HImass')
                halocat_HImass, cencat_HImass, satcat_HImass = HImodelz.assignHI(halocat, cencat, satcat)
                halocat_rsdpos, cencat_rsdpos, satcat_rsdpos = HImodelz.assignrsd(rsdfac, halocat, cencat, satcat, los=los)

                if rank == 0: print('Creating HI mesh in real space for bias')
                #h1mesh = HImodelz.createmesh_catalog(bs, nc, halocat, cencat, satcat, mode=mode, position='Position', weight='HImass')
                positions = [halocat['Position']]
                weights = [halocat_HImass]
                h1mesh = HImodelz.createmesh(bs, nc, positions, weights)

                if rank == 0: print('Creating HI mesh in redshift space')
                if mode=='halos':
                    positions = [halocat_rsdpos]
                    weights = [halocat_HImass]
                if mode=='galaxies':
                    positions = [cencat_rsdpos, satcat_rsdpos]
                    weights = [cencat_HImass, satcat_HImass]
                h1meshz = HImodelz.createmesh(bs, nc, positions, weights)

                
            calc_pk1d(aa, h1meshz, outfolder)
            calc_pkmu(aa, h1meshz, outfolder, los=los, Nmu=8)
            calc_pkll(aa, h1meshz, outfolder, los=los)
            calc_bias(aa, h1mesh, outfolder)
            #calc_pkmu(aa, h1meshz, outfolder, los=los, Nmu=5)
                
    sys.exit(-1)
