import numpy as np 
import h5py 
from tqdm import tqdm 

for direc in ['MZZ',"MXX_P_MYY","MXZ_MYZ","MXY_MXX_M_MYY"]:

    fnc = h5py.File("ak135/" + direc + "/Data/axisem_output.nc4","r")
    fout = h5py.File("ak135/" + direc + "/Data/axisem_fields.h5","w")

    for d in ['disp_s','disp_p','disp_z']:

        dsetname = 'Snapshots/' + d
        if dsetname not in fnc.keys():
            continue
        
        print('transpose %s in %s'%(d,direc))
        disp = fnc[dsetname]
        nt = disp.shape[0]; npts = disp.shape[1]

        dset = fout.create_dataset(d,(npts,nt),dtype=np.float32,chunks = True)
        
        for i in tqdm(range(nt)):
            dset[:,i] = disp[i,:]
            #print(np.sum(disp[i,:]))
        
    fnc.close()
    fout.close()

