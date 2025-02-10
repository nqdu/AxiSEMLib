from database import AxiBasicDB
import numpy as np
import os 
import sys 

def main():
    if len(sys.argv) != 2:
        print("Usage: syn_seismogram.py workdir")
        print("example: python syn_seismogram.py axisem/SOLVER/ak135.21 ")
        exit(1)

    basedir = sys.argv[1] + "/"

    # create dir
    os.makedirs("SEISMOGRAMS",exist_ok=True)

    db:AxiBasicDB = AxiBasicDB()
    db.read_basic(basedir + "/MZZ/Data/axisem_output.nc4")
    db.set_iodata(basedir)

    cmtfile = "CMTSOLUTION"
    stacords = np.loadtxt(basedir + "/MZZ/STATIONS",dtype = str)
    t = np.arange(db.nt) * db.dtsamp + db.t0
    nsta = stacords.shape[0]

    for i in range(nsta):
        print(i+1,nsta)
        stla,stlo = np.float32(stacords[i,2:4])
        ue,un,uz = db.syn_seismo(stla,stlo,0.,'enz',basedir + cmtfile)
        name = stacords[i,1] + "." + stacords[i,0]
        newname = "SEISMOGRAMS/" + name
        dataout = np.zeros((len(un),2)) 
        dataout[:,0] = t 
        dataout[:,1] = ue 
        np.savetxt(f'{newname}.BXE.dat',dataout)
        dataout[:,1] = un
        np.savetxt(f'{newname}.BXN.dat',dataout)
        dataout[:,1] = uz
        np.savetxt(f'{newname}.BXZ.dat',dataout)

if __name__ == "__main__":
    main()