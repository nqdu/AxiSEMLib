#from database1 import AxiBasicDB as AxiBasicDB1
from bak.database import AxiBasicDB
import numpy as np 
from obspy import read_events
import matplotlib.pyplot as plt 
import os 

basedir = '../SOLVER/ak135_th/'

# create dir
os.makedirs("SEISMOGRAMS",exist_ok=True)

db:AxiBasicDB = AxiBasicDB()
db.read_basic(basedir + "/MZZ/Data/axisem_output.nc4")
db.set_iodata(basedir)

cmtfile = "/CMTSOLUTION"
cat = read_events(basedir + cmtfile)[0]
org = cat.origins[0]
starttime = org.time
xs = np.array([org.longitude,org.latitude,org.depth])
mzz,mxx,myy,mxz,myz,mxy = db.read_cmt(basedir + cmtfile)
mij = np.array([mxx,myy,mzz,2 * myz,2 * mxz,2 * mxy])

stacords = np.loadtxt(basedir + "/MZZ/STATIONS",dtype = str)
t = np.arange(db.nt) * db.dtsamp - db.shift
nsta = stacords.shape[0]

for i in range(nsta):
    print(i+1,nsta)
    stla,stlo = np.float32(stacords[i,2:4])
    ue,un,uz = db.syn_seismo(stla,stlo,0.,'enz',basedir + cmtfile)
    name = stacords[i,0] + "_" + stacords[i,1]
    newname = "SEISMOGRAMS/" + name
    dataout = np.zeros((len(un),2)) 
    dataout = np.zeros((len(un),2)) 
    dataout[:,0] = t 
    dataout[:,1] = ue 
    np.savetxt(f'{newname}.BXE.dat',dataout)
    dataout[:,1] = un
    np.savetxt(f'{newname}.BXN.dat',dataout)
    dataout[:,1] = uz
    np.savetxt(f'{newname}.BXZ.dat',dataout)
