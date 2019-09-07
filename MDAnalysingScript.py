from handle import Handle
from analyse import Analyse
from resultprocessor import ResultProcessor
from mpl_toolkits.mplot3d import Axes3D
import MDAnalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib as im
import handle
import analyse
import fit
import visualisation
import resultprocessor
matplotlib.use('QT5Agg')


"""
Some characteristics of the five handles, specifically for .pdb files.
In .pdb files, residues can be identified through their property "SegName".
Strands can be identified by their property "ResName".
Residues on the handle strand are numbered from 1 to 20 towards the top.
Residues on the staple strand (including the two free bases) are numbered from 22 to 1 towards the top.
"""

# location of my files
linux_loc = "/home/tristandewaard/TU_OneDriveSync/UNI/JAAR_TRES/QUARTIEL_QUATRO/BEP"
windows_loc = "D:/ONEDRIVE_TUE/OneDrive - TU Eindhoven/UNI/JAAR_TRES/QUARTIEL_QUATRO/BEP"

# handle numbering is clockwise, middle handle last.
handleNumbers = range(1, 6)

# code defining the complementarity of free bases in the handle w.r.t. scaffold strand bases.
compCode = ["11", "00", "10", "111", "01"]

# location codes
locCode = ["I", "II", "III", "IV", "V"]

# colour pairs for every handle in the format (stapleStrandColour, handleStrandColour).
handleColours = [("Light Blue", "Light Green"),
                 ("Blue", "White"),
                 ("Yellow", "Pink"),
                 ("Dark Purple", "Light Purple"),
                 ("Magenta", "Cyan")]

# characteristic colours for drawing the handle vectors/planes
handleDrawColours = ["green", "cyan", "orange", "blue", "magenta"]

# handle segname numbers for stapleStrands and handleStrands respectively
# when going from the bottom to the top.
stapleStrandSegNames = len(handleNumbers) * [range(22, 0, -1)]
handleStrandSegNames = len(handleNumbers) * [range(1, 21)]

# segname numbers for the "handle starts"
handleStartSegNames = [4879, 5351, 5759, 1752, 5319]


# handle strand "atom" names for stapleStrands and handleStrands respectively, from handle 1 to 5.
stapleStrandNames = [range(18991, 18947, -2),
                     range(21157, 21113, -2),
                     range(22163, 22119, -2),
                     range(21453, 21409, -2),
                     range(20547, 20503, -2)]
handleStrandNames = [range(287, 327, 2),
                     range(127, 167, 2),
                     range(167, 207, 2),
                     range(247, 287, 2),
                     range(207, 247, 2)]

# handle resname numbers for stapleStrands and handleStrands respectively
stapleStrandResNames = [80, 113, 127, 117, 104]
handleStrandResNames = [7, 3, 4, 6, 5]

# resname for the scaffold, needed to calculate the "handle starts"
scaffoldResname = len(handleNumbers) * [8]

# Data for the handles' underlying scaffold vector:
# key = "scaffoldstrand" gives the segnames on scaffold strand per handle
# key = "staplestrand" gives the segnames on staplestand per handle
# EVERYTHING FROM - TO + DIRECTION => i.e. THE CONTINUATION OF HANDLE STAPLE STRAND ON SCAFFOLD.
scafVecSegNames = {"scaffoldstrand": [[range(4888, 4872, -1), range(2220, 2217, -1)],
                                      [range(5360, 5341, -1)],
                                      [range(5768, 5752, -1), range(1332, 1329, -1)],
                                      [range(1761, 1742, -1)],
                                      [range(5328, 5312, -1), range(1776, 1773, -1)]],
                   "staplestrands": [[24, range(24, 32), range(23, 33)],
                                     [24, range(25, 33), range(23, 31), range(9, 11)],
                                     [24, range(24, 32), range(23, 33)],
                                     [45, range(25, 33), range(23, 31), range(9, 11)],
                                     [24, range(24, 32), range(23, 33)]]}

scafVecResNames = {"scaffoldstrand": [8, 8, 8, 8, 8],
                   "staplestrands": [[78, 79, 80],
                                     [100, 101, 113, 102],
                                     [125, 126, 127],
                                     [104, 105, 117, 106],
                                     [102, 103, 104]]}
# old properties for the scaffold vector
dirvecScafEndSegnames = [2209, 5333, 1321, 1734, 1765]
dirvecStapEndSegnames = [41, 19, 41, 19, 41]
dirvecStapEndResnames = [80, 102, 127, 106, 104]

proplist = [handleStrandSegNames, stapleStrandSegNames, handleStrandResNames,
                              stapleStrandResNames, scaffoldResname, handleStartSegNames, handleDrawColours,
                              compCode, dirvecScafEndSegnames, dirvecStapEndSegnames, dirvecStapEndResnames,
                               scafVecSegNames["scaffoldstrand"], scafVecSegNames["staplestrands"],
                               scafVecResNames["scaffoldstrand"], scafVecResNames["staplestrands"]]


def init_handles(u, proplist=proplist, handleNumbers=handleNumbers):
    im.reload(handle)
    hs = []
    for i in range(len(handleNumbers)):
        h = Handle.with_properties(u, *[prop[i] for prop in proplist])
        hs.append(h)
    return hs

# test = MDAnalysis.Universe(windows_loc+"/myResults/firstSim/mytest.psf", windows_loc+"/myPythonFiles/analysed_universes/StijnSimulated.pdb")
# print(len(test.trajectory))

# %% StijnSimulated

PSF = "/StijnResultaten/simulated/mytest.psf"
PDB = "/StijnResultaten/simulated/trajectory_sim.dat.pdb"

#make MDAnalysis universe object
try:
    u = MDAnalysis.Universe(windows_loc + PSF, windows_loc + PDB)
except:
    u = MDAnalysis.Universe(linux_loc + PSF, linux_loc + PDB)

# %% analyse
im.reload(analyse)
im.reload(fit)
im.reload(visualisation)
im.reload(resultprocessor)

hs = init_handles(u)

stijn_analysis = Analyse.from_universe(u, hs, save=True, rotate=True, name="StijnSimulated")
#stijn_analysis.do_primary_frame_analysis(framerange=range(len(u.trajectory)), save=True, debugplots=False)

stijn_results = ResultProcessor(stijn_analysis, rid="20190607-154308")
stijn_results.multi_histplot2d(3, 4, nbins=[16, 16], grid=False, xlim=[-10, 10], ylim=[-10, 10], excludehframes=None)
stijn_results.multi_histplot1d(0, density=True, grid=True, xlim=[-180, 180], excludehframes=None)
stijn_results.multi_histplot1d(1, density=True, grid=True, xlim=[0, 180], excludehframes=None)
stijn_results.theta_corrplot(nbins=40, grid=False, xlim=[0, np.pi], ylim=[0, 0.7])
stijn_results.multi_histplot1d(3, grid=True, xlim=[-10, 10], density=True, excludehframes=None)
stijn_results.multi_histplot1d(4, grid=True, xlim=[-10, 10], density=True, excludehframes=None)
stijn_results.multi_histplot1d(5, grid=True, xlim=[0, 11], density=True, excludehframes=None)
stijn_results.lineplot_overframes(3, method="mavg", grid=True, binding=True, numbinding=True, excludehframes=None)
stijn_results.lineplot_overframes(4, method="mavg", grid=True, binding=True, numbinding=True, excludehframes=None)
stijn_results.lineplot_overframes(5, method="mavg", grid=True, binding=True, numbinding=True, excludehframes=None)
stijn_results.bargraph_binding(count=True)
stijn_results.bargraph_binding(count=False)
stijn_results.plot_surfaceprops()


# %% myCompleteSim

PSF1 = "/myResults/firstSim/mytest.psf"
PDB1 = "/myResults/firstSim/trajectory_sim_cont.dat.pdb"
PSF2 = "/myResults/secondSim/mytest.psf"
PDB2 = "/myResults/secondSim/trajectory_sim.dat.pdb"

#make MDAnalysis universe object
try:
    u = MDAnalysis.Universe(windows_loc + PSF1, [windows_loc + PDB1, windows_loc + PDB2])
except:
    u = MDAnalysis.Universe(linux_loc + PSF1, [linux_loc + PDB1, linux_loc + PDB2])

# %% analyse

im.reload(analyse)
im.reload(fit)
im.reload(visualisation)
im.reload(resultprocessor)

hs = init_handles(u)
complete_sim_analysis = Analyse.from_universe(u, hs, save=True, rotate=True, name="myCompleteSim")
# arr = complete_sim_analysis.do_primary_frame_analysis(framerange=range(len(u.trajectory)), debugplots=False, save=True)
# complete_sim_analysis.visualise_frames([1749])
# complete_sim_analysis.openVMD(1750)

complete_sim_results = ResultProcessor(complete_sim_analysis, '20190604-141658')
# complete_sim_results.lineplot_overframes(3, ylim=[-10, 10], method="mavg", grid=True, binding=True, numbinding=True, excludehframes=(4, [850, 2000]), legend=True)
# complete_sim_results.lineplot_overframes(4, ylim=[-10, 10], method="mavg", grid=True, binding=True, numbinding=True, excludehframes=(4, [850, 2000]), legend=True)
# complete_sim_results.lineplot_overframes(5, ylim=[-10, 10], method="mavg", grid=True, binding=True, numbinding=True, excludehframes=None, legend=True)
# complete_sim_results.multi_densplot1d(0, nbins=15, grid=True, xlim=[-np.pi, np.pi], ylim=[0, 0.4], excludehframes=(4, [850, 2000]), showh=True)
# complete_sim_results.multi_densplot1d(1, nbins=32, grid=True, xlim=[0, np.pi], ylim=[0, 1.3], excludehframes=(4, [850, 2000]), showh=False)
# complete_sim_results.multithetacorrplot(nbins=32, grid=True, xlim=[0, np.pi], ylim=[0, 0.7], excludehframes=(4, [850, 2000]))
# complete_sim_results.theta_corrplot(nbins=32, grid=False, xlim=[0, np.pi], ylim=[0, 0.7])
# complete_sim_results.plot_surfaceprops()
# complete_sim_results.bargraph_binding(count=True)
# complete_sim_results.bargraph_binding(count=False)
# complete_sim_results.multi_histplot2d(3, 4, nbins=[20, 20], grid=False, xlim=[-10, 10], ylim=[-10, 10], excludehframes=(4, [850, 2000]))
# complete_sim_results.histplot_2ddata(3, 4, nbins=[20, 20], grid=False, xlim=[-10, 10], ylim=[-10, 10])

complete_sim_results.multithetacorrplot(hs=[hs[0]], nbins=32, grid=True, xlim=[0, np.pi], ylim=[0, 0.5], excludehframes=(4, [850, 2000]))
# complete_sim_results.multi_densplot1d(0, hs=[hs[0]], nbins=15, grid=True, xlim=[-np.pi, np.pi], ylim=[0, 0.3], excludehframes=(4, [850, 2000]), showh=True)


# %% BaseRepSim

PDB = "/myResults/baseRepSim/trajectory_sim.dat.pdb"
PSF = "/myResults/baseRepSim/mytest.psf"

#make MDAnalysis universe object
try:
    u = MDAnalysis.Universe(windows_loc + PSF, windows_loc + PDB)
except:
    u = MDAnalysis.Universe(linux_loc + PSF, linux_loc + PDB)

# %% analyse
hs = init_handles(u)
for i in range(len(hs)):
    hs[i].code = locCode[i]

baserep_analysis = Analyse.from_universe(u, hs, save=False, rotate=True, name="baseRepSim")
# baserep_analysis.do_primary_frame_analysis(framerange=range(len(u.trajectory)), debugplots=False, save=True)

baserep_results = ResultProcessor(baserep_analysis, "20190609-203055")
# baserep_results.lineplot_overframes(3, ylim=[-10, 10], method="mavg", grid=True, binding=True, numbinding=True, excludehframes=(2, [0, 400]), legend=False)
# baserep_results.lineplot_overframes(4, ylim=[-10, 10], method="mavg", grid=True, binding=True, numbinding=True, excludehframes=(2, [0, 400]), legend=True)
# baserep_results.lineplot_overframes(5, ylim=[-10, 10], method="mavg", grid=True, binding=True, numbinding=True, excludehframes=None, legend=True)
# baserep_results.multi_densplot1d(0, nbins=15, grid=True, xlim=[-np.pi, np.pi], ylim=[0, 0.4], excludehframes=(2, [0, 400]), showh=True)
# baserep_results.multi_densplot1d(1, nbins=25, grid=True, xlim=[0, np.pi], ylim=[0, 1.5], excludehframes=(2, [0, 400]), showh=False)
# baserep_results.multithetacorrplot(nbins=32, grid=True, xlim=[0, np.pi], ylim=[0, 0.7], excludehframes=(2, [0, 400]))
# baserep_results.plot_surfaceprops()
# baserep_results.bargraph_binding(count=True)
# baserep_results.bargraph_binding(count=False)
# baserep_results.multi_histplot2d(3, 4, nbins=[17, 17], grid=False, xlim=[-10, 10], ylim=[-10, 10], excludehframes=(2, [0, 400]))
baserep_results.histplot_2ddata(3, 4, nbins=[17, 17], grid=False, xlim=[-10, 10], ylim=[-10, 10])

# %%
im.reload(resultprocessor)
pathb = "BondLengths/"
pathn = "NeighLengths/"
bnd_results = ResultProcessor(extraresultfiles=[pathb + "myCompleteSim/myCompleteSim.txt",
                                                pathn + "myCompleteSim/myCompleteSim.txt"])
bnd_results.plot_bondlen()


