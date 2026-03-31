from model import DefectSquareLattice, plot_disorder_figure, plot_bott_phase_diagram
from matplotlib import pyplot as plt


if __name__ == "__main__":
    doInterpolation = False # Max be slow : also, be sure to view the computation without interpolation, as interpolation often causes artifacts such as turning a single peak into two peaks.
    doShow = False # Warning: may lag
    doSavePng = True 
    doSaveSvg = False
    save_directory = './Plots/'

    doFig3 = False
    doFig4 = False
    doFig5 = False
    doFig6 = False
    doFig7 = False
    doFig13 = True # Computation intensive, data is included in repo.
    doBottPhaseDiagram = False

    Lx = Ly = 25 # Adjust for quicker/slower computation. Figures in the manuscript are done using Lx = Ly = 25 

    # Figure 3
    if doFig3:
        Lattice = DefectSquareLattice(Lx, Ly, "vacancy", True)
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=True, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig3.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig3.svg')
        if doShow: plt.show(); plt.close('all')

    # Figure 4 (a-d)
    if doFig4:
        Lattice = DefectSquareLattice(Lx - 1, Ly - 1, "schottky", True, schottky_distance = 1)
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=True, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig4i.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig4i.svg')
        if doShow: plt.show(); plt.close('all')

    # Figure 4 (e-h)
    if doFig4:
        Lattice = DefectSquareLattice(Lx - 1, Ly - 1, "schottky", True, schottky_distance = 11)
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=True, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig4ii.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig4ii.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 5 (i)
    if doFig5:
        Lattice = DefectSquareLattice(Lx, Ly, "substitution", True)
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=False, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig5i.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig5i.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 5 (ii)
    if doFig5:
        Lattice = DefectSquareLattice(Lx, Ly, "substitution", True)
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=True, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig5ii.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig5ii.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 6 (i)
    if doFig6:
        Lattice = DefectSquareLattice(Lx - 1, Ly - 1, "interstitial", True)
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=False, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig6i.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig6i.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 6 (ii)
    if doFig6:
        Lattice = DefectSquareLattice(Lx, Ly, "frenkel_pair", True)
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=False, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig6ii.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig6ii.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 7 (a-b)
    if doFig7:
        Lattice = DefectSquareLattice(25, 25, 'vacancy', True, doSquareDefect=True)
        Lattice.plot_spectrum_ldos([1.0, 2.5], doInterpolation=doInterpolation, plot_type='imshow')
        if doSavePng: plt.savefig(save_directory + 'fig7ab.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig7ab.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 7 (c)
    if doFig7:
        Lattice = DefectSquareLattice(25, 25, 'substitution', True, doSquareDefect=True, sqdWidth=5)
        Lattice.plot_spectrum_ldos([1.0], [2.5], doInterpolation=doInterpolation, plot_type='imshow')
        if doSavePng: plt.savefig(save_directory + 'fig7c.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig7c.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 7 (d)
    if doFig7:
        Lattice = DefectSquareLattice(25, 25, 'substitution', True, doSquareDefect=True, sqdWidth=17)
        Lattice.plot_spectrum_ldos([2.5], [1.0], doInterpolation=doInterpolation, plot_type='imshow')
        if doSavePng: plt.savefig(save_directory + 'fig7d.png')
        if doSaveSvg: plt.savefig(save_directory + 'fig7d.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 13 (a-e)
    if doFig13: 
        plot_disorder_figure(1.0, -1.0, Lx, Ly, 'onsite', 0.50, 50, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig13i.png')
        if doSavePng: plt.savefig(save_directory + 'fig13i.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 13 (f-j)
    if doFig13: 
        plot_disorder_figure(1.0, -1.0, Lx, Ly, 'mass', 0.25, 50, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig13ii.png')
        if doSavePng: plt.savefig(save_directory + 'fig13ii.svg')
        if doShow: plt.show(); plt.close('all')

    # Fig 13 (k-o)
    if doFig13: 
        plot_disorder_figure(1.0, -1.0, Lx, Ly, 'hopping', 0.25, 50, doInterpolation=doInterpolation)
        if doSavePng: plt.savefig(save_directory + 'fig13iii.png')
        if doSavePng: plt.savefig(save_directory + 'fig13iii.svg')
        if doShow: plt.show(); plt.close('all')


    # Phase diagram for a given r0 and R (see Eq. 6)
    if doBottPhaseDiagram:
        plot_bott_phase_diagram((-5., 5.), 51, 1, 1)
        if doSavePng: plt.savefig(save_directory + 'bott_phase_diagram.png')
        if doSaveSvg: plt.savefig(save_directory + 'bott_phase_diagram.svg')
        if doShow: plt.show(); plt.close('all')