import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import colors
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.stats as stat
import itertools as it
import operator as op

matplotlib.use('QT5Agg')
font = {'family': 'serif',
        'serif': ['Palatino'],
        'weight': 'light',
        'size': 22}

matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=False)


class ResultProcessor(object):

    def __init__(self, analyse_object=None, rid=None, extraresultfiles=None):
        if analyse_object is not None and rid is not None:
            self.resultdir = "RESULTS/"+analyse_object.name
            self.outdir = "GRAPHS/"+analyse_object.name
            self.handles = analyse_object.handles

            for f in os.listdir(self.resultdir):
                print("Found {}".format(f))
                if rid in f:
                    self.p_file = self.resultdir+"/"+f
            print("Result processing from {}".format(self.p_file))
            inf = open(self.p_file)
            self.p_lines = inf.readlines()
            inf.close()
        else:
            # for processing the output of multiple simulations
            self.p_lines = []
            if extraresultfiles is not None:
                if isinstance(extraresultfiles, list):
                    for f in extraresultfiles:
                        inf = open(f)
                        print("Also processing output from {} !".format(f))
                        self.p_lines.extend(inf.readlines())
                        inf.close()
                else:
                    inf = open(extraresultfiles)
                    print("Also processing output from {} !".format(extraresultfiles))
                    self.p_lines.extend(inf.readlines())

        self.latexn = [r'$\phi$', r'$\theta$', r'$\kappa$', r'$x$', r'$y$', r'$z$', r'Binding State', r'RMSE', r'$N_{x}$', r'N_{y}']
        self.names = ['Phi', 'Theta', 'Kappa', 'X', 'Y', 'Z', 'Binding State', 'RMSE', 'X order', 'Y order']
        self.units = ['(deg.)', '(deg.)', '(deg.)', '(sim. units)', '(sim. units)', '(sim. units)', '-', '-', '-', '-']

    def __get_handle_result(self, i_h, i_d):
        """
        Extracts the data with index i_d for handle with index i_h
        :param i_h: handle index
        :param i_d: data index
        :return: lst: list of the data over time
        """
        lst = []
        for i_f in range(len(self.p_lines)):
            splitted = self.p_lines[i_f].split('\t\t')
            d = splitted[i_h].split(' ')[i_d]
            if i_d == 6:
                lst.append(d)
            else:
                lst.append(float(d))

        return lst

    def multi_densplot1d(self,  i_d, hs=None, nbins='fd', grid=True, xlim=None, ylim=None, excludehframes=None, showh=False):
        latn = self.latexn[i_d]
        unit = self.units[i_d]
        if hs is None:
            hs = self.handles
        fig, ax = plt.subplots(1)
        for i_h in range(len(hs)):
            lst = self.__get_handle_result(i_h, i_d)
            if i_d <= 2:
                lst = np.deg2rad(lst)
            if i_h == excludehframes[0]:
                begin, end = excludehframes[1]
                lst = lst[begin:end]
            if nbins is not 'fd':
                xtot = xlim[1] - xlim[0]
                binstep = xtot / nbins
                binsarray = np.arange(xlim[0], xlim[1] + np.pi / 128, binstep)
            else:
                binsarray = 'fd'
            p, bins = np.histogram(lst, bins=binsarray, density=True)
            print("#bins used = {}".format(len(bins) - 1))
            x, y = [], []
            area = 0
            for i in range(len(p)):
                binmid = (bins[i] + bins[i + 1]) / 2
                x.append(binmid)
                y.append(p[i])
                area += (bins[i + 1] - bins[i]) * p[i]
            print(area)
            ax.plot(x, y, color=hs[i_h].draw_colour, linewidth=3, linestyle='-', marker='o', markersize=10,
                           label="Handle {}".format(hs[i_h].code))
        if showh:
            ax.axhline(1 / (2 * np.pi), color='r', linewidth=3, linestyle='--',
                       label=r"$P(${}$)$".format(latn) + r" = $\frac{1}{2\pi}$ $rad.^{-1}$")
        ax.title.set_text("{} Probability Density".format(latn))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.001, 0.1))
        ax.set_xticks(np.arange(xlim[0], xlim[1] + np.pi / 128, np.pi / 8))
        ticks = [r"$-\pi$", r"$-\frac{7\pi}{8}$", r"$-\frac{3\pi}{4}$", r"$-\frac{5\pi}{8}$", r"$-\frac{\pi}{2}$",
                            r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$",
                            "$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$",
                            r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$"]
        if i_d == 1:
            ticks = ticks[8:]
        ax.set_xticklabels(ticks)

        ax.title.set_text("{} Probability Density".format(latn))

        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.1)
        ax.legend(loc='best')
        fig.text(0.5, 0.01, "{} {}".format(latn, " (rad.)"), ha='center')
        fig.text(0.04, 0.6, r"$P(${}$)$".format(latn) + " (rad.$^{-1}$)", rotation='vertical')
        plt.grid(grid)
        plt.show()

        fig.set_size_inches(16, 9)
        path = self.outdir + "/{}/multi{}new.png".format(self.names[i_d], self.names[i_d])
        ResultProcessor.__mkdirs(path)
        print("Saved {}.".format(path))
        fig.savefig(path, dpi=100)


    def multi_histplot1d(self, i_d, nbins='fd', grid=True, xlim=None, ylim=None,
                         density=True, excludehframes=None, thetacorr=False):
        latn = self.latexn[i_d]
        unit = self.units[i_d]
        hs = self.handles
        plotdata = []
        labellist = []
        colourlist = []
        fig, ax = plt.subplots(1, tight_layout=True)
        for i_h in range(len(hs)):
            lst = self.__get_handle_result(i_h, i_d)
            if i_h == excludehframes[0]:
                begin, end = excludehframes[1]
                lst = lst[begin:end]
            label = "Handle " + hs[i_h].code
            colour = hs[i_h].draw_colour
            labellist.append(label)
            colourlist.append(colour)
            if density:
                kde = stat.gaussian_kde(lst, bw_method='scott')
                print("Bandwith factor = {}".format(kde.factor))
                # label += " $bw$ = {}".format(round(kde.factor, 2))
                xs = np.linspace(xlim[0], xlim[1] + 90, 200)
                if i_d == 1 and thetacorr:
                    cor = np.sin(np.deg2rad(xs))
                    mean = np.mean(kde(xs))
                    dens = kde.evaluate(xs) / cor
                else:
                    dens = kde.evaluate(xs)
                    mean = np.mean(dens)

                ax.plot(xs, dens, linewidth=3, color=colour, label=label)
            else:
                plotdata.append(lst)
        if density:
            plt.ylabel('$P(${}$)$ ({}'.format(latn, unit[1:-1])+r'$^{-1}$)')
            title = r'Gaussian Kernel Density Estimate of {}'.format(latn)

            if excludehframes is not None:
                if thetacorr:
                    path = self.outdir + "/{}/{}multi1ddensCorexcl{}.png".format(self.names[i_d], self.names[i_d],
                                                                          excludehframes)
                    title += " (Corrected)"
                else:
                    path = self.outdir + "/{}/{}multi1ddensexcl{}.png".format(self.names[i_d], self.names[i_d], excludehframes)
            else:
                if thetacorr:
                    path = self.outdir + "/{}/{}multi1ddensCor.png".format(self.names[i_d], self.names[i_d])
                    title += " (Corrected)"
                else:
                    path = self.outdir + "/{}/{}multi1ddens.png".format(self.names[i_d], self.names[i_d])
            plt.title(title)
            ax.grid(grid)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend()
        else:
            N, bins, patches = ax.hist(plotdata, histtype='bar', bins=nbins,
                                       color=colourlist, label=labellist, edgecolor='k', range=xlim)
            ax.legend()
            print("#bins used = {}".format(len(bins) - 1))
            plt.ylabel(r'$n_{occurences}$')
            plt.title(r'{} Histogram'.format(latn))
            path = self.outdir + "/{}/{}multi1dhist.png".format(self.names[i_d], self.names[i_d])

        plt.xlabel("{} {}".format(latn, unit))
        plt.show()

        fig.set_size_inches(16, 9)
        ResultProcessor.__mkdirs(path)
        print("Saved {}.".format(path))
        fig.savefig(path, dpi=100)

    def theta_corrplot(self, i_d=1, nbins='fd', grid=True, xlim=None, ylim=None, excludehframes=4):
        fig = plt.figure()
        gs = gridspec.GridSpec(6, 7, wspace=0.5, hspace=2.2)
        cols = [(0, 2), (2, 4), (4, 6), (1, 3), (3, 5)]
        latn = self.latexn[i_d]
        hs = self.handles
        for i_h in range(len(hs)):
            if i_h <= 2:
                curr_ax = fig.add_subplot(gs[:3, cols[i_h][0]:cols[i_h][1]])
            else:
                curr_ax = fig.add_subplot(gs[3:, cols[i_h][0]:cols[i_h][1]])

            theta = self.__get_handle_result(i_h, i_d)
            theta_rad = np.deg2rad(theta)
            if i_h == excludehframes:
                sli = int(len(theta_rad)/2)
                theta_rad = theta_rad[sli:]
            if nbins is not 'fd':
                xtot = xlim[1] - xlim[0]
                binstep = xtot / nbins
                print(xtot, nbins, binstep)
                binsarray = np.arange(xlim[0], xlim[1] + np.pi/128, binstep)
            else:
                binsarray = 'fd'
            ptheta, bins, bars = curr_ax.hist(theta_rad, align='mid', density=True, bins=binsarray, range=xlim, edgecolor='k', color=hs[i_h].draw_colour)
            area = 0
            for i in range(len(ptheta)):
                # area of ptheta * sin(theta)
                area += (bins[i + 1] - bins[i]) * bars[i].get_height()
                binmid = (bins[i] + bins[i+1])/2
                bars[i].set_height(bars[i].get_height()/np.sin(binmid))
            # calc how many times the AUC is higher than 1/(2*pi)
            multiplier = area / (1/(2*np.pi))
            _ = [b.set_height(b.get_height()/multiplier) for b in bars]
            newarea = 0
            for i in range(len(ptheta)):
                newarea += (bins[i+1] - bins[i])*bars[i].get_height()
            print(newarea)
            lower = curr_ax.axhline(1/(4*np.pi), color='k', linewidth=3, linestyle='--', label=r"$P(\theta) = \frac{1}{4\pi}$")
            upper = curr_ax.axhline(1/(2*np.pi), color='r', linewidth=3, linestyle='--', label=r"$P(\theta) = \frac{1}{2\pi}$")
            curr_ax.title.set_text("Handle {}".format(hs[i_h].code))
            curr_ax.set_xlim(xlim)
            curr_ax.set_ylim(ylim)
            curr_ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.001, 0.1))
            curr_ax.set_xticks(np.arange(xlim[0], xlim[1] + np.pi/128, np.pi/8))
            curr_ax.set_xticklabels(["$0$", r"$\frac{\pi}{8}$",  r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$",
                                     r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$"])
            if i_h in [1, 2, 4]:
                plt.setp(curr_ax, yticklabels=[])
            print("#bins used = {}".format(len(bins) - 1))
        fig.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)
        fig.legend(handles=(upper, lower), loc='lower right')
        plt.suptitle("{} Probability Density (Corrected)".format(latn))
        fig.text(0.5, 0.01, "{} {}".format(latn, "(rad.)"), ha='center')
        fig.text(0.04, 0.6, r"$P(\theta)$ (rad.$^{-2}$)", rotation='vertical')
        plt.show()

        fig.set_size_inches(16, 9)
        path = self.outdir + "/{}/multi{}Corrhandle.png".format(self.names[i_d], self.names[i_d])
        ResultProcessor.__mkdirs(path)
        print("Saved {}.".format(path))
        fig.savefig(path, dpi=100)

    def multithetacorrplot(self, hs=None, i_d=1, nbins='fd', grid=True, xlim=None, ylim=None, excludehframes=(2, [0, 400])):
        fig, ax = plt.subplots()
        latn = self.latexn[i_d]
        if hs is None:
            hs = self.handles
        for i_h in range(len(hs)):
            theta = self.__get_handle_result(i_h, i_d)
            theta_rad = np.deg2rad(theta)
            if i_h == excludehframes[0]:
                begin, end = excludehframes[1]
                theta_rad = theta_rad[begin:end]
            if nbins is not 'fd':
                xtot = xlim[1] - xlim[0]
                binstep = xtot / nbins
                binsarray = np.arange(xlim[0], xlim[1] + np.pi/128, binstep)
            else:
                binsarray = 'fd'
            # ptheta, bins, bars = plt.hist(theta_rad, align='mid', density=True, bins=binsarray, range=xlim, edgecolor='k', color=hs[i_h].draw_colour)
            ptheta, bins = np.histogram(theta_rad, bins=binsarray, density=True)
            area = 0
            x = []
            for i in range(len(ptheta)):
                # area of ptheta * sin(theta)
                area += (bins[i + 1] - bins[i]) * ptheta[i]
                binmid = (bins[i] + bins[i+1])/2
                ptheta[i] = ptheta[i]/np.sin(binmid)
                x.append(binmid)
            # calc how many times the AUC is higher than 1/(2*pi)
            multiplier = area / (1/(2*np.pi))
            y = np.array(ptheta)/multiplier

            ax.plot(x, y, color=hs[i_h].draw_colour, linewidth=3, linestyle='-', marker='o', markersize=10, label="Handle {}".format(hs[i_h].code))
            ax.title.set_text("{} Probability Density (Corrected)".format(latn))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.001, 0.1))
            ax.set_xticks(np.arange(xlim[0], xlim[1] + np.pi/128, np.pi/8))
            ax.set_xticklabels(["$0$", r"$\frac{\pi}{8}$",  r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$",
                                     r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$"])
            print("#bins used = {}".format(len(bins) - 1))
        ax.axhline(1 / (4 * np.pi), color='k', linewidth=3, linestyle='--',
                           label=r"$P'(\theta) = \frac{1}{4\pi}$ $rad.^{-2}$")
        ax.axhline(1 / (2 * np.pi), color='r', linewidth=3, linestyle='--',
                           label=r"$P'(\theta) = \frac{1}{2\pi}$ $rad.^{-2}$")
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.1)
        ax.legend(loc='best')
        fig.text(0.5, 0.01, "{} {}".format(latn, "(rad.)"), ha='center')
        fig.text(0.04, 0.6, r"$P'(\theta)$ (rad.$^{-2}$)", rotation='vertical')
        plt.grid(grid)
        plt.show()

        fig.set_size_inches(16, 9)
        path = self.outdir + "/{}/multi{}Corr.png".format(self.names[i_d], self.names[i_d])
        ResultProcessor.__mkdirs(path)
        print("Saved {}.".format(path))
        fig.savefig(path, dpi=100)


    def histplot_1ddata(self, i_d, nbins='fd', grid=True, xlim=None):
        latn = self.latexn[i_d]
        unit = self.units[i_d]
        hs = self.handles

        for i_h in range(len(hs)):
            lst = self.__get_handle_result(i_h, i_d)

            fig, ax = plt.subplots(1, tight_layout=True)
            N, bins, patches = ax.hist(lst, bins=nbins, edgecolor='k', range=xlim)

            print("#bins used = {}".format(len(bins) - 1))

            fracs = N / N.max()
            norm = colors.Normalize(fracs.min(), fracs.max())
            viridis = cm.get_cmap('viridis')
            for thisfrac, thispatch in zip(fracs, patches):
                color = viridis(norm(thisfrac))
                thispatch.set_facecolor(color)

            ax.set_xlim([min(lst), max(lst)])
            plt.xlabel("{} {}".format(latn, unit))
            plt.ylabel(r'$n_{occurences}$')
            plt.title(r'{} Histogram for handle {}'.format(latn, hs[i_h].code))
            plt.grid(grid)
            plt.show()

            fig.set_size_inches(16, 9)
            path = self.outdir + "/{}/1dhist{}handle{}.png".format(self.names[i_d], self.names[i_d], hs[i_h].code)
            ResultProcessor.__mkdirs(path)
            print("Saved {}.".format(path))
            fig.savefig(path, dpi=100)

    def histplot_2ddata(self, i_d1, i_d2, nbins='fd', grid=True, xlim=None, ylim=None):
        latn1 = self.latexn[i_d1]
        latn2 = self.latexn[i_d2]
        unit1 = self.units[i_d1]
        unit2 = self.units[i_d2]
        hs = self.handles

        for i_h in range(len(hs)):
            lst1 = self.__get_handle_result(i_h, i_d1)
            lst2 = self.__get_handle_result(i_h, i_d2)
            if nbins == 'fd':
                _, binsx, _ = plt.hist(lst1, bins='fd')
                _, binsy, _ = plt.hist(lst2, bins='fd')
                nxbins = len(binsx) - 1
                nybins = len(binsy) - 1
                plt.close()
            else:
                nxbins, nybins = nbins[0], nbins[1]

            fig, ax = plt.subplots(1, tight_layout=True)
            counts, xedges, yedges, im = ax.hist2d(lst1, lst2, bins=(nxbins, nybins), range=[xlim, ylim])
            ax.scatter(np.mean(lst1), np.mean(lst2), s=150, color='r', marker='o', label="Mean $(x, y)$")
            print("HANDLE {}, mean_x = {}, mean_y = {}".format(hs[i_h].code, np.mean(lst1), np.mean(lst2)))
            ax.scatter(0, 0, s=300, color='w', marker='+')
            print("#bins used = ({}, {})".format(nxbins, nybins))

            fig.legend()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.colorbar(im, ax=ax)

            plt.xlabel("{} {}".format(latn1, unit1))
            plt.ylabel("{} {}".format(latn2, unit2))
            plt.title(r'{}, {} 2D Histogram for handle {}'.format(latn1, latn2, hs[i_h].code))
            plt.grid(grid)
            plt.show()
            fig.subplots_adjust(left=0.1, right=1, top=0.95, bottom=0.05)

            fig.set_size_inches(12, 9)
            folname = self.names[i_d1]+self.names[i_d2]
            path = self.outdir + "/{}/2dhist{}handle{}.png".format(folname, folname, hs[i_h].code)
            ResultProcessor.__mkdirs(path)
            print("Saved {}.".format(path))
            fig.savefig(path, dpi=100)

    def multi_histplot2d(self, i_d1, i_d2, nbins='fd', grid=True, xlim=None, ylim=None, excludehframes=None):
        latn1 = self.latexn[i_d1]
        latn2 = self.latexn[i_d2]
        unit1 = self.units[i_d1]
        unit2 = self.units[i_d2]
        hs = self.handles

        fig = plt.figure()
        gs = gridspec.GridSpec(6, 7, wspace=0.4, hspace=1.5)
        cols = [(0, 2), (2, 4), (4, 6), (1, 3), (3, 5)]
        images = []
        for i_h in range(len(hs)):
            lst1 = self.__get_handle_result(i_h, i_d1)
            lst2 = self.__get_handle_result(i_h, i_d2)
            if excludehframes is not None and i_h == excludehframes[0]:
                begin, end = excludehframes[1]
                lst1 = lst1[begin:end]
                lst2 = lst2[begin:end]
            if nbins == 'fd':
                tempfig, tempax = plt.subplots(1)
                _, binsx, _ = tempax.hist(lst1, bins='fd', range=xlim)
                _, binsy, _ = tempax.hist(lst2, bins='fd', range=ylim)
                nxbins = len(binsx) - 1
                nybins = len(binsy) - 1
                plt.close()
            else:
                nxbins, nybins = nbins[0], nbins[1]
            if i_h <= 2:
                curr_ax = fig.add_subplot(gs[:3, cols[i_h][0]:cols[i_h][1]])
            else:
                curr_ax = fig.add_subplot(gs[3:, cols[i_h][0]:cols[i_h][1]])
            counts, xedges, yedges, im = curr_ax.hist2d(lst1, lst2, bins=(nxbins, nybins), range=[xlim, ylim])
            m = curr_ax.scatter(np.mean(lst1), np.mean(lst2), s=150, color='r', marker='o', label="Mean $(x, y)$")
            o = curr_ax.scatter(0, 0, s=300, color='w', marker='+')
            images.append(im)
            if i_h in [1, 2, 4]:
                plt.setp(curr_ax, yticklabels=[])
            curr_ax.title.set_text("Handle {}".format(hs[i_h].code))
            print("#bins used = ({}, {})".format(nxbins, nybins))

        plt.suptitle(r'{}, {} 2D Histograms'.format(latn1, latn2))
        plt.grid(grid)
        fig.legend(handles=(m, o))
        fig.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        # global colourbar
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        fig.colorbar(images[0], cax=cbar_ax)

        fig.text(0.5, 0.01, "{} {}".format(latn1, unit1), ha='center')
        fig.text(0.01, 0.5, "{} {}".format(latn2, unit2), rotation='vertical')
        fig.tight_layout()
        plt.show()

        fig.set_size_inches(16, 10)
        folname = self.names[i_d1]+self.names[i_d2]
        if excludehframes is not None:
            path = self.outdir + "/{}/multi2dhist{}excl{}.png".format(folname, folname, excludehframes)
        else:
            path = self.outdir + "/{}/multi2dhist{}.png".format(folname, folname)
        ResultProcessor.__mkdirs(path)
        fig.savefig(path, dpi=100)
        print("Saved {}.".format(path))

    def lineplot_overframes(self, i_d, method="mavg", win_size=20, sigma=5, grid=True,
                            binding=False, numbinding=False, excludehframes=None, legend=False, ylim=None):
        latn = self.latexn[i_d]
        unit = self.units[i_d]
        hs = self.handles

        for i_h in range(len(hs)):
            lst = self.__get_handle_result(i_h, i_d)
            t = range(len(lst))
            if excludehframes is not None and i_h == excludehframes[0]:
                begin, end = excludehframes[1]
                lst = lst[begin:end]
                t = t[begin:end]
            fig, ax = plt.subplots(1)
            line = ResultProcessor.__make_moving_avgplot(ax, t, lst, method, win_size=win_size, sigma=sigma)
            ax.axhline(np.mean(lst), linewidth=3, color='r', linestyle='--', label='Mean')
            ax.text((t[0]+t[-1])/2, 8, r"$\mu$ = {} {}".format(round(np.mean(lst), 2), unit[1:-1]), horizontalalignment='center')
            ax.set_xlim([t[0], t[-1]])
            ax.set_ylim(ylim)
            if binding:
                x_data = np.array(line[0].get_xdata())
                y_data = np.array(line[0].get_ydata())
                # line.pop(0).remove()
                bind_data = self.__get_handle_result(i_h, 6)
                if excludehframes is not None and i_h == excludehframes[0]:
                    begin, end = excludehframes[1]
                    bind_data = bind_data[begin:end]
                _, _, _, _, \
                unbndi, onebndi, twobndi, threebndi, \
                _1bndi, _2bndi, _3bndi, \
                _, _, _ = ResultProcessor.__extract_bnddata(bind_data)

                if numbinding:
                    for i in unbndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='k', label='Unbound')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='k', label='Unbound')
                    for i in onebndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='g', label='One bound base')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='g', label='One bound base')
                    for i in twobndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='b', label='Two bound bases')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='b', label='Two bound bases')
                    for i in threebndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='r', label='Three bound bases')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='r', label='Three bound bases')
                    if legend:
                        leg = ax.legend(['-', 'Mean', 'Unbound', 'One bound base', 'Two bound bases', 'Three bound bases'], loc='best')
                        colors = ['grey', 'red' ,'black', 'green', 'blue', 'red']
                        for i, j in enumerate(leg.legendHandles):
                            j.set_color(colors[i])
                else:
                    for i in unbndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='k', label='Unbound')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='k', label='Unbound')
                    for i in _1bndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='g', label='Base 1 bound')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='g', label='Base 1 bound')
                    for i in _2bndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='b', label='Base 2 bound')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='b', label='Base 2 bound')
                    for i in _3bndi:
                        ax.plot(x_data[i], y_data[i], linewidth=3, linestyle='-', color='r', label='Base 3 bound')
                        if isinstance(i, int):
                            ax.scatter(x_data[i], y_data[i], s=5, marker='s', color='r', label='Base 3 bound')
                    if legend:
                        leg = ax.legend(['-', 'Unbound', 'Base 1 bound', 'Base 2 bound', 'Base 3 bound'])
                        colors = ['grey', 'black', 'green', 'blue', 'red']
                        for i, j in enumerate(leg.legendHandles):
                            j.set_color(colors[i])
            if method == "mavg":
                n = r"Moving Average ($N$ = {})".format(win_size)
            elif method == "gau":
                n = r"Gaussian Smoothed ($\sigma$ = {})".format(sigma)
            plt.title(r'{} of {} for Handle {}'.format(n, self.names[i_d], hs[i_h].code))
            plt.ylabel("{} {}".format(latn, unit))
            plt.xlabel("Frame number")
            plt.grid(grid)
            plt.show()
            fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.1)
            if excludehframes is not None and i_h == excludehframes[0]:
                path = self.outdir + "/FramePlots/{}handle{}excl.png".format(self.names[i_d], hs[i_h].code)
            else:
                path = self.outdir + "/FramePlots/{}handle{}.png".format(self.names[i_d], hs[i_h].code)
            ResultProcessor.__mkdirs(path)
            fig.set_size_inches(16, 9)
            print("Saved {}.".format(path))
            fig.savefig(path, dpi=100)

    def bargraph_binding(self, count=True):
        hs = self.handles

        fig, ax = plt.subplots(1, tight_layout=True)
        for i_h in range(len(hs)):
            lst = self.__get_handle_result(i_h, 6)
            unbndcnt, onebndcnt, twobndcnt, threebndcnt, \
                _, _, _, _, \
                _, _, _, \
                _1bndcnt, _2bndcnt, _3bndcnt = ResultProcessor.__extract_bnddata(lst)

            w = 0.3
            if count:
                labels = ["No Bases Bound", "One Base Bound", "Two Bases Bound", "Three Bases Bound"]
                cnts = [unbndcnt, onebndcnt, twobndcnt, threebndcnt]
                ind = 2*np.arange(len(labels))
                ax.bar(ind + w*i_h, cnts, width=w, label="Handle {}".format(hs[i_h].code),
                       color=hs[i_h].draw_colour, edgecolor='k')
                ax.set_xticks(ind + w*2)
                ax.set_xticklabels(labels)
                ax.legend()
            else:
                labels = ["Base 1 Bound", "Base 2 Bound", "Base 3 Bound"]
                cnts = [_1bndcnt, _2bndcnt, _3bndcnt]
                ind = 2*np.arange(len(labels))
                ax.bar(ind + w*i_h, cnts, width=w, label="Handle {}".format(hs[i_h].code),
                       color=hs[i_h].draw_colour, edgecolor='k')
                ax.set_xticks(ind + w*2)
                ax.set_xticklabels(labels)
                ax.legend()

        plt.ylabel(r'$n_{occurrences}$')
        plt.title("Binding State Bar Graph")
        plt.show()
        fig.set_size_inches(16, 9)
        if count:
            path = self.outdir + "/BNDcountbargraph.png"
        else:
            path = self.outdir + "/BNDstatebargraph.png"
        ResultProcessor.__mkdirs(path)
        print("Saved {}.".format(path))
        fig.savefig(path, dpi=100)

    def plot_bondlen(self):
        """
        First half of p_lines needs to be bond lengths, second half neighbour lengths
        :return: None
        """
        sep = int(len(self.p_lines)/2)
        bonddata = self.p_lines[:sep]
        neighdata = self.p_lines[sep:]

        bondmin = [float(l[:-1].split(' ')[0]) for l in bonddata]
        bondmax = [float(l[:-1].split(' ')[1]) for l in bonddata]
        bondmean = [float(l[:-1].split(' ')[2]) for l in bonddata]

        bondmeanmean = np.mean(bondmean)
        bondminmean = np.mean(bondmin)
        bondmaxmean = np.mean(bondmax)
        bondabsmax = np.max(bondmax)

        neighmin = [float(l[:-1].split(' ')[0]) for l in neighdata]
        neighmax = [float(l[:-1].split(' ')[1]) for l in neighdata]
        neighmean = [float(l[:-1].split(' ')[2]) for l in neighdata]

        neighmeanmean = np.mean(neighmean)
        neighminmean = np.mean(neighmin)
        neighmaxmean = np.mean(neighmax)
        neighabsmin = np.min(neighmin)

        n = len(bonddata)
        x = range(n)

        fig, ax = plt.subplots(1)

        ax.scatter(x, bondmean, marker='.', color='g', label="Bound Bases")
        ax.scatter(x, neighmean, marker='.', color='orange', label="Neighbour Bases")

        ax.axhline(bondmeanmean, color='k', linewidth=3)
        ax.axhline(neighmeanmean, color='k', linewidth=3)

        ax.axhline(bondmaxmean,  color='r', linestyle='--', linewidth=3)
        ax.axhline(neighmaxmean, color='r', linestyle='--', linewidth=3)

        ax.axhline(bondminmean,  color='b', linestyle='--', linewidth=3)
        ax.axhline(neighminmean, color='b', linestyle='--', linewidth=3)
        ax.axhline(0.69, color='c', linestyle=':', linewidth=5)

        ax.fill_between(x, bondminmean, bondmaxmean, color='g', alpha=0.4)
        ax.fill_between(x, neighminmean, neighmaxmean, color='orange', alpha=0.4)
        ax.fill_between(x, neighminmean, bondmaxmean, color='r', alpha=0.5 )

        ax.axhline(neighabsmin, color='r', linewidth=3)
        ax.axhline(bondabsmax, color='b', linewidth=3)
        ax.set_xlim([x[0], x[-1]])

        plt.title("Handle inter-base distances")
        ax.legend()
        ax.set_ylabel("Distance (sim. units.)")
        ax.set_xlabel("Handle #")

        fig.set_size_inches(16, 9)
        plt.show()
        path = "GRAPHS/BondLenPlot.png"
        ResultProcessor.__mkdirs(path)
        print("Saved {}.".format(path))
        fig.savefig(path, dpi=100)

    def plot_surfaceprops(self):
        rmse = self.__get_handle_result(0, 7)
        rmsd = self.__get_handle_result(0, 10)
        x_order = self.__get_handle_result(0, 8)
        y_order = self.__get_handle_result(0, 9)
        fig, axes = plt.subplots(ncols=2, nrows=2, tight_layout=False)
        n = len(rmse)
        x = range(n)
        axes = axes.flatten()
        axes[0].plot(x, rmse, linewidth=2, color='k', label='RMSE')
        axes[0].set_xlim([x[0], x[-1]])
        axes[0].title.set_text(r'Surface Fit $RMSE$')
        axes[0].set_xlabel("Frame Number")
        axes[0].set_ylabel("$RMSE$ (sim. units)")

        axes[1].plot(x, rmsd, linewidth=2, color='k', label='RMSE')
        axes[1].set_xlim([x[0], x[-1]])
        axes[1].title.set_text(r"System's $RMSD$")
        axes[1].set_xlabel("Frame Number")
        axes[1].set_ylabel(r"$RMSD$ (sim. units)")

        bins = np.arange(1, 9, 1) - 0.5
        axes[2].hist(x_order, bins=bins, color='r', edgecolor='k', align='mid')
        axes[2].set_xticks(bins + 0.5)
        axes[2].title.set_text(r"Surface $x'$ Order Histogram")
        axes[2].set_xlabel(r"$N_{x'}$")
        axes[2].set_ylabel(r'$n_{occurrences}$')

        axes[3].hist(y_order, bins=bins, color='b', edgecolor='k', align='mid')
        axes[3].set_xticks(bins + 0.5)
        axes[3].title.set_text(r"Surface $y'$ Order Histogram")
        axes[3].set_xlabel(r"$N_{y'}$")
        axes[3].set_ylabel(r'$n_{occurrences}$')
        plt.suptitle("Surface Properties")
        plt.subplots_adjust(hspace=0.4)
        fig.set_size_inches(18, 9)
        plt.show()
        path = self.outdir + "/SurfPropPlot.png"
        ResultProcessor.__mkdirs(path)
        print("Saved {}.".format(path))
        fig.savefig(path, dpi=100)


    @staticmethod
    def __extract_bnddata(lst):
        unbndcnt = 0        # nothing bound counter
        onebndcnt = 0       # one base is bound counter
        twobndcnt = 0       # two bases are bound counter
        threebndcnt = 0     # three bases are bound counter

        _1bndcnt = 0        # base 1 is bound counter
        _2bndcnt = 0        # base 2 is bound counter
        _3bndcnt = 0        # base 3 is bound counter

        unbndi = []     # nothing bound frame indexes
        onebndi = []    # one base bound frame indexes
        twobndi = []    # two bases bound frame indexes
        threebndi = []  # three bases are bound frame indexes

        _1bndi = []     # base 1 bound frame indexes
        _2bndi = []     # base 2 bound frame indexes
        _3bndi = []     # base 3 bound frame indexes
        i = 0
        for s in lst:
            if "1" in s:
                _1bnd = True
                _1bndcnt += _1bnd
                _1bndi.append(i)
            else:
                _1bnd = False
            if "2" in s:
                _2bnd = True
                _2bndcnt += _2bnd
                _2bndi.append(i)
            else:
                _2bnd = False
            if "3" in s:
                _3bnd = True
                _3bndcnt += _3bnd
                _3bndi.append(i)
            else:
                _3bnd = False

            tot_bnd = sum([_1bnd, _2bnd, _3bnd])

            if tot_bnd == 0:
                unbndcnt += 1
                unbndi.append(i)
            elif tot_bnd == 1:
                onebndcnt += 1
                onebndi.append(i)
            elif tot_bnd == 2:
                twobndcnt += 1
                twobndi.append(i)
            elif tot_bnd == 3:
                threebndcnt += 1
                threebndi.append(i)

            i += 1

        # convert the index lists to list of list where sublist only has consecutive indexes
        unbndi = ResultProcessor.__splice_indexlist(unbndi)
        onebndi = ResultProcessor.__splice_indexlist(onebndi)
        twobndi = ResultProcessor.__splice_indexlist(twobndi)
        threebndi = ResultProcessor.__splice_indexlist(threebndi)
        _1bndi = ResultProcessor.__splice_indexlist(_1bndi)
        _2bndi = ResultProcessor.__splice_indexlist(_2bndi)
        _3bndi = ResultProcessor.__splice_indexlist(_3bndi)

        return unbndcnt, onebndcnt, twobndcnt, threebndcnt,\
            unbndi, onebndi, twobndi, threebndi,\
            _1bndi, _2bndi, _3bndi, \
            _1bndcnt, _2bndcnt, _3bndcnt

    @staticmethod
    def __splice_indexlist(indexlist):
        """
        Changes a list of indexes to a list of lists,
        where every sublist has successive indexes.
        :param indexlist: the list of indexes
        :return:
        """
        newl = []
        i = 1
        j = 0
        if len(indexlist) == 1:
             newl = indexlist
        else:
            for k, g in it.groupby(enumerate(indexlist), lambda ix: ix[0] - ix[1]):
                newl.append(list(map(op.itemgetter(1), g)))

        return newl


    @staticmethod
    def __make_moving_avgplot(ax, x, y, method, win_size=20, sigma=5):
        # make the masking filters
        avg_mask = np.ones(win_size) / win_size
        gau_x = np.linspace(-2.7 * sigma, 2.7 * sigma, 6 * sigma)
        gau_mask = ResultProcessor.__gaussfunc(gau_x, sigma)

        y_avg = np.convolve(y, avg_mask, mode='same')
        y_gau = np.convolve(y, gau_mask, mode='same')
        if method == "gau":
            line = ax.plot(x, y_gau, linewidth=2, color='grey', label='Gauss. Conv.')
            ax.fill_between(x, y_gau, y, where=y >= y_gau, facecolor='blue', alpha=0.4, interpolate=True)
            ax.fill_between(x, y_gau, y, where=y <= y_gau, facecolor='blue', alpha=0.4, interpolate=True)

        elif method == "mavg":
            line = ax.plot(x, y_avg, linewidth=2, color='grey', label='Moving Avg.')
            ax.fill_between(x, y_avg, y, where=y >= y_avg, facecolor='blue', alpha=0.4, interpolate=True)
            ax.fill_between(x, y_avg, y, where=y <= y_avg, facecolor='blue', alpha=0.4, interpolate=True)

        return line

    @staticmethod
    def __gaussfunc(x, sigma):
        return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x ** 2) / (2 * sigma ** 2))

    @staticmethod
    def __mkdirs(path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != exc.errno.EEXIST:
                    raise
