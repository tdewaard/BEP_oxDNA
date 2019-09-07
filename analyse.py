from fit import Fit
from visualisation import Visualisation as vis
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
import subprocess as sub
import time
import handle
import fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math as m
import MDAnalysis
import os


class Analyse(object):
    def __init__(self, universe, handles, rotate, name):
        self.Universe = universe
        self.handles = handles
        self.rotate = rotate
        self.name = name

    @classmethod
    def from_universe(cls, universe, handles, save=False, rotate=True, name="temp"):
        path = "analysed_universes/{}.pdb".format(name)
        if save:
            Analyse.__save_universe(path, universe, rotate=rotate, overwrite=False)

        return cls(universe, handles, rotate, name)

    @staticmethod
    def __mkdirs(path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: 	# Guard against race condition
                if exc.errno != exc.errno.EEXIST:
                    raise
    @staticmethod
    def __save(path, lines, overwrite=False):
        Analyse.__mkdirs(path)
        try:
            if os.path.isfile(path) and not overwrite:
                print("Skipped saving {}".format(path))
            elif overwrite and os.path.isfile(path):
                with open(path, 'w') as outFile:
                    outFile.write('\n'.join(lines))
                print("Overwritten {}".format(path))
            else:
                with open(path, 'w') as outFile:
                    outFile.write('\n'.join(lines))
                print("Saved {}".format(path))
        except IOError as exception:
            raise IOError("{}: {}".format(path, exception.strerror))
        return None

    @staticmethod
    def __save_universe(path, universe, overwrite=False, rotate=True):
        Analyse.__mkdirs(path)
        try:
            if os.path.isfile(path) and not overwrite:
                print("Skipped saving the universe trajectory, {} already exists.".format(path))
            elif overwrite and os.path.isfile(path):
                print("Overwriting universe to {}".format(path))
                with MDAnalysis.Writer(path, n_atoms=universe.atoms.n_atoms) as W:
                    for k, ts in enumerate(universe.trajectory):
                        print("Writing frame {}...".format(ts))
                        if rotate:
                            universe.atoms.rotateby(90, [0, 1, 0])
                        W.write(universe)
                print("Overwritten {}".format(path))
            else:
                print("Saving universe to {}".format(path))
                with MDAnalysis.Writer(path, n_atoms=universe.atoms.n_atoms) as W:
                    for k, ts in enumerate(universe.trajectory):
                        print("Writing frame {}...".format(ts))
                        if rotate:
                            universe.atoms.rotateby(90, [0, 1, 0])
                        W.write(universe)
                print("Saved {}".format(path))
        except IOError as exception:
            raise IOError("{}: {}".format(path, exception.strerror))

    def get_initial_xy(self):
        """
        Used to calculate the average X, Y location of handles in the initial configuration.
        :return: mean x and y coordinates of the handle
        """
        xyarray = np.zeros((len(self.Universe.trajectory), 2, len(self.handles)))
        for k in range(len(self.Universe.trajectory)):
            f = self.Universe.trajectory[k]
            # rotate the system if necessary
            if self.rotate:
                self.Universe.atoms.rotateby(90, [0, 1, 0])
            print(20 * "-" + "Analysing frame {}".format(k + 1) + 20 * "-")
            for i_h in range(len(self.handles)):
                x, y = self.handles[i_h].get_avg_xy()
                xyarray[k, :, i_h] = x, y
        print(20 * "=" + "DONE" + 20 * "=")
        mean_x = np.mean(xyarray[:, 0, :], axis=0)
        mean_y = np.mean(xyarray[:, 1, :], axis=0)

        return mean_x, mean_y

    def do_bond_length_analysis(self, framerange=range(3), save=True, neighbours=False):
        if neighbours:
            p = "NeighLengths"
        else:
            p = "BondLengths"
        path = "{}/{}/".format(p, self.name) + self.name + ".txt"
        outlist = []
        for k in framerange:
            print(20 * "-" + "Analysing frame {}".format(k + 1) + 20 * "-")
            f = self.Universe.trajectory[k]
            hs = self.handles
            for i_h in range(len(hs)):
                dists = hs[i_h].get_interbase_dists(neighbours=neighbours)
                mean_dists = np.mean(dists)
                min_dists = min(dists)
                max_dists = max(dists)
                line = "{} {} {}".format(min_dists, max_dists, mean_dists)
                outlist.append(line)
        print(30 * "=" + "DONE" + 30 * "=")
        #print("MIN: {} MAX: {} MEAN: {}".format(min(outlist), max(outlist), np.mean(outlist)))
        Analyse.__save(path, outlist, overwrite=True)

        return outlist

    def do_primary_frame_analysis(self, framerange=range(3), save=True, debugplots=False):
        """
        Does the angle analysis for the angles and projections
        and writes them to a file. Also stores x,y values of handle end projection on plane.
        :param framerange: range of frames to be analysed
        :param save: whether to save to file or not
        :param debugplots whether to show plots or not
        :return: resultarray
        """
        path = "RESULTS/{}/".format(self.name) + self.name + "{}.txt".format(time.strftime("%Y%m%d-%H%M%S"))
        h_result_array = np.zeros((len(self.Universe.trajectory), 11, len(self.handles)), dtype=object)

        # declare the reference positions from the first frame
        f = self.Universe.trajectory[0]

        if self.rotate:
            self.Universe.atoms.rotateby(90, [0, 1, 0])

        ref_centre_of_mass = self.Universe.atoms.center_of_mass()
        ref0 = f.positions - ref_centre_of_mass

        for k in framerange:
            f = self.Universe.trajectory[k]

            if self.rotate:
                self.Universe.atoms.rotateby(90, [0, 1, 0])

            print("Aligning frame {}...".format(k + 1))
            mobile_centre_of_mass = self.Universe.atoms.center_of_mass()
            mobile0 = self.Universe.atoms.positions - mobile_centre_of_mass
            R, the_rmsd = align.rotation_matrix(mobile0, ref0)
            self.Universe.atoms.rotate(R)
            print("Alignment complete !")

            print(20 * "-" + "Analysing frame {}".format(k + 1) + 20 * "-")

            # fit 2D polynomial through the scaffold
            scaffold = self.__get_scaffold_atoms()
            xx, yy, zz, fit, rmse, x_order, y_order = Analyse.__fit_surf2atoms(scaffold)

            hs = self.handles
            for i_h in range(len(hs)):
                # find the closest point on the surface
                closest = fit.get_closest_surfXY(hs[i_h].start_base.positions, num=True)

                # find the surface normal
                surf_norm = fit.get_surf_norm([closest], start=hs[i_h].start_base.positions[0, :])

                # find a surface normal's corresponding plane
                h_len = hs[i_h].calc_len()    # use the handle length to get the drawing dimensions of the plane
                _, coords = fit.surf_norm2plane(len_tuple=(1.5*h_len, 1.5*h_len))

                # find a handle's corresponding scaffold directional vector
                scafdirvec, scafcoords = hs[i_h].fit_vector2scaffold()

                # get a handle's primary directional vector
                pd_vec = hs[i_h].get_primary_dir_vec(normalised=False)

                # get a handle's absolute directional vector from surf norm begin
                absd_vec = hs[i_h].get_absolute_dir_vec(vecbegin=surf_norm[0, :3], normalised=False)

                # project the scaffold directional vector onto the plane
                proj_scaf_vec = Analyse.__project_vec(surf_norm[0, :], scafdirvec[0, :], normalised=True)

                # get the perpendicular vector to the surface normal and projected scaffold vector
                perpen_vec = Analyse.__get_perpen_vec(surf_norm[0, :], proj_scaf_vec[0, :])

                # project the handle directional vector onto the plane
                proj_dir_vec = Analyse.__project_vec(surf_norm[0, :], absd_vec[0, :], normalised=False)

                # calculate the angle theta (out-plane) absddirvec -> surf norm
                theta = Analyse.__calc_vec_angle(absd_vec[0, 3:], surf_norm[0, 3:])

                # calculate the angle kappa (out-plane) primary dirvec -> surf norm
                kappa = Analyse.__calc_vec_angle(pd_vec[0, 3:], surf_norm[0, 3:])

                # find the local x, y, z. x-axis = proj_scaf_vec, y-axis = perpen_vec, z-axis = surf_norm
                x, y, z = (np.dot(proj_dir_vec[0, 3:], proj_scaf_vec[0, 3:]),
                           np.dot(proj_dir_vec[0, 3:], perpen_vec[0, 3:]),
                           np.dot(absd_vec[0, 3:], surf_norm[0, 3:]))

                # calculate the angle phi (in-plane)of projected dirvec, counter-clockwise from x-axis = +
                phi = np.rad2deg(np.arctan2(y, x))

                # get the binding situation of the free bases with the scaffold bases.
                bind_code = hs[i_h].get_binding_code(thres=0.69)

                h_result_array[k, :, i_h] = phi, theta, kappa, x, y, z, bind_code, rmse, x_order, y_order, the_rmsd

                if debugplots:
                    print("PHI = {}, THETA = {}, (X,Y,Z) = {}, BIND = {} , CODE = {}".format(phi, theta, (x, y, z), bind_code, hs[i_h].code))
                    # some accuracy testing  of the vectors
                    print(10 * "=" + ">" + "  accuracy = {}, should be 0.".format(
                        np.dot(fit.norms[0, 3:], proj_scaf_vec[0, 3:])))
                    print(10 * "=" + ">" + "  accuracy = {}, should be 0.".format(
                        np.dot(fit.norms[0, 3:], perpen_vec[0, 3:])))
                    print(10 * "=" + ">" + "  accuracy = {}, should be 0.".format(
                        np.dot(proj_scaf_vec[0, 3:], perpen_vec[0, 3:])))
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # Analyse.__make_vectorplot(ax, proj_dir_vec, color='y', linewidth=5, length=1)
                    # Analyse.__make_vectorplot(ax, perpen_vec, color='k', linewidth=5, length=7)
                    # Analyse.__make_vectorplot(ax, proj_scaf_vec, color='g', linewidth=5, length=7)
                    # Analyse.__make_vectorplot(ax, scafdirvec, color='b', linewidth=5, length=7)
                    # Analyse.__make_vectorplot(ax, pd_vec, color='r', linewidth=5, length=1)
                    # Analyse.__make_vectorplot(ax, absd_vec, color='m', linewidth=5, length=1)
                    # Analyse.__make_vectorplot(ax, surf_norm, color='c', linewidth=5, length=7)
                    # Analyse.__make_surfplot(ax, coords[:, :, 0, 0], coords[:, :, 1, 0], coords[:, :, 2, 0], color='GnBu')
                    # Analyse.__make_scatterplot(ax, scaffold.positions[:, 0], scaffold.positions[:, 1], scaffold.positions[:, 2])
                    # Analyse.__make_surfplot(ax, xx, yy, zz, color='winter')
                    # Analyse.__make_scatterplot(ax, hs[i_h].start_base.positions[:, 0], hs[i_h].start_base.positions[:, 1],
                    #                            hs[i_h].start_base.positions[:, 2], size=200)
                    # Analyse.__make_scatterplot(ax, scafcoords[:, 0], scafcoords[:, 1],  scafcoords[:, 2])
                    # Analyse.__make_scatterplot(ax, hs[i_h].atoms.positions[:, 0], hs[i_h].atoms.positions[:, 1],
                    #                            hs[i_h].atoms.positions[:, 2])
                    # for i in range(len(hs[i_h].free_bases)):
                    #     Analyse.__make_scatterplot(ax, hs[i_h].free_bases[i].positions[:, 0],
                    #                                hs[i_h].free_bases[i].positions[:, 1],
                    #                                hs[i_h].free_bases[i].positions[:, 2], color='b', size=80)
                    #     Analyse.__make_scatterplot(ax, hs[i_h].scaf_comp_bases[i].positions[:, 0],
                    #                                hs[i_h].scaf_comp_bases[i].positions[:, 1],
                    #                                hs[i_h].scaf_comp_bases[i].positions[:, 2], color='g', size=80)
                    plt.show()

            if save:
                Analyse.__write_frameresult(path, h_result_array[k, :, :])
            print(15 * "-" + "Analysing frame {} complete!".format(k + 1) + 15 * "-")
        print(30 * "=" + "DONE" + 30 * "=")
        return h_result_array

    @staticmethod
    def __write_frameresult(path, result):
        Analyse.__mkdirs(path)
        f = open(path, 'a+')
        line = ""
        for i_h in range(result.shape[1]):
            h_line = ""
            for d in result[:, i_h]:
                h_line += "{} ".format(d)
            line += h_line[:-1] + "\t\t"
        f.write(line + "\n")
        f.close()
        print("Frame results saved to {}.".format(path))
        return None

    def visualise_frames(self, framerange=range(3)):
        # declare the reference positions from the first frame
        f = self.Universe.trajectory[0]

        if self.rotate:
            self.Universe.atoms.rotateby(90, [0, 1, 0])

        ref_centre_of_mass = self.Universe.atoms.center_of_mass()
        ref0 = f.positions - ref_centre_of_mass

        for k in framerange:
            path = "VMDscripts/{}/frame{}/".format(self.name, k + 1)
            Analyse.__mkdirs(path)
            print("Preparing visualisation of frame {}...".format(k + 1))

            f = self.Universe.trajectory[k]
            if self.rotate:
                self.Universe.atoms.rotateby(90, [0, 1, 0])

            mobile_centre_of_mass = self.Universe.atoms.center_of_mass()
            mobile0 = self.Universe.atoms.positions - mobile_centre_of_mass
            R, the_rmsd = align.rotation_matrix(mobile0, ref0)
            self.Universe.atoms.rotate(R)

            print("Saving the universe as .pdb...")
            with MDAnalysis.Writer(path + "universe.pdb", n_atoms=self.Universe.atoms.n_atoms) as W:
                W.write(self.Universe.atoms)
            print("Saved {}".format(path + "universe.pdb"))
            xx, yy, zz, fit, _, _, _ = Analyse.__fit_surf2atoms(self.__get_scaffold_atoms())

            hs = self.handles
            surfcoords = [xx, yy, zz]
            surfmat = 'Transparent'
            surfcolor = 'red'

            vecarraylist = []
            veccolorlist = []
            vecmat = 'BrushedMetal'
            veclenlist = []

            planecoordlist = []
            planecolorlist = []
            planemat = 'Glass1'
            for i_h in range(len(hs)):
                # find the surface normal
                closest = fit.get_closest_surfXY(hs[i_h].start_base.positions, num=True)
                surf_norm = fit.get_surf_norm([closest], start=hs[i_h].start_base.positions[0, :])
                vecarraylist.append(surf_norm[0, :])
                veccolorlist.append(hs[i_h].draw_colour)
                veclenlist.append(10)

                # find a surface normal's corresponding plane
                h_len = hs[i_h].calc_len()    # use the handle length to get the drawing dimensions of the plane
                _, coords = fit.surf_norm2plane(len_tuple=(1.5*h_len, 1.5*h_len))
                x, y, z = coords[:, :, 0, 0], coords[:, :, 1, 0], coords[:, :, 2, 0]
                planecoordlist.append([x, y, z])
                planecolorlist.append(hs[i_h].draw_colour)

                # find a handle's corresponding scaffold directional vector
                scafdirvec, _ = hs[i_h].fit_vector2scaffold()
                vecarraylist.append(scafdirvec[0, :])
                veccolorlist.append('blue')
                veclenlist.append(10)

                # get a handle's absolute directional vector from surf norm begin
                absd_vec = hs[i_h].get_absolute_dir_vec(vecbegin=surf_norm[0, :3], normalised=False)
                vecarraylist.append(absd_vec[0, :])
                veccolorlist.append('pink')
                veclenlist.append(1)

                # project the scaffold directional vector onto the plane
                proj_scaf_vec = Analyse.__project_vec(surf_norm[0, :], scafdirvec[0, :], normalised=True)
                vecarraylist.append(proj_scaf_vec[0, :])
                veccolorlist.append('green')
                veclenlist.append(10)

                # get the perpendicular vector to the surface normal and projected scaffold vector
                perpen_vec = Analyse.__get_perpen_vec(surf_norm[0, :], proj_scaf_vec[0, :])
                vecarraylist.append(perpen_vec[0, :])
                veccolorlist.append('white')
                veclenlist.append(10)

                # project the handle directional vector onto the plane
                proj_dir_vec = Analyse.__project_vec(surf_norm[0, :], absd_vec[0, :], normalised=False)
                vecarraylist.append(proj_dir_vec[0, :])
                veccolorlist.append('yellow')
                veclenlist.append(1)

            my_vis = vis(path, self, vecarraylist, veccolorlist, vecmat, veclenlist,
                         surfcoords, surfcolor, surfmat, planecoordlist, planecolorlist, planemat)
            my_vis.save2tcl()
            my_vis.copytemplate()

    def openVMD(self, frame):
        path = "VMDscripts/{}/frame{}/".format(self.name, frame)
        abspath = os.path.abspath(path)
        print("Opening VMD from folder {}...".format(path))
        h = sub.run(["vmd", "-e", "runme.vmd"], cwd=abspath)
        print(h)

        return None

    def __get_scaffold_atoms(self):
        handle_atoms = sum([h.atoms for h in self.handles])
        scaffold = self.Universe.atoms.subtract(handle_atoms)

        return scaffold

    @staticmethod
    def __fit_surf2atoms(atomgroup):
        my_fit = Fit(atomgroup, maxordertuple=(10, 10), gridtuple=(100, 100))
        xx, yy, zz, _, best_RMSE, x_order, y_order = my_fit.make_fit()

        return xx, yy, zz, my_fit, best_RMSE, x_order, y_order

    @staticmethod
    def __calc_vec_angle(vec1, vec2):
        # calculate vector magnitudes
        len_vec1 = np.linalg.norm(vec1)
        len_vec2 = np.linalg.norm(vec2)
        # calculate dot product between vectors
        dot = np.dot(vec1, vec2)
        # Now calculate the angle between the vectors
        ang = np.arccos(dot / (len_vec1 * len_vec2))  # is in radians

        return (ang / (2 * m.pi)) * 360     # in degrees

    @staticmethod
    def __project_vec(plane_norm, vec, normalised=True):
        """
        Projects a vector vec onto a plane with normal plane_norm
        :param plane_norm: plane normal
        :param vec: vector
        :return: vectorarray
        """
        # projection of u onto plane with normal n = u - ((u . n)/||n||^2)n
        proj = vec[3:] - ((np.dot(vec[3:], plane_norm[3:]))/(np.linalg.norm(plane_norm[3:]))**2) * plane_norm[3:]
        if normalised:
            projected = np.array([np.concatenate([plane_norm[:3], proj/np.linalg.norm(proj)])])
        else:
            projected = np.array([np.concatenate([plane_norm[:3], proj])])

        return projected

    @staticmethod
    def __get_perpen_vec(plane_norm, proj_vec):
        """
        Calculates the vector perpendicular to the plane normal,
        and the scaffold vector projected onto the plane
        :param plane_norm: plane normal vector
        :param proj_vec: dir vec of scaffold projected on plane
        :return: vectorarray
        """
        perp = np.cross(plane_norm[3:], proj_vec[3:])

        return np.array([np.concatenate([plane_norm[:3], perp])])

    @staticmethod
    def __make_scatterplot(ax, x, y, z, color='r', size=10):
        ax.scatter(x, y, z, s=size, color=color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        return None

    @staticmethod
    def __make_vectorplot(ax, vec_array, color='r', length=10, linewidth=1):
        ax.quiver(vec_array[:, 0], vec_array[:, 1], vec_array[:, 2],
                  vec_array[:, 3], vec_array[:, 4], vec_array[:, 5], color=color, length=length, linewidth=linewidth)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        return None

    @staticmethod
    def __make_surfplot(ax, xx, yy, zz, color='winter'):
        ax.plot_surface(xx, yy, zz, cmap=color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")


