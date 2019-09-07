import itertools as it
import numpy as np
import math as m
import matplotlib.pyplot as plt
import os


class Handle(object):
    """
    Defines some properties of handles, taking .pdb properties as input and
    translates them to the actual properties.
    Can be extended with MDAnalyis import if necessary.
    """
    def __init__(self, data):

        self.atoms, self.bases, self.staple_atoms, self.staple_bases,\
            self.handle_atoms, self.handle_bases, self.start_base,\
            self.scaf_vec_bases, self.full_scaf_vec_bases, self.scaf_comp_bases, self.free_bases, \
            self.draw_colour, self.code = data

        # stores own directional vectors
        self.dirvecs = np.zeros((1, 1))
        self.primdirvec = np.zeros((1, 1))
        self.absdirvec = np.zeros((1, 1))
        self.scaf_dirvec = np.zeros((1, 1))

        # stores some calculated properties
        self.length = 0

    @classmethod
    def with_properties(cls, u, handleSegNames, stapleSegNames, handleResNames, stapleResnames,
                        scaffoldResname, handleStartSegName, handleColour, compCode, dirvecScafEndSegname,
                        dirvecStapEndSegname, dirvecStapEndResname, scafVecSegNamesScafStrand, scafVecSegNamesStapStrand,
                               scafVecResNamesScafStrand, scafVecResNamesStapStrand):

        # translate the incorrect properties from the .PDB file to the actual properties
        handleStrandResNumbers = handleSegNames
        stapleStrandResNumbers = stapleSegNames

        handleStrandNumber = handleResNames
        stapleStrandNumber = stapleResnames

        scaffoldStrandNumber = scaffoldResname
        startResNumber = handleStartSegName

        # select the desired atoms
        atoms = Handle.__get_all_atoms(u, stapleStrandNumber, handleStrandNumber, stapleStrandResNumbers)
        bases = Handle.__filter_bases(atoms)
        staple_atoms = Handle.__get_staple_atoms(u, stapleStrandNumber, stapleStrandResNumbers)
        staple_bases = Handle.__filter_bases(staple_atoms)
        handle_atoms = Handle.__get_handle_atoms(u, handleStrandNumber)
        handle_bases = Handle.__filter_bases(handle_atoms)
        start_base = Handle.__get_start_base(u, scaffoldStrandNumber, startResNumber)
        scaf_vec_bases = Handle.__get_scaf_vec_bases(u, start_base, stapleStrandResNumbers,
                                                     stapleStrandNumber, scaffoldStrandNumber,
                                                     dirvecScafEndSegname, dirvecStapEndSegname,
                                                     dirvecStapEndResname)
        full_scaf_vec_bases = Handle.__get_full_scaf_vec_bases(u, scafVecSegNamesScafStrand, scafVecSegNamesStapStrand,
                               scafVecResNamesScafStrand, scafVecResNamesStapStrand)

        scaf_comp_bases = Handle.__get_scaf_comp_bases(u, scafVecSegNamesScafStrand, scafVecResNamesScafStrand)
        free_bases = Handle.__get_free_bases(staple_bases, stapleStrandResNumbers)

        data = (atoms, bases, staple_atoms, staple_bases, handle_atoms,
                handle_bases, start_base, scaf_vec_bases, full_scaf_vec_bases,
                scaf_comp_bases, free_bases, handleColour, compCode)

        return cls(data)

    @staticmethod
    def __get_atom(universe, selector):
        return universe.select_atoms(selector)

    @staticmethod
    def __filter_bases(atomgroup):
        return atomgroup.select_atoms("not type A")

    @staticmethod
    def __get_all_resnames(handleStrandNumber, stapleStrandNumber):
        h = handleStrandNumber
        s = stapleStrandNumber
        return [h, s]

    @staticmethod
    def __get_all_names(handleBaseNumbers, stapleBaseNumbers):
        h = handleBaseNumbers
        s = stapleBaseNumbers
        return [i for i in it.chain(h, s)]

    @staticmethod
    def __get_all_segids(handleStrandResNumbers, stapleStrandResNumbers):
        h = handleStrandResNumbers
        s = stapleStrandResNumbers
        return [i for i in it.chain(h,s)]

    @staticmethod
    def __get_handle_strand_selector(handleStrandNumber):
        s = ""
        s += "resname {}".format(handleStrandNumber)

        return s

    @staticmethod
    def __get_staple_strand_selector(stapleStrandNumber, stapleStrandResNumbers):
        s = ""
        s += "resname {}".format(stapleStrandNumber)
        segs = stapleStrandResNumbers
        ss = ""
        for seg in segs:
            ss += "segid {} or ".format(seg)
        return s + " and ({})".format(ss[:-4])

    @staticmethod
    def __get_selector(stapleStrandNumber, handleStrandNumber, stapleStrandResNumbers):
        hss = Handle.__get_handle_strand_selector(handleStrandNumber)
        sss = Handle.__get_staple_strand_selector(stapleStrandNumber, stapleStrandResNumbers)
        return "(({}) or ({}))".format(hss, sss)

    @staticmethod
    def __get_start_selector(scaffoldStrandNumber, startResNumber):
        # first approach: get the base "before" the first free base
        # s = "(not type A)" \
        #     " and (resname {})" \
        #     " and (segid {})".format(self.stapleStrandNumber, self.stapleStrandResNumbers[0] + 1)
        # second approach: get the scaffold base "before" the base that can form a base pair w/ free bases
        s = "(not type A)" \
            " and (resname {})" \
            " and (segid {})".format(scaffoldStrandNumber, startResNumber)
        return s

    @staticmethod
    def __get_scaf_vec_bases(universe, start_base, stapleStrandResNumbers, stapleStrandNumber, scaffoldStrandNumber,
                             dirvecScafEndSegname, dirvecStapEndSegname, dirvecStapEndResname):

        selec_stap_begin = "(not type A) and (resname {}) and (segid {})".format(stapleStrandNumber, stapleStrandResNumbers[0]+1)
        selec_stap_end = "(not type A) and (resname {}) and (segid {})".format(dirvecStapEndResname, dirvecStapEndSegname)
        selec_scaf_end = "(not type A) and (resname {}) and (segid {})".format(scaffoldStrandNumber, dirvecScafEndSegname)

        stap_begin = Handle.__get_atom(universe, selec_stap_begin)
        scaf_begin = start_base
        stap_end = Handle.__get_atom(universe, selec_stap_end)
        scaf_end = Handle.__get_atom(universe, selec_scaf_end)

        return stap_begin, scaf_begin, stap_end, scaf_end

    @staticmethod
    def __get_full_scaf_vec_bases(universe, scafVecSegNamesScafStrand, scafVecSegNamesStapStrand,
                               scafVecResNamesScafStrand, scafVecResNamesStapStrand):
        scaf_strand_selec = "(not type A) and (resname {}) and (".format(scafVecResNamesScafStrand)
        for sn in scafVecSegNamesScafStrand:
            if isinstance(sn, range):
                for i in sn:
                    scaf_strand_selec += "segid {} or ".format(i)
            else:
                scaf_strand_selec += "segid {} or ".format(sn)
        stap_strand_selec = "(not type A) and ("
        for i_r in range(len(scafVecResNamesStapStrand)):
            if isinstance(scafVecSegNamesStapStrand[i_r], range):
                temp_selec = "resname {} and (".format(scafVecResNamesStapStrand[i_r])
                for i in scafVecSegNamesStapStrand[i_r]:
                    temp_selec += "segid {} or ".format(i)
                stap_strand_selec += temp_selec[:-4]+")) or ("
            else:
                stap_strand_selec += "(resname {} and segid {}) or (".format(scafVecResNamesStapStrand[i_r],
                                                                      scafVecSegNamesStapStrand[i_r])
        stap_bases = Handle.__get_atom(universe, stap_strand_selec[:-5]+")")
        scaf_bases = Handle.__get_atom(universe, scaf_strand_selec[:-4]+")")
        return stap_bases + scaf_bases

    @staticmethod
    def __get_scaf_comp_bases(universe, scafVecSegNamesScafStrand, scafVecResNamesScafStrand):
        scaf_strand_selec = "(not type A) and (resname {})".format(scafVecResNamesScafStrand)
        scaf_strand_bases = Handle.__get_atom(universe, scaf_strand_selec)

        compbase_A_selec = "segid {}".format(scafVecSegNamesScafStrand[0][6])
        compbase_B_selec = "segid {}".format(scafVecSegNamesScafStrand[0][7])
        compbase_C_selec = "segid {}".format(scafVecSegNamesScafStrand[0][8])
        compbase_A = Handle.__get_atom(scaf_strand_bases, compbase_A_selec)
        compbase_B = Handle.__get_atom(scaf_strand_bases, compbase_B_selec)
        compbase_C = Handle.__get_atom(scaf_strand_bases, compbase_C_selec)

        return compbase_A, compbase_B, compbase_C

    @staticmethod
    def __get_free_bases(staple_bases, stapleStrandResNumbers):
        # the two free bases and the first staple strand base bound to the handle strand base, from top to bottom!
        first_selec = "segid {}".format(stapleStrandResNumbers[2])
        sec_selec = "segid {}".format(stapleStrandResNumbers[1])
        third_selec = "segid {}".format(stapleStrandResNumbers[0])
        first = Handle.__get_atom(staple_bases, first_selec)
        sec = Handle.__get_atom(staple_bases, sec_selec)
        third = Handle.__get_atom(staple_bases, third_selec)

        return first, sec, third

    @staticmethod
    def __get_start_base(universe, scaffoldStrandNumber, startResNumber):
        start = Handle.__get_atom(universe, Handle.__get_start_selector(scaffoldStrandNumber, startResNumber))
        return start

    @staticmethod
    def __get_all_atoms(universe, stapleStrandNumber, handleStrandNumber, stapleStrandResNumbers):
        atoms = Handle.__get_atom(universe, Handle.__get_selector(stapleStrandNumber, handleStrandNumber, stapleStrandResNumbers))
        return atoms

    @staticmethod
    def __get_staple_atoms(universe, stapleStrandNumber, stapleStrandResNumbers):
        stapleatoms = Handle.__get_atom(universe, Handle.__get_staple_strand_selector(stapleStrandNumber, stapleStrandResNumbers))
        return stapleatoms

    @staticmethod
    def __get_handle_atoms(universe, handleStrandNumber):
        handleatoms = Handle.__get_atom(universe, Handle.__get_handle_strand_selector(handleStrandNumber))
        return handleatoms

    @staticmethod
    def __calc_middle(atom1, atom2):
        try:
            mid = atom1.position + (atom2.position - atom1.position)/2
        except:
            mid = atom1.positions + (atom2.positions - atom1.positions)/2

        return mid

    @staticmethod
    def __calc_dist(atom1, atom2):
        try:
            d = np.linalg.norm(atom1.position - atom2.position)
        except:
            d = np.linalg.norm(atom1.positions - atom2.positions)

        return d

    def get_interbase_dists(self, neighbours=False):
        """
        For testing purposes,
        calculate the distances between staple and handle strand bases 3 bp away from handle top and 3 away from
        the bottom of the handle strand
        :return: array with distances
        """
        dists = np.zeros((len(self.handle_bases), 2))
        i = 3
        j = len(self.staple_bases) - 3
        while i < len(self.handle_bases) - 3:
            d1 = Handle.__calc_dist(self.handle_bases[i], self.staple_bases[j-3])
            d2 = Handle.__calc_dist(self.handle_bases[i], self.staple_bases[j-2])
            dists[i, :] = d1, d2
            i += 1
            j -= 1
        if neighbours:
            dists = dists[:, 1]
        else:
            dists = dists[:, 0]

        return dists[np.nonzero(dists)]

    def get_dir_vecs(self, normalised=True):
        """
        Calculate all directional vectors between the middle points of all handle bases
        :return:
        """
        vecs = np.zeros((len(self.handle_bases)-1, 6))
        i = 0
        j = len(self.staple_bases)
        while i < (len(self.handle_bases)-1):
            vec_begin = Handle.__calc_middle(self.handle_bases[i], self.staple_bases[j-3])
            vec_end = Handle.__calc_middle(self.handle_bases[i+1], self.staple_bases[j-4])
            dir_vec = vec_end - vec_begin
            if normalised:
                vecs[i, :] = np.concatenate([vec_begin, dir_vec/np.linalg.norm(dir_vec)])
            else:
                vecs[i, :] = np.concatenate([vec_begin, dir_vec])
            i += 1
            j -= 1
        print("DONE: directional vectors calculated")
        self.dirvecs = vecs
        return vecs

    def get_primary_dir_vec(self, normalised=True):
        """
        Sum all the dir vecs to get the primary dir vec, from the first to last base
        starting at the base above the free bases
        :return:
        """
        dvecs = self.get_dir_vecs(normalised=False)
        vec_begin = dvecs[0, 0:3]
        summed = dvecs.sum(axis=0)
        vec = summed[3:]
        if normalised:
            prim_dir_vec = np.array([np.concatenate([vec_begin, vec/np.linalg.norm(vec)])])
        else:
            prim_dir_vec = np.array([np.concatenate([vec_begin, vec])])
        self.primdirvec = prim_dir_vec
        print("DONE: primary directional vector calculated")
        return prim_dir_vec

    def get_absolute_dir_vec(self, vecbegin=None, normalised=True):
        """
        Calculate the absolute directional vector;
        from the middle of handle_start base and its complementary base
        to the middle of the last staple and handle base (normalised).
        Optionally pick a different beginning of the vector.
        :return:
        """
        if vecbegin is None:
            vec_begin = Handle.__calc_middle(self.scaf_vec_bases[0], self.scaf_vec_bases[1])
        else:
            vec_begin = vecbegin
        last_handlestrand_base = self.handle_bases[-1]
        last_staplestrand_base = self.staple_bases[0]
        vec_end = Handle.__calc_middle(last_handlestrand_base, last_staplestrand_base)
        vec = vec_end - vec_begin

        if normalised:
            abs_dir_vec = np.array([np.concatenate([vec_begin, vec/np.linalg.norm(vec)])])
        else:
            abs_dir_vec = np.array([np.concatenate([vec_begin, vec])])
        self.absdirvec = abs_dir_vec
        print("DONE: absolute directional vector calculated")
        return abs_dir_vec

    def get_scaffold_dir_vec(self):
        """
        Use a handle's start base and a base 18 steps further on the scaffold
        to get a handle's corresponding scaffold directional vector
        :return: vector array (start x y z and vec components as columns)
        """
        vec_begin = Handle.__calc_middle(self.scaf_vec_bases[0], self.scaf_vec_bases[1])
        vec_end = Handle.__calc_middle(self.scaf_vec_bases[2], self.scaf_vec_bases[3])
        vec = vec_end - vec_begin
        scaf_dir_vec = np.concatenate([vec_begin, vec/np.linalg.norm(vec)], axis=1)

        self.scaf_dirvec = scaf_dir_vec
        print("DONE: scaffold directional vector calculated")
        return scaf_dir_vec

    def fit_vector2scaffold(self):
        scaf_coords = self.full_scaf_vec_bases.positions
        middle = scaf_coords.mean(axis=0)
        _, _, vvm = np.linalg.svd(scaf_coords - middle)
        vec = vvm[0]
        vecn = vec/np.linalg.norm(vec)
        scafvec = np.array([np.concatenate([middle, vecn])])

        print("DONE: scaffold directional vector has been fitted")
        self.scaf_dirvec = scafvec
        return scafvec, scaf_coords

    def get_binding_code(self, thres=0.69):
        """
        Calculate which free bases are bound to which (complementary) scaffold bases.
        :param: thres: the threshold in sim. units for selecting if a base is bound or not
        :return: binding code string, e.g. "1a" means a is bound to first, returns "0" if nothing is bound
        """
        # first, sec, third = self.free_bases
        # a, b, c = self.scaf_comp_bases
        fbs = self.free_bases
        fb_codes = ["1", "2", "3"]
        cbs = self.scaf_comp_bases
        cb_codes = ["a", "b", "c"]
        b_code = ""
        for i_fb in range(len(fbs)):
            for i_cb in range(len(cbs)):
                d = Handle.__calc_dist(fbs[i_fb], cbs[i_cb])
                if d <= thres:
                    b_code += "{}{}".format(fb_codes[i_fb], cb_codes[i_cb])
        if len(b_code) == 0:
            b_code = "0"

        return b_code

    def get_avg_xy(self):
        """
        For testing purposes,
        Get the average X, Y value of all the points in between the bases
        :return:
        """
        i = 0
        j = len(self.staple_bases) - 3
        coords = np.zeros((len(self.handle_bases), 3))
        while i < len(self.handle_bases):
            mid_coord = Handle.__calc_middle(self.handle_bases[i], self.staple_bases[j])
            coords[i, :] = mid_coord
            i += 1
            j -= 1
        mean_x, mean_y = (np.mean(coords[:, 0]), np.mean(coords[:, 1]))

        return mean_x, mean_y

    def calc_len(self):
        """
        Calculates the approximate length of the handle (excl. free bases), useful for determining
        the necessary size of its corresponding "scaffold plane"
        :return:
        """
        dirvecs = self.get_dir_vecs(normalised=False)
        tot_len = 0
        for v in dirvecs[:, 3:]:
            length = np.linalg.norm(v)
            tot_len += length

        self.length = tot_len
        return tot_len

    @staticmethod
    def __mkdirs(path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: 	# Guard against race condition
                if exc.errno != exc.errno.EEXIST:
                    raise
    @staticmethod
    def __save(path, lines, overwrite=True):
        Handle.__mkdirs(path)
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
