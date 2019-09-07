import numpy as np
import os


class Visualisation(object):

    def __init__(self, path, analyse_object, vecarraylist=None, veccolorlist=None, vecmat=None, veclenlist=None,
                 surfcoords=None, surfcolor=None, surfmat=None, planecoordlist=None,
                 planecolorlist=None, planemat=None):
        self.vecarraylist = vecarraylist
        self.veccolorlist = veccolorlist
        self.vecmat = vecmat
        self.veclenlist = veclenlist

        self.surfcoords = surfcoords
        self.surfcolor = surfcolor
        self.surfmat = surfmat

        self.planecoordlist = planecoordlist
        self.planecolorlist = planecolorlist
        self.planemat = planemat

        self.path = path
        self.handles = analyse_object.handles

    @staticmethod
    def __make_triangle_mesh_lines(x, y, z, moln):
        lines = []
        for x_index in range(len(x)-1):
            for y_index in range(len(y)-1):
                # only use the middle part of the surface
                if x.shape[0] > 50 and not ((14 < x_index < (range(len(x)-1)[-1] - 15)) and (14 < y_index < (range(len(y)-1)[-1] - 15))):
                    continue

                a = np.array([x[y_index, x_index], y[y_index, x_index], z[y_index, x_index]])
                b = np.array([x[y_index+1, x_index], y[y_index+1, x_index], z[y_index+1, x_index]])
                c = np.array([x[y_index+1, x_index+1], y[y_index+1, x_index+1], z[y_index+1, x_index+1]])
                d = np.array([x[y_index, x_index+1], y[y_index, x_index+1], z[y_index, x_index+1]])

                n_a = np.cross(d-a, b-a)
                n_a = n_a/np.linalg.norm(n_a)

                n_b = np.cross(a-b, c-b)
                n_b = n_b/np.linalg.norm(n_b)

                n_c = np.cross(b-c, d-c)
                n_c = n_c/np.linalg.norm(n_c)

                n_d = np.cross(c-d, a-d)
                n_d = n_d/np.linalg.norm(n_d)

                fdict1 = {"xa": str(a[0]), "ya": str(a[1]), "za": str(a[2]),
                          "xb": str(b[0]), "yb": str(b[1]), "zb": str(b[2]),
                          "xd": str(d[0]), "yd": str(d[1]), "zd": str(d[2]),
                          "nxa": str(n_a[0]), "nya": str(n_a[1]), "nza": str(n_a[2]),
                          "nxb": str(n_b[0]), "nyb": str(n_b[1]), "nzb": str(n_b[2]),
                          "nxd": str(n_d[0]), "nyd": str(n_d[1]), "nzd": str(n_d[2])}
                fdict2 = {"xc": str(c[0]), "yc": str(c[1]), "zc": str(c[2]),
                          "xd": str(d[0]), "yd": str(d[1]), "zd": str(d[2]),
                          "xb": str(b[0]), "yb": str(b[1]), "zb": str(b[2]),
                          "nxc": str(n_c[0]), "nyc": str(n_c[1]), "nzc": str(n_c[2]),
                          "nxd": str(n_d[0]), "nyd": str(n_d[1]), "nzd": str(n_d[2]),
                          "nxb": str(n_b[0]), "nyb": str(n_b[1]), "nzb": str(n_b[2])}

                ABD = 'graphics '+str(moln)+' trinorm "{xa} {ya} {za}" ' \
                                                    '"{xb} {yb} {zb}" ' \
                                                    '"{xd} {yd} {zd}" ' \
                                                    '"{nxa} {nya} {nza}" ' \
                                                    '"{nxb} {nyb} {nzb}" ' \
                                                    '"{nxd} {nyd} {nzd}"'.format(**fdict1)
                CDB = 'graphics '+str(moln)+' trinorm "{xc} {yc} {zc}" ' \
                                                    '"{xd} {yd} {zd}" ' \
                                                    '"{xb} {yb} {zb}" ' \
                                                    '"{nxc} {nyc} {nzc}" ' \
                                                    '"{nxd} {nyd} {nzd}" ' \
                                                    '"{nxb} {nyb} {nzb}"'.format(**fdict2)
                lines.append(ABD+"\n"+CDB)

        return lines

    @staticmethod
    def __make_vector_lines(vecarray, moln, size=15):
        if size == 1:
            cylrad = 0.15
            conerad = 0.45
        else:
            cylrad = size/100
            conerad = size/20
        lines = []
        x1, y1, z1 = vecarray[:3]
        x2, y2, z2 = vecarray[:3] + (0.8*size) * vecarray[3:]
        x3, y3, z3 = vecarray[:3] + size * vecarray[3:]
        cyl = 'graphics '+str(moln)+' cylinder "{} {} {}" "{} {} {}" ' \
              'radius "{}" resolution "6" filled "yes"'.format(x1, y1, z1, x2, y2, z2, cylrad)
        cap = 'graphics '+str(moln)+' cone "{} {} {}" "{} {} {}" ' \
                               'radius "{}" resolution "6"'.format(x2, y2, z2, x3, y3, z3, conerad)
        lines.append(cyl + "\n" + cap)

        return lines

    def save2tcl(self):
        hs = self.handles
        moln = 0

        # generate the surface
        if self.surfcoords is not None and self.surfcolor is not None and self.surfmat is not None:
            print("Preparing the Surface for visualisation...")
            moln += 1
            surf_lines = ['graphics {} materials on'.format(moln)]
            surf_lines += Visualisation.surf2lines(self.surfcoords, self.surfmat, self.surfcolor, moln)
            Visualisation.__save(self.path + "surface.tcl", surf_lines)
        else:
            print("Skipping Surface visualisation.")

        # generate the planes
        if self.planecoordlist is not None and self.planecolorlist is not None and self.planemat is not None:
            print("Preparing the Planes for visualisation...")
            for i_h in range(len(hs)):
                moln += 1
                plane_lines = ['graphics {} materials on'.format(moln)]
                plane_lines += Visualisation.surf2lines(self.planecoordlist[i_h], self.planemat,
                                                        self.planecolorlist[i_h], moln)
                Visualisation.__save(self.path + "planeHandle{}.tcl".format(hs[i_h].code), plane_lines)
        else:
            print("Skipping Plane visualisation.")

        # generate the vectors
        if self.vecarraylist is not None and self.veccolorlist is not None and self.veclenlist is not None:
            print("Preparing the Vectors for visualisation...")
            vecsperhandle = len(self.vecarraylist) / len(hs)
            for i_h in range(len(hs)):
                moln += 1
                vec_lines = ['graphics {} materials on'.format(moln)]
                begin = int(i_h * vecsperhandle)
                end = int(begin + vecsperhandle - 1)
                for i in range(begin, end + 1):
                    vec_lines += Visualisation.vector2lines(self.vecarraylist[i], self.veccolorlist[i],
                                                            self.vecmat, self.veclenlist[i], moln)
                Visualisation.__save(self.path + "vectorsHandle{}.tcl".format(hs[i_h].code), vec_lines)
        else:
            print("Skipping Vector visualisation.")

    def copytemplate(self):
        # saves the .vmd script for automatic nice representations
        tempf = open("VIS_TEMPLATE.vmd")
        lines = tempf.readlines()
        tempf.close()
        outf = open(self.path + "runme.vmd", "w+")
        outf.write(''.join(lines))
        print("Saved {}.".format(self.path + "runme.vmd"))
        outf.close()
        return None


    @staticmethod
    def surf2lines(surfcoords, surfmat, surfcol, moln):
        lines = ['graphics {} color {}'.format(moln, surfcol),
                 'graphics {} material {}'.format(moln, surfmat)]
        x, y, z = (surfcoords[0], surfcoords[1], surfcoords[2])
        lines.extend(Visualisation.__make_triangle_mesh_lines(x, y, z, moln))

        return lines

    @staticmethod
    def vector2lines(vecsarray, veccol, vecmat, veclen, moln):
        lines = ['graphics {} color {}'.format(moln, veccol),
                 'graphics {} material {}'.format(moln, vecmat)]
        lines.extend(Visualisation.__make_vector_lines(vecsarray, moln, size=veclen))

        return lines

    @staticmethod
    def __flattenlist(listoflists):
        return [val for sublist in listoflists for val in sublist]

    @staticmethod
    def __save(path, lines):
        try:
            with open(path, 'w+') as outFile:
                outFile.write('\n'.join(lines))
            print("Saved {}".format(path))
        except IOError as exception:
            raise IOError("{}: {}".format(path, exception.strerror))
        return None
