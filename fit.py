import numpy as np
from numpy.polynomial import polynomial
import sympy as sym
import scipy.optimize
import os


class Fit(object):
    """
    A fit object, where all methods used for fitting a 2D polynomial surface are present.
    """

    def __init__(self, atomgroup, maxordertuple=(3, 3), gridtuple=(20, 20)):

        self.atoms = atomgroup
        self.x, self.y, self.z = (atomgroup.positions[:, 0], atomgroup.positions[:, 1], atomgroup.positions[:, 2])

        self.grid = gridtuple       # defines how many grid points in x and y directions, resp.
        self.maxorder = maxordertuple     # defines the max order in x and y direction, resp.

        # surface processing/fitting results will be stored in these instance properties
        self.fit_result = 0    # fit results are stored for error calculation
        self.exprstr = ""       # string for the equation of the best-fit 2D polynomial

        # keeps track of final coefficient matrix c, final x,y,z values of the fit,
        # its corresponding norm vectors and their planes
        self.c = np.zeros((1,1))
        self.xx = np.zeros((1,1))
        self.yy = np.zeros((1,1))
        self.zz = np.zeros((1,1))
        self.norms = np.zeros((1,1))
        self.normplanes = np.zeros((1,1))

    def __polyfit2dvander(self, order):
        """
        Performs least square optimisation of a 2D polynomial,
        where a 2D Van der Monde-coefficient matrix is optimised.
        :param order: x and y order tuple
        :return: coefficient matrix of shape (x_order + 1, y_order + 1)
        """
        deg = np.asarray([order[0], order[1]])
        vander = polynomial.polyvander2d(self.x, self.y, deg)
        vander = vander.reshape((-1, vander.shape[-1]))
        f = self.z.reshape((vander.shape[0],))
        c = np.linalg.lstsq(vander, f, rcond=0)[0]
        #self.c = c.reshape(deg + 1)

        return c.reshape(deg + 1)

    def __coeff2expr(self):
        """
        Transforms a 2D Van der Monde-coefficient matrix to a
        symbolic SymPy expression
        :return: symbolic sympy expression e.g. x**2 + y**2 - z (= 0)
        """
        e = "-z +"
        for i in range(self.c.shape[0]):
            for j in range(self.c.shape[1]):
                e += " {} * x**{} * y**{} +".format(str(self.c[i, j]), str(i), str(j))
        self.exprstr = e[:-2]
        expr = sym.sympify(self.exprstr)
        return expr

    @staticmethod
    def __calc_gradient(expr):
        """
        Calculates a symbolic gradient vector for the expression expr
        :param expr: symbolic expression of the surface
        :return: np.array (vector) gradient
        """
        x, y, z = sym.symbols('x y z')
        dfdx = sym.diff(expr, x)
        dfdy = sym.diff(expr, y)
        dfdz = sym.diff(expr, z)
        return np.array([-1*dfdx, -1*dfdy, -1*dfdz])

    @staticmethod
    def __evalexpr(expr, xx, yy):
        """
        Finds z value for given x, y value on the surface
        :param expr: surface expression
        :param xx: x value
        :param yy: y value
        :return: z value
        """
        x, y, z = sym.symbols('x y z')
        zz = expr.evalf(subs={x: xx, y: yy, z: 0})

        return zz

    def __vanderfit(self, order):
        """
        Calculates the surface on a 3D meshgrid
        :param order: ordertuple of the surface
        :return: x y and z values and coefficient matrix
        """

        c = self.__polyfit2dvander(order)

        nx, ny = self.grid
        xx, yy = np.meshgrid(np.linspace(self.x.min(), self.x.max(), nx),
                             np.linspace(self.y.min(), self.y.max(), ny))

        # store the fit result for error calculation, zz is used to plot the surface
        self.fit_result = polynomial.polyval2d(self.x, self.y, c)
        zz = polynomial.polyval2d(xx, yy, c)

        return xx, yy, zz, c

    def make_fit(self):
        """
        Performs the actual fitting to the data.
        Optimises the RMSE for different polynomial orders in x and y direction
        :return: x y and z values and coefficient matrix of best fit
        """
        RMSEs = []
        print("Starting RMSE optimisation...")
        for x_order in range(1, self.maxorder[0]+1):
            for y_order in range(1, self.maxorder[1]+1):
                _, _, _, _ = self.__vanderfit((x_order, y_order))
                _, RMSE = self.__calc_error()
                RMSEs.append([RMSE, (x_order, y_order)])
        best_result = sorted(RMSEs, key=lambda x: x[0])[0]
        best_RMSE = best_result[0]
        best_orders = best_result[1]
        print("N_x = {} and N_y = {}, RMSE = {}".format(best_orders[0], best_orders[1], best_RMSE))
        self.xx, self.yy, self.zz, self.c = self.__vanderfit(best_orders)
        return self.xx, self.yy, self.zz, self.c, best_RMSE, best_orders[0], best_orders[1]

    @staticmethod
    def print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))

    @staticmethod
    def cost_func(coeff, data, cshape, squaredscalar):
        """
        Cost function for optimisation
        :param coeff: coefficient matrix
        :return: residuals
        """
        x, y, z = data[0], data[1], data[2]

        coeff = np.reshape(coeff, cshape)
        pred_z = polynomial.polyval2d(x, y, coeff)
        if squaredscalar:
            resid = (pred_z - z)**2
            resid = resid.sum()
        else:
            resid = pred_z - z
        return resid

    def optimise(self, use_fitresult=True):
        if use_fitresult:
            theta_0 = self.c.flatten()
        else:
            theta_0 = np.ones(self.c.shape).flatten()

        num_coeff = len(theta_0)

        lb = np.full(shape=num_coeff, fill_value=-np.inf, dtype=np.float)
        ub = np.full(shape=num_coeff, fill_value=np.inf, dtype=np.float)
        bnds = (lb, ub)
        data = [self.x, self.y, self.z]
        lst_sqrs_result = scipy.optimize.least_squares(Fit.cost_func, theta_0,
                                                       args=(data, self.c.shape, False),
                                                       bounds=bnds,
                                                       method='trf',
                                                       loss='linear',
                                                       ftol=1e-10,
                                                       gtol=1e-10)
        # minimizer_kwargs = {"args": (data, self.c.shape, True)}
        # lst_sqrs_result = scipy.optimize.basinhopping(Fit.cost_func, theta_0,
        #                                               minimizer_kwargs=minimizer_kwargs, callback=Fit.print_fun)

        resid = Fit.cost_func(lst_sqrs_result.x, [self.x, self.y, self.z], self.c.shape, False)
        _, rmse = self.__calc_error(resid=resid)
        self.c = np.reshape(lst_sqrs_result.x, self.c.shape)
        self.zz = polynomial.polyval2d(self.xx, self.yy, self.c)
        print("N_x = {} and N_y = {}, RMSE = {}".format(self.c.shape[0] - 1, self.c.shape[1] - 1, rmse))
        # print('====================')
        # print('lst_sqrs_result =\n{}'.format(lst_sqrs_result,))
        # print('====================')

        return self.xx, self.yy, self.zz, self.c, rmse, self.c.shape[0] - 1, self.c.shape[1] - 1

    @staticmethod
    def dist_func(x, f):
        return f(*x)

    def get_closest_surfXY(self, coord, num=True):
        """
        Find the closest points on the surface for the coordinates in coord
        Do not use analytical method, it is too slow.
        :param coord: list of coords
        :param num: numeric optimisation of dist func if True, else analytical method
        :return: nearest [x, y, z] on surface
        """
        expr = self.__coeff2expr()
        print("Calculating nearest surface coordinate...")
        for co in coord:
            if num:
                x, y, z = sym.symbols('x y z')
                a, b, c = sym.symbols('a b c')

                z_ = sym.solve(expr, z)[0]
                dist_func = (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2
                dist = dist_func.subs({a: co[0], b: co[1], c: co[2], z: z_})

                f = sym.lambdify([x, y], dist)

                init_guess = np.array([co[0], co[1]])
                res = scipy.optimize.minimize(Fit.dist_func, init_guess,
                                              method=None, args=f, bounds=None,
                                              options={'gtol': 1e-9, 'disp':True})
                p = sym.lambdify([x, y], z_)
                nearestXYZ = [res.x[0], res.x[1], p(*res.x)]
                return nearestXYZ
            else:
                x, y, z = sym.symbols('x y z')
                a, b, c = sym.symbols('a b c')
                L = sym.symbols('L')
                dist_func = (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2

                dist = dist_func.subs({a: co[0], b: co[1], c: co[2]})
                dist_grad = Fit.__calc_gradient(dist)
                surf_grad = Fit.__calc_gradient(expr)
                lagrange_eq = dist_grad - L*surf_grad

                # solve the last row for lambda, and substitute it in other rows
                solve_for_L = sym.solve(lagrange_eq[2], L)[0]
                L_subsed = np.array([lagrange_eq[0].subs({L: solve_for_L}),
                                     lagrange_eq[1].subs({L: solve_for_L}),
                                     0])

                # surface expression as z =
                z_ = sym.solve(expr, z)[0]
                z_subsed = np.array([L_subsed[0].subs({z: z_}),
                                     L_subsed[1].subs({z: z_}),
                                     0])

                simpl = np.array([sym.simplify(z_subsed[0]),
                                  sym.simplify(z_subsed[1]),
                                  0])

                nearest = sym.solve((simpl[0], simpl[1]), [x, y], dict=True)
                nearestXYZ = [nearest.x, nearest.y, co[2]]
        print("DONE: Found nearest surface coordinate!")
        return nearestXYZ

    def get_surf_norm(self, coordlist, start=None):
        """
        Calculates the surface normal for at the projection of (x,y) coordinates on the surface
        :param coordlist: a list of coordinates of where to calculate surface normal at
        :return: np.array where every row represents x, y, z value of vector start
                and x, y, z components of vector
        """
        x, y, z = sym.symbols('x y z')
        expr = self.__coeff2expr()

        normal = np.zeros((len(coordlist), 6))
        for k, coord in enumerate(coordlist):
            xx, yy = (coord[0], coord[1])
            zz = self.__evalexpr(expr, xx, yy)
            grad = self.__calc_gradient(expr)

            n = np.array([grad[0].evalf(subs={x: xx, y: yy, z: zz}),
                          grad[1].evalf(subs={x: xx, y: yy, z: zz}),
                          grad[2]], dtype=np.float32)
            le = np.linalg.norm(n)
            if start is None:
                start = [xx, yy, zz]
            normal[k, :] = np.concatenate([start, n/le])
            print("DONE: surface normal vector calculated")
        self.norms = normal

        return normal

    def surf_norm2plane(self, len_tuple=(10, 10), gridtuple=(10, 10)):
        """
        Calculates the equation, x, y, z lists for the planes
        related to the surf_norm elements in the input array
        :param norm_array: array with normal vectors
        :param len_tuple length of the plane in x and y from middle
        :return: a list containing the sympy plane expressions, and lists containing x, y and z values, which all are
        numpy arrays
        """
        x, y = sym.symbols('x y')
        expr_list = []
        coordsarray = np.zeros((gridtuple[0], gridtuple[1], 3, len(self.norms)))
        for k, n in enumerate(self.norms):
            # a*x + b*y + c*z = d, d = a*x_0 + b*y_0 + c*z_0
            fdict = {"x0": str(n[0]), "y0": str(n[1]), "z0": str(n[2]), "a": str(n[3]), "b": str(n[4]), "c": str(n[5])}
            expr_str = "({a}*x + {b}*y - {a}*{x0} - {b}*{y0} - {c}*{z0})/ -{c}".format(**fdict) # z = ...
            expr = sym.sympify(expr_str)
            expr_list.append(sym.sympify(expr))
            x_min = n[0] - len_tuple[0]/2
            x_max = n[0] + len_tuple[0]/2
            y_min = n[1] - len_tuple[1]/2
            y_max = n[1] + len_tuple[1]/2
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, gridtuple[0]),
                                 np.linspace(y_min, y_max, gridtuple[1]))
            f = sym.lambdify([x, y], expr)
            zz = f(xx, yy)

            coordsarray[:, :, 0, k], coordsarray[:, :, 1, k], coordsarray[:, :, 2, k] = xx, yy, zz
            print("DONE: surface normal plane calculated")

        self.normplanes = coordsarray

        return expr_list, coordsarray

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
        Fit.__mkdirs(path)
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
        return

    def __calc_error(self, resid=None):
        """
        Calculates the RMSE and residuals for a fit result
        :return: residuals and RMSE
        """
        if resid is None:
            resid = self.fit_result - self.z
        SE = np.square(resid)       # squared errors
        MSE = np.mean(SE)           # mean squared errors
        RMSE = np.sqrt(MSE)         # Root Mean Squared Error, RMSE

        #print("RMSE: {}".format(RMSE))

        return resid, RMSE
