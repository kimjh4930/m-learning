import numpy as np
import cvxopt
import cvxopt.solvers

from cvxopt.base import matrix

q = cvxopt.matrix(np.ones(5) * -1)

print q
