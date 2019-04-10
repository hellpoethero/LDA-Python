import numpy as np


GAMMA_ACCURATE = True
INV810 = 1.234567901234568E-003
HALFLN2 = 3.465735902799726e-001
HALFLN2PI = 9.189385332046727E-001


def log_sum(log_a, log_b):
	if log_a < log_b:
		v = log_b + np.log(1 + np.exp(log_a - log_b))
	else:
		v = log_a + np.log(1 + np.exp(log_b - log_a))
	return v


def digamma(x):
	x = x + 6
	p = 1 / (x * x)
	p = (((0.004166666666667 * p - 0.003968253986254) * p + 0.008333333333333) * p - 0.083333333333333) * p

	p = p + np.log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6)
	return p


def lgamma(x):
	lnx = np.log(x)
	einvx = np.exp(1. / x)

	if GAMMA_ACCURATE:
		prec = x * x * x
		prec *= prec
		prec = INV810 / prec
		return x * (lnx - 1. + .5 * np.log(x * (einvx - 1. / einvx) / 2. + prec))- .5 * lnx + HALFLN2PI
	else:
		return x * (1.5 * lnx - 1. + .5 * np.log(einvx - 1. / einvx) - HALFLN2) - .5 * lnx + HALFLN2PI

