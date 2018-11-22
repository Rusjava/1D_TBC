# ------------------------------------------Asymmetrical waveguide with wave and Gaussian initial condition -- stable

import math
import cmath
import numpy as np
import io
import matplotlib.pyplot as plt

lam = 0.1  # -----------------Wavelength, nm
rho = 0.00054

f1Si = 14.142
f2Si = 0.141

den1 = 2.3  # --------------------------------------SiO2
M1 = 32.0

delta1 = den1 * rho / M1 * lam**2 * f1Si
beta1 = den1 * rho / M1 * lam**2 * f2Si

RMIN = 30  # ------------------------Gap semi-thickness
RMAX = 100  # ------------Maximum x
angle = 0  # ------------------------Incidence angle, mrad
ZMAX = 1e6  # ----------------Waveguide length, nm

h = 0.5  # ----------------------------- Transversal step
tau_int = 1e3  # ----------------------------- Longitudinal step
sprsn = 1  # ----------------------------ARRAY thinning(long range)
sprsm = 1  # ----------------------------ARRAY thinning
alp1 = -delta1 + 1j * beta1
alp0 = 0

# Vacuum outside
# k = 2 * pi / lam

# Vacuum inside
k = cmath.sqrt(1 + alp1) * 2 * math.pi / lam
alp1 = -alp1 / (1 + alp1)

MMAX = int(round(2. * RMAX / h))
muMAX = math.floor((MMAX + 2) / sprsm)
MMIN = int(round((RMAX - RMIN) / h))
MMIN2 = int(round((RMAX + RMIN) / h))
NMAX = int(round(ZMAX / tau_int))
if sprsn != 1:
    nuMAX = math.floor(NMAX / sprsn) + 1
else:
    nuMAX = NMAX

r = np.r_[0:MMAX+2] * h
z = tau_int * np.r_[0:NMAX]

# -----------------------------------------------Epsilon(r)

# ------------------------------PLANE WAVE
u0 = np.exp(1j * k * math.sin(angle) * r)

# ---------------------------------------GAUSSIAN
# WAIST = RMIN
# u0 =numpy.exp(-(r - RMAX)**2 / WAIST**2) * (numpy.sign(2 * RMIN - numpy.abs(r - RMAX)) + 1) / 2

# -------------------------------------
u = np.copy(u0)

# ----------------Creating main matrices
utop = np.zeros(NMAX,dtype=complex)
ubottom = np.zeros(NMAX,dtype=complex)
alp = np.zeros(MMAX+2,dtype=complex)
beta = np.zeros(NMAX,dtype=complex)
gg = np.zeros(NMAX,dtype=complex)
uplot = np.zeros((muMAX,nuMAX),dtype=complex)
# ----------------------------------------------MARCHING - - old TBC
utop[0] = u[MMAX]
ubottom[0] = u[1]
zplot = np.zeros(nuMAX)
rplot = r[sprsm * np.r_[0:muMAX]]
P = np.ones(MMAX + 2,dtype=complex)
Q = np.ones(MMAX + 2,dtype=complex)
c1 = (k*h)**2
alp0 *= c1
alp1 *= c1

# Initializing sparse field amplitude array
nuu = 1
zplot[0] = z[0]
uplot[0:muMAX, 0] = np.exp(1j * k * z[0]) * u[sprsm * np.r_[0:muMAX]]

c0 = 4. * 1j * k * h**2 / tau_int
ci = 2. - c0
cci = 2. + c0

# ----------------------------------------------MARCHING - new TBC
beta0 = -1j * 2. * cmath.sqrt(c0 - c0**2 / 4.)
phi = -1. / 2. - (-1.)**np.r_[0:NMAX+1] + ((-1.)**np.r_[0:NMAX+1]) / 2. * ((1. + c0 / 4.) / (1. - c0 / 4.))**np.r_[1:NMAX+2]
beta[0] = phi[0]
gg[0] = 1.
qq = 1j * 2. * cmath.sin(k * math.sin(angle) * h) / beta0
yy = cmath.cos(k * math.sin(angle) * h)
mm = cmath.exp(1j * k * math.sin(angle) * h)

for cntn in np.r_[0:NMAX-1]:

    alp[MMIN2 + 1:MMAX + 2] = alp0
    alp[MMIN:MMIN2 + 1] = alp1
    alp[0:MMIN] = alp0

# Top and bottom boundary conditions
    gg[cntn + 1] = ((c0 + 2. - 2. * yy) / (c0 - 2. + 2. * yy))**cntn

    SS = -np.dot(ubottom[0:cntn], beta[0:cntn+]) - ((qq-1)*gg[cntn+1] - np.dot(gg[0:cntn+1], beta[0:cntn+1])) * ubottom[0]
    SS1 = -np.dot(utop[0:cntn], beta[0:cntn+]) + ((qq+1)*gg[cntn+1] + np.dot(gg[0:cntn+1], beta[0:cntn+1])) * utop[0]

    beta[cntn + 1] = (np.dot(phi[0:cntn+1], beta[0:cntn+1]) + phi[cntn+1])/(cntn+1)

# Initial condition at the bottom
    c = ci - alp[1]
    cconj = cci - alp[1]
    d = u[2] - cconj * u[1] + u[0]

    P[0] = -(c - beta0) / 2.
    Q[0] = -(d - beta0 * SS) / 2.

# Preparation for marching
    for cntm in np.r_[0:MMAX]:
        c = ci - alp[cntm + 1]
        cconj = cci - alp[cntm + 1]
        d = u[cntm + 2] - cconj * u[cntm + 1] + u[cntm]

        P[cntm + 1] = -1. / (c + P[cntm])
        Q[cntm + 1] = -(Q[cntm] + d) * P[cntm + 1]

# Initial condition at the top
    u[MMAX + 1] = (beta0 * SS1 + Q[MMAX-1] - (P[MMAX-1] + beta0) * Q[MMAX]) / (1. - (beta0 + P[MMAX-1]) * P[MMAX])

# Solving the system
    for cntm in np.r_[MMAX+1:0: -1]:
        u[cntm-1] = Q[cntm-1] - P[cntm-1] * u[cntm]

# Preserving boundary values
    utop[cntn + 1] = u[MMAX]
    ubottom[cntn + 1] = u[1]

# Sparsing
    if cntn / sprsn - math.floor(cntn / sprsn) == 0:
        zplot[nuu] = z[cntn]
        uplot[0:muMAX, nuu] = np.exp(1j * k * z[cntn]) * u[sprsm * np.r_[0:muMAX]]
        nuu = nuu + 1
    # Printing the execution progress
    progress = int(round(1.*cntn / NMAX * 100))
    print(str(progress) + " %")

rplot = rplot - RMAX

# Preparing the title string
buf = io.StringIO()
buf.write("|u|: $\lambda =$ %1.2f nm   $XMAX =$ %4.2f $\mu$m  $XMIN =$ %4.2f $\mu$m  $ZMAX =$ %5.0f $\mu$m " % (lam, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3))

# Plotting the field amplitude in a color chart
fig, gplot = plt.subplots()
gplot.set_title(buf.getvalue())
X, Y = np.meshgrid(zplot * 1e-6, rplot * 1e-3)
cset = gplot.pcolormesh(X, Y, np.log10(np.abs(uplot)**2), cmap='jet')
fig.colorbar(cset)
gplot.set_xlabel('z, mm')
gplot.set_ylabel('x, $\mu$m')
plt.show()
