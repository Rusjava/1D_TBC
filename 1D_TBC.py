# ------------------------------------------Asymmetrical waveguide
#          with wave and Gaussian initial condition -- stable

import math
import cmath
import numpy as np
import io
import matplotlib.pyplot as plt

lam = 0.1
rho = 0.00054

f1Si = 14.142
f2Si = 0.141

den1 = 2.3 # --------------------------------------SiO2
M1 = 32.0

delta1 = den1 * rho / M1 * lam**2 * f1Si
beta1 = den1 * rho / M1 * lam**2 * f2Si

RMIN = 30 # ------------------------Gap semi-thickness
RMAX = 100 # ------------Maximum x
angle = 0 # ------------------------Angle in mrad

ZMAX = 1e6 # ----------------waveguide length

h = 0.5 # ----------------------------- << lam * f / R
tau_int = 1e3 # ----------------------------- INSIDE waveguide
sprsn = 1 # ----------------------------ARRAY thinning(long range)
sprsm = 1 # ----------------------------ARRAY thinning
alp1 = -delta1 + 1j * beta1
alp0 = 0

# Vacuum outside
# k = 2 * pi / lam

# Vacuum inside
k = cmath.sqrt(1 + alp1) * 2 * math.pi / lam
alp1 = -alp1 / (1 + alp1)

MMAX = int(np.round(2 * RMAX / h))
muMAX = math.floor((MMAX + 2) / sprsm)
MMIN = int(np.round((RMAX - RMIN) / h))
MMIN2 = int(np.round((RMAX + RMIN) / h))
NMAX = int(np.round(ZMAX / tau_int))
nuMAX = math.floor((NMAX + 2) / sprsn)

r = np.r_[0:MMAX+2] * h
z = tau_int * np.r_[0:NMAX]

# -----------------------------------------------Epsilon(r)

# ------------------------------PLANE WAVE
u0 = np.exp(1j * k * math.sin(angle) * r)

# ---------------------------------------GAUSSIAN
#WAIST = RMIN
#u0 =numpy.exp(-(r - RMAX)**2 / WAIST**2) * (numpy.sign(2 * RMIN - numpy.abs(r - RMAX)) + 1) / 2
# -------------------------------------
u = u0
#----------------Creating main matrices
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

nuu = 0

c0 = 4. * 1j * k * h**2 / tau_int
ci = 2. - c0
cci = 2. + c0

# ----------------------------------------------MARCHING - new TBC
beta0 = -1j * 2. * cmath.sqrt(c0 - c0**2 / 4.)
phi = -1 / 2. - (-1)**np.r_[0:NMAX+1] + ((-1)**np.r_[0:NMAX+1]) / 2. * ((1 + c0 / 4.) / (1 - c0 / 4.))**np.r_[1:NMAX+2]
beta[0] = phi[0]
gg[0] = 1
qq = 2. * cmath.sin(k * math.sin(angle) * h) / beta0 * 1j
yy = cmath.cos(k * math.sin(angle) * h)
mm = cmath.exp(1j * k * math.sin(angle) * h)

for cntn in np.r_[0:NMAX]:

    alp[MMIN2 + 1: MMAX + 2]=alp0
    alp[MMIN: MMIN2 + 1]=alp1
    alp[0: MMIN]=alp0

# Top and bottom boundary conditions
    gg[cntn + 1] = ((c0 + 2 - 2 * yy) / (c0 - 2 + 2 * yy))**cntn

    SS = -np.dot(ubottom[0:cntn], beta[0:cntn]) - ((qq-1)*gg[cntn+1]-np.dot(gg[0:cntn], beta[0:cntn])) * ubottom[0]
    SS1 = -np.dot(utop[0:cntn], beta[0:cntn]) + ((qq+1)*gg[cntn+1]+np.dot(gg[0:cntn], beta[0:cntn])) * utop[0]

    beta[cntn + 1] = (np.dot(phi[0:cntn], beta[0:cntn]) + phi[cntn+1])/(cntn+1)

# Initial condition at the bottom
    c = ci - alp[1]
    cconj = cci - alp[1]
    d = u[2] - cconj * u[1] + u[0]

    P[0] = -(c - beta0) / 2
    Q[0] = -(d - beta0 * SS) / 2

# Preparation for marching
    for cntm in np.r_[0:MMAX]:
        c = ci - alp[cntm + 1]
        cconj = cci - alp[cntm + 1]
        d = u[cntm + 2] - cconj * u[cntm + 1] + u[cntm]

        P[cntm + 1] = -1 / (c + P[cntm])
        Q[cntm + 1] = -(Q[cntm] + d) * P[cntm + 1]

# Initial condition at the top
    u[MMAX + 1] = (beta0 * SS1 + Q[MMAX-1] - (P[MMAX-1] + beta0) * Q[MMAX]) / (1 - (beta0 + P[MMAX-1]) * P[MMAX])

# Solving the system
    for cntm in np.r_[MMAX+1:0: -1]:
        u[cntm] = Q[cntm] - P[cntm] * u[cntm + 1]

# Preserving boundary values
    utop[cntn + 1] = u[MMAX]
    ubottom[cntn + 1] = u[1]

# Sparsing
    if cntn / sprsn - math.floor(cntn / sprsn) == 0:
        zplot[nuu] = z[cntn]
        uplot[0:muMAX, nuu]=np.exp(1j * k * z[cntn]) * u[sprsm * np.r_[0:muMAX]]
        nuu = nuu + 1
    progress = int(np.round(cntn / NMAX * 100))

rplot = rplot - RMAX

buf = io.StringIO()
buf.write("|u|: lambda = %1.2f nm   XMAX =%4.2f mum  XMIN =%4.2f mum  ZMAX =%5.0f mum " % (lam, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3))

fig, gplot = plt.subplots()
gplot.title(buf)
X, Y = np.meshgrid(zplot * 1e-6, rplot * 1e-3)
gplot.pcolormesh(X, Y, math.log10(np.abs(uplot)**2), cmap='jet')
gplot.colorbar()
gplot.xlabel('z, mm')
gplot.ylabel('x, \mum')
