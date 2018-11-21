# ------------------------------------------Asymmetrical waveguide
#          with wave and Gaussian initial condition -- stable

import math;
import cmath;
import numpy;
import io;
import matplotlib.pyplot as plt;

lam = 0.1;
rho = 0.00054;

f1Si = 14.142;
f2Si = 0.141;

den1 = 2.3; # --------------------------------------SiO2
M1 = 32.0;

delta1 = den1 * rho / M1 * lam**2 * f1Si;
beta1 = den1 * rho / M1 * lam**2 * f2Si;

RMIN = 30; # ------------------------Gap semi-thickness
RMAX = 100; # ------------Maximin x
angle = 0; # ------------------------Angle in mrad

ZMAX = 1e6; # ----------------waveguide length

h = 0.5; # ----------------------------- << lam * f / R
tau_int = 1e3; # ----------------------------- INSIDE waveguide
sprsn = 1; # ----------------------------ARRAY thinning(long range)
sprsm = 1; # ----------------------------ARRAY thinning
alp1 = -delta1 + 1j * beta1;
alp0 = 0;

# Vacuum outside
# k = 2 * pi / lam;

# Vacuum inside
k = math.sqrt(1 + alp1) * 2 * math.pi / lam;
alp1 = -alp1 / (1 + alp1);

MMAX = math.round(2 * RMAX / h);
muMAX = math.floor((MMAX + 2) / sprsm);
MMIN = math.round((RMAX - RMIN) / h);
MMIN2 = math.round((RMAX + RMIN) / h);
NMAX = math.round(ZMAX / tau_int);
nuMAX = math.floor((NMAX + 2) / sprsn);

r = numpy.r_[0:MMAX+2] * h;
z = tau_int * numpy.r_[0:NMAX];

# -----------------------------------------------Epsilon(r)

# ------------------------------PLANE WAVE
u0 = numpy.exp(1j * k * math.sin(angle) * r);

# ---------------------------------------GAUSSIAN
#WAIST = RMIN;
#u0 =numpy.exp(-(r - RMAX)**2 / WAIST**2) * (numpy.sign(2 * RMIN - numpy.abs(r - RMAX)) + 1) / 2;
# -------------------------------------
u = u0;
#----------------Creating main matrices
utop = numpy.zeros(NMAX);
ubottom = numpy.zeros(NMAX);
alp = numpy.zeros(MMAX+2);
beta = numpy.zeros(NMAX);
gg = numpy.zeros(NMAX);
uplot = numpy.zeros(muMAX,nuMAX);
# ----------------------------------------------MARCHING - - old TBC
utop[0] = u[MMAX];
ubottom[0] = u[1];
zplot = numpy.zeros(nuMAX);
rplot = r[sprsm * numpy.r_[0:muMAX]];
P = numpy.ones(MMAX + 2);
Q = numpy.ones(MMAX + 2);
c1 = (k*h)**2;
alp0 *= c1;
alp1 *= c1;

nuu = 0;

c0 = 4 * 1j * k * h**2 / tau_int;
ci = 2 - c0;
cci = 2 + c0;

# ----------------------------------------------MARCHING - - new TBC
beta0 = -1j * 2 * cmath.sqrt(c0 - c0**2 / 4);
phi = -1 / 2 - (-1)**numpy.r_[0:NMAX+1] + ((-1)**numpy.r_[0:NMAX+1]) / 2. * ((1 + c0 / 4) / (1 - c0 / 4))**numpy.r_[1:NMAX+2];
beta[0] = phi[0];
gg[0] = 1;
qq = 2 * 1j * math.sin(k * math.sin(angle) * h) / beta0;
yy = math.cos(k * math.sin(angle) * h);
mm = cmath.exp(1j * k * math.sin(angle) * h);

for cntn in numpy.r_[0:NMAX]:

    alp[MMIN2 + 1: MMAX + 2]=alp0;
    alp[MMIN: MMIN2 + 1]=alp1;
    alp[0: MMIN]=alp0;

# Top and bottom boundary conditions
    gg[cntn + 1] = ((c0 + 2 - 2 * yy) / (c0 - 2 + 2 * yy))**cntn;

    SS = -numpy.dot(ubottom, beta) - ((qq-1)*gg[cntn+1]-numpy.dot(gg[0:cntn], beta)) * ubottom[0];
    SS1 = -numpy.dot(utop, beta) + ((qq+1)*gg[cntn+1]+numpy.dot(gg[0:cntn], beta)) * utop[0];

    beta[cntn + 1] = (numpy.dot(phi[0:cntn], beta) + phi[cntn+1])/(cntn+1);

# Initial condition at the bottom
    c = ci - alp[1];
    cconj = cci - alp[1];
    d = u[2] - cconj * u[1] + u[0];

    P[0] = -(c - beta0) / 2;
    Q[0] = -(d - beta0 * SS) / 2;

# Preparation for marching
    for cntm in numpy.r_[0:MMAX]:
        c = ci - alp[cntm + 1];
        cconj = cci - alp[cntm + 1];
        d = u(cntm + 2) - cconj * u(cntm + 1) + u(cntm);

        P[cntm + 1] = -1 / (c + P[cntm]);
        Q[cntm + 1] = -(Q[cntm] + d) * P[cntm + 1];

# Initial condition at the top
    u[MMAX + 1] = (beta0 * SS1 + Q[MMAX-1] - (P[MMAX-1] + beta0) * Q[MMAX]) / (1 - (beta0 + P[MMAX-1]) * P[MMAX]);

# Solving the system
    for cntm in numpy.r_[MMAX+1:0: -1]:
        u[cntm] = Q[cntm] - P[cntm] * u[cntm + 1];

# Preserving boundary values
    utop[cntn + 1] = u[MMAX];
    ubottom[cntn + 1] = u[1];

# Sparsing
    if cntn / sprsn - math.floor(cntn / sprsn) == 0:
        zplot[nuu] = z[cntn];
        uplot[0:muMAX, nuu]=numpy.exp(1j * k * z[cntn]) * u[sprsm * numpy.r_[0:muMAX]];
        nuu = nuu + 1;
    progress = math.round(cntn / NMAX * 100)

rplot = rplot - RMAX;

buf = io.StringIO();
buf.write("|u|: lambda = %1.2f nm   XMAX =%4.2f mum  XMIN =%4.2f mum  ZMAX =%5.0f mum " % (lam, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3));

fig = plt.figure();
fig.suptitle(buf);
X, Y = numpy.meshgrid(zplot * 1e-6, rplot * 1e-3);
plt.pcolor(X, Y, math.log10(numpy.abs(uplot)**2));
colormap('jet')
shading
interp
colorbar
xlabel('z, mm')
ylabel('x, \mum')
title(STRING)