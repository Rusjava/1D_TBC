# ------------------------------------------Asymmetrical waveguide
#          with wave and Gaussian initial condition -- stable

import math;
import cmath;
import numpy

lam = 0.1;
rho = 0.00054;

f1Si = 14.142;
f2Si = 0.141;

den1 = 2.3; # --------------------------------------SiO2
M1 = 32.0;

delta1 = den1 * rho / M1 * lam**2 * f1Si;
beta1 = den1 * rho / M1 * lam**2 * f2Si;

RMIN = 30; # ------------------------Gap semithickness
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

r = numpy.r_['c',0:MMAX+1] * h;
z=tau_int * numpy.r_['c',1:NMAX];

# -----------------------------------------------Epsilon(r)

# ------------------------------PLANE WAVE
u0 = numpy.exp(1j * k * math.sin(angle) * r);

# ---------------------------------------GAUSSIAN
#WAIST = RMIN;
#u0 = numpy.multiply(numpy.exp(-(r - RMAX)**2 / WAIST**2), (numpy.sign(2 * RMIN - numpy.abs(r - RMAX)) + 1) / 2);
# -------------------------------------
u = u0.H;
#----------------Creating main matrices
utop=numpy.matrix(numpy.zeros([NMAX,1]));
ubottom=numpy.zeros([NMAX,1]);
# ----------------------------------------------MARCHING - - old TBC
utop[0] = u[MMAX];
ubottom[0] = u[1];
zplot = numpy.zeros([nuMAX,1]);
rplot=numpy.r_['c',sprsm * numpy.r_[1:muMAX]];
P = numpy.ones(1, MMAX + 1);
Q = numpy.ones(1, MMAX + 1);
c1 = k**2 * h**2;
alp0 = c1 * alp0;
alp1 = c1 * alp1;

nuu = 0;

c0 = 4 * 1j * k * h ^ 2 / tau_int;
ci = 2 - c0;
cci = 2 + c0;

# ----------------------------------------------MARCHING - - new TBC
beta0 = -1j * 2 * cmath.sqrt(c0 - c0 ^ 2 / 4);
phi = -1 / 2 - (-1). ^ (0:NMAX) + ((-1). ^ (0:NMAX)) / 2. * ((1 + c0 / 4) / (1 - c0 / 4)). ^ (1:NMAX+1);
beta(1) = phi(1);
gg(1) = 1;
qq = 2 * 1j * math.sin(k * math.sin(angle) * h) / beta0;
yy = math.cos(k * math.sin(angle) * h);
mm = math.exp(1j * k * math.sin(angle) * h);

for cntn=1:NMAX,

    alp(MMIN2 + 2: MMAX + 2)=alp0;
    alp(MMIN: MMIN2 + 1)=alp1;
    alp(1: MMIN)=alp0;

# Top and bottom bondary conditions
    gg(cntn + 1) = ((c0 + 2 - 2 * yy) / (c0 - 2 + 2 * yy)) ^ cntn;

    SS = -ubottom * flipud(beta.')-((qq-1)*gg(cntn+1)-gg(1:cntn)*flipud(beta.'))*ubottom(1);
    SS1 = -utop * flipud(beta.')+((qq+1)*gg(cntn+1)+gg(1:cntn)*flipud(beta.'))*utop(1);

    beta(cntn + 1) = (phi(1:cntn) * flipud(beta.')+phi(cntn+1))/(cntn+1);

# Initial condition at the bottom
    c = ci - alp(2);
    cconj = cci - alp(2);
    d = u(3) - cconj * u(2) + u(1);

    P(1) = -(c - beta0) / 2;
    Q(1) = -(d - beta0 * SS) / 2;

# Preparation for marching
    for cntm=1:MMAX,
        c = ci - alp(cntm + 1);
        cconj = cci - alp(cntm + 1);
        d = u(cntm + 2) - cconj * u(cntm + 1) + u(cntm);

        P(cntm + 1) = -1 / (c + P(cntm));
        Q(cntm + 1) = -(Q(cntm) + d) * P(cntm + 1);

# Initial condition at the top
    u(MMAX + 2) = (beta0 * SS1 + Q(MMAX) - (P(MMAX) + beta0) * Q(MMAX + 1)) / (1 - (beta0 + P(MMAX)) * P(MMAX + 1));

# Solving the system
    for cntm=MMAX+1:-1: 1:
        u(cntm) = Q(cntm) - P(cntm) * u(cntm + 1);

# Preserving boudary values
    utop(cntn + 1) = u(MMAX + 1);
    ubottom(cntn + 1) = u(2);

# Sparsing
    if cntn / sprsn - floor(cntn / sprsn) == 0
        nuu = nuu + 1;
        zplot(nuu) = z(cntn);
        uplot1(1: muMAX, nuu)=cmath.exp(1j * k * z(cntn)) * u(sprsm * (1:muMAX));

    progress = math.round(cntn / NMAX * 100);

rplot = rplot - RMAX;

STRING = [sprintf('|u|: lambda = %1.2f nm   XMAX =%4.2f mum  XMIN =%4.2f mum  ZMAX =%5.0f mum '
                  , lam, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3)];

figure
pcolor(zplot * 1e-6, rplot * 1e-3, math.log10(abs(uplot1). ^ 2))
colormap('jet')
shading
interp
colorbar
xlabel('z, mm')
ylabel('x, \mum')
title(STRING)