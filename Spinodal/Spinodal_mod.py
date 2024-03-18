import numpy as np
from scipy.optimize import least_squares
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('sos_paper_data.xlsx')
Tbl_atm = pd.read_excel('ref_atm_data.xlsx')


def prepare_data(data, code='ABchZl'):
    numeric = data.iloc[:, 2:7]
    text = data.iloc[:, 0:2]
    Table = data
    Ilcode = code
    return numeric, text, Table, Ilcode


num, txt, Tb1, ILcode = prepare_data(data, code='ABchZl')

indc = data[data.iloc[:, 1] == ILcode].index
numL = data[data.iloc[:, 1] == ILcode]
indP = numL[numL.iloc[:, 3] > 120].index
T = numL.loc[indP, 'temperature']

P = numL.loc[indP, 'pressure'] * 10 ** -3
ce = numL.loc[indP, 'sos']
Tu = T.round().unique()
Tu.sort()

### Speed of sound fitting
from scipy.optimize import curve_fit


#  Definite function of speed of sound as a function of pressure
def cspin(p, *b):
    return (1 + b[0] * p) ** b[1]


def cs(T, data):
    coefficients = np.polyfit(data['Te'], data['ce'], 2)
    return np.polyval(coefficients, T)


ce0 = cs(Tu, Tbl_atm)
Pf = np.arange(0.1, 201, 5).reshape(-1, 1)


def plot_data(Tu, T, P, ce, ce0, Pf):
    clr = []
    plt.gca().set_prop_cycle(None)
    b = np.zeros((len(Tu), 2))
    ce_f = []

    for j in range(len(Tu)):
        ind = np.where(np.round(T) == Tu[j])[0]
        pl = plt.plot(P.iloc[ind], ce.iloc[ind], 'o', markersize=4)
        clr.append(pl[0].get_color())

        p = np.polyfit((P.iloc[ind] - 0.1), (ce.iloc[ind] / ce0[j]) ** 3, 1)
        b0 = [p[0], 1 / 3]
        b, _ = curve_fit(cspin, P.iloc[ind] - 0.1, ce.iloc[ind] / ce0[j], p0=b0)
        cef_j = ce0[j] * cspin(Pf - 0.1, *b)
        ce_f.append(cef_j)
        plt.plot(Pf, ce_f[j], '-', color=clr[j])

    FS = 12
    FN = 'Times New Roman'
    plt.xlabel('$P,~\mathrm{MPa}$', fontsize=12, fontname=FN, labelpad=10)
    plt.ylabel('$c,~\mathrm{m/s}$', fontsize=12, fontname=FN, labelpad=10)
    plt.xticks(fontsize=12, fontname=FN)
    plt.yticks(fontsize=12, fontname=FN)
    plt.show()
    return ce_f, FS, FN


# Example of usage
cef, FS, FN = plot_data(Tu, T, P, ce, ce0, Pf)
p = np.polyfit(Tbl_atm['Te'] / 1e3, Tbl_atm['rhoe'] / 1e3, 1)
b0 = [p[1], p[0], 1, 1]
'''Definition of Daridon function, and residuals'''


def Daridon(T, *b):
    return (b[0] + b[1] * T ** (1 / b[2])) ** (1 / b[3])


def residualsDaridon(b, T, y):
    return Daridon(T, *b) - y


b0 = [p[1], p[0], 1, 1]
T_res = Tbl_atm['Te'].values / 1e3
y_res = Tbl_atm['rhoe'].values / 1e3

result = least_squares(residualsDaridon, b0, args=(T_res, y_res), method='trf')

brho = result.x


def dDaridon(T, *b):
    return -(b[1] / (b[2] * b[3])) * T ** (1 / b[2] - 1) / (b[0] + b[1] * T ** (1 / b[3]))


'''Alpha_p and alpha_p derivative definition'''


def alphaP(T, *b):
    return dDaridon(T / 1e3, *b) / 1e3


def dalphaP(T, *b):
    return alphaP(T, *b) ** 2 * (1 - (1 - 1 / b[2]) / (b[3] * alphaP(T, *b) * T))


def rhos(T, *b):
    return 1e3 * Daridon(T / 1e3, *brho)


def Cps(T, *bcp):
    return 1e3 * Daridon(T / 1e3, *bcp)


'''Heat capacity fitting'''
p = np.polyfit(Tbl_atm['Te_2'] / 1e3, Tbl_atm['Cpe'] / 1e3, 1)
b0 = [p[0], p[1], 1, 1]

TCpAtm = Tbl_atm['Te_2'].values / 1e3
CpAtm = Tbl_atm['Cpe'].values / 1e3

result = least_squares(residualsDaridon, b0, args=(TCpAtm, CpAtm), method='trf')
'''Fitting function coefficients'''
bcp = result.x


def kappaTs(T):
    return (1. / np.square(cs(T, Tbl_atm)) + T * np.square(alphaP(T, *brho)) / Cps(T, *bcp) / rhos(T, *brho))


'''T has more than 40 records by default, we replace fo with Tu, which refers to a specific isobar '''
T = Tu
ind = np.arange(len(T))
'''Create empty container for results, dim is Pf x T'''
rhoa = np.zeros((len(Pf), len(T)))
'''zeros_like - thanks to that you don't have to write zeros every time'''
Cpf = np.zeros_like(rhoa)
alpha = np.zeros_like(rhoa)
dalpha = np.zeros_like(rhoa)
dCpf = np.zeros_like(rhoa)
drhoa = np.zeros_like(rhoa)
dCpfe = np.zeros_like(rhoa)
drhoae = np.zeros_like(rhoa)
kappaTa = np.zeros_like(rhoa)
'''Add to the first line the values for p = atm'''
rhoa[0, :] = rhos(T, *brho)
Cpf[0, :] = Cps(T, *bcp)
alpha[0, :] = alphaP(T, *brho)
dalpha[0, :] = dalphaP(T, *brho)
cef = np.array(cef)

TCpAtm = Tbl_atm['Te_2'].values / 1e3
CpAtm = Tbl_atm['Cpe'].values / 1e3

result = least_squares(residualsDaridon, b0, args=(TCpAtm, CpAtm), method='trf')

bcp = result.x

for j in range(1, len(Pf)):
    ind = np.arange(len(T))
    h = (Pf[j] - Pf[j - 1]) * 1e6
    dCpf[j, ind] = -(T[ind] / rhoa[j - 1, ind]) * (alpha[j - 1, ind] ** 2 + dalpha[j - 1, ind])
    Cpf[j, ind] = Cpf[j - 1, ind] + dCpf[j, ind]
    drhoa[j, ind] = (T[ind] * alpha[j - 1, ind] ** 2 / Cpf[j - 1, ind] + 1 / (cef[ind, j - 1].flatten()) ** 2)
    rhoa[j, ind] = rhoa[j - 1, ind] + drhoa[j, ind] * h
    pj = np.polyfit(T[ind] / 1e3, rhoa[j, ind] / 1e3, 1)
    b0j = [pj[1], pj[0], 1, 1]
    res = least_squares(residualsDaridon, b0j, args=(T[ind] / 1e3, rhoa[j, ind] / 1e3), method='trf')
    brhoj = res.x
    '''This is an old function that is not optimized. 
    The least_squares predefined earlier performs much better. '''
    # brhoj, _ = curve_fit(Daridon, T[ind] / 1e3, rhoa[j, ind] / 1e3, p0=b0j,maxfev = 80000)
    print(f"brhoj before corrector:{brhoj} ----- iteration: {j}")
    alpha[j, ind] = alphaP(*brhoj, T[ind])
    dalpha[j, ind] = dalphaP(T[ind], *brhoj)
    ###Corrector
    dCpfe[j, ind] = -(T[ind] / rhoa[j, ind]) * (alpha[j, ind] ** 2 + dalpha[j, ind])
    Cpf[j, ind] = Cpf[j - 1, ind] + (dCpf[j, ind] + dCpfe[j, ind]) * h / 2
    drhoae[j, ind] = (T[ind] * alpha[j, ind] ** 2 / Cpf[j, ind] + 1 / (cef[ind, j - 1].flatten() ** 2))
    rhoa[j, ind] = rhoa[j - 1, ind] + (drhoa[j, ind] + drhoae[j, ind]) * h / 2
    pj = np.polyfit(T[ind] / 1e3, rhoa[j, ind] / 1e3, 1)
    b0j = [pj[1], pj[0], 1, 1]
    res = least_squares(residualsDaridon, b0j, args=(T[ind] / 1e3, rhoa[j, ind] / 1e3), method='trf')
    brhoj = res.x
    print(f"brhoj After corrector:{brhoj} ----- iteration: {j}")
    alpha[j, ind] = alphaP(*brhoj, T[ind])
    dalpha[j, ind] = dalphaP(T[ind], *brhoj)
    kappaTa[j, ind] = (1 / (cef[ind, j - 1].flatten()) ** 2 + T[ind] * alpha[j, ind] ** 2 / Cpf[j, ind] / rhoa[j, ind])


# Load D4 data
# D4 = scipy.io.loadmat('D4.mat')['D4']
# D4Hir = scipy.io.loadmat('D4hir.mat')['D4Hir']
#
# # Extract unique values
# TuD4 = np.unique(D4[:, 0])
# TuD4Hir = np.unique(D4Hir[:, 0])
def load_D4_data(d4_fn='D4.mat', d4hir_fn='D4hir.mat'):
    # Load D4 data
    D4 = scipy.io.loadmat(d4_fn)['D4']
    D4Hir = scipy.io.loadmat(d4hir_fn)['D4Hir']

    # Extract unique values
    TuD4 = np.unique(D4[:, 0])
    TuD4Hir = np.unique(D4Hir[:, 0])

    return D4, D4Hir, TuD4, TuD4Hir


def plot_graph(Pf, rhoa, D4, TuD4, D4Hir, TuD4Hir, FS=12):
    # Creating a new graph
    plt.figure()
    plt.plot(Pf, rhoa)
    # Adding points from D4
    for j in range(4):
        ind = np.where(D4[:, 0] == TuD4[j])
        plt.errorbar(D4[ind[0], 1] * 1e-3, D4[ind[0], 2], yerr=5 * D4[ind[0], 2] / D4[ind[0], 2], fmt='x')
    # Adding points with D4Hir
    for j in range(1):
        ind = np.where(D4Hir[:, 0] == TuD4Hir[j])
        plt.errorbar(D4Hir[ind[0], 1] * 1e-3, D4Hir[ind[0], 2], yerr=2.9 * D4Hir[ind[0], 2] / D4Hir[ind[0], 2], fmt='+')
    # Adding axis labels
    plt.xlabel('$P,~\mathrm{MPa}$', fontsize=FS)
    plt.ylabel('$\\rho,~\\mathrm{kg\cdot m^{-3}}$', fontsize=FS)

    # Setting the font
    plt.rc('font', family='Times New Roman', size=FS)
    # Displaying the graph
    plt.show()


D4, D4Hir, TuD4, TuD4Hir = load_D4_data()
plot_graph(Pf, rhoa, D4, TuD4, D4Hir, TuD4Hir)
'''You can indicate the name of the liquid under consideration - 
the program will create the appropriate folder and save the results for the density there '''

import os
def save_file(il_name='Test'):

    directory_path = il_name
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    data = pd.DataFrame(rhoa)
    Pf_copy = Pf.copy()
    Pf_copy[1:] = Pf_copy[1:].astype(int)

    header_T = [str(int(temp)) + ' K' for temp in T]
    data.insert(0, 'P (MPa)', Pf_copy)
    data.columns = ['P (MPa)'] + header_T

    data.to_excel(os.path.join(directory_path, il_name + '_RHO_DATA_SPINODAL.xlsx'), index=False)
'''To save- just type in I.E. PyCharm save_file('My_name'). Script should create folder 
../folderWithScript/<My_name>/My_name_RHO_DATA_SPINODAL.XLSX 
'''