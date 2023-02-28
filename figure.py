import numpy as np
import matplotlib.pyplot as plt
import json
import qiskit.quantum_info as qi

exact  = json.load(open('data/exact_result_J0.25_B1.dat'))
data80   = json.load(open('data/trial_results_shots_80.dat'))
data800   = json.load(open('data/trial_results_shots_800.dat'))
data8000  = json.load(open('data/trial_results_shots_8000.dat'))
data80000   = json.load(open('data/trial_results_shots_80000.dat'))

Sx = np.matrix([[0,1],[1,0]])
Sy = np.matrix([[0,-1j],[1j,0]])
Sz = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1,0],[0,1]])

def makeStatevectorExact(data):
    length = len(data['times'])
    states = []

    for i in range(length):
        sz = data['Sz'][i]
        sx = data['Sx'][i]
        sy = data['Sy'][i]
        rho = (I + sx*Sx + sy*Sy + sz*Sz )/2
        states.append(rho)

    return states

def makeStatevector(data):
    length = len(data['times'])
    states = []

    for i in range(length):
        sz = data['Sz_0'][i]
        sx = data['Sx_0'][i]
        sy = data['Sy_0'][i]
        rho = (I + sx*Sx + sy*Sy + sz*Sz )/2
        states.append(rho)

    return states[:-1]

def infidelity(state1, state2):
    dt = 0.05
    infidelitySum = 0

    for i in range(len(state1)):
        infidelitySum += 1 - np.trace(np.matmul(state1[i].H,state2[i]))**2
        infidelitySum = infidelitySum * dt

    return infidelitySum

samples = [10**5, 10**6, 10**7, 10**8]
ins = np.zeros(4)
ins[0] = infidelity(makeStatevector(data80),makeStatevectorExact(exact))
ins[1] = infidelity(makeStatevector(data800),makeStatevectorExact(exact))
ins[2] = infidelity(makeStatevector(data8000),makeStatevectorExact(exact))
ins[3] = infidelity(makeStatevector(data80000),makeStatevectorExact(exact))


# Figure 2 in the paper
plt.figure(1)
plt.plot(samples, ins)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Samples")
plt.ylabel("$\Delta_F(T)$")


# Figure 3 in the paper
# Plot of Sx
n = 60
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(exact['times'][:n],exact['Sx'][:n],label ="Exact",linestyle='dashed',linewidth=1.2,color='black')
ax[0].errorbar(data80['times'][:n],data80['Sx_0'][:n],yerr=data80['err_Sx_0'][:n],label="pVQD: 80 shots",marker='o',linestyle='',elinewidth=1,color='g',capsize=2,markersize=3.5)
ax[0].errorbar(data800['times'][:n],data800['Sx_0'][:n],yerr=data800['err_Sx_0'][:n],label="pVQD: 800 shots",marker='o',linestyle='',elinewidth=1,color='r',capsize=2,markersize=3.5)
ax[0].errorbar(data8000['times'][:n],data8000['Sx_0'][:n],yerr=data8000['err_Sx_0'][:n],label="pVQD: 8000 shots",marker='o',linestyle='',elinewidth=1,color='b',capsize=2,markersize=3.5)
ax[0].errorbar(data80000['times'][:n],data80000['Sx_0'][:n],yerr=data80000['err_Sx_0'][:n],label="pVQD: 80000 shots",marker='o',linestyle='',elinewidth=1,color='purple',capsize=2,markersize=3.5)


ax[0].set(ylabel=r"$\langle\sigma_{x}\rangle_{1}$")
ax[0].set_ylim(ymax=1.1,ymin=-1.1)

# Plot of Sz

ax[1].plot(exact['times'][:n],exact['Sz'][:n],label ="Exact",linestyle='dashed',linewidth=1.2,color='black')
ax[1].errorbar(data80['times'][:n],data80['Sz_0'][:n],yerr=data80['err_Sz_0'][:n],label="pVQD: 80 shots",marker='o',linestyle='',elinewidth=1,color='g',capsize=2,markersize=3.5)
ax[1].errorbar(data800['times'][:n],data800['Sz_0'][:n],yerr=data800['err_Sz_0'][:n],label="pVQD: 800 shots",marker='o',linestyle='',elinewidth=1,color='r',capsize=2,markersize=3.5)
ax[1].errorbar(data8000['times'][:n],data8000['Sz_0'][:n],yerr=data8000['err_Sz_0'][:n],label="pVQD: 8000 shots",marker='o',linestyle='',elinewidth=1,color='b',capsize=2,markersize=3.5)
ax[1].errorbar(data80000['times'][:n],data80000['Sz_0'][:n],yerr=data80000['err_Sz_0'][:n],label="pVQD: 80000 shots",marker='o',linestyle='',elinewidth=1,color='purple',capsize=2,markersize=3.5)

ax[1].set(ylabel=r"$\langle\sigma_{z}\rangle_{1}$",xlabel=r'$t$')
ax[1].set_ylim(ymax=1.1,ymin=-1.1)

# Legend above the plots
lines, labels = ax[0].get_legend_handles_labels()
ax[0].legend(lines , labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),ncol=2, fancybox=True, shadow=False)



# figure 4 of the paper
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(data800['times'][:n],data800['iter_number'][:n],marker = 'o', linestyle ="",markersize=3.5, color = 'b')
ax[0].set(ylabel="Optimization steps")

ax[1].plot(data800['times'][:n],np.subtract(1,data800['init_F'][:n]),marker = 'o', linestyle ="",markersize=3.5, color = 'b')
ax[1].plot(data800['times'][:n],np.subtract(1, data800['final_F'][:n]),marker = 'o', linestyle ="",markersize=3.5, color = 'r')
ax[1].set(ylabel="Infidelity", yscale = "log")

plt.show()
