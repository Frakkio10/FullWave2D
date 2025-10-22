#%%
simname = 'test_LV'
from fullwave2d.core.wrapper import InputData
inp = InputData.load_pickle(simname)
from fullwave2d.core.wrapper import OutputData
out = OutputData(simname)
import numpy as np
out.ez # electric field pattern at the last time step
out.ampl # amplitude over time (in units of 10 dt!)
out.phase # phase over time (in units of 10 dt!)
import pickle
import matplotlib.pyplot as plt
##fig, ax = plt.subplots()
#ax.plot(out.ampl)
#plt.show()
print(out.ampl[599])
print(out.phase[599])
dir(inp)
fig, ax = plt.subplots()
ax.plot(out.ampl)
plt.show()

# %%

with open("output_A_P.dat", "r") as f:
    contenu = f.read()
print(contenu)
#%%

with open("output.dat", "r") as f:
    contenu = f.read()
plt.imshow(f)
# %%

with open("output_A_P.dat", "r") as f:
    a = list()
    p = list()
    for line in f:
        if "#" in line:
            # on saute la ligne
            continue
        data = line.split()
        a.append(data[0])
        p.append(data[1])
fig, ax = plt.subplots()
ax.plot(a)
plt.show()

# %%
