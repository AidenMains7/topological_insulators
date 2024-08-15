import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")

file = "./ProjectCode_1/PhaseDiagram/phase_diagram_data.npy"


data = np.load(file)


M = data[0]
B_tilde = data[1]
bott = data[2]

print(M.size/(np.unique(M).size))


#unique values
M_vals = np.unique(M)
B_tilde_vals = np.unique(B_tilde)

#bounds for plot
x_bounds = (M_vals.min(), M_vals.max())
y_bounds = (B_tilde_vals.min(), B_tilde_vals.max())

#organize the bott array into a surface over the meshgrid
arrs = np.split(bott, M_vals.size)
Z = np.stack(arrs, axis=0)

fig = plt.figure(figsize=(10,10))
plt.imshow(Z, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin='lower', aspect='auto', cmap='viridis')
cbar = plt.colorbar(label='Bott Index')
cbar.set_label('Bott Index', rotation=0)

x_ticks = np.linspace(x_bounds[0], x_bounds[1], 5)
y_ticks = np.linspace(y_bounds[0], y_bounds[1], 5)
plt.xticks(ticks=x_ticks, labels=np.round(x_ticks, 2))
plt.yticks(ticks=y_ticks, labels=np.round(y_ticks, 2))

plt.xlabel("M")
plt.ylabel("B_tilde", rotation=0)


plt.show()



