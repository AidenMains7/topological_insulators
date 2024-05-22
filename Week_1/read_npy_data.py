import numpy as np
import matplotlib.pyplot as plt

data = np.load(".\\ProjectCode\\PhaseDiagram\\phase_diagram_data.npy")

listofM = np.split(data, 51, axis=1)
length = len(listofM)

print(length)
print(listofM[40][:])


doPlot = False
if doPlot:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(listofM)):
            ax.plot(listofM[i][0], listofM[i][1], listofM[i][2], label=f'{listofM[i][0,0]}')
    ax.set_xlabel("M")
    ax.set_ylabel("B_tilde")

    plt.show()
