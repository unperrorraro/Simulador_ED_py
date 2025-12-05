import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter,PillowWriter
_animacion_global = None
def animar(frames_A, frames_F,frames_D,frames_O,frames_V,guardar_gif=False, filename="fresas.gif"):
    fig, axs = plt.subplots(1,5,figsize=(12,4))
    global _animacion_global

    A_all = np.array(frames_A)
    F_all = np.array(frames_F)
    D_all = np.array(frames_D)
    O_all = np.array(frames_O)
    V_all = np.array(frames_V)

    vminA, vmaxA = A_all.min(), A_all.max()
    vminF, vmaxF = F_all.min(), F_all.max()
    vminD, vmaxD = D_all.min(), D_all.max()
    vminO, vmaxO = O_all.min(), O_all.max()
    vminV, vmaxV = V_all.min(), V_all.max()

    imA = axs[0].imshow(frames_A[0], origin='lower', cmap='magma',
                    vmin=vminA, vmax=vmaxA)

    imF = axs[1].imshow(frames_F[0], origin='lower', cmap='bone',
                    vmin=vminF, vmax=vmaxF)

    imD = axs[2].imshow(frames_D[0], origin='lower', cmap='magma',
                    vmin=vminD, vmax=vmaxD)

    imO = axs[3].imshow(frames_O[0], origin='lower', cmap='magma',
                    vmin=vminO, vmax=vmaxO)

    imV = axs[4].imshow(frames_V[0], origin='lower', cmap='magma',
                    vmin=vminV, vmax=vmaxV)

    for ax, title in zip(axs, ["A(x,y,t)", "F(x,y,t)", "D(x,y,t)", "O(x,y,t)", "V(x,y,t)"]):
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    def update(i):  
        imA.set_array(frames_A[i])
        imF.set_array(frames_F[i])  
        imD.set_array(frames_D[i]) 
        imO.set_array(frames_O[i])    
        imV.set_array(frames_V[i]) 


        return [imA, imF, imD,imO,imV]

    ani = FuncAnimation(fig, update, frames=len(frames_A), interval=80)

    if guardar_gif:
        writer = PillowWriter(fps=15)
        ani.save(filename, writer=writer)
        print(f"Animación guardada en {filename}")
    _animacion_global = ani   # <- Mantener referencia viva
    plt.show()
    return ani



def plot_tiempos(times,  mean_A, mean_F, mean_D, mean_O, mean_V):
    plt.figure(figsize=(7,4))
    plt.plot(times, mean_A, 'y-', label='⟨A⟩')
    plt.plot(times, mean_F, 'g-', label='⟨F⟩')
    plt.plot(times, mean_D, 'k-', label='⟨D⟩')
    plt.plot(times, mean_O, 'b-', label='⟨O⟩')
    plt.plot(times, mean_V, 'r-', label='⟨V⟩')
    plt.grid()
    plt.xlabel("Tiempo")
    plt.ylabel("Media espacial")
    plt.legend()
    plt.title("Evolución temporal de promedios")
    plt.show()


