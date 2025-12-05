import json
from modelo import run_simulation
from animacion import animar, plot_tiempos
from make_mask import make_mask
import matplotlib.pyplot as plt
import time
def main():

    # Cargar parámetros desde JSON
    with open("parametros.json", "r") as f:
        cfg = json.load(f)

    params = cfg["params"]
    init_vals = cfg["init_vals"]
    D = cfg["D"]
    area_p = cfg["S_campo"]

    make_mask(area = area_p)
    frames_A, frames_F,frames_D,frames_O,frames_V, times, mean_A, mean_F, mean_D, mean_O, mean_V = run_simulation(
        params,
        init_vals,
        D,
      
        cfg["Nx"],
        cfg["Ny"],
        cfg["T"],
        cfg["dt"],
        cfg["mask_path"]
     )

    # Graficar animación
    animar(frames_A, frames_F,frames_D,frames_O,frames_V,True)

    # Graficar evolución temporal
    plot_tiempos(times,  mean_A, mean_F, mean_D, mean_O, mean_V)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
