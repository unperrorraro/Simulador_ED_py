
import numpy as np
from PIL import Image

def make_mask(area = 1000, espacio = 0.50,siz =256,siz_planta = 0,border=4):
    # Parámetros físicos del terreno
    #

    if (area == None):
        area_m2 = 1000
    else:
        area_m2 = area                   # área en m^2

    spacing_m = espacio                 # separación entre plantas en metros
    field_side_m = area_m2 ** 0.5    # lado del campo en metros (~31.62)

  
    # Parámetros de la imagen
    img_size = siz                  # resolución deseada
    meters_per_pixel = field_side_m / img_size
    spacing_px = spacing_m / meters_per_pixel
    if (siz_planta == 0):
        planta_px = np.round(spacing_px / 3)
    else:
        planta_px = siz_planta/meters_per_pixel
    


    print(f"Separación entre plantas en píxeles: {spacing_px:.2f}")
    print(f"Tamaño de plantas en píxeles: {planta_px:.2f}")

    # Crear la máscara vacía
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

  
    centers_i = np.arange(0, siz, int(round(spacing_px)))
    centers_j = np.arange(0, siz, int(round(spacing_px)))

    # Dibujar círculos
    for i_c in centers_i:
        for j_c in centers_j:
            # Definir sub-cuadrado para evitar recorrer toda la imagen
            i_min = max(0, int(i_c - planta_px))
            i_max = min(siz, int(i_c + planta_px + 1))
            j_min = max(0, int(j_c - planta_px))
            j_max = min(siz, int(j_c + planta_px + 1))

            # Crear malla local
            I, J = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')
            # Distancia al centro
            dist2 = (I - i_c)**2 + (J - j_c)**2
            # Píxeles dentro del radio
            mask[I, J] = np.where(dist2 <= planta_px**2, 255, mask[I, J])
            

    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    # Guardar como PNG
    img = Image.fromarray(mask)
    img.save("campo_fresas_mask.png")

    print("Máscara generada: campo_fresas_mask.png")

make_mask(siz= 256);

