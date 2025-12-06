 ![simulacion detallada, area 1000 m2 en gif](/edp/save-media/fresas_detallado_1000.gif)
 
# Uso:


## Clonar repositorio:
```
mkdir ProyectoEDO
cd ProyectoEDO
git clone https://github.com/unperrorraro/Simulador_ED_py
```
## Entorno virtual

### Linux y MAC
```
python -m venv env
source env/bin/activate
```
### Windows

```
python -m venv env
env\Scripts\activate
```
## Instalar dependencias
```
cd Simulador_ED_py
pip install -r requeriments.txt
```
## EDO


### Cambiar Parametros

Los parámetros del sistema de EDOs se cambian durante la ejecución del script

### Simular
```
python EDO.py
```

## EDP

### Cambiar Parametros

Los parámetros del sistema de EDPs se cambian antes de la ejecución del modelo.
Los parametros estan almacenados en "edp/parametros.json"


### Parametros especiales


- S_campo : "Superficie del campo" en la practica cambia la escala de la máscara de fresas
- Nx,Ny : Dimensiones de las matrizas utilizadas en el modelo y resolucion de los frames generados **(Aumenta mucho el tiempo de renderizado)**
- T : Tiempo **(En dias)** Que dura la simulacion
- mask_path : por si se quiere usar una mascara distina a la generada. (si se quiere por defecto : campo_fresas_mask.png) 

### JSONs

+ save-media/parametros.json.bak : copia de seguridad de parametros de simulación ligera (3-5 min)
+ save-media/parametros_detallado.json.bak : copia de seguridad de parametros de simulación pesad (1H 30min - 1H 45min)


### Simular
```
cd edp
python main.py
```

# Ejemplos:

simulacion detallada, area 1000 m2
 ![simulacion detallada, area 1000 m2 en gif](/edp/save-media/fresas_detallado_1000.gif)

simulacion detallada, area 300 m2 
 ![simulacion detallada, area 300 m2 en gif](/edp/save-media/fresas_detallado_300.gif)

 simulacion no detallada, area 1000 m2
 ![simulacion no detallada, area 1000 m2 en gif](/edp/save-media/fresas_no_detallado_1000.gif)

 simulacion no detallada, area 300 m2
 ![simulacion no detallada, area 300 m2 en gif](/edp/save-media/fresas_no_detallado_300.gif)




