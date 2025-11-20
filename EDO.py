import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Button, TextBox
from matplotlib import axes

def dagger(a):
    return 0 if a<0 else a

# λ(t,T,A,L) ; modeliza cuantas chinches, cada cuanto y cunatas vaces se hechan las chinches

# Pulsos periódicos modelados como ventanas de duración pulse_width
def Lambda(t, Periodo, Amplitud, Limite, pulse_width=0.1):
    if Periodo <= 0:
        return 0.0
    # n = número de pulso actual (0,1,2,...). Añadimos small tol para robustez.
    n = int(np.floor((t + 1e-9) / Periodo))
    if (n < Limite) and (abs(t - n*Periodo) <= pulse_width/2):
        # devolvemos una tasa elevada durante pulse_width; la integral sobre el pulso ~ Amplitud
        return Amplitud / pulse_width
    return 0.0


rho1 = 0.33
rho2 = 0.30
r1 = 0.5 #Provisional
r2 = 0.02 #Provisional
r3 = 0.02 #Provisional
r4 = 0.03   #Provisional  
r5 = 0.05 #Provisional
k1 = 6000 
k2 = 8500
h0 = 0.005
h1 = 0.001 #Provisional
h2 = 0.0049
h3 = 0.001 #Provisional
alpha = 5e-05 #Provisional 
beta  = 3e-05  #Provisional
gamma = 5e-05  #Provisional 
delta = 2e-04  #Provisional
epsilon = 1e-03 #Provisional
dseta = 1e-03 #Provisional
nu = 0.0005  #Provisional
#Lambda = 0.05 #Provisional
psi = 1e-04 #Provisional
A_min = 6000.0 #Provisional
eta = 0.0005  #Provisional

# Parametros de λ();

T = 30 # Cada cuantos dias se hechan las chinches //Provisional
A = 2 # Cuantas chinches se hechan //Provisional
L = 10 # Cantidad de veces que se hechan las chinches //Provisional

A0 = 7000 
F0 = 2000
D0 = 100 #Provisional
O0 = 1 #Provisional
V0 = 6 #Provisional

def edo_huerta(t, y, r1,r2,r3,r4,r5,k1,k2,h0,h1,h2,h3,alpha,beta,gamma,delta,epsilon,dseta,nu,T,A,L,psi,A_min,eta):
    A,F,D,O,V = y
    dA = (r1 * A * (1 - (A / k1))) + ((alpha * A *F) / (1 + (h0 * A))) - (nu * A) - ((V * eta * dagger(A-A_min)) / (1 + (h1 * dagger(A-A_min))))
    dF = (r2 * F * (1 - (F / k2))) + ((beta * A * F) / (1 + (h2 * A))) - (gamma * F * D)
    dD = (- r3 * D) +((delta * D * F)) - (epsilon * O * D)
    dO = ((Lambda(t,T,A,L)) -( r4*O )+ (dseta*D*O))
    dV = (-r5*V+((V*psi*dagger(A-A_min))/(1+(h3*dagger(A-A_min)))))
    return [dA, dF, dD, dO, dV]

var = [A0, F0, D0, O0, V0, 
       r1, r2, r3, r4, r5,
       k1, k2, h0, h1, h2, h3,
       alpha, beta, gamma, delta, epsilon, 
       dseta, nu, T , A ,L, psi, A_min, eta]
variables_names = ["A0", "F0", "D0", "O0", "V0", 
                   "r1", "r2", "r3", "r4", "r5",
                   "k1", "k2", "h0", "h1", "h2", "h3", 
                   "alpha", "beta", "gamma", "delta", 
                   "epsilon", "dseta", "nu", "Lambda(Periodo)",
                   "Lambda(Amplitud)","Lambda(Limite)","Psi", "A_min", "eta"]

variables_symbols = ["A0", "F0", "D0", "O0", "V0",
                     "r1", "r2", "r3", "r4", "r5", 
                     "k1", "k2", "h0", "h1", "h2", "h3"
                     , "α", "β", "γ", "δ", "ε", "ζ", "μ"
                     , "λ_T", "λ_A", "λ_L", "ψ", "A_", "η"]

t_max = 360
t_eval = np.linspace(0, t_max, 1000 )

sol = solve_ivp(
    edo_huerta, (0, t_max), [A0, F0, D0, O0, V0],
    args=(r1, r2, r3, r4, r5, k1, k2, h0, h1, h2, h3, alpha, beta, gamma, delta, epsilon, dseta, nu, T,A,L, psi, A_min, eta),
    t_eval=t_eval, rtol=1e-6, atol=1e-8
    )

fig = plt.figure(figsize=(15, 8))
fig2, ax2 = plt.subplots(figsize=(10, 6))

line0, = ax2.plot(sol.t, sol.y[0], label='A')
line1, = ax2.plot(sol.t, sol.y[1], label='F')
line2, = ax2.plot(sol.t, sol.y[2], label='D')
line3, = ax2.plot(sol.t, sol.y[3], label='O')
line4, = ax2.plot(sol.t, sol.y[4], label='V')


suppress_update = False


ax2.set_xlabel('Time')
ax2.set_ylabel('Values')
ax2.set_title('EDO Solutions')
ax2.legend()
ax2.grid(True)

fig.subplots_adjust(left=0.1, right=0.2)

textboxes = []
axe = []



n = len(var)
cols = 3
rows = int(np.ceil(n / cols))

# Coordenadas base para las columnas
col_width = 0.18
start_x = 0.02
start_y = 0.25
offset_x = 0.0
offset_y = 0.0
box_height = 0.06  # altura de cada fila

index = 0
for c in range(cols):
    for r in range(rows):
        if index >= n:
            break
        
        x = start_x + c * col_width + offset_x
        y = start_y + (rows - r) * box_height + offset_y

        aux = fig.add_axes([x, y, col_width - 0.02, box_height * 0.9])
        tb = TextBox(
            ax=aux,
            label=variables_symbols[index] + " ",
            initial=str(var[index])
        )

        textboxes.append(tb)
        axe.append(aux)
        index += 1





# handler común para los TextBox que respeta la bandera suppress_update
def _textbox_handler(text):
    if suppress_update:
        return
    update(None)

# conectar el handler a todos los TextBox
for tb in textboxes:
    tb.on_submit(_textbox_handler)



# Función que actualiza la solución leyendo las cajas de texto
def update(arg=None):
    # Leer valores de las cajas; si falla la conversión se mantiene el valor previo
    vals = [None]*len(textboxes)
    for idx in range(len(textboxes)):
        txt = textboxes[idx].text
        try:
            vals[idx] = float(txt)
        except:
            vals[idx] = var[idx]

    # convertir L (índice 25) a entero (nº de pulsos)
    L_val = int(round(vals[25])) if len(vals) > 25 else int(round(var[25]))

    # Preparar condiciones iniciales a partir de las primeras 5 cajas
    y0 = [vals[0], vals[1], vals[2], vals[3], vals[4]]

    # Llamar a solve_ivp con los parámetros leídos desde las cajas
    try:
        sol2 = solve_ivp(
            edo_huerta, (0, t_max), y0,
            args=(
                vals[5],   # r1
                vals[6],   # r2
                vals[7],   # r3
                vals[8],   # r4
                vals[9],   # r5
                vals[10],  # k1
                vals[11],  # k2
                vals[12],  # h0
                vals[13],  # h1
                vals[14],  # h2
                vals[15],  # h3
                vals[16],  # alpha
                vals[17],  # beta
                vals[18],  # gamma 
                vals[19],  # delta
                vals[20],  # epsilon
                vals[21],  # dseta
                vals[22],  # nu

                vals[23],  # Lambda_Periodo (T)
                vals[24],  # Lambda_Amplitud (A)
                vals[25],    # Lambda_Limite (L)

                vals[26],  # psi
                vals[27],  # A_min
                vals[28]   # eta 
            ),
            t_eval=t_eval, rtol=1e-6, atol=1e-8
        )
    except Exception as e:
        print("Error al integrar:", e)
        return

    # Verificar que la integración fue exitosa
    if not getattr(sol2, "success", True):
        print("solve_ivp falló o no convergió. mensaje:", getattr(sol2, "message", None))
        return

    # Actualizar X e Y de las líneas (evita shape mismatch)
    # las nuevas x serán sol2.t
    line0.set_xdata(sol2.t); line0.set_ydata(sol2.y[0])
    line1.set_xdata(sol2.t); line1.set_ydata(sol2.y[1])
    line2.set_xdata(sol2.t); line2.set_ydata(sol2.y[2])
    line3.set_xdata(sol2.t); line3.set_ydata(sol2.y[3])
    line4.set_xdata(sol2.t); line4.set_ydata(sol2.y[4])

    # ajustar límites en x e y automáticamente
    ax2.relim()
    ax2.autoscale_view()

    # Forzar redraw
    fig2.canvas.draw_idle()# Botón reset: vuelve a los valores iniciales de 'var'



def reset(event):
    plt.close('all')  # cierra todas las ventanas de matplotlib
    os.execv(sys.executable, [sys.executable] + sys.argv)

reset_ax = fig.add_axes([0.8, 0.03, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset')

reset_button.on_clicked(reset);

plt.show()
