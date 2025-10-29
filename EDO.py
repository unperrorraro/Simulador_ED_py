import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Button, Slider
from matplotlib import axes

def dagger(a):
    return a<0 ? 0 : a

r1 = 2
r2 = 2
r3 = 2
r4 = 2
r5 = 2;
k1 = 2
k2 = 2
h0 = 2
h1 = 2
alpha = 2
beta = 2
gamma = 2
delta = 2
epsilon = 2
dseta = 2
nu = 2
Lambda = 2
psi = 2

A0 = 2
F0 = 2
D0 = 2
O0 = 2
V0 = 2



def edo_huerta(t, y, r0,r2,r3,r4,k1,k2,h0,alpha,beta,gamma,delta,epsilon,dseta,nu,Lambda):
    A,F,D,O,V = y
    dA = r0*A*(1-(A/k1)) +((alpha*A*F)/(1+(h0*A))) - nu*A - (V*psi*dagger(A-A0))/(1+(h1*dagger(A-A0)))

    dF = r1*F*(1-(F/k2)) +((beta*A*F)) - gamma*F*D
    dD = (r2*D*-1) +((delta*D*F)) - epsilon*O*D
    dO = Lambda - r3*O + (dseta*D*O)
    dV = -r5*V+(V*psi*dagger(A-A0))/(1+(h1*dagger(A-A0)))
    return [dA, dF, dD,dO,dV]

var =              [  A0,  F0,  D0,  O0,V0,  r1,  r2,  r3,  r4,  r5,  k1,  k2,  h0,h1,  alpha,  beta,   gamma,  delta, epsilon,   dseta,  nu,  Lambda,psi]
variables_names  = [ "A0","F0","D0","O0","V0","r1","r2","r3","r4","r5","k1","k2","h0","h1","alpha","beta","gamma","delta","epsilon","dseta","nu","Lambda","Psi"]
variables_symbols  =["A0","F0","D0","O0","V0","r1","r2","r3","r4","r5","k1","k2","h0","h1","α",    "β",  "γ",     "δ",   "ε",      "ζ",    "μ",  "λ", "ψ"]

t_max = 300

t_eval = np.linspace(0, t_max, 2000)

sol = solve_ivp(
    edo_huerta, (0, t_max), [A0, F0, D0, O0,V0],
    args=(r1,r2,r3,r4,r5,k1,k2,h0,h1,alpha,beta,gamma,delta,epsilon,dseta,nu,Lambda,psi),
    t_eval=t_eval, rtol=1e-6, atol=1e-8
    )
    
fig = plt.figure()
fig2, ax2 = plt.subplots()

line0, = ax2.plot(sol.t, sol.y[0])
line1, = ax2.plot(sol.t, sol.y[1])
line2, = ax2.plot(sol.t, sol.y[2])
line3, = ax2.plot(sol.t, sol.y[3])
line4, = ax2.plot(sol.t, sol.y[4])

fig.subplots_adjust(left=0.1,right=0.2)

sliders = []
axe = []


for i in range(len(var)):

    aux =fig.add_axes([0.05*i, 0.25, 0.0225, 0.63])
    aux_slider  =Slider(
    ax=aux,
    label=variables_symbols[i],
    valmin=0,
    valmax=20,
    valinit=var[i],
    orientation="vertical")
    sliders.append(aux_slider)
    axe.append(aux)


# The function to be called anytime a slider's value changes
def update():

    sol = solve_ivp(
    edo_huerta, (0, t_max), [sliders[0].val, sliders[1].val, sliders[2].val, sliders[3].val,sliders[4].val],
    args=(
    sliders[5].val, #r1
    sliders[6].val,#r2
    sliders[7].val,#r3
    sliders[8].val,#r4
    sliders[9].val,#r5
    sliders[10].val, #k1
    sliders[11].val, #k2
    sliders[12].val, #h0
    sliders[13].val, #h1
    sliders[14].val, #alpha
    sliders[15].val, #beta
    sliders[16].val, #gamma 
    sliders[17].val, #delta
    sliders[18].val, #epsilon
    sliders[19].val, #dseta
    sliders[20].val, #nu
    sliders[21].val, #Lambda
    sliders[22].val), #psi
    t_eval=t_eval, rtol=1e-6, atol=1e-8
    )
    

    line0.set_ydata(sol.y[0])
    line1.set_ydata(sol.y[1])
    line2.set_ydata(sol.y[2])
    line3.set_ydata(sol.y[3])
    line4.set_ydata(sol.y[4])
    fig2.canvas.draw_idle()

for i in sliders:
    i.on_changed(update)

plt.show()








