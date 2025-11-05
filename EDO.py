import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Button, Slider
from matplotlib import axes

def dagger(a):
    return 0 if a<0 else a

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
Lambda = 0.05 #Provisional
psi = 1e-04 #Provisional
A_min = 6000.0 #Provisional
eta = 0.0005  #Provisional

A0 = 7000 
F0 = 2000
D0 = 100 #Provisional
O0 = 0 #Provisional
V0 = 6 #Provisional

def edo_huerta(t, y, r1,r2,r3,r4,r5,k1,k2,h0,h1,h2,h3,alpha,beta,gamma,delta,epsilon,dseta,nu,Lambda,psi,A_min,eta):
    A,F,D,O,V = y
    dA = (r1*A*(1-(A/k1))) +((alpha*A*F)/(1+(h0*A))) - (nu*A) - ((V*eta*dagger(A-A_min))/(1+(h1*dagger(A-A_min))))
    dF = (r2*F*(1-(F/k2))) +((beta*A*F)/(1+(h2*A))) - (gamma*F*D)
    dD = (r3*D*-1) +((delta*D*F)) - (epsilon*O*D)
    dO = ((Lambda) -( r4*O )+ (dseta*D*O))
    dV = (-r5*V+((V*psi*dagger(A-A_min))/(1+(h3*dagger(A-A_min)))))
    return [dA, dF, dD, dO, dV]

var = [A0, F0, D0, O0, V0, r1, r2, r3, r4, r5, k1, k2, h0, h1, h2, h3, alpha, beta, gamma, delta, epsilon, dseta, nu, Lambda, psi, A_min, eta]
variables_names = ["A0", "F0", "D0", "O0", "V0", "r1", "r2", "r3", "r4", "r5", "k1", "k2", "h0", "h1", "h2", "h3", "alpha", "beta", "gamma", "delta", "epsilon", "dseta", "nu", "Lambda", "Psi", "A_min", "eta"]
variables_symbols = ["A0", "F0", "D0", "O0", "V0", "r1", "r2", "r3", "r4", "r5", "k1", "k2", "h0", "h1", "h2", "h3", "α", "β", "γ", "δ", "ε", "ζ", "μ", "λ", "ψ", "A_", "η"]

t_max = 600
t_eval = np.linspace(0, t_max, 2000)

sol = solve_ivp(
    edo_huerta, (0, t_max), [A0, F0, D0, O0, V0],
    args=(r1, r2, r3, r4, r5, k1, k2, h0, h1, h2, h3, alpha, beta, gamma, delta, epsilon, dseta, nu, Lambda, psi, A_min, eta),
    t_eval=t_eval, rtol=1e-6, atol=1e-8
)

fig = plt.figure(figsize=(15, 8))
fig2, ax2 = plt.subplots(figsize=(10, 6))

line0, = ax2.plot(sol.t, sol.y[0], label='A')
line1, = ax2.plot(sol.t, sol.y[1], label='F')
line2, = ax2.plot(sol.t, sol.y[2], label='D')
line3, = ax2.plot(sol.t, sol.y[3], label='O')
line4, = ax2.plot(sol.t, sol.y[4], label='V')

ax2.set_xlabel('Time')
ax2.set_ylabel('Values')
ax2.set_title('EDO Solutions')
ax2.legend()
ax2.grid(True)

fig.subplots_adjust(left=0.1, right=0.2)

sliders = []
axe = []

for i in range(len(var)):
    aux = fig.add_axes([0.035*i, 0.25, 0.0225, 0.63])
    aux_slider = Slider(
        ax=aux,
        label=variables_symbols[i],
        valmin=0,
        valmax=var[i]*2,
        valinit=var[i],
        orientation="vertical"
    )
    sliders.append(aux_slider)
    axe.append(aux)

# The function to be called anytime a slider's value changes
def update(val):
    sol = solve_ivp(
        edo_huerta, (0, t_max), 
        [sliders[0].val, sliders[1].val, sliders[2].val, sliders[3].val, sliders[4].val],
        args=(
            sliders[5].val,   # r1
            sliders[6].val,   # r2
            sliders[7].val,   # r3
            sliders[8].val,   # r4
            sliders[9].val,   # r5
            sliders[10].val,  # k1
            sliders[11].val,  # k2
            sliders[12].val,  # h0
            sliders[13].val,  # h1
            sliders[14].val,  # h2
            sliders[15].val,  # h3
            sliders[16].val,  # alpha
            sliders[17].val,  # beta
            sliders[18].val,  # gamma 
            sliders[19].val,  # delta
            sliders[20].val,  # epsilon
            sliders[21].val,  # dseta
            sliders[22].val,  # nu
            sliders[23].val,  # Lambda
            sliders[24].val,  # psi
            sliders[25].val,  # A_min
            sliders[26].val   # eta 
        ),
        t_eval=t_eval, rtol=1e-6, atol=1e-8
    )
    
    line0.set_ydata(sol.y[0])
    line1.set_ydata(sol.y[1])
    line2.set_ydata(sol.y[2])
    line3.set_ydata(sol.y[3])
    line4.set_ydata(sol.y[4])
    
    # Update y-axis limits
    ax2.relim()
    ax2.autoscale_view()
    
    fig2.canvas.draw_idle()

# Connect sliders to update function (pass the function reference, not the result)
for i in sliders:
    i.on_changed(update)

plt.show()
