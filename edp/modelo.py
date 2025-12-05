import numpy as np
from PIL import Image

def dagger(a):
    return np.where(a < 0, 0, a)
# λ(t,T,A,L) ; modeliza cuantas chinches, cada cuanto y cunatas vaces se hechan las chinches

# Pulsos periódicos modelados como ventanas de duración pulse_width
def Lambda(t, Periodo, Amplitud, Limite, pulse_width=0.1):
    if Periodo <= 0:
        return 0.0
    n = int(np.floor((t + 1e-9) / Periodo))
    if (n < Limite) and (abs(t - n*Periodo) <= pulse_width/2):

        return Amplitud / pulse_width
    return 0.0


# ------------------------
# funcion a para el laplaciano no homogeneo
# ------------------------
#


def a_func(x,y, radius = 0):
    xc = 0.5 * (x.max() + x.min())
    yc = 0.5 * (y.max() + y.min())
    Lx = (x.max() - x.min()) / 2
    Ly = (y.max() - y.min()) / 2
    if (radius == 0):
        L = (Lx + Ly) / 2 
    else:
        L = radius
    X , Y = np.meshgrid(x,y, indexing='ij')

    R2 = (X - xc)**2 + (Y - yc)**2
    return (np.exp((-R2)/(np.pow(L,2))))



# ------------------------
# Cargar máscara
# ------------------------
def load_mask(image_path, Nx, Ny):
    img = Image.open(image_path).convert('L').resize((Nx, Ny))
    mask = np.array(img) / 255.0
    return np.where(mask > 0.3, 1.0, 0.0)

# ------------------------
# Laplaciano con máscara
# ------------------------
#
def laplacian(u, dx, dy):
    lap = (
        (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2 +
        (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dy**2
    )

    # Discutir bordes: los ponemos a 0
    lap[0, :] = lap[-1, :] = 0
    lap[:, 0] = lap[:, -1] = 0

    # Aplicar máscara
    

    return lap

def laplacian_masked(u, dx, dy, mask):
    lap = laplacian(u,dx,dy)
    
    return lap * mask

def laplac_heterogeneo(u,dx,dy,x,y):


    X, Y = np.meshgrid(x, y, indexing='ij')
    a = a_func(x,y);

    # grad(u)
    du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)

    # grad(a)
    da_dx = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2*dx)
    da_dy = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2*dy)

  # div(a grad(u)) = grad(u) * div(a) + a * lap(u)
    result = a *  laplacian(u,dx,dy) + da_dx * du_dx + da_dy * du_dy
    return result

def laplac_heterogeneo_masked(u,dx,dy,x,y,mask):

    result = laplac_heterogeneo(u,dx,dy,x,y) * mask
    return result











# ------------------------
# Paso explícito completo
# ------------------------
def step_explicit_masked(A, F, D, O,V,t, params,diff_param,init_vals, dx, dy, dt, mask,x,y):
    lapA = laplac_heterogeneo(A, dx, dy,x,y)
    lapF = laplacian_masked(V,dx,dy,mask) 
    lapD = laplacian(D, dx, dy)
    lapO = laplacian(O, dx, dy)
    lapV = laplacian(V, dx, dy)

    A0,F0,D0,O0,V0 = init_vals["A0"],init_vals["F0"],init_vals["D0"],init_vals["O0"],init_vals["V0"]
    r1,r2,r3,r4,r5 = params["r1"],params["r2"],params["r3"],params["r4"],params["r5"]
    k1,k2 = params["k1"],params["k2"]
    h0,h1,h2,h3 = params["h0"],params["h1"],params["h2"],params["h3"]
    alpha,beta,gamma,delta,epsilon = params["alpha"],params["beta"],params["gamma"],params["delta"],params["epsilon"]
    dseta,nu,T_period,Amp,Limit = params["dseta"],params["nu"],params["T_period"],params["Amp"],params["Limit"]
    psi,A_min,eta = params["psi"],params["A_min"],params["eta"]
    da,df,dd,do,dv = diff_param["da"], diff_param["df"], diff_param["dd"], diff_param["do"], diff_param["dv"] 

    lambda_temp = Lambda(t, T_period, Amp, Limit)
    dagg_Amin = dagger(A-A_min)
   
  
    radius = 0.1

    # ------------------------
    # reusamos a func para focalizar lambda 
    # ------------------------

    lambda_t = lambda_temp * a_func(x,y,radius)

    # Ecuaciones
    reacA = ((r1 * A * (1 - (A / k1)))
            + ((alpha * A *F) / (1 + (h0 * A))) 
            - (nu * A) - ((V * eta * dagg_Amin) / (1 + (h1 *dagg_Amin))))
    reacF = ((r2 * F * (1 - (F / k2))) 
            + ((beta * A * F) / (1 + (h2 * A))) 
            - (gamma * F * D))
    reacD = ((- r3 * D) 
            + ((delta * D * F)) 
            - (epsilon * O * D))
    reacO = (lambda_t
            -( r4*O )
            + (dseta*D*O))
    reacV = (-r5*V+((V*psi*dagg_Amin)/(1+(h3*dagg_Amin))))


    An = np.maximum((A + dt*(da*lapA + reacA)), 0)
    Fn = np.maximum(mask*(F + dt*(df*lapF + reacF)), 0)
    Dn = np.maximum((D + dt*(dd*lapD + reacD)), 0)
    On = np.maximum((O + dt*(do*lapO + reacO)), 0)
    Vn = np.maximum((V + dt*(dv*lapV + reacV)), 0)

    return An, Fn, Dn, On, Vn

# ------------------------
# Simulación principal
# ------------------------
def run_simulation(params, init_vals, disp_coeff,
                   Nx, Ny,T_final=360, dt=None, mask_path="campo_fresas_mask.png"):

    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/(Nx-1), Ly/(Ny-1)

    if dt is None:
        dt = 0.25 * min(dx,dy)**2 / max(disp_coeff.values())

    Nt = int(T_final/dt)

    # Malla
    x = np.linspace(0,Lx,Nx)
    y = np.linspace(0,Ly,Ny)
    radius = 0.1

    # ------------------------
    # reusamos a func para la distribucion de A0 y O0
    # ------------------------

    # Máscara
    mask = load_mask(mask_path, Nx, Ny)

    # Inicialización:
    A = init_vals["A0"] *  a_func(x,y,radius)
    F = init_vals["F0"] * mask
    D = init_vals["D0"] * np.ones((Nx, Ny))
    O = init_vals["O0"] *  a_func(x,y,radius)
    V = init_vals["V0"] * np.ones((Nx, Ny))

  
    # Almacenamiento
    frames_A, frames_F, frames_D, frames_O, frames_V = [], [], [], [], []
    mean_A, mean_F, mean_D, mean_O, mean_V = [], [], [], [], []
    times = []


    save_step = max(1, Nt//200)

    for n in range(Nt):
        t = n*dt
        A,F,D,O,V = step_explicit_masked(A,F,D,O,V,t, params, disp_coeff,init_vals, dx, dy, dt, mask, x, y)

        if n % save_step == 0:


            times.append(t)
            frames_A.append(A.copy())
            frames_F.append(F.copy())
            frames_D.append(D.copy())
            frames_O.append(O.copy())
            frames_V.append(V.copy())
            
            t = n*dt
            # Promedios en región con máscara
            mean_A.append(np.mean(A[mask == 1]))
            mean_F.append(np.mean(F[mask == 1]))
            mean_D.append(np.mean(D[mask == 1]))
            mean_O.append(np.mean(O[mask == 1]))
            mean_V.append(np.mean(V[mask == 1]))

            print(f"t={t:.2f}/{T_final}")

    return frames_A, frames_F,frames_D,frames_O,frames_V, times, mean_A, mean_F, mean_D, mean_O, mean_V
