import numpy as np
from rsc import RSC

if __name__ == "__main__":
    # All naming conventions between this run file and the class are consistent for ease of understanding
    gamma_new: float = 60.
    delta_new: float = 120.
    epsilon_new: float = 5.
    zeta_new: float = 2.5
    ghk_scl: float = 40000.
    VNa: float = 60.
    gNaMax: float = 200.
    rate_dict = {
        'alpha': {'max': 80., 'Vhalf': 30., 'k': 23.},
        'beta': {'max': 15., 'Vhalf': -47., 'k': -6.},
        'iota': {'max': 0.2, 'Vhalf': -50., 'k': 10.},
        'kappa': {'max': 1.5, 'Vhalf': -90., 'k': -12.},
        'lambda': {'max': 1.5, 'Vhalf': 10., 'k': 10.},
        'mu': {'max': 0.2, 'Vhalf': -90., 'k': -12.},
        'eta': {'max': 1., 'Vhalf': -30., 'k': 10.},
        'theta': {'max': 0.6, 'Vhalf': -70., 'k': -30.}
    }
    y_new = [0.913, 0.079, 0.002, 0.000, 0., 0., 0.004, 0.001, 0.000, 0.000, 0., 0.] 
    ts = 25.
    tr = 30.
    te = 225.
    dt = 0.001
    time = np.arange(0., 250., dt)
    resurg_steps = np.arange(-90., -20., 10.)

    rsc_instance = RSC(
        init_states=y_new,
        resurg_steps=resurg_steps,
        time_series=time,
        rate_dict=rate_dict,
        gamma_new=gamma_new,
        delta_new=delta_new,
        epsilon_new=epsilon_new,
        zeta_new=zeta_new,
        ghk_scl=ghk_scl,
        VNa=VNa,
        gNaMax=gNaMax
    )
    rsc_instance.run_sweep_protocol()
    rsc_instance.plot_sweeps()