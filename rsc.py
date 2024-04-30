from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from numpy import float128
from scipy.integrate import odeint

@dataclass
class RSC:
    init_states: list
    resurg_steps: np.array
    time_series: np.array
    rate_dict: dict
    gamma_new: float
    delta_new: float
    epsilon_new: float
    zeta_new: float
    ghk_scl: float
    VNa: float
    gNaMax: float
    currents: dict=field(default_factory=lambda: {})

    """ These are the values from Raman 2023, but let's import our own values
    gamma_new: float = 60.
    delta_new: float = 120.
    epsilon_new: float = 5.
    zeta_new: float = 2.5
    ghk_scl: float = 40000.
    VNa: float = 60.0
    rate_dict = field(default_factory=lambda: {
        'alpha': {'max': 80., 'Vhalf': 30., 'k': 23.},
        'beta': {'max': 15., 'Vhalf': -47., 'k': -6.},
        'iota': {'max': 0.2, 'Vhalf': -50., 'k': 10.},
        'kappa': {'max': 1.5, 'Vhalf': -90., 'k': -12.},
        'lambda': {'max': 1.5, 'Vhalf': 10., 'k': 10.},
        'mu': {'max': 0.2, 'Vhalf': -90., 'k': -12.},
        'eta': {'max': 1., 'Vhalf': -30., 'k': 10.},
        'theta': {'max': 0.6, 'Vhalf': -70., 'k': -30.}
    })
    """
    
    def a_new(self, voltage: float) -> float:
        return (self.rate_dict['lambda']['max'] / self.rate_dict['iota']['max'])**(1/3)
    def b_new(self, voltage: float) -> float:
        return (self.rate_dict['mu']['max'] / self.rate_dict['kappa']['max'])**(1/3)

    def rate(self, const: str, Vm: float, modifier=False, *args, **kwargs) -> float:
        rate_max = self.rate_dict[const]['max']
        if modifier == True:
            if kwargs["mod"] == "a":
                rate_max *= (self.a_new(Vm) ** kwargs["mod_exp"])
            elif kwargs["mod"] == "b":
                rate_max *= (self.b_new(Vm) ** kwargs["mod_exp"])
            else:
                print(f"Wrong mod - you entered {kwargs['mod']}, needs to be a or b")
        Vhalf = self.rate_dict[const]['Vhalf']
        k = self.rate_dict[const]['k']
        exp_val = float128(-(Vm - Vhalf) / k)
        return rate_max / ( np.exp(exp_val) + 1 )

    # New ghk function based on labview screenshot
    def ghk(self, Vm: float, Na_i: float=9., Na_e: float=154., F: float=96485., RT: float=(8.3145 * 298)) -> float:
        if Vm == 0.:
            Vm += -0.001
        Vm /= 1000.
        top_0 = float128(F**2 * Vm)
        top_1 = float128(top_0 / RT)
        bottom_0 = Na_e / 1000
        bottom_1 = Na_i / 1000
        bottom_2 = float128(bottom_1 - (bottom_0 * (np.exp(-float128(F * Vm) / RT))))
        bottom_3 = float128(1 - np.exp(-float128(F * Vm) / RT))
        final = float128((top_1 * bottom_2) / bottom_3)
        #print(final)
        return final

    # Have to find a new way of expressing this, cannot use boolean logic within jit...
    def find_voltage(self, time: float, resurg_voltage: float, start: float=1., resurg_start: float=6., resurg_end: float=225.) -> float:
        if (time < start) or (time > resurg_end):
            return -90.
        elif (time > start) and (time < resurg_start):
            return 30.
        else:
            return resurg_voltage
            
    # Forward transitions in C states
    def dC0C1(self, voltage: float) -> float:
        return 3 * self.rate('alpha', voltage)
            
    def dC1C2(self, voltage: float) -> float:
        return 2 * self.rate('alpha', voltage)
            
    def dC2C3(self, voltage: float) -> float:
        return self.rate('alpha', voltage)

    # Backward transitions in C states
    def dC1C0(self, voltage: float) -> float:
        return self.rate('beta', voltage)
            
    def dC2C1(self, voltage: float) -> float:
        return 2 * self.rate('beta', voltage)
            
    def dC3C2(self, voltage: float) -> float:
        return 3 * self.rate('beta', voltage)
        
    # Forward transitions in I states
    def dI0I1(self, voltage: float) -> float:
        return 3 * self.rate('alpha', voltage, modifier=True, mod="a", mod_exp=1)
            
    def dI1I2(self, voltage: float) -> float:
        return 2 * self.rate('alpha', voltage, modifier=True, mod="a", mod_exp=1)
            
    def dI2I3(self, voltage: float) -> float:
        return self.rate('alpha', voltage, modifier=True, mod="a", mod_exp=1)

    # Backward transitions in I states
    def dI1I0(self, voltage: float) -> float:
        return self.rate('beta', voltage, modifier=True, mod="b", mod_exp=1)
            
    def dI2I1(self, voltage: float) -> float:
        return 2 * self.rate('beta', voltage, modifier=True, mod="b", mod_exp=1)
            
    def dI3I2(self, voltage: float) -> float:
        return 3 * self.rate('beta', voltage, modifier=True, mod="b", mod_exp=1)

    # C to I
    def dC0I0(self, voltage: float) -> float:
        return self.rate('iota', voltage)

    def dC1I1(self, voltage: float) -> float:
        return self.rate('iota', voltage, modifier=True, mod="a", mod_exp=1)

    def dC2I2(self, voltage: float) -> float:
        return self.rate('iota', voltage, modifier=True, mod="a", mod_exp=2)

    def dC3I3(self, voltage: float) -> float:
        return self.rate('iota', voltage, modifier=True, mod="a", mod_exp=3)

    # I to C
    def dI0C0(self, voltage: float) -> float:
        return self.rate('kappa', voltage)

    def dI1C1(self, voltage: float) -> float:
        return self.rate('kappa', voltage, modifier=True, mod="b", mod_exp=1)

    def dI2C2(self, voltage: float) -> float:
        return self.rate('kappa', voltage, modifier=True, mod="b", mod_exp=2)

    def dI3C3(self, voltage: float) -> float:
        return self.rate('kappa', voltage, modifier=True, mod="b", mod_exp=3)

    # Open forward
    def dC3OO(self, voltage: float) -> float:
        return self.gamma_new

    def dOOOB(self, voltage: float) -> float:
        return self.epsilon_new

    def dI3OI(self, voltage: float) -> float:
        return self.gamma_new

    # Open backward
    def dOOC3(self, voltage: float) -> float:
        return self.delta_new

    def dOBOO(self, voltage: float) -> float:
        zeta = self.zeta_new * (-self.ghk(voltage) / self.ghk_scl)
        #print(f'ghk: {ghk(voltage)}, ghk_scl: {ghk_scl}, zeta: {zeta}')
        if zeta <= 0.25:
            return 0.25
        else:
            return zeta

    def dOII3(self, voltage: float) -> float:
        return self.delta_new

    # Open to I
    def dOOOI(self, voltage: float) -> float:
        return self.rate('lambda', voltage)

    def dOIOO(self, voltage: float) -> float:
        return self.rate('mu', voltage)

    # Open block inactivation
    def dOBOBI(self, voltage: float) -> float:
        return self.rate('eta', voltage)

    def dOBIOB(self, voltage: float) -> float:
        return self.rate('theta', voltage)


    # Function for diffeqs
    def run_sweep(self, states, time: float, resurg_voltage: float):# -> jax.numpy.array:
        new_states = np.zeros(len(states))
        voltage = self.find_voltage(time, resurg_voltage)

        C0, C1, C2, C3, OO, OB, I0, I1, I2, I3, OI, OBI = states
        # Forward C
        C0_C1 = C0 * self.dC0C1(voltage)
        C1_C2 = C1 * self.dC1C2(voltage)
        C2_C3 = C2 * self.dC2C3(voltage)
        # Backward C
        C1_C0 = C1 * self.dC1C0(voltage)
        C2_C1 = C2 * self.dC2C1(voltage)
        C3_C2 = C3 * self.dC3C2(voltage)
        # Forward I
        I0_I1 = I0 * self.dI0I1(voltage)
        I1_I2 = I1 * self.dI1I2(voltage)
        I2_I3 = I2 * self.dI2I3(voltage)
        # Backward I
        I1_I0 = I1 * self.dI1I0(voltage)
        I2_I1 = I2 * self.dI2I1(voltage)
        I3_I2 = I3 * self.dI3I2(voltage)
        # C to I
        C0_I0 = C0 * self.dC0I0(voltage)
        C1_I1 = C1 * self.dC1I1(voltage)
        C2_I2 = C2 * self.dC2I2(voltage)
        C3_I3 = C3 * self.dC3I3(voltage)
        # I to C
        I0_C0 = I0 * self.dI0C0(voltage)
        I1_C1 = I1 * self.dI1C1(voltage)
        I2_C2 = I2 * self.dI2C2(voltage)
        I3_C3 = I3 * self.dI3C3(voltage)
        # Open Forward
        C3_OO = C3 * self.dC3OO(voltage)
        OO_OB = OO * self.dOOOB(voltage)
        I3_OI = I3 * self.dI3OI(voltage)
        # Open Backward
        OO_C3 = OO * self.dOOC3(voltage)
        OB_OO = OB * self.dOBOO(voltage)
        OI_I3 = OI * self.dOII3(voltage)
        # Open to I
        OO_OI = OO * self.dOOOI(voltage)
        OI_OO = OI * self.dOIOO(voltage)
        # OBI
        OB_OBI = OB * self.dOBOBI(voltage)
        OBI_OB = OBI * self.dOBIOB(voltage)


        new_states[0] = float128(C1_C0 + I0_C0 - (C0_C1 + C0_I0))
        new_states[1] = float128(C0_C1 + C2_C1 + I1_C1 - (C1_C0 + C1_C2 + C1_I1))
        new_states[2] = float128(C1_C2 + C3_C2 + I2_C2 - (C2_C1 + C2_C3 + C2_I2))
        new_states[3] = float128(C2_C3 + OO_C3 + I3_C3 - (C3_C2 + C3_OO + C3_I3))
        new_states[4] = float128(C3_OO + OI_OO + OB_OO - (OO_C3 + OO_OB + OO_OI))
        new_states[5] = float128(OO_OB + OBI_OB - (OB_OO + OB_OBI))
        new_states[6] = float128(C0_I0 + I1_I0 - (I0_C0 + I0_I1))
        new_states[7] = float128(I0_I1 + I2_I1 + C1_I1 - (I1_I0 + I1_I2 + I1_C1))
        new_states[8] = float128(I1_I2 + I3_I2 + C2_I2 - (I2_I1 + I2_I3 + I2_C2))
        new_states[9] = float128(I2_I3 + OI_I3 + C3_I3 - (I3_I2 + I3_OI + I3_C3))
        new_states[10] = float128(I3_OI + OO_OI - (OI_I3 + OI_OO))
        new_states[11] = float128(OB_OBI - OBI_OB)

        return new_states

    def run_sweep_protocol(self):
        for voltage_ in self.resurg_steps:
            self.currents[voltage_] = odeint(self.run_sweep, self.init_states, self.time_series, (voltage_,))
    
    def plot_sweeps(self):
        plt.figure(0, figsize=(12,12), dpi=800)
        for key in self.currents.keys():
            plt.plot(self.time_series, [(self.gNaMax * self.currents[key][:,4][i]) * (self.find_voltage(self.time_series[i], key) - self.VNa) for i in range(len(self.time_series))], label=f'mv={key}')
        plt.legend()
        plt.savefig('./rsc.png')
            



if __name__ == "__main__":
    print('you called the rsc file... this will eventually run tests on the model\nfor now call run_rsc to see model output')