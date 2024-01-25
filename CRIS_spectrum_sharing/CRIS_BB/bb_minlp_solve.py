import numpy as np
from pyomo.environ import *

def minlp_solve(K, L, N, M, J, var_pi, num_subchannel, pu_power, pu_freq_usage, noise_power, G, F, H):
    ptx_channel = np.zeros(shape=(K, M, N), dtype='complex')
    stx_channel = np.zeros(shape=(L, J, N), dtype='complex')   
    inter_stx_channel = np.zeros(shape = (K, J, N), dtype='complex')
    inter_ptx_channel = np.zeros(shape = (L, M, N), dtype='complex')

    # combined two channel: H,G
    for i in range(K):
        H_ = np.eye(N) * np.conj(H[:, i]).T
        temp = np.dot(H_, G)
        ptx_channel[i, :, :] = np.conj(temp).T

    # combined two channel: H,F
    for i in range(L):
        H_ = np.eye(N) * np.conj(H[:, K+i]).T
        temp = np.dot(H_, F)
        stx_channel[i, :, :] = np.conj(temp).T

    # interference from STx to PU : H,F
    for i in range(K):
        H_ = np.eye(N) * np.conj(H[:, i]).T
        temp = np.dot(H_, F)
        inter_stx_channel[i, :, :] = np.conj(temp).T

    # interference from PTx to PU : H,G
    for i in range(L):
        H_ = np.eye(N) * np.conj(H[:, K+i]).T
        temp = np.dot(H_, G)
        inter_ptx_channel[i, :, :] = np.conj(temp).T

    # multiply theta variable real part
    def RIS_part_real(users_channel, user_index, idx_m, ris_phase_shift):

        return sum((users_channel[user_index, idx_m, j].real*cos(ris_phase_shift[j]) + users_channel[user_index, idx_m, j].imag*sin(ris_phase_shift[j])) for j in range(N))
    
    # multiply theta variable imag part
    def RIS_part_imag( users_channel, user_index, idx_m, ris_phase_shift):

        return sum((users_channel[user_index, idx_m, j].real*sin(ris_phase_shift[j]) - users_channel[user_index, idx_m, j].imag*cos(ris_phase_shift[j])) for j in range(N))

    # multiply theta variable square part
    def calc_channel(users_channel, user_index, ris_phase_shift):

        return sum((RIS_part_real(users_channel, user_index, idx_m, ris_phase_shift))**2 + (RIS_part_imag(users_channel, user_index, idx_m, ris_phase_shift))**2 for idx_m in range(users_channel.shape[1]))

    def object_func(model):
        SE = 0
        # SUs' spectral efficiency
        for i in range(L):
            for j in range(num_subchannel):
                usage_condition = model.su_spectrum_usage[i, j]

                channel = calc_channel(stx_channel, i, model.ris_phase_shift)
                received_power = model.su_power[i] * channel
                total = usage_condition * received_power
                inter_pu_channel = calc_channel(inter_ptx_channel, i, model.ris_phase_shift)
                interference = 0
                for g in range(K+L):
                    if g < K:
                        interference += pu_power[g] * inter_pu_channel * pu_freq_usage[g, j]
                    else:
                        if g == K + i:
                            continue
                        else:
                            interference += model.su_power[g-K] * channel * model.su_spectrum_usage[g-K, j]

                SE += log10(1 + total/(interference + noise_power)) / log10(2)

        return SE

    # primary user SINR constraint
    def constraint_pu_capacity(model, idx):
        SINR = 0
        for i in range(num_subchannel):
            usage_subchannel = pu_freq_usage[idx, i]
            if usage_subchannel == 1:
                channel = calc_channel(ptx_channel, idx, model.ris_phase_shift)
                received_power = pu_power[idx] * channel
                total = usage_subchannel * received_power
                interference = 0
                inter_su_channel = calc_channel(inter_stx_channel, idx, model.ris_phase_shift)
                # calculate the interference
                for j in range(L):
                    interference += model.su_power[j] * inter_su_channel * model.su_spectrum_usage[j, i]

                SINR += total/(interference + noise_power)

         # PU is idle, then ignore the constraint
        if value(SINR) == 0:
            return Constraint.Skip

        return SINR >= var_pi
    
    # every SU can use only one spectrum
    def constraint_user_spectrum(model, idx):
        spectrum_usage = 0
        for j in range(num_subchannel):
            spectrum_usage += model.su_spectrum_usage[idx, j]

        return spectrum_usage == 1
    
    def constraint_ris_phaseshift(model, idx):
        theta = sqrt(cos(model.ris_phase_shift[idx])**2 + sin(model.ris_phase_shift[idx])**2)

        return theta == 1

    model = ConcreteModel(name="cpz_test_BB")

    # optimization variable
    model.L = Set(initialize=[i for i in range(L)])
    model.N = Set(initialize=[i for i in range(N)])
    model.num_subchannel = Set(initialize=[i for i in range(num_subchannel)])

    # NonNegativeReals means non-zero real
    model.su_spectrum_usage = Var(model.L, model.num_subchannel, bounds=(0, 1), within=Binary, initialize=1)
    
    model.su_power = Var(model.L, bounds=(0.01, 1), within=Reals, initialize=0.01)
    model.ris_phase_shift = Var(model.N, bounds=(0, 2*np.pi), within=Reals, initialize=0)

    # Add Constraints
    model.cons = ConstraintList()

    # add pu capacity constraint
    for idx in range(K):
        model.cons.add(constraint_pu_capacity(model, idx))
    
    # add spectrum constraint
    for idx in range(L):
        model.cons.add(constraint_user_spectrum(model, idx))

    # add ris phase shift constraint
    for idx in range(N):
        model.cons.add(constraint_ris_phaseshift(model, idx))

    # objective function
    model.obj = Objective(expr=object_func, sense=maximize)
    model.cons.display()

    # solver
    solver_path = '/home/cpz/Couenne-0.5.8/build/bin/bonmin'

    # solve math model by solver
    opt = SolverFactory('bonmin', executable=solver_path)

    # 150 240 390 520
    opt.options['max_iter'] = 158
    solution = None
    try:
        solution = opt.solve(model, tee=True)

        # check the solver status
        if solution.solver.status != SolverStatus.okay:
            print("Solver failed to converge.")
        else:
            pass
        
    except Exception as e:
        print("An error occurred:", str(e))

    # model.display()
    time = solution.solver.time
    objective = model.obj()
    Power = np.array(list(model.su_power.get_values().values()))
    Theta_R = np.array(list(model.ris_phase_shift.get_values().values()))
    Usage = np.array(list(model.su_spectrum_usage.get_values().values()))
    
    # terminate condition: optimal, feasible, infeasible
    if solution == None:
        solver_state = "infeasible"
    else:
        solver_state = solution.solver.termination_condition

    return objective, Power, Theta_R, Usage , solver_state, time
