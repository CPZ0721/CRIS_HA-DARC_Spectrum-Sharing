import numpy as np
from pyomo.environ import *
import math

def minlp_solve(L, N, J, num_subchannel, noise_power, F, H, bandwidth):
    users_channel = np.zeros(shape=(L, J, N), dtype='complex')

    # combined two channel: H,F
    for i in range(L):
        H_ = np.eye(N) * np.conj(H[:, i]).T
        temp = np.dot(H_, F)
        users_channel[i, :, :] = np.conj(temp).T

    # multiply theta variable real part
    def RIS_part_real(model, user_index, idx_m):

        return sum((users_channel[user_index, idx_m, j].real*cos(model.ris_phase_shift[j]) + users_channel[user_index, idx_m, j].imag*sin(model.ris_phase_shift[j])) for j in range(N))
    
    # multiply theta variable imag part
    def RIS_part_imag(model, user_index, idx_m):

        return sum((users_channel[user_index, idx_m, j].real*sin(model.ris_phase_shift[j]) - users_channel[user_index, idx_m, j].imag*cos(model.ris_phase_shift[j])) for j in range(N))

    # multiply theta variable square part
    def calc_channel(model, user_index):

        return sum((RIS_part_real(model, user_index, idx_m))**2 + (RIS_part_imag(model, user_index, idx_m))**2 for idx_m in range(J))

    # calculate the total capacity of secondary system
    def object_func(model):
        total_capacity = 0
        for i in range(L):
            for j in range(num_subchannel):
                usage_condition = model.su_spectrum_usage[i, j]
                channel = calc_channel(model, i)
                received_power = model.su_power[i] * channel
                total = usage_condition * received_power
                interference = 0
                for g in range(L):
                    if g == i:
                        continue
                    else:
                        interference += model.su_power[g] * channel * model.su_spectrum_usage[g, j]
                    

                total_capacity += bandwidth * log10(1 + total/(interference + noise_power * bandwidth)) / log10(2)

        return total_capacity
    
    # constraint of su capacity
    def constraint_su_capacity(model, idx):
        capacity = 0
        for j in range(num_subchannel):
            usage_condition = model.su_spectrum_usage[idx, j]
            channel = calc_channel(model, idx)
            received_power = model.su_power[idx] * channel
            total = usage_condition * received_power
            interference = 0
            for g in range(L):
                    if g == i:
                        continue
                    else:
                        interference += model.su_power[g] * channel * model.su_spectrum_usage[g, j]

            capacity += bandwidth * log10(1 + total/(interference + noise_power * bandwidth)) / log10(2)

        return capacity >= 1e5
    
    # every SU at least use one subchannel
    def constraint_user_spectrum(model, idx):
        spectrum_usage = 0
        for j in range(num_subchannel):
            spectrum_usage += model.su_spectrum_usage[idx, j]

        return spectrum_usage >= 1

    # constraint of RIS phase shift
    def constraint_ris_phaseshift(model, idx):
        theta = sqrt(cos(model.ris_phase_shift[idx])**2 + sin(model.ris_phase_shift[idx])**2)

        return theta == 1

    model = ConcreteModel(name="cpz_test_BB")

    # optimization variable
    model.L = Set(initialize=[i for i in range(L)])
    model.N = Set(initialize=[i for i in range(N)])
    model.user_num = Set(initialize=[i for i in range(L)])
    model.num_subchannel = Set(initialize=[i for i in range(num_subchannel)])
    
    # NonNegativeReals means non-zero real
    model.su_spectrum_usage = Var(model.L, model.num_subchannel, bounds=(
        0, 1), within=Binary, initialize=1)
    
    model.su_power = Var(model.L, bounds=(0.01, 1), within=Reals, initialize=0.01)
    model.ris_phase_shift = Var(model.N, bounds=(0, 2*np.pi), within=Reals, initialize=0)

    # Add Constraints
    model.cons = ConstraintList()

    # add su capacity constraint
    for idx in range(L):
        model.cons.add(constraint_su_capacity(model, idx))

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
    solution = None
    try:
        solution = opt.solve(model, tee=True)

        # 檢查求解器狀態
        if solution.solver.status != SolverStatus.okay:
            print("Solver failed to converge.")
        else:
            # 繼續執行後續程式碼
            # ...
            pass
    except Exception as e:
        print("An error occurred:", str(e))
    print("=====================================")

    # computing time
    time = solution.solver.time
    objective = model.obj()

    # Solutions
    Power = np.array(list(model.su_power.get_values().values()))
    Theta_R = np.array(list(model.ris_phase_shift.get_values().values()))
    Usage = np.array(list(model.su_spectrum_usage.get_values().values()))

    # terminate condition: optimal, feasible, infeasible
    if solution == None:
        solver_state = "infeasible"
    else:
        solver_state = solution.solver.termination_condition

    return objective, Power, Theta_R, Usage , solver_state, time
