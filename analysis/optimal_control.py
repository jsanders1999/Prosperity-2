import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm
from numba import njit

from utils.unpackUtils import read_csv_files, split_activities_df, split_trade_history_df
from utils.dataUtils import calc_weighted_mid_price

edges_amth = np.array([-5.5, -5.4, -5.3, -5.2, -5.1, -5. , -4.9, -4.8, -4.7, -4.6, -4.5,
       -4.4, -4.3, -4.2, -4.1, -4. , -3.9, -3.8, -3.7, -3.6, -3.5, -3.4,
       -3.3, -3.2, -3.1, -3. , -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3,
       -2.2, -2.1, -2. , -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2,
       -1.1, -1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
        0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
        1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
        2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
        3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
        4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
        5.5])

probs_amth = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.04540454, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.06990699, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.13511351, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.00350035, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.1290129 , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.06410641, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.04360436, 0.        , 0.        , 0.        , 0.        ])

edges_starf = np.array([-5.5, -5.4, -5.3, -5.2, -5.1, -5. , -4.9, -4.8, -4.7, -4.6, -4.5,
       -4.4, -4.3, -4.2, -4.1, -4. , -3.9, -3.8, -3.7, -3.6, -3.5, -3.4,
       -3.3, -3.2, -3.1, -3. , -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3,
       -2.2, -2.1, -2. , -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2,
       -1.1, -1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
        0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
        1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
        2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
        3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
        4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
        5.5])

probs_starf = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 2.00020002e-04, 1.90019002e-03, 3.60036004e-03,
       4.00040004e-04, 3.70037004e-03, 2.55025503e-02, 5.43054305e-02,
       1.30113011e-01, 5.10051005e-03, 3.90039004e-03, 1.40014001e-03,
       1.90019002e-03, 4.10041004e-03, 7.00070007e-04, 2.20022002e-03,
       5.50055006e-03, 1.87018702e-02, 2.58025803e-02, 3.07030703e-02,
       3.05030503e-02, 1.49014901e-02, 6.50065007e-03, 1.00010001e-03,
       1.30013001e-03, 8.00080008e-04, 1.00010001e-03, 9.00090009e-04,
       5.30053005e-03, 8.10081008e-03, 1.30013001e-03, 3.00030003e-03,
       4.00040004e-04, 1.90019002e-03, 3.00030003e-04, 0.00000000e+00,
       1.90019002e-03, 3.50035004e-03, 3.42034203e-02, 2.40024002e-03,
       2.80028003e-03, 9.00090009e-04, 9.00090009e-04, 0.00000000e+00,
       1.00010001e-03, 0.00000000e+00, 1.70017002e-03, 1.55015502e-02,
       2.00020002e-03, 1.00010001e-03, 1.80018002e-03, 1.10011001e-03,
       2.00020002e-03, 4.00040004e-04, 3.00030003e-04, 2.12021202e-02,
       4.18041804e-02, 3.98039804e-02, 1.08010801e-02, 3.50035004e-03,
       6.10061006e-03, 4.50045005e-03, 2.20022002e-03, 6.00060006e-04,
       1.00010001e-04, 8.00080008e-04, 5.30053005e-03, 1.85018502e-02,
       1.35013501e-02, 5.60056006e-03, 2.40024002e-03, 1.40014001e-03,
       3.00030003e-04, 3.20032003e-03, 1.10011001e-03, 1.00010001e-03,
       4.70047005e-03, 3.45034503e-02, 1.03510351e-01, 9.50095010e-03,
       1.55015502e-02, 6.30063006e-03, 1.70017002e-03, 9.00090009e-04,
       0.00000000e+00, 2.70027003e-03, 1.00010001e-04, 1.00010001e-04,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00])

@njit
def factorial(n):
    if n==0:
        return 1.0
    else:
        result = 1.0
        for i in range(1, n+1):
            result *= i
        return result

def generate_edges_probs_from_data_files(trade_file_paths, activities_file_paths, product, n_subdivs = 20, max_volume = 20, filter_noise=True):
    differences = []
    volumes = []
    total_timestamps = 0
    for trade_file_path, activity_file_path in zip(trade_file_paths, activities_file_paths):
        activities_df, trade_history_df = read_csv_files(activity_file_path, trade_file_path)
        product_activities_dfs = split_activities_df(activities_df)
        product_history_dfs = split_trade_history_df(trade_history_df)

        print(product_history_dfs)

        trades_df = product_history_dfs[product]
        activity_df = product_activities_dfs[product]

        trades = trades_df['price']
        trade_timestamps = trades_df['timestamp']
        if product == "AMETHYSTS":
            value = 10000*np.ones(trades.shape)
            activity_timestamps = activity_df['timestamp']
        else:
            activity_timestamps = activity_df['timestamp']
            weighted_mid_price = calc_weighted_mid_price(activity_df)
            #mid_price = np.array(activity_df['mid_price'])
            #filtered_weighted_mid_price, _ = calc_filtered_weighted_mid_price(weighted_mid_price, window, bound)
            value = weighted_mid_price #np.round(weighted_mid_price-0.5)+0.5 #filtered_weighted_mid_price
            relevant_inds  = []
            for i, timestamp in enumerate(trade_timestamps):
                relevant_inds += [list(activity_timestamps).index(timestamp)]
            value = value[relevant_inds]

        differences += [trades - value]
        volumes += [np.where(trades_df['quantity']<=max_volume, trades_df['quantity'], max_volume)]
        total_timestamps += round(np.array(activity_timestamps)[-1]/100)


    differences = np.concatenate(differences)
    volumes = np.concatenate(volumes)

    fig, ax = plt.subplots()

    bins = (np.linspace(np.floor(differences.min())-0.5, np.ceil(differences.max())+0.5, n_subdivs*(round(np.ceil(differences.max()) - np.floor(differences.min()))+1)+1),
            np.linspace(volumes.min()-0.5, volumes.max()+0.5, round(volumes.max() - volumes.min())+2))
    h, xedges, yedges, img = ax.hist2d(differences, volumes, bins=bins, density=True, cmap = 'Blues')
    print(f"{h=}")
    fig.colorbar(img, ax = ax)

    fig, ax = plt.subplots()
    ax.set_title(f"Trade histogram for {product}")
    probs = np.histogram(differences, bins=xedges, density=False, weights=volumes)[0]/(total_timestamps)
    #probs = h@(yedges[1:]-0.5)/np.array(activity_timestamps)[-1]
    if filter_noise:
        probs = np.where(probs>0.02*np.sum(probs), probs, 0.0)
    print(f"for {product=} {probs=}")


    print(f"{xedges=}")
    ax.plot((xedges[:-1]+xedges[1:])/2, probs)

    h, xedges, yedges = np.histogram2d(differences, volumes, bins=bins, density=False)

    ax.scatter((xedges[:-1]+xedges[1:])/2, np.sum(h*(yedges[1:]-0.5), axis = 1)/(total_timestamps), color = 'red')
    return probs, xedges


# #@njit
# def gamma_p(delta_p, edges, probs):
#     delta_p = np.max([delta_p, 0.0])
#     return np.sum(probs[edges[1:] >= delta_p]) 

@njit
def gamma_p(delta_p, edges, probs):
    result = 0.0
    delta_p = max(delta_p, 0.0)  # Avoid using np.max for a scalar
    for i in range(len(edges)):
        if edges[i] >= delta_p:
            result += probs[i]
    return result

# #@njit
# def gamma_n(delta_n, edges, probs):
#     delta_n = np.max([delta_n, 0.0])
#     return np.sum(probs[edges[:-1] <= -delta_n])

@njit
def gamma_n(delta_n, edges, probs):
    result = 0.0
    delta_n = max(delta_n, 0.0)  # Avoid using np.max for a scalar
    for i in range(len(edges) - 1):
        if edges[i] <= -delta_n:
            result += probs[i]
    return result

#@njit
def E_gain_next_turn(delta_p, delta_n, edges, probs):
    return delta_p * gamma_p(delta_p, edges, probs)+ delta_n*gamma_n(delta_n, edges, probs)

def E_gain_next_turn_inv_const(delta_p, delta_n, edges, probs, q, Q):
    lam_p = gamma_p(delta_p, edges, probs)
    lam_n = gamma_n(delta_n, edges, probs)
    E_dN_p = np.sum(np.array([k*lam_p**k/factorial(k)*np.exp(-lam_p) for k in range(0, Q+q+1)]))
    E_dN_n = np.sum(np.array([k*lam_n**k/factorial(k)*np.exp(-lam_n) for k in range(0, Q-q+1)]))

    return delta_p * E_dN_p + delta_n * E_dN_n

# def E_gain_next_turn_langevin(delta_p, delta_n, edges, probs, q, Q, eta, St, Se):
#     lam_p = gamma_p(delta_p, edges, probs)
#     lam_n = gamma_n(delta_n, edges, probs)
#     E_dN_p = np.sum(np.array([k*lam_p**k/factorial(k)*np.exp(-lam_p) for k in range(0, Q+q+1)]))
#     E_dN_n = np.sum(np.array([k*lam_n**k/factorial(k)*np.exp(-lam_n) for k in range(0, Q-q+1)]))

#     return delta_p * E_dN_p + delta_n * E_dN_n +q*(St-Se)*(eta-1) + (St-Se)*(eta-1)*(E_dN_n-E_dN_p)

@njit
def E_gain_next_turn_langevin(delta_p, delta_n, edges, probs, q, Q, eta, St, Se):
    lam_p = gamma_p(delta_p, edges, probs)
    lam_n = gamma_n(delta_n, edges, probs)
    E_dN_p = np.sum(np.array([k*lam_p**k/factorial(k)*np.exp(-lam_p) for k in range(0, Q+q+1)]))
    E_dN_n = np.sum(np.array([k*lam_n**k/factorial(k)*np.exp(-lam_n) for k in range(0, Q-q+1)]))

    return delta_p * E_dN_p + delta_n * E_dN_n +q*(St-Se)*(eta-1) + (St-Se)*(eta-1)*((q+Q)*E_dN_n-(-q+Q)*E_dN_p)


#@njit
def pois_pmf(k, lam):
    return lam**k/factorial(k)*np.exp(-lam)

# @njit
# def compute_addition_to_q_prob(shape, q, dN_p_probs, dN_n_probs):
#     addition_to_q_prob = np.zeros(shape, dtype = float)
#     for p_p, k_p in zip(dN_p_probs, range(len(dN_p_probs))):
#         for p_n, k_n in zip(dN_n_probs, range(len(dN_n_probs))):
#             addition_to_q_prob[q+k_n-k_p] += old_q_probs[q]*p_p*p_n
#     return addition_to_q_prob

def E_gain_end(policy_p, policy_n, edges, probs, q, Q, T):
    
    E = np.zeros((T), dtype = float)
    q_probs = np.zeros((2*Q+1, T), dtype = float)

    def calc_dN_p_probs(t, q):
        indices = np.arange(0, q+Q+1)
        dN_p_probs = np.zeros((q+Q+1), dtype = float)
        lam = gamma_p(policy_p(t,q), edges, probs)
        #calculate probabilities up to 1 before the limit
        for k in indices[:-1]:
            dN_p_probs[k] = pois_pmf(k, lam)
        #calculate probabilities for the limit
        dN_p_probs[-1] = 1 - np.sum(dN_p_probs)
        return dN_p_probs, indices
    
    def calc_dN_n_probs(t, q):
        indices = np.arange(0, Q+1-q)
        dN_n_probs = np.zeros((Q+1-q), dtype = float)
        lam = gamma_n(policy_n(t,q), edges, probs)
        #calculate probabilities up to 1 before the limit
        for k in indices[:-1]:
            dN_n_probs[k] = pois_pmf(k, lam)
        #calculate probabilities for the limit
        dN_n_probs[-1] = 1 - np.sum(dN_n_probs)
        return dN_n_probs, indices

    def propagate_qprob(old_q_prob, t):
        new_q_prob = np.zeros_like(old_q_prob, dtype = float)
        for prob, q in zip(old_q_prob, range(-Q,Q+1)):
            addition_to_q_prob = np.zeros_like(old_q_prob, dtype = float)
            if prob>0:
                dN_p_probs, indices_p = calc_dN_p_probs(t, q) #probability of selling, so q goed down
                dN_n_probs, indices_n = calc_dN_n_probs(t, q) #probability of buying, so q goes up
                #dQ_probs = compute_Z_probs(dN_n_probs, dN_p_probs) #np.convolve(dN_n_probs, dN_p_probs[::-1], mode = 'full')
                for p_p, k_p in zip(dN_p_probs, indices_p):
                    for p_n, k_n in zip(dN_n_probs, indices_n):
                        addition_to_q_prob[q+k_n-k_p+Q] += prob*p_p*p_n
                        #addition_to_q_prob[q+k_p-k_n+Q] += prob*p_p*p_n
                #new_q_prob += prob*dQ_probs 
                
                # print("")
                # print(f"{q=}, {t=}")
                # print(f"{dN_p_probs=}")
                # print(f"{dN_n_probs=}")
                #print(f"{dQ_probs=}")
                # if prob>1e-3 and t%10 == 0 or t//10 == 0:
                #     fig, ax = plt.subplots(3,1)
                #     ax[0].set_title(f"q = {q}, t = {t}, prob = {prob}")
                #     ax[0].bar(indices_p, dN_p_probs)
                #     ax[1].bar(indices_n, dN_n_probs)
                #     ax[2].bar(range(-Q,Q+1), addition_to_q_prob)
                #     plt.plot()
                #     plt.show()
            new_q_prob += addition_to_q_prob
                
        
        # if t%100 == 0:
        #     #print(f"{new_q_prob=}")
        #     fig, ax = plt.subplots()
        #     ax.bar(range(-Q,Q+1), new_q_prob)
        #     ax.set_title(f"t = {t}")
        #     #plt.show()
                
        #print(f"{t=}, {np.sum(new_q_prob)=}")
        return new_q_prob
    
    def expected_value(old_q_prob, t ):
        return np.sum([prob*(policy_p(t, q)*gamma_p(policy_p(t,q), edges, probs) + policy_n(t, q)*gamma_n(policy_n(t,q), edges, probs)) for prob, q in zip(old_q_prob, range(-Q,Q+1))])

    old_q_prob = np.zeros((2*Q+1), dtype = float)
    old_q_prob[q+Q] = 1

    for t in range(0, T):
        E[t] = expected_value(old_q_prob, t)
        new_q_prob = propagate_qprob(old_q_prob, t)
        q_probs[:,t] = new_q_prob
        old_q_prob = new_q_prob

    return q_probs, E

def construct_policy_p(base = 2, steps_dicts= [{4:0, 5:200, 6:400}, {4:0, 5:200, 6:400}, {5:0, 6:200, 7:400}] ):
    def policy(t,q):
        res = base
        for steps in steps_dicts:
            q_step = np.array(list(steps.keys()))
            t_step = np.array(list(steps.values()))
            inds = np.where(t>t_step)
            if q_step[inds].size == 0:
                q_bdry = 0
            else:
                q_bdry = q_step[inds][-1]
            res += np.heaviside(q-q_bdry, 1)
        return res
    return np.vectorize(policy)


def construct_policy_n(base = 2, steps_dicts= [{-4:0, -5:200, -6:400}, {-4:0, -5:200, -6:400}, {-5:0, -6:200, -7:400}] ):
    def policy(t,q):
        res = base
        for steps in steps_dicts:
            q_step = np.array(list(steps.keys()))
            t_step = np.array(list(steps.values()))
            inds = np.where(t>t_step)
            if q_step[inds].size == 0:
                q_bdry = 0
            else:
                q_bdry = q_step[inds][-1]
            res += np.heaviside(q_bdry-q, 1)
        return res
    return np.vectorize(policy)

def plot_policies(policy_p, policy_n, T, Q):
    t_arr = np.arange(0, T, 1)
    q_arr = np.arange(-Q, Q+1, 1)
    t_mesh, q_mesh = np.meshgrid(t_arr, q_arr, indexing='xy')
    im_policy_p = policy_p(t_mesh, q_mesh)
    im_policy_n = policy_n(t_mesh, q_mesh)
    fig, ax = plt.subplots(2,1)
    fig.colorbar(ax[0].imshow(im_policy_p, cmap='Blues', origin='lower', aspect='auto', interpolation= "None"  ))
    fig.colorbar(ax[1].imshow(im_policy_n, cmap='Reds', origin='lower', aspect='auto', interpolation= "None"  ))

def plot_next_turn_gain(delta_p, delta_n, edges, probs):
    x = np.linspace(0, 8, 1000)
    lam_b = np.empty_like(x)
    lam_n = np.empty_like(x)
    for i, x_i in enumerate(x):
        lam_b[i] = gamma_p(x_i, edges, probs)
        lam_n[i] = gamma_n(x_i, edges, probs)

    plt.plot(x, lam_b, label = "lam_b")
    plt.plot(-x, lam_n, label = "lam_n")

    y = np.array([0,1,2,3,4,5,6])
    lam_b = np.empty_like(y, dtype = float)
    lam_n = np.empty_like(y, dtype = float)
    for i, y_i in enumerate(y):
        lam_b[i] = gamma_p(y_i, edges, probs)
        lam_n[i] = gamma_n(y_i, edges, probs)
    
    plt.scatter(y, lam_b, label = "lam_b", color = "blue")
    plt.scatter(-y, lam_n, label = "lam_n", color = "red")
    plt.legend()

    plt.show()

    fig, ax = plt.subplots()
    E_free = np.empty((len(delta_p), len(delta_n)))
    E_const = np.empty((len(delta_p), len(delta_n)))
    for i, dp in enumerate(delta_p):
        for j, dn in enumerate(delta_n):
            E_free[i,j] = E_gain_next_turn(dp, dn, edges, probs)
            E_const[i,j] = E_gain_next_turn_inv_const(dp, dn, edges, probs, q, Q)
    maxindex = np.unravel_index(np.argmax(E_const, axis=None), E_const.shape)
    max_delta_p = delta_p[maxindex[0]]
    max_delta_n = delta_n[maxindex[1]]

    
    print(max_delta_p, -max_delta_n)
    print(f"E(dq) = {-gamma_p(max_delta_p, edges, probs)+gamma_n(max_delta_n, edges, probs)}")
    print(maxindex)
    fig.colorbar(ax.imshow(E_const, cmap='Blues', origin='lower',
                        extent=[delta_n[0]-0.5,delta_n[-1]+0.5,delta_p[0]-0.5,delta_p[-1]+0.5]))
    ax.set_xlabel("delta_n")
    ax.set_ylabel("delta_p")
    fig, ax = plt.subplots(2,1)
    ax[0].plot(delta_n, E_const[0,:])
    ax[0].set_title("delta_n")
    ax[1].plot(delta_p, E_const[:,0])
    ax[1].set_title("delta_p")

    plt.show()


def plot_next_turn_gain_langevin(delta_p, delta_n, edges, probs, q, Q, eta, St, Se, plots = True):

    E_langevin = np.empty((len(delta_p), len(delta_n)))
    
    for i, dp in enumerate(delta_p):
        for j, dn in enumerate(delta_n):
            E_langevin[i,j] = E_gain_next_turn_langevin(dp, dn, edges, probs, q, Q, eta, St, Se)
    maxindex = np.unravel_index(np.argmax(E_langevin, axis=None), E_langevin.shape)
    max_delta_p = delta_p[maxindex[0]]
    max_delta_n = delta_n[maxindex[1]]
    if plots:
        print(max_delta_p, -max_delta_n)
        print(f"E(dq) = {-gamma_p(max_delta_p, edges, probs)+gamma_n(max_delta_n, edges, probs)}")

        fig, ax = plt.subplots()

        fig.colorbar(ax.imshow(E_langevin, cmap='Blues', origin='lower',
                            extent=[delta_n[0]-0.5,delta_n[-1]+0.5,delta_p[0]-0.5,delta_p[-1]+0.5]))
        ax.set_xlabel("delta_n")
        ax.set_ylabel("delta_p")
        plt.show()

    return max_delta_p, max_delta_n, E_langevin[maxindex]



def run_model(bounds):
    E_T = np.zeros((len(bounds)), dtype = float)
    for i, bound in tqdm(enumerate(bounds)):
        #TODO: fic p and n discrepancy. (p=lose q, n = gain q)
        policy_n = construct_policy_p(base = 2, steps_dicts= [{bound:0}, {bound:0}])
        policy_p = construct_policy_n(base  =2, steps_dicts= [{-bound:0}, {-bound:0}])
        #plot_policies(policy_p, policy_n, T, Q)
        q_probs, E = E_gain_end(policy_p, policy_n, edges, probs, q, Q, T)
        fig1, ax1 = plt.subplots()
        fig1.colorbar(ax1.imshow(np.log(q_probs+1e-5), cmap='Blues', origin='lower', aspect='auto', interpolation= "None"  ))
        ax1.set_title(f"bound = {bound}")
        fig2, ax2 = plt.subplots()
        ax2.plot(E)
        ax2.set_title(f"bound = {bound}")
        E_T[i] = np.sum(E)
    return E_T


if __name__ == "__main__":

    # x = [0,0, 1]
    # y = [0,1, 0]
    # print(np.convolve(x,y, mode = "same"))
    # raise Error


    Q = 60
    T = 1000
    q = 0
    product = "GIFT_BASKET"

    # trade_file_paths_r1 = [f"data/ProvidedData/trades_round_1_day_{i}_nn.csv" for i in [-2, -1, 0]]
    # activities_file_paths_r1 = [f"data/ProvidedData/prices_round_1_day_{i}.csv" for i in [-2, -1, 0]]

    trade_file_paths_r3 = [f"data/ProvidedData/trades_round_3_day_{i}_nn.csv" for i in [0,1,2]]
    activities_file_paths_r3 = [f"data/ProvidedData/prices_round_3_day_{i}.csv" for i in [0,1,2]]

    probs, edges = generate_edges_probs_from_data_files(trade_file_paths_r3, activities_file_paths_r3, product, n_subdivs = 50, max_volume = 20)

    plt.show()

    if product == "STARFRUIT":
        delta_p = np.linspace(0.5,6.5,600+1) #[0,1,2,3,4,5,6]
        delta_n = np.linspace(0.5,6.5,600+1)
        bound = 14
        policy_p = construct_policy_p(base = 3.5, steps_dicts= [{bound:0}])
        policy_n = construct_policy_n(base = 2.5, steps_dicts= [{-bound:0}])

    elif product == "AMETHYSTS":
        delta_p = np.linspace(0,6,60+1) #[0,1,2,3,4,5,6]
        delta_n = np.linspace(0,6,60+1)

        #bound1 = 16
        #bound2 = 18
        #policy_p = construct_policy_p(base = 2, steps_dicts= [{bound1:0}, {bound1:0}, {bound2:0}])
        #policy_n = construct_policy_n(base = 2, steps_dicts= [{-bound1:0}, {-bound1:0}, {-bound2:0}])

        bound = 17
        policy_p = construct_policy_p(base = 2, steps_dicts= [{bound:0}, {bound:0}])
        policy_n = construct_policy_n(base = 2, steps_dicts= [{-bound:0}, {-bound:0}])
    else:
        delta_p = np.linspace(0,7,70+1)
        delta_n = np.linspace(0,7,70+1)
    
    #plot_policies(policy_p, policy_n, T, Q)
    print(probs)



    #probs = 60/1000*(np.where(edges[1:]>=-5.5, 1, 0)*np.where(edges[:-1]<-5.5, 1, 0) + np.where(edges[1:]>=5.5, 1, 0)*np.where(edges[:-1]<5.5, 1, 0))
    #print(probs)


    #plot_next_turn_gain(delta_p, delta_n, edges, probs)
    St_arr = np.linspace(-250.0,250.0,51)
    q_arr = np.linspace(-59,59,60, dtype=int)

    opt_dp = np.empty((len(St_arr), len(q_arr)), dtype = float)
    opt_dn = np.empty((len(St_arr), len(q_arr)), dtype = float)
    opt_E = np.empty((len(St_arr), len(q_arr)), dtype = float)

    conditions_n = np.zeros((len(St_arr), len(q_arr)), dtype = float)
    conditions_p = np.zeros((len(St_arr), len(q_arr)), dtype = float)
    eta = 0.996
    C1 = 3
    C2 = 6

    for i, St in tqdm(enumerate(St_arr)):
        for j, q in tqdm(enumerate(q_arr)):
            opt_dp[i,j], opt_dn[i,j], opt_E[i,j] = plot_next_turn_gain_langevin(delta_p, delta_n, edges, probs, q, Q, eta, St, 0.0, plots = False)
            if St*(Q+q)*(1-eta)>= C2:
                conditions_n[i,j] = 2
            elif St*(Q+q)*(1-eta)>= C1:
                conditions_n[i,j] = 1
            
            if St*(-Q+q)*(1-eta)>= C2:
                conditions_p[i,j] = 2
            elif St*(-Q+q)*(1-eta)>= C1:
                conditions_p[i,j] = 1


    
    fig, ax = plt.subplots(2,3)
    fig.colorbar(ax[0,0].imshow(opt_dp, cmap='Blues', origin='lower', aspect='auto', interpolation= "None"  ))
    fig.colorbar(ax[0,1].imshow(opt_dn, cmap='Reds', origin='lower', aspect='auto', interpolation= "None"  ))
    fig.colorbar(ax[0,2].imshow(opt_E, cmap='Greens', origin='lower', aspect='auto', interpolation= "None"  ))
    fig.colorbar(ax[1,0].imshow(conditions_p, cmap='Blues', origin='lower', aspect='auto', interpolation= "None"  ))
    fig.colorbar(ax[1,1].imshow(conditions_n, cmap='Reds', origin='lower', aspect='auto', interpolation= "None"  ))
    #set the x and y axis values for the subplots up to 1 decimal
    for i in range(3):
        ax[0,i].set_xticks(np.arange(0, len(q_arr), 10))
        ax[0,i].set_xticklabels(np.round(q_arr[::10], 1))
        ax[0,i].set_yticks(np.arange(0, len(St_arr), 10))
        ax[0,i].set_yticklabels(np.round(St_arr[::10], 1))
        ax[1,i].set_xticks(np.arange(0, len(q_arr), 10))
        ax[1,i].set_xticklabels(np.round(q_arr[::10], 1))
        ax[1,i].set_yticks(np.arange(0, len(St_arr), 10))
        ax[1,i].set_yticklabels(np.round(St_arr[::10], 1))


    #set the titles of the subplots
    ax[0,0].set_title("optimal delta_p")
    ax[0,1].set_title("optimal delta_n")
    ax[0,2].set_title("E or optimal point")

    #set axis labels
    ax[0,0].set_ylabel("St")
    ax[0,0].set_xlabel("q")
    ax[0,1].set_ylabel("St")
    ax[0,1].set_xlabel("q")
    ax[0,2].set_ylabel("St")
    ax[0,2].set_xlabel("q")

    plt.show()

    #run_model([5])
    #plt.show()

    

    # E = run_model([9,10,11,12,13,14,15,16,17,18,19,20])
    # print(E)
    # print(np.argmax(E), np.max(E))
    # plt.show()
  


    