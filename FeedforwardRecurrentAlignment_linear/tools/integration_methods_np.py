import numpy as np


def forward_euler_constInp(inputs,
                  w_rec,
                  start_activity,
                  delta_t,
                  time_steps,
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float):
    """ Theano implementation of the forward euler method with input constant in time.
    For explanation see numpy implementation. """

    proc_inputs = inputs[0,:]
    num_neurons = proc_inputs.shape[0]


    # Calculate the activities of all neurons for all times by scanning over "time".
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_frames, num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = np.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity

    fn_euler=lambda tl_neuronal_activity,proc_inputs,delta_t,tau,w_rec: (
            tl_neuronal_activity + (delta_t / tau) * (
            - tl_neuronal_activity + nonlinearity(np.dot(tl_neuronal_activity, w_rec.T) + proc_inputs)))


    res=[]
    res.append(t_start_activity)
    for istep in range(k):
        res.append(fn_euler(res[-1],proc_inputs,delta_t,tau,w_rec))
    return np.asarray(res)[1:]


def runge_kutta_explicit(inputs,
                  w_rec,
                  start_activity,
                  delta_t,
                  time_steps,
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float):
    """ Theano implementation of the runge kutta method, 4th order.
    For explanation see numpy implementation. """

    proc_inputs = inputs[0,:]
    num_neurons = inputs.shape[0]



    # Calculate the activities of all neurons for all times by scanning over "time".
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_frames, num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = np.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity

    def fprime(x, proc_inputs, tau, nonlinearity, w_rec):
        f = (-x + nonlinearity(proc_inputs + np.dot(w_rec, x)))/tau
        return f

    def rk4step(delta_t, x, fprime, proc_inputs, tau, nonlinearity, w_rec):
        k1 = fprime(x, proc_inputs, tau, nonlinearity, w_rec)
        k2 = fprime(x + 0.5*k1*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        k3 = fprime(x + 0.5*k2*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        k4 = fprime(x + k3*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        x_new = x + delta_t*( (1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4 )
        return x_new

    fn_rke= lambda tl_neuronal_activity,proc_inputs,delta_t,tau,w_rec: rk4step(
                    delta_t, tl_neuronal_activity, fprime, proc_inputs, tau, nonlinearity, w_rec)

#    #replaces theano.scan, only store last result
#    res = np.copy(t_start_activity)
#    for istep in range(k):
#        res = fn_rke(res,proc_inputs,delta_t,tau,w_rec)
#    return res

    #replaces theano.scan, store all res
    res=[]
    res.append(t_start_activity)
    for istep in range(k):
        res.append(fn_rke(res[-1],proc_inputs,delta_t,tau,w_rec))
    return np.asarray(res)[1:]


def runge_kutta2(inputs,
                  w_rec,
                  start_activity,
                  delta_t,
                  time_steps,
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float):
    """ Theano implementation of the runge kutta method, 2nd order.  """

    proc_inputs = inputs[0,:]
    num_neurons = proc_inputs.shape[0]

    def fprime(x, proc_inputs, tau, nonlinearity, w_rec):
        f = (-x + nonlinearity(proc_inputs + np.dot(w_rec, x)))/tau
        return f

    def rk2step(delta_t, x, fprime, proc_inputs, tau, nonlinearity, w_rec):
        k1 = fprime(x, proc_inputs, tau, nonlinearity, w_rec)
        x_new = x + delta_t*fprime(x+k1*0.5*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        return x_new

    fn_rk2=lambda tl_neuronal_activity,proc_inputs,delta_t,tau,w_rec: rk2step(
                delta_t, tl_neuronal_activity, fprime, proc_inputs, tau, nonlinearity, w_rec)

    # Calculate the activities of all neurons for all times by scanning over "time".
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = np.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity


    #replaces theano.scan, store all res
    res=[]
    res.append(t_start_activity)
    for istep in range(k):
        res.append(fn_rk2(res[-1],proc_inputs,delta_t,tau,w_rec))
    return np.asarray(res)[1:]


