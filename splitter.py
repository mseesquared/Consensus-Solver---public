"""
This file is intended to eventually be merged into (or rewritten)
for the CVXPY project.
"""

from __future__ import division
import cvxpy
import itertools
import pymetis
import numpy as np
from multiprocessing import Process, Pipe
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class Subsystem(cvxpy.Problem):

  def __init__(self, objective, constraints, public_vars, local_params, params):
    super(Subsystem, self).__init__(objective, constraints)
    self.public_vars = public_vars
    self.local_params = local_params
    self.params = params

  def update_params(self, param_values):
    for i, param in enumerate(self.params):
      param.value = param_values[i]
      self.local_params[i].value += self.public_vars[i].value - param.value

class Consensus(cvxpy.Problem):

  def __init__(self, objective, constraints=None, num_procs=2, max_iters=50):
    super(Consensus, self).__init__(objective, constraints)
    self.num_procs = num_procs  # Number of consensus problems
    self.max_iters = max_iters
    self.rho = 1.0              # Penalty parameter
    self.public_vars = []
    self.subsystems = []
    self.obj_history = []
    self.time_history = []

# Stop early if variance and change in variables is low enough
    self.min_variance = 1e-5
    self.min_change = 1e-5

  # Splitting a problem into a consensus problem
  # Given an objective, which is a sum of functions, returns the partition list
  # per function and list of public variables (variables are edges, functions are
  # vertices).
  def split(self, random_split=False):
    objective_fn = self.objective.args[0]
    constraints = self.constraints
    nparts = self.num_procs
    if not isinstance(objective_fn, cvxpy.atoms.affine.add_expr.AddExpression):
      objective_fn += 0

    num_funcs = len(objective_fn.args)
    num_components = len(objective_fn.args) + len(constraints)

    # The functions are indexed from 0 to num_funcs-1
    var_sets = [frozenset(func.variables()) for func in objective_fn.args]
    var_sets += [frozenset(constraint.variables()) for constraint in constraints]
    all_vars = self.variables()
    adj_list = [[] for _ in range(num_components)] # adj_list contains indices, not actual functions
    funcs_per_var = {}
    for var in all_vars:

      # find all functions that contain this variable
      funcs_per_var[var] = [i for i in range(num_components) if var
                            in var_sets[i]]

      # add an edge between any two of them
      for pair in itertools.permutations(funcs_per_var[var], 2):
        adj_list[pair[0]].append(pair[1])

    if not random_split:
      partition_per_func = pymetis.part_graph(nparts, adjacency=adj_list)[1]
    else:
      partition_per_func = np.random.randint(0, nparts, size=num_components)
    public_vars = []
    public_vars_per_partition = [[] for _ in range(nparts)]
    for var in all_vars:
      partitions_per_var = list(set([partition_per_func[i] for i in
                                     funcs_per_var[var]]))
      # If this is a public variable
      if len(partitions_per_var) > 1:
        public_vars.append(var)
        for partition in partitions_per_var:
          public_vars_per_partition[partition].append(var)

    # Index of functions that belong to each subproblem
    funcs_per_partition = [[] for _ in range(nparts)]
    for i in range(num_components):
      funcs_per_partition[partition_per_func[i]].append(i)
    subsystems = []
    for i in range(nparts):
      func_indices = [index for index in funcs_per_partition[i] if index < num_funcs]
      constrs = [constraints[index - num_funcs] for index in
                 funcs_per_partition[i] if index >= num_funcs]
      sub_objective = sum([objective_fn.args[func_index] for func_index in
                           func_indices])
      params = []
      local_params = []
      for var in public_vars_per_partition[i]:
        param = cvxpy.Parameter(*var.size, sign=var.sign, value=np.zeros(var.size))
        local_param = cvxpy.Parameter(*var.size, sign=var.sign, value=np.zeros(var.size))
        params.append(param)
        local_params.append(local_param)
        # Add prox term
        sub_objective += self.rho / 2 * cvxpy.sum_squares(var - param + local_param)
      # TODO Only deals with minimization problem for now
      subsystems.append(Subsystem(cvxpy.Minimize(sub_objective),
                                  constraints=constrs,
                                  public_vars=public_vars_per_partition[i],
                                  local_params=local_params, params=params))

    return [partition_per_func, public_vars, subsystems]

  # Solves a problem using ADMM/Consensus
  def consensus_solve(self, random_splitting=False):
    # Split problem
    splits, self.public_vars, self.subsystems = self.split(random_splitting)
    self.split_info = (splits, self.public_vars, self.subsystems)

    print("splits: %s\n" % str(splits))

    # Setting up pipes
    [pipes, procs] = self.start_pipes()

    # Run ADMM loop
    var = self.admm_loop(pipes)

    # Clean up
    [p.terminate() for p in procs]

    return self.objective.value

  # Set up pipes for each subproblem's prox
  def start_pipes(self):
    pipes = []
    procs = []
    for i in range(self.num_procs):
      local, remote = Pipe()
      pipes += [local]
      procs += [Process(target=self.run_process, args=(self.subsystems[i], remote))]
      procs[-1].start()
    return (pipes, procs)

  # In the main pipe, loop through the iterations for each subproblem
  def admm_loop(self, pipes):
    variances = [0]
    self.obj_history = []
    self.time_history = []
    for iteration in range(self.max_iters):
      # Gather.
      start_iter = time.time()
      max_change = 0
      # Sum the public variables values from the subsystems and average each by
      # the number of partition of each public variable.
      for var in self.public_vars:
        var._values_from_subsystems = []
      for i, pipe in enumerate(pipes):
        pipe_recv = pipe.recv()
        # print("Master pipe_recv: "); print(pipe_recv)
        for j, recv_value in enumerate(pipe_recv):
          self.subsystems[i].public_vars[j]._values_from_subsystems.append(recv_value)
      for var in self.public_vars:
        new_value = np.mean(var._values_from_subsystems, axis=0)
        if var.value is not None:
          max_change = max(max_change, np.linalg.norm(var.value - new_value))
        var.value = new_value
        var.variance = np.var(var._values_from_subsystems)

      variance = sum([var.variance for var in self.public_vars])
      variances.append(variance)
      print("iter %d, variances %f, change %f" % (iteration, variances[-1], max_change))

      # Stop early if variance is smaller than self.min_variance
      # print("checking if we're done " + str((variance < self.min_variance) and (delta <
      # self.min_change) and iteration > 1) + "\n")
      terminating = (iteration == self.max_iters - 1)
      # terminating = (iteration == self.max_iters - 1 or ((variance < self.min_variance) and
      # (max_change < self.min_change) and iteration > 1))
      # Scatter.
      # print("Master pipe_send: "); print([var.value for var in self.subsystems[i].public_vars])
      for pipe in pipes:
        pipe.send([[var.value for var in self.subsystems[i].public_vars], terminating])
      elapsed = time.time() - start_iter
      print("iter %d, took %s seconds" % (iteration, elapsed))
      self.time_history.append(elapsed)

      # Temporary: collect the variable values and objective function values
      # from all the subsystems
      saved_public_vars_values = [var.value for var in self.public_vars]
      for i, pipe in enumerate(pipes):
        pipe_recv = pipe.recv()
        for j, recv_value in enumerate(pipe_recv):
          self.subsystems[i].variables()[j].value = recv_value
      # The public variables values have been overwritten. We restore it here.
      for i, var in enumerate(self.public_vars):
        var.value = saved_public_vars_values[i]
      self.obj_history.append(self.objective.value)

      if terminating:
        break

    # Collect the variable values and objective function values from all the
    # subsystems
    saved_public_vars_values = [var.value for var in self.public_vars]
    for i, pipe in enumerate(pipes):
      pipe_recv = pipe.recv()
      for j, recv_value in enumerate(pipe_recv):
        self.subsystems[i].variables()[j].value = recv_value
    # The public variables values have been overwritten. We restore it here.
    for i, var in enumerate(self.public_vars):
      var.value = saved_public_vars_values[i]
    # print all variable values
    print("Final variable values")
    sorted_vars = sorted([var for var in self.variables()], key=lambda var: var.id)
    print([var.value for var in sorted_vars])
    print("Final objective values = %g" % self.objective.value)

    return variances

  # Indefinite minimization loop that run each subsystem
  def run_process(self, subsystem, pipe):
    while True:
      try:
        # subsystem.solve(solver=cvxpy.SCS, use_indirect=True)
        subsystem.solve(solver=cvxpy.ECOS)
        # print("Variable values"); print([(var.name(), var.value) for var in subsystem.variables()])
        # print("Parameter values"); print([(param.name(), param.value) for param in subsystem.parameters()])
        pipe.send([var.value for var in subsystem.public_vars])
        [param_values, terminating] = pipe.recv()
        subsystem.update_params(param_values)

        # Temporary: send all variable values back
        pipe.send([var.value for var in subsystem.variables()])
        if terminating:
          break
      except (Exception, e):
        print("exception! \n")
        print(e)
        break
    # Send all variable values back to master
    print("sending back variables")
    print(str([var.value for var in subsystem.variables()]))

    pipe.send([var.value for var in subsystem.variables()])

  # Plot obj_history
  def plot_history(self):
    plt.plot(self.obj_history, 'r-')
    plt.xlabel('iteration')
    plt.ylabel('objective function value')
    # plt.legend(['optimal cut'], loc='upper right')
    # plt.legend(['random cut'], loc='upper right')
    plt.show()

