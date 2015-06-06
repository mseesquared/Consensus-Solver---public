from __future__ import division
import cvxpy as cvx
import numpy as np
import ../src/splitter
from cvxpy.tests.base_test import BaseTest
import unittest
import time
import matplotlib.pyplot as plt


class TestConsensus(BaseTest):

  def run_and_test(self, c, name, random_splitting=False):
    print('\n\n\n### Test: ' + name + ' ###\n')
    consensus_start = time.time()
    result = c.consensus_solve(random_splitting)
    consensus_time = time.time() - consensus_start

    cvx_start = time.time()
    c.solve()
    cvx_time = time.time() - cvx_start

    print("splitter took {} seconds\n".format(consensus_time))
    print("cvxpy took {} seconds\n".format(cvx_time))

    # Assert correctness
    self.assertAlmostEqual(result, c.solve(), places=2)


  # def test_single_var(self):
  def single_var(self):
    V1 = cvx.Variable()
    # objective_fn = cvx.norm(V1 + V2 + V3) + cvx.norm(V1 + V2 + V4) + cvx.norm(V4 + V5 + 1)
    # objective_fn = cvx.norm(V1 + V2 + 1) + cvx.norm(V1 + V2 + 1)
    objective_fn = cvx.norm(V1 + 1) + cvx.norm(V1 + 1)
    c = splitter.Consensus(cvx.Minimize(objective_fn), num_procs=2)

    self.run_and_test(c, 'one_dim')

  def single_constraint(self):
  # def test_single_constraint(self):
    V1 = cvx.Variable()
    # objective_fn = cvx.norm(V1 + V2 + V3) + cvx.norm(V1 + V2 + V4) + cvx.norm(V4 + V5 + 1)
    # objective_fn = cvx.norm(V1 + V2 + 1) + cvx.norm(V1 + V2 + 1)
    objective_fn = cvx.norm(V1 + 1) + cvx.norm(V1 + 1)
    constr = [V1 >= 3]
    c = splitter.Consensus(cvx.Minimize(objective_fn), constraints=constr, num_procs=2)

    self.run_and_test(c, 'one_dim_with_1_constraint')

  # def test_single_var_with_1_constraint(self):
  def single_var_with_1_constraint(self):
    V1 = cvx.Variable()
    objective_fn = cvx.norm(V1)
    constraints = [V1 >= 2]
    c = splitter.Consensus(cvx.Minimize(objective_fn), constraints, num_procs=2)
    self.run_and_test(c, 'one_dim_with_1_constraint')

  # Test a single commodity flow problem
  def test_commodity(self):
    m = 60
    n = 27
    m_a = 30
    n_a = 12
    n_b = 12
    R, cap = self.flow_data(m, m_a, n, n_a, n_b)
    constr = []
    ind_flows = []

    #  Individual flows
    for i in range(n):
      ind_flows.append(cvx.Variable())
      constr.append(ind_flows[i] >= 0)
      
    # Capacity constraints
    for i in range(m):
      sum_flow = 0
      for j in range(n):
        if R[i][j] > 0:
          sum_flow = sum_flow + ind_flows[j]

      constr.append(sum_flow <= cap[i])

    # Want to maximize flows
    # objective = cvx.Minimize(-sum([cvx.sqrt(max(ind_flows[i], 0)) for i in range(n)]))
    objective = cvx.Minimize(-sum(ind_flows))

    c = splitter.Consensus(objective, constr, num_procs=2, max_iters=100)
    # self.run_and_test(c, 'one commodity flow', random_splitting=False)
    self.plot(c)

  # Test a dual commodity flow problem
  # def test_dual_commodity(self):
  def dual_commodity(self):
    m = 40
    n = 17
    k = 12
    m_a = 20
    n_a = 6
    n_b = 6
    R, cap = self.flow_data(m, m_a, n, n_a, n_b)
    constr = []
    # all_flows = cvx.Variable(n)
    ind_flows = []

    #  Individual flows
    for i in range(n):
      ind_flows.append(cvx.Variable())
      # constr.append(ind_flows[i] == all_flows[i])
      constr.append(ind_flows[i] >= 0)
      
    # Capacity constraints
    for i in range(m):
      sum_flow = 0
      for j in range(n):
        if R[i][j] > 0:
          sum_flow = sum_flow + ind_flows[j]

      constr.append(sum_flow <= cap[i])
    # Capacity constraints
    # for i in range(m):
      # constr.append(R[i] * all_flows <= cap[i])

    # Want to maximize flows
    # objective = cvx.Minimize(-sum([cvx.sqrt(max(ind_flows[i], 0)) for i in range(n)]))
    objective = cvx.Minimize(-sum(ind_flows))

    c = splitter.Consensus(objective, constr, num_procs=2, max_iters=200)
    # self.run_and_test(c, 'two commodity flow', random_splitting=False)

  # m = number of edges
  # n = number of flows
  # m_a = cut of the graph
  # n_a, b = number of flows in each half
  def flow_data(self, m, m_a, n, n_a, n_b):
    R = []
    np.random.seed(1)
    for i in range(m - m_a):
      # [a only, b only, both]
      flows = [np.minimum(1, np.random.randint(0, 3, (1, n_a))), np.zeros((1, n_b)), \
        np.random.randint(0, 2, (1, n - n_a - n_b))]
      R.append(np.concatenate(flows, axis=1)[0])
    for i in range(m_a):
      # [a only, b only, both]
      flows = [np.zeros((1, n_a)), np.minimum(1, np.random.randint(0, 3, (1, n_b))), \
        np.random.randint(0, 2, (1, n - n_a - n_b))]
      R.append(np.concatenate(flows, axis=1)[0])

    print R
    c = 3 * np.random.rand(m) + 0.5;

    return (R, c)

  # def test_single_var_with_1_leq_constraint(self):
  def single_var_with_1_leq_constraint(self):
    V1 = cvx.Variable()
    objective_fn = cvx.norm(V1)
    constraints = [V1 <= 2]
    c = splitter.Consensus(cvx.Minimize(objective_fn), constraints, num_procs=2)
    self.run_and_test(c, 'one_dim_with_1_leq_constraints')

  # def test_single_var_with_2_constraints(self):
  # def single_var_with_2_constraints(self):
    V1 = cvx.Variable()
    objective_fn = cvx.norm(V1)
    constraints = [V1 <= 2, V1 >= -1]
    c = splitter.Consensus(cvx.Minimize(objective_fn), constraints, num_procs=2)
    self.run_and_test(c, 'one_dim_with_2_constraints')

  # def test_multi_var_dim_1(self):
  def multi_var_dim_1(self):
    V1 = cvx.Variable()
    V2 = cvx.Variable()
    V3 = cvx.Variable()
    V4 = cvx.Variable(2)
    V5 = cvx.Variable()
    objective_fn = cvx.norm(V1 + V2 + V3 + 4) + cvx.norm(V1 + V2 + sum(V4)/2) + cvx.norm(sum(V4 +
    np.ones((2, 1))) + V5 + 1)
    c = splitter.Consensus(cvx.Minimize(objective_fn), num_procs=2)
    result = c.consensus_solve()
    self.assertAlmostEqual(result, c.solve(), places=2)
    self.assertAlmostEqual(result, 0, places=2)
    self.run_and_test(c, 'multi_var_dim_1')

  # def test_multi_var_multi_dim(self):
  def multi_var_multi_dim(self):
    V1 = cvx.Variable(10)
    V2 = cvx.Variable(10)
    V3 = cvx.Variable(10)
    V4 = cvx.Variable(10)
    V5 = cvx.Variable(10)
    objective_fn = cvx.norm(V1 + V2 + V3 + 4) + cvx.norm(V1 + V2 + V4/2) + cvx.norm(V4 + V5 + 1) + 5
    c = splitter.Consensus(cvx.Minimize(objective_fn), num_procs=2)
    result = c.consensus_solve()
    self.assertAlmostEqual(result, c.solve(), places=2)
    self.assertAlmostEqual(result, 5, places=2)
    self.run_and_test(c, 'multi_var_multi_dim')

  # def test_multi_var_multi_dim_big(self):
  def multi_var_multi_dim_big(self):
    V1 = cvx.Variable(5)
    V2 = cvx.Variable(5)
    V3 = cvx.Variable(5)
    V4 = cvx.Variable(5)
    V5 = cvx.Variable(5)
    V6 = cvx.Variable(5)
    V7 = cvx.Variable(5)
    objective_fn = cvx.norm(V1 + V2 + V3 + 4) + cvx.norm(V1 + V2 + V4/2) + cvx.norm(V4 + V5 + 1) + \
    cvx.norm(V3 + V6 + V7) + cvx.norm(V2 - V6 + V7 - 3)
    c = splitter.Consensus(cvx.Minimize(objective_fn), num_procs=2)
    self.run_and_test(c, 'multi_var_multi_dim')

  # def test_multi_var_multi_dim_big_random_splitting(self):
  def multi_var_multi_dim_big_random_splitting(self):
    V1 = cvx.Variable(5)
    V2 = cvx.Variable(5)
    V3 = cvx.Variable(5)
    V4 = cvx.Variable(5)
    V5 = cvx.Variable(5)
    V6 = cvx.Variable(5)
    V7 = cvx.Variable(5)
    objective_fn = cvx.norm(V1 + V2 + V3 + 4) + cvx.norm(V1 + V2 + V4/2) + cvx.norm(V4 + V5 + 1) + \
    cvx.norm(V3 + V6 + V7) + cvx.norm(V2 - V6 + V7 - 3)
    c = splitter.Consensus(cvx.Minimize(objective_fn), num_procs=2, max_iters=20)
    self.run_and_test(c, 'multi_var_multi_dim', random_splitting=True)

  # def test_svm1(self):
  def svm1(self):
    NUM_PROCS = 4
    DATA_SETS = 4
    SPLIT_SIZE = 100
    MAX_ITER = 100

    # Problem data.
    np.random.seed(1)
    N = NUM_PROCS*SPLIT_SIZE
    n = 100
    sigma = np.sqrt(n-1)
    data = []
    offset = np.random.randn(n-1, 1)
    for i in range(DATA_SETS//2):
      samples = np.random.normal(1.0, sigma, (n-1, SPLIT_SIZE))
      data += [(1, offset + samples)]
    for i in range(DATA_SETS//2):
      samples = np.random.normal(-1.0, sigma, (n-1, SPLIT_SIZE))
      data += [(-1, offset + samples)]
    data = data*(NUM_PROCS//DATA_SETS)

  # Loss function
    f = []
    x = cvx.Variable(n)
    for split in data:
      label, sample = split
      slack = cvx.pos(1 - label*(sample.T*x[:-1] - x[-1]))
      f += [(1/N)*cvx.sum_entries(slack)]

    c = splitter.Consensus(cvx.Minimize(sum(f)), num_procs = NUM_PROCS, max_iters = MAX_ITER)
    self.run_and_test(c, 'svm')

  # def test_svm2(self):
  def svm2(self):
    NUM_PROCS = 8
    DATA_SETS = 8
    SPLIT_SIZE = 100
    MAX_ITER = 100

  # Problem data.
    np.random.seed(1)
    N = NUM_PROCS*SPLIT_SIZE
    n = 200
    sigma = np.sqrt(n-1)
    data = []
    offset = np.random.randn(n-1, 1)
    for i in range(DATA_SETS//2):
      samples = np.random.normal(1.0, sigma, (n-1, SPLIT_SIZE))
      data += [(1, offset + samples)]
    for i in range(DATA_SETS//2):
      samples = np.random.normal(-1.0, sigma, (n-1, SPLIT_SIZE))
      data += [(-1, offset + samples)]
    data = data*(NUM_PROCS//DATA_SETS)

  # Loss function
    f = []
    x = cvx.Variable(n)
    for split in data:
      label, sample = split
      slack = cvx.pos(1 - label*(sample.T*x[:-1] - x[-1]))
      f += [(1/N)*cvx.sum_entries(slack)]

    c = splitter.Consensus(cvx.Minimize(sum(f)), num_procs = NUM_PROCS, max_iters = MAX_ITER)
    self.run_and_test(c, 'svm 2')

  # least squares
  # def test_dumbbell(self):
  def dumbbell(self):
    cluster1 = []
    cluster2 = []
    objective_fn = 0
    for i in range(10):
      cluster1.append(cvx.Variable(5))
      cluster2.append(cvx.Variable(5))

    complicating_var = cvx.Variable(5)

    for i in range(4):
      problem1 = 0
      problem2 = 0
      for j in range(10):
        A = np.random.randn(5)
        B = np.random.randn(5)
        problem1 = problem1 + cluster1[j] - A
        problem2 = problem2 + cluster2[j] - B

      objective_fn  = objective_fn + cvx.sum_squares(problem1) + cvx.sum_squares(problem2)

    objective_fn = objective_fn + cvx.sum_squares(sum(cluster1) - complicating_var) + cvx.sum_squares(sum(cluster2) - complicating_var)

    c = splitter.Consensus(cvx.Minimize(objective_fn), num_procs = 2, max_iters = 100)
    self.run_and_test(c, 'dumbbell')

  # def test_least_squares(self):
  def random_splitting_plot_graph(self):
    V1 = cvx.Variable(5)
    V2 = cvx.Variable(5)
    V3 = cvx.Variable(5)
    V4 = cvx.Variable(5)
    V5 = cvx.Variable(5)
    V6 = cvx.Variable(5)
    V7 = cvx.Variable(5)
    objective_fn = cvx.norm(V1 + V2 + V3 + 4) + cvx.norm(V1 + V2 + V4/2) + cvx.norm(V4 + V5 + 1) + \
    cvx.norm(V3 + V6 + V7) + cvx.norm(V2 - V6 + V7 - 3)
    c = splitter.Consensus(cvx.Minimize(objective_fn), num_procs=2)
    c.consensus_solve()
    c.plot_history()
    c.consensus_solve(random_splitting=True)
    c.plot_history()

  # Plotting code, given a consensus problem
  def plot(self, c):
    p_star = c.solve()
    print "# optimal is: %d" % p_star

    c.consensus_solve()
    print "\nnumber of global variables for spectral cut: %d\n" % len(c.public_vars)
    obj_spectral_cut = np.array(c.obj_history)
    time_spectral_cut = np.cumsum(c.time_history)
    obj_random_cut = []
    time_random_cut = []
    for _ in range(5):
      c.consensus_solve(random_splitting=True)
      print "\nnumber of public variables: %d\n" % len(c.public_vars)
      obj_random_cut.append(np.array(c.obj_history))
      time_random_cut.append(np.cumsum(c.time_history))

    # Plot objective value vs iteration
    plt.plot(np.abs(obj_spectral_cut - p_star))
    for i in range(5):
      plt.plot(np.abs(obj_random_cut[i] - p_star))
    plt.yscale('log')
    plt.xlabel('iteration', fontsize=14)
    plt.ylabel(r'$|f - f^\star|$', fontsize=16)
    legend = ['spectral cut'] + ['random cut ' + str(i+1) for i in range(5)]
    plt.legend(legend, loc='lower right')
    # plt.legend(legend, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    # plt.savefig('routing.png', bbox_inches='tight')
    plt.show()

    # Plot objective value vs time
    plt.plot(time_spectral_cut, np.abs(obj_spectral_cut - p_star))
    for i in range(5):
      plt.plot(time_random_cut[i], np.abs(obj_random_cut[i] - p_star))
    plt.yscale('log')
    plt.xlabel('time', fontsize=14)
    plt.ylabel(r'$|f - f^\star|$', fontsize=16)
    legend = ['spectral cut'] + ['random cut ' + str(i+1) for i in range(5)]
    plt.legend(legend, loc='lower right')
    plt.show()

if __name__ == '__main__':
  unittest.main()


