import import_bfts

import numpy as np

from algorithms.bfts import BFTS 
import algorithms.posteriors as p

import scipy.stats as sp

from environments.csv_dist import csv_dist_bandit, csv_dist_means
from algorithms.posteriors import TDistribution

import matplotlib.pyplot as plt

import pickle

dir_ = "./school-real"

m = 10
time = 2000

def create_t_dist():
   return TDistribution(.5)

def pdf(x, posterior, rewards):
   n = len(rewards)
   freedom = posterior.freedom(n)
   sigma = np.sqrt(np.var(rewards)/freedom)
   mu = np.mean(rewards)
   return sp.t.pdf((x - mu)/sigma, freedom) / sigma

csv_dist_fn = "./brute-force-reward.csv"
bandit = csv_dist_bandit(csv_dist_fn)

np.random.seed(1)

posteriors = [create_t_dist() for x in range(len(bandit.arms))]

algo = BFTS(bandit, m, posteriors)

#init posteriors
total_inits = 0
for i in range(len(bandit.arms)):
   for j in range(posteriors[i].times_to_init()):
      reward = bandit.play(i)
      algo.add_reward(i, reward)
      total_inits = total_inits + 1

#post-init BFTS steps
for t in range(1, time + 1 - total_inits):
   (J_t, arm, reward) = algo.step(t)
   J_t = [str(i) for i in J_t]

   if t % 100 == 0:
      #plot posteriors
      fig = plt.figure()
      
      for i in range(len(bandit.arms)):
         p = posteriors[i]
         rewards = algo.rewards_per_arm[i]
        
         #!!! Would be cleaner to look at the posteriors (mean and variance),
         #    to determine these bounds
         x = np.linspace(0.97, .987, 500)
         
         pdf_ = pdf(x, p, rewards)
         plt.plot(x, pdf_)
      
      plt.xlabel("x")
      plt.ylabel("Density")
      plt.title(f"BFTS posteriors")
      #!!! Very hacky way to store interactive figures (will not work across
      #    python versions), but OK for this experiment :)
      fn = dir_+"/posterior-fig-"+str(t)+".pickle"
      with open(fn, 'wb') as f:
        pickle.dump(fig, f)
