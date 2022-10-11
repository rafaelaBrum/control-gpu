import random
import math
import sys

import logging


class Poisson:
    """
    lambda : the average number of events that happen in one second
    seed : to initialize the uniform number generator
    """

    def __init__(self, p_lambda, k=1, seed=None):
        self.random = random

        if seed is not None:
            self.random.seed(a=seed)
        else:
            seed = random.randrange(sys.maxsize)
            rng = random.Random(seed)
            self.random = rng

        logging.info("Poison SEED: {} ".format(seed))

        ''' 
        The average number of failures expected  to
        happen each second in a   Poisson Process, which is also
        called event rate or rate parameter. 
        '''
        self.p_lambda = p_lambda
        self.k = k

    def random_uniform(self):
        return self.random.uniform(0.0, 1.0)

    def __events_arrival_probability(self):
        return (self.p_lambda ** self.k * math.exp(-self.p_lambda)) / math.factorial(self.k)

    def event_happened(self):
        return self.random_uniform() <= self.__events_arrival_probability()

    def sample(self):
        return math.exp(1.0 - self.random_uniform()) / self.p_lambda

    @staticmethod
    def expected_cost(n, c, lambda_var, gamma, epsilon, var_r):

        current_sum = 0.0
        for k in range(int((epsilon * var_r) / n)):
            current_sum += (math.exp(-(lambda_var * (k - 1)) / epsilon) * math.floor(k / epsilon) * (1 - gamma)) + (
                math.exp(-(lambda_var * (k - 1)) / epsilon) * (k / epsilon))

        return c * n * (math.exp(lambda_var / epsilon) - 1) * current_sum
