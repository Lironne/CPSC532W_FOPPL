import torch,os 
import torch.distributions as dist

from daphne import daphne

dirn = os.path.dirname(os.path.abspath(__file__))


def grad_u(x):
    return 

def leapfrog(X,R,T,eps):
    return

def hmc(X,S,T,eps,M):
    for s in range(S):
        r = dist.normal.Normal(0,M).sample()

        u = dist.uniform.Uniform(0,1).sample()
           
    return 


if __name__ == '__main__':

    for i in range(1,5):

        steps = 100
        eps = -1e4
        M = 0

        filename = dirn + '/programs/{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        samples = hmc(graph, steps, eps, M)