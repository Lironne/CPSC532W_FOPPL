import os, torch, utils

from daphne import daphne
from collections.abc import Iterable
from evaluation_based_sampling import evaluate_program
from tests import is_tol, run_prob_test,load_truth

dirn = os.path.dirname(os.path.abspath(__file__))

def optimizer_step(Q,g):
    for v in g:
        lambda_v = get_parameters(Q[v])
        
    
    return Q

def elbo_grad(G,logW):
    for v in G:
        for l in range(L):
            if v in G[l]:
                F[l][v] = G[l][v], logW
            else:
                F[l][v],G[l][v] = 0,0
        b = torch.sum(torch.cov(F[:][v], G[:][v]))/torch.sum(torch.var(G[:][v]))
        g = torch.sum(F[:][v]- b * G[:][v])/L
    return g
                




def bbvi(ast,T,L):

    G = torch.zeros((T,L))

    for t in range(T):
        for l in range(L):
            r,sig = evaluate_program(ast)
            #store grad, sig
        g_hat = elbo_grad(G[t],logW)

    return None


if __name__ == '__main__':

    #run_deterministic_tests()
    #run_probabilistic_tests()


    for i in range(1,5):

        L,T = 100,100

        filename = dirn + '/programs/{}.daphne'
        ast = daphne(['desugar', '-i', filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        bbvi(ast)


        #evaluate_program(ast)
    
       # utils.draw_hists("Eval Based", i, stream, n_samples)