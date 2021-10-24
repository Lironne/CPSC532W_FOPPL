import os,torch,utils
from daphne import daphne
from evaluation_based_sampling import evaluate_program

dirn = os.path.dirname(os.path.abspath(__file__))

def likelihood_weighting(ast, L):
    sig = torch.zeros(L)
    r = torch.zeros(L)
    for l in range(L):
       r[l] , sig[l] = evaluate_program(ast,sig[l])
    return r, sig

if __name__ == '__main__':

    for i in range(1,5):

        n_samples=1000

        filename = dirn + '/programs/{}.daphne'
        ast = daphne(['desugar', '-i', filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        samples, weights = likelihood_weighting(ast, n_samples)
        samples = samples * weights
    
        utils.draw_hists("Eval Based", i, samples, n_samples)