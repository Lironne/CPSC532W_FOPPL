import os,torch,utils
from daphne import daphne
from evaluation_based_sampling import evaluate_program

dirn = os.path.dirname(os.path.abspath(__file__))

def likelihood_weighting(ast, L):
    sig = torch.zeros(L,1)
    r = []
    for l in range(L):
       c , sig[l] = evaluate_program(ast,sig[l])
       r.append(c)
    return r, sig

def get_stream(ast):
    """Return a stream of  weighted samples"""
    while True:
        yield likelihood_weighting(ast)

if __name__ == '__main__':

    for i in range(1,5):

        iter = 10
        L = 100

        filename = dirn + '/programs/{}.daphne'
        ast = daphne(['desugar', '-i', filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        #stream = get_stream(ast)
        samples, weights = likelihood_weighting(ast, L)
        print('samples lengh: ',len(samples), 'weights: ', weights.size())
        #samples = samples * weights 
        
    
        #utils.gen_hists("IS", i, samples)