import os, torch, utils

from daphne import daphne
from collections.abc import Iterable
from tests import is_tol, run_prob_test,load_truth

dirn = os.path.dirname(os.path.abspath(__file__))







if __name__ == '__main__':

    #run_deterministic_tests()
    #run_probabilistic_tests()


    for i in range(1,5):

        n_samples=1000

        filename = dirn + '/programs/{}.daphne'
        ast = daphne(['desugar', '-i', filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))


        #evaluate_program(ast)
    
       # utils.draw_hists("Eval Based", i, stream, n_samples)