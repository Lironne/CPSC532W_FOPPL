import os, torch, utils
import numpy as np

import primitives as pr
from daphne import daphne
from collections.abc import Iterable
from tests import is_tol, run_prob_test,load_truth

dirn = os.path.dirname(os.path.abspath(__file__))

def is_ast_c(ast):
    if isinstance(ast, Iterable):
        return False
    else:
        return torch.is_tensor(ast) or type(ast) == float or type(ast) == int or pr.is_primitive(ast) or type(ast) == bool

def add_context(vals, c, context):
    for i in range(len(c)):
        context[vals[i]] = c[i]
    return context

def evaluate_program_observe(dist, exp, context, sig):

    if torch.is_tensor(exp):
        c = exp
    else: 
        c, sig = evaluate_program_help(exp, context, sig)
    
    d, sig = evaluate_program_help(dist, context, sig)
    if not torch.is_tensor(exp):
        c, sig = torch.tensor(c)

    W = d.log_prob(c)
    W = W.unsqueeze(0) if W.dim() == 0 else W
    sig += W
    return c, sig

def evaluate_program_sample(dist, context, sig):
    try:
        # single sample
        return evaluate_program_help(dist, context, sig)[0].sample().item(), sig
    except:
        # multi bath sample
        return evaluate_program_help(dist, context, sig)[0].sample(), sig

def evaluate_program_let(exp_1,exp_0, context, sig):
    context[exp_1[0]], sig = evaluate_program_help(exp_1[1],context, sig)
    return evaluate_program_help(exp_0, context,sig)
    
def evaluate_program_bool(exp_1,exp_2,exp_3, context, sig):
    e_bool, sig = evaluate_program_help(exp_1, context, sig)
    if e_bool:
        return evaluate_program_help(exp_2, context, sig)
    else:
        return evaluate_program_help(exp_3, context, sig) 

def evaluate_program_help(ast,context,sig=0):
    # 8: case c
    if is_ast_c(ast):                                                         
        if pr.is_primitive(ast):                                             
            return context[ast[0]]
        elif type(ast) == bool:
            return torch.tensor([float(ast)]), sig
        else: 
            return ast, sig  
    # 4: case sample                          
    elif ast[0] == 'sample':                                                 
        return evaluate_program_sample(ast[1],context, sig)
    # 12: case let
    elif ast[0] == 'let':                                                    
        return evaluate_program_let(ast[1],ast[2], context, sig)
    # 6: case observe
    elif ast[0] == 'observe':                                                
        return evaluate_program_observe(ast[1],ast[2], context, sig)
    # 15: case if
    elif ast[0] == 'if':                                                     
        return evaluate_program_bool(ast[1],ast[2],ast[3], context, sig)
    # 10: case v
    elif isinstance(ast, str):                                               
        return context[ast], sig
    # 21: case (e_0,...,e_n)
    elif isinstance(ast, Iterable):                                          
        n = len(ast)
        c = []
        for i in range(1,n):
            c_i, sig = evaluate_program_help(ast[i], context, sig)
            c.append(c_i)
        # 28: case c
        if pr.is_primitive(ast[0]):                                          
            operator = context[ast[0]]
            return operator(*c), sig
        elif torch.is_tensor(ast[0]):
            return ast[0], sig
        # 25: case f
        else: 
            vals, e_0 = context[ast[0]]
            context = add_context(vals, c ,context)                                
            return evaluate_program_help(e_0, context, sig)                   

def evaluate_program(ast, sig=0):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """  
    context = pr.get_primitives()

    for exp in ast:
        if exp[0] == 'defn':
            context[exp[1]] = (exp[2], exp[3])
        else:
           return evaluate_program_help(exp, context, sig)

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0]
    
def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        filename = dirn + '/programs/tests/deterministic/test_{}.daphne'
        ast = daphne(['desugar', '-i', filename.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!    
        filename = dirn + '/programs/tests/probabilistic/test_{}.daphne'    
        ast = daphne(['desugar', '-i', filename.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    #run_deterministic_tests()
    #run_probabilistic_tests()


    for i in range(1,5):

        n_samples=1000

        filename = dirn + '/programs/{}.daphne'
        ast = daphne(['desugar', '-i', filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        stream = get_stream(ast)
    
        #utils.draw_hists("Eval Based", i, stream, n_samples)