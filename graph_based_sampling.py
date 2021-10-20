import torch
import os
import torch.distributions as dist
import operator
import math

from daphne import daphne

from primitives import context as pr
from tests import is_tol, run_prob_test,load_truth



dirn = os.path.dirname(os.path.abspath(__file__))
# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
        'sqrt': torch.sqrt,
        'vector': lambda *x: torch.tensor(x),
        '+' : operator.add,
        '-' : operator.sub,
        '*' : operator.mul,
        '/' : lambda a,b: b / a,  # use operator.div for Python 2
        '%' : operator.mod,
        '^' : operator.xor, 
        'get': lambda a,x: a[x],
        #'put': lambda a,i,x: a[i] = x,
        'first': lambda a: a[0],
        'last': lambda a: x[-1],
        'append': lambda a,x: a.append(x),
        '<': lambda a,b: b < a, 
        '>':lambda a,b: b > a, 
        'normal': torch.distributions.normal.Normal,
        'beta': torch.distributions.beta.Beta,
        'uniform': torch.distributions.uniform.Uniform,
        'exponential': torch.distributions.exponential.Exponential,
        'discrete': torch.distributions.categorical.Categorical,
        'second': lambda a: a[1],
        'rest': lambda a: a[1:],
        'mat-transpose': lambda a: a.t(),
        # 'mat-mul': torch.mat_mul,
        # 'mat-repmat': mat_remat,[]),
        # 'mat-add': (mat_add, []), 
        # 'mat-tanh': (lambda a: torch.tanh(a[0]), [])'
       }

def top_sort(V, E):

    deg_count = {0:[]}

    for v in V:
        if v in E:
            deg_v = len(E[v])
            if deg_v in deg_count:
                deg_count[deg_v].append(v)
            else:
                deg_count[deg_v] = [v]
        else:
            deg_count[0].append(v)  
    return deg_count



def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."

    print('graph: ', graph)
    
    user_defn = graph[0]
    graph_struct = graph[1]
    ret_exp = graph[2]

    var_order = top_sort(graph_struct['V'], graph_struct['A'])
    link_func = graph_struct['P']

    var_val = {}

    for deg in var_order:
        V = var_order[deg]
        for v in V:
            print('var_val: ', var_val)
            exp = link_func[v]
            op = exp[0]
            if op == 'if':
            


            ret = deterministic_eval(exp[1])
            var_val[v] = ret.sample()

    return var_val[ret_exp]



def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        filename = dirn + '/programs/tests/deterministic/test_{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!  
        filename = dirn +  '/programs/tests/probabilistic/test_5.daphne'
        graph = daphne(['graph', '-i', filename.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    #run_deterministic_tests()
    run_probabilistic_tests()




    for i in range(1,5):
        filename = dirn + '/programs/{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    