import torch
import numpy as np
import torch.distributions as dist
import os, math, copy, operator


from collections.abc import Iterable
from daphne import daphne


from primitives import vector_two, get, put, hash_map, mat_remat, mat_mul
from tests import is_tol, run_prob_test,load_truth
from evaluation_based_sampling import evaluate_program_help


dirn = os.path.dirname(os.path.abspath(__file__))
# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'sqrt': lambda x: torch.sqrt(x[0]),
        'vector': lambda *x: vector,
        '+' : operator.add,
        '-' : operator.sub,
        '*' : operator.mul,
        '/' : lambda a,b: b / a,  # use operator.div for Python 2
        '%' : operator.mod,
        '^' : operator.xor, 
        'get': get,
        'put': put,
        'first': lambda a: x[0] if len(x) > 1 else x[0][0],
        'last': lambda a: x[-1] if len(x) > 1 else x[0][-1],
        'second': lambda x: x[1] if len(x) > 1 else x[0][1],
        'rest': lambda x: x[1:] if len(x) > 1 else x[0][1:] ,
        'append': lambda x,a: torch.cat((a,torch.Tensor([x]))), 
        '<': lambda a,b: b < a, 
        '>': lambda a,b: b > a, 
        'normal': lambda a,b: torch.distributions.normal.Normal(a,b),
        'beta': lambda a,b: torch.distributions.beta.Beta(a,b),
        'uniform': lambda a,b: torch.distributions.uniform.Uniform(a,b),
        'exponential': lambda a: torch.distributions.exponential.Exponential(a[0]),
        'discrete': lambda a: torch.distributions.categorical.Categorical(torch.flatten(a[0])),
        #'discrete': lambda a: print('in disc? ', a),
        'mat-transpose': lambda a: a[0].t(),
        'mat-mul':mat_mul,
        'mat-repmat': mat_remat,
        'mat-add': torch.add, 
        'mat-tanh': lambda a: torch.tanh(a[0]),
       }

def topologicalSortUtil(v,E,neighbours,visited,stack):
 
    # Mark the current node as visited.
    visited[v] = 1
 
    # Recur for all the vertices adjacent to this vertex
    for i in neighbours:
        if visited[i] == 0:
            if i in E:
                topologicalSortUtil(i,E,E[i],visited,stack)
            else: 
                    topologicalSortUtil(i,E,[],visited,stack)
 
        # Push current vertex to stack which stores result
    stack.insert(0,v)

def top_sort(V,E):

    n = len(V)
    visited = dict(zip(V, np.zeros(n)))
    stack =[]
 
    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for i in range(n):
        if visited[V[i]] == 0:
            if V[i] in E:
                topologicalSortUtil(V[i],E,E[V[i]],visited,stack)
            else: 
                topologicalSortUtil(V[i],E,[],visited,stack)
 
    return stack



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
    var_order = top_sort(graph_struct['V'],graph_struct['A'])
    link_func = graph_struct['P']

    for fn in user_defn:
        env[fn] = user_defn[fn]

    for var in graph_struct['Y']:
        env[var] = graph_struct['Y'][var]

    for v in var_order:
        exp = link_func[v]
        print('v: ', v, 'link func: ', exp[1])
        eval_exp, sig = evaluate_program_help(exp[1], env)
        env[v] = eval_exp.sample()     
    
    sample, sig = evaluate_program_help(ret_exp, env)
    try:
        return sample.item()
    except:
        return sample



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
        filename = dirn +  '/programs/tests/probabilistic/test_{}.daphne'
        graph = daphne(['graph', '-i', filename.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    #run_deterministic_tests()
    #run_probabilistic_tests()




    for i in range(1,5):
        filename = dirn + '/programs/{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    