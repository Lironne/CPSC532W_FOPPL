import os, torch, operator,copy, utils
import numpy as np
import torch.distributions as dist

from daphne import daphne
from primitives import vector, put, get, mat_mul
from tests import is_tol, run_prob_test,load_truth
from evaluation_based_sampling import evaluate_program_help

dirn = os.path.dirname(os.path.abspath(__file__))

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = { 'sqrt': lambda * x: torch.sqrt(torch.FloatTensor(x)),
        'vector': lambda *x: vector(x),
        'hash-map': lambda *x: dict(zip(x[::2],x[1::2])),
        '/' : lambda a,b: a / b,
        '+' : operator.add,
        '-' : operator.sub,
        '*' : operator.mul,
        '%' : operator.mod,
        '^' : operator.xor, 
        'get': get,
        'put': put,
        'first': lambda x: x[0],
        'last': lambda x: x[-1],
        'second': lambda x: x[1],
        'rest': lambda x: x[1:],
        'append': lambda x,a: torch.cat((x,torch.Tensor([a]))), 
        '<': lambda a,b: a < b, 
        '>': lambda a,b: a > b, 
        '=': lambda a,b: a == b,
        'and': lambda a,b: a and b,
        'or': lambda a,b: a or b,
        'normal': dist.normal.Normal,
        'beta': dist.beta.Beta,
        'uniform': dist.uniform.Uniform,
        'exponential': dist.exponential.Exponential,
        'flip': dist.bernoulli.Bernoulli,
        'dirichlet': dist.dirichlet.Dirichlet,
        'gamma' : dist.gamma.Gamma,
        'dirac': lambda *x: print(x),
        'discrete': lambda a: dist.categorical.Categorical(torch.flatten(a)),
        'mat-transpose': lambda t: torch.transpose(t,0,1),
        'mat-repmat': lambda t,d0,d1: t.repeat(d0,d1),
        'mat-mul': mat_mul,
        'mat-add': torch.add, 
        'mat-tanh': torch.tanh,
       }


# modified from an algorithm taken from
# https://www.geeksforgeeks.org/topological-sorting/
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


def deterministic_eval(exp, env):
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


def evaluate_graph(graph, observe=True):

    user_defn, G = graph[0], graph[1]
    var_order = top_sort(G['V'],G['A'])
    
    for fn in user_defn:
        env[fn] = user_defn[fn]

    for var in G['Y']:
        env[var] = G['Y'][var]

    for v in var_order:
        link_fn = G['P'][v]
        if not observe and link_fn[0] == 'observe*':
            continue
        # store result
        dist, _ = evaluate_program_help(link_fn[1], env)
        sample = dist.sample()
        sample = sample.unsqueeze(0) if sample.dim() == 0 else sample
        env[v] = sample
        G['Y'][v] = dist
    return copy.deepcopy(graph), copy.deepcopy(env)


def sample_from_graph(graph, context):
    ret_exp = graph[2]
    return evaluate_program_help(ret_exp, context)[0]

    

def sample_from_joint(graph, observe=True):
    "This function does ancestral sampling starting from the prior."
    
    ret_exp =  graph[2]
    evaluate_graph(graph,observe)

    return evaluate_program_help(ret_exp,env)[0]

    # try:
    #     # single sample
    #     return evaluate_program_help(ret_exp, env)[0]
    # except:
    #     # multi batch sample
    #     return evaluate_program_help(ret_exp, env)[0]



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

        n_samples = 1000

        filename = dirn + '/programs/{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        sample_from_joint(graph) 
        print('graph:', graph)

        #stream = get_stream(graph)
         
        #utils.draw_hists("Graph Based", i, stream, n_samples) 

    