import torch
import math, operator, copy

from functools import reduce
from collections.abc import Iterable

def mat_mul(a,b):
    try:
        return a @ b
    except:
        a = torch.transpose(a,0,1)
        return a @ b

    
def hash_map(a,b):
    if type(a) == tuple:
        c = a[0]
        val = a[1]
        c[b] = val
        return c
    elif isinstance(a, Iterable):
        return (a,b)
    else:
        c = {}
        c[b] = a 
        return c

def put(a,b):
    if isinstance(b, Iterable):
        if type(b) == torch.Tensor:
            b.numpy()
        b[int(a[1])] = a[0]
        return b
    else:
        return (a,b)

def vector(a):
    try:
        #has more than 1 element
        if a[1]:                     
            return torch.stack(a)
    except:
        try:
            # list of distributions
            if a[1]:
                return a             
        except:
            # vector base case
            return a[0]     
    


context = { 'sqrt':  torch.sqrt,
            'vector': lambda *x: vector(x),
            '+' : operator.add,
            '-' : operator.sub,
            '*' : operator.mul,
            '/' : lambda a,b: b / a,  # use operator.div for Python 2
            '%' : operator.mod,
            '^' : operator.xor, 
            'get': lambda a,b: a[b.long()],
            'put': put,
            'first': lambda x: x[0],
            'last': lambda x: x[-1],
            'second': lambda x: x[1],
            'rest': lambda x: x[1:],
            'append': lambda x,a: torch.cat((x,torch.Tensor([a]))), 
            '<': lambda a,b: b < a, 
            '>': lambda a,b: b > a, 
            'normal': lambda a,b: torch.distributions.normal.Normal(a,b),
            'beta': lambda a,b: torch.distributions.beta.Beta(a,b),
            'uniform': lambda a,b: torch.distributions.uniform.Uniform(a,b),
            'exponential': lambda a: torch.distributions.exponential.Exponential(a),
            'discrete': lambda a: torch.distributions.categorical.Categorical(torch.flatten(a)),
            'mat-transpose': lambda t: torch.transpose(t,0,1),
            'mat-repmat': lambda t,d0,d1: t.repeat(d0.int(),d1.int()),
            'mat-mul': mat_mul,
            'mat-add': torch.add, 
            'mat-tanh': torch.tanh,
            }

def get_primitives():
    return copy.deepcopy(context)

def is_primitive(operator):
    return operator in context
