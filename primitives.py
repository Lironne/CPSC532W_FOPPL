import torch
import math, operator, copy

from functools import reduce
from collections.abc import Iterable

def mat_mul(a,b):
    try:
        return a @ b
    except:
        return b @ a.t()

def hash_map(a):
    it = iter(a)
    m = {}
    for key, val in zip(it,it):
        if isinstance(key, str):
            m[key] = val
        else:
            m[key.item()] = val 
    return m

def append(a,e):
    try:
        return torch.cat((a,torch.tensor([e])))
    except:
        # a is list
        try:
            a.append(e)
            return a
        except:
            return [a[0],e]


def put(a,i,e):
    if torch.is_tensor(i):
        a[i.int().item()] = e
    else:
        a[i] = e
    return a

def get(a,i):
    if torch.is_tensor(i):
        return a[i.int().item()]
    else:
        return a[i]

def vector(a):   
    try:
        return torch.stack(a)                        
    except:
        if isinstance(a[0],float) or isinstance(a[0],int):
            return torch.FloatTensor(a)
        else:
            return a


context = { 'sqrt': lambda * x: torch.sqrt(torch.FloatTensor(x)),
            'vector': lambda *x: vector(x),
            'hash-map': lambda *x: dict(zip(x[::2],x[1::2])),
            '+' : operator.add,
            '-' : operator.sub,
            '*' : operator.mul,
            '/' : lambda a,b: a / b,  # use operator.div for Python 2
            '%' : operator.mod,
            '^' : operator.xor, 
            'put': put,
            'append': append,
            'get': get,
            'first': lambda x: x[0],
            'last': lambda x: x[-1],
            'second': lambda x: x[1],
            'rest': lambda x: x[1:],
            '<': lambda a,b: a < b, 
            '>': lambda a,b: a > b, 
            'normal': lambda a,b: torch.distributions.normal.Normal(a,b),
            'beta': lambda a,b: torch.distributions.beta.Beta(a,b),
            'uniform': lambda a,b: torch.distributions.uniform.Uniform(a,b),
            'exponential': lambda a: torch.distributions.exponential.Exponential(a),
            'discrete': lambda a: torch.distributions.categorical.Categorical(torch.flatten(a)),
            'mat-transpose': lambda t: t.t(),
            'mat-repmat': lambda t,d0,d1: t.repeat(d0,d1),
            'mat-mul': mat_mul,
            'mat-add': torch.add, 
            'mat-tanh': torch.tanh,
            }

def get_primitives():
    return copy.deepcopy(context)

def is_primitive(operator):
    return operator in context
