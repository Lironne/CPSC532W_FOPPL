import torch
import operator
import math
import copy

from functools import reduce
from collections.abc import Iterable

def mat_mul(a,b):
    try:
        return b @ a
    except:
        try:
            a = a.t() 
            return a @ b
        except:  
            shape = a.size()
            b = torch.reshape(b, (shape[0],shape[0]))
            return b @ a

def transpose(a):
    m = a[0]
    return m.t()

def mat_add(a,b):
    a = a.squeeze()
    b = b.squeeze()
    return torch.add(a,b)

def mat_remat(a,b):
    if type(a) == tuple:
        x = a[0].int().item()
        d = a[1].int().item() 
        return b.repeat(d,x)
    else:
        return (a,b)

def second(x):  
    if len(x) > 1 :
        return x[1]
    else: 
        return x[0][1]

def last(x):
    if len(x) > 1:
        return x[len(x) - 1]
    else:
        return x[0][-1]

def rest(x):
    if len(x) > 1:
        return x[1:]
    else:
        return x[0][1:]

def first(x):
    if len(x) > 1:
        return x[0]
    else:
        return x[0][0]

def append(x,a):
    return torch.cat((a,torch.Tensor([x])))
    

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

def get(a,b):
    if isinstance(a,list):
        a = a[0]
    if torch.is_tensor(b):
        return torch.index_select(b, 0 ,a.int())
    else:
        return b[a.int()]

def vector(a): 
    if torch.is_tensor(a[0]) and len(a) > 1:
        try:
            return torch.stack(a)
        except:
           return a
    elif torch.is_tensor(a[0]) and a[0].size():
        return torch.Tensor(a[0])
    elif torch.is_tensor(a[0]):
        return torch.Tensor(a)
    else:
        return a


context = {
    '+' : (operator.add, []),
    '-' : (operator.sub, []),
    '*' : (operator.mul, []),
    '/' : ((lambda a,b: b / a), []),  # use operator.div for Python 2
    '%' : (operator.mod, []),
    '^' : (operator.xor, []),
    'sqrt': (lambda x: torch.sqrt(x[0]) , []),
    'vector': (vector, []), 
    'get': (get,[]),
    'put': (put,[]),
    'first': (first ,[]),
    'last': (last,[]),
    'append': (append, []),
    'hash-map': (hash_map,[]),
    '<': (lambda a,b: b < a, []),
    '>':(lambda a,b: b > a, []),
    'normal': (lambda a,b: torch.distributions.normal.Normal(b,a), []),
    'beta': (lambda a,b: torch.distributions.beta.Beta(b,a), []),
    'uniform': (lambda a,b: torch.distributions.uniform.Uniform(b,a), []),
    'exponential': (lambda a: torch.distributions.exponential.Exponential(a),[]),
    'discrete': (lambda a: torch.distributions.categorical.Categorical(torch.flatten(a[0])),[]),
    'second': (second,[]),
    'rest': (rest, []),
    'mat-transpose': (transpose ,[]),
    'mat-mul':(mat_mul,[]),
    'mat-repmat': (mat_remat,[]),
    'mat-add': (mat_add, []), 
    'mat-tanh': (lambda a: torch.tanh(a[0]), [])
    }

def get_primitives():
    c = copy.deepcopy(context)
    return c

def is_primitive(operator):
    return operator in context
