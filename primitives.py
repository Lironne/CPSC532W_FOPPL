import torch
import operator
import math
import copy

from functools import reduce
from collections.abc import Iterable


def mat_mul_tail(a,b):
    print('a: ', a, 'b: ', b)
    try:
        return a(b)
    except:
        return b @ a 

def mat_mul(a,b):
    print('a: ', a, 'b: ', b)
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
    elif torch.is_tensor(b):
        return torch.index_select(b, 0 ,a.int())
    else:
        return b[a.int()]

def vector_two(a):
    print('a: ', a)
    try: 
        x = a[0]
        y = a[1]
        return torch.stack([x,y])
    except:
        try:
            if y.size():
                y = torch.unsqueeze(y,1)
            else: 
                y = torch.unsqueeze(y,0) 
            #print('x: ', x, 'y: ', y)
            return torch.cat((x,y)) 
        except: 
            if isinstance(a[0], list):
                try: 
                    a[0].append(a[1])
                    return a[0]
                except:
                    return a[0][0]
            else:
                return list(a)
        



def vector_one(a): 
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
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    '/' : lambda a,b: b / a,  # use operator.div for Python 2
    '%' : operator.mod,
    '^' : operator.xor, 
    'sqrt': lambda x: torch.sqrt(x[0]),
    'vector': vector_one, 
    'get': get,
    'put': put,
    'first': lambda a: x[0] if len(x) > 1 else x[0][0],
    'last': lambda a: x[-1] if len(x) > 1 else x[0][-1],
    'second': lambda x: x[1] if len(x) > 1 else x[0][1],
    'rest': lambda x: x[1:] if len(x) > 1 else x[0][1:] ,
    'append': lambda x,a: torch.cat((a,torch.Tensor([x]))), 
    'hash-map': hash_map,
    '<': lambda a,b: b < a,
    '>':lambda a,b: b > a,
    'normal': lambda a,b: torch.distributions.normal.Normal(a,b),
    'beta': lambda a,b: torch.distributions.beta.Beta(a,b),
    'uniform': lambda a,b: torch.distributions.uniform.Uniform(a,b),
    'exponential': lambda a: torch.distributions.exponential.Exponential(a),
    'discrete': lambda a: torch.distributions.categorical.Categorical(torch.flatten(a[0])),
    'mat-transpose': lambda a: a[0].t(),
    'mat-mul':mat_mul,
    'mat-repmat': mat_remat,
    'mat-add': mat_add, 
    'mat-tanh': lambda a: torch.tanh(a[0]),
    }

def get_primitives():
    c = copy.deepcopy(context)
    return c

def is_primitive(operator):
    return operator in context
