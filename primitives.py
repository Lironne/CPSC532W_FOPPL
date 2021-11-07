import torch
import operator, copy

from collections.abc import Iterable
import distributions as d
import torch.distributions as dist


def append(a,e):
    try:
        return torch.cat((a,torch.tensor([e])))
    except:
        # a is list
        try:
            a.append(e)
            return a
        # list base case
        except:
            return [a[0],e]


def put(a,i,e):
    if torch.is_tensor(i):
        a[i.long()] = e
    else:
        a[i] = e
    return a

def get(a,i):
    if torch.is_tensor(i):
        return a[i.long()]
    else:
        return a[i]

def vector(a):   
    try:
        return torch.stack(a)                        
    except:
        # if isinstance(a[0],float) or isinstance(a[0],int):
        #     return torch.FloatTensor(a)
        # else:
        return a

def mat_mul(a,b):
    try:
        return a @ b
    except:
        print('b size', b.size(), 'a size: ', a.size())
        return b @ torch.transpose(a,0,1)


context = { 'sqrt': torch.sqrt,
    	    'abs': torch.abs,
            'vector': lambda *x: vector(x),
            'hash-map': lambda *x: dict(zip(x[::2],x[1::2])),
            '/' : lambda a,b: a / b, 
            '+' : operator.add,
            '-' : operator.sub,
            '*' : operator.mul,
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
            '=': lambda a,b: a == b,
            'and': lambda a,b: a and b,
            'or': lambda a,b: a or b,
            'normal': d.Normal,
            'beta': dist.beta.Beta,
            'uniform': dist.uniform.Uniform,
            'uniform-continuous': d.Uniform,
            'exponential': dist.exponential.Exponential,
            'flip': d.Bernoulli,
            'dirichlet': d.Dirichlet,
            'gamma' : d.Gamma,
            'discrete': lambda a: d.Categorical(torch.flatten(a)),
            'mat-transpose': lambda t: torch.transpose(t,0,1),
            'mat-repmat': lambda t,d0,d1: t.repeat(d0.int(),d1.int()),
            'mat-mul': mat_mul,
            'mat-add': torch.add, 
            'mat-tanh': torch.tanh,
            'Q': {},
            'G': {}
            }

def get_primitives():
    return copy.deepcopy(context)

def is_primitive(operator):
    return operator in context
