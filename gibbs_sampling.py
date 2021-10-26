import torch, os, copy
import torch.distributions as dist

from daphne import daphne
from evaluation_based_sampling import evaluate_program, evaluate_program_help, evaluate_program_observe
from graph_based_sampling import deterministic_eval, evaluate_graph, sample_from_graph


dirn = os.path.dirname(os.path.abspath(__file__))

def accept(graph,linkfn,v, context):

    dist, _ = evaluate_program_help(linkfn[1], context)
    sample = dist.sample()    
    alpha = dist.log_prob(context[v]) - dist.log_prob(sample)
    sample_context = copy.deepcopy(context)
    sample_context[v] = sample

    if sample.dim() == 0:
        sample = sample.unsqueeze(0)
    if alpha.dim() == 0:
        alpha = alpha.unsqueeze(0)

    _ ,  score_p = evaluate_program_observe(linkfn[1],sample, context, torch.tensor([0.0]))
    _ , score_c = evaluate_program_observe(linkfn[1], sample,sample_context, torch.tensor([0.0]))
    alpha += score_c
    alpha -= score_p

    if v in graph['A']:
        for var in graph['A'][v]:
            _, score_p = evaluate_program_observe(graph['P'][var][1],context[var],context, torch.tensor([0.0]))
            _, score_c = evaluate_program_observe(graph['P'][var][1], context[var],sample_context, torch.tensor([0.0]))
            alpha += score_c
            alpha -= score_p

    return sample, torch.exp(alpha) 

def gibbs_step(G, context):
    samples = []
    graph = G[1]

    for v in graph['V']:
        sample = sample_from_graph(G,context)
        eval_v, alpha = accept(graph,graph['P'][v],v,context)

        u = dist.uniform.Uniform(0,1).sample()

        if u < alpha:
            context[v] = eval_v
            samples.append(sample)       

    return (samples, copy.deepcopy(context))

def gibbs(X,S):
    samples = []
    G, context = evaluate_graph(X)
    for _ in range(S):
        
        s, context = gibbs_step(G,context)
        samples.extend(s)
    return samples

if __name__ == '__main__':

    for i in range(1,5):

        steps = 100

        filename = dirn + '/programs/{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        samples = gibbs(graph, steps)
        print('samples?', samples)
