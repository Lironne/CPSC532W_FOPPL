import torch,os,copy, utils
import torch.distributions as dist

from daphne import daphne
from graph_based_sampling import evaluate_graph, sample_from_graph, sample_from_joint

dirn = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    torch.cuda.set_device(0)

def H(G,X,R,x,r,M):
    U_X = torch.tensor([0.0]).cuda()
    U_x = torch.tensor([0.0]).cuda()
    for v in G['V']:
        U_X += G['Y'][v].log_prob(X[v]).cuda()
        U_x += G['Y'][v].log_prob(x[v]).cuda()
    H_XR = -U_X + 0.5 * R.t() @ M @ R
    H_xr = -U_x + 0.5 * r.t() @ M @ r

    return torch.exp(H_XR-H_xr).cuda()

def grad_u(graph,X,context): 
    # turn on grad
    [context[v].requires_grad_(True).cuda() for v in graph['V']]
    U = torch.tensor([0.0], requires_grad=True).cuda()

    # compute E_U
    for var in graph['V']:
        U += graph['Y'][var].log_prob(context[var]).cuda()

    g_U = torch.autograd.grad(U,[context[v] for v in X])
    # turn off
    [context[v].grad.zero_() if context[v].grad else None for v in graph['V']]
    return torch.tensor(g_U).cuda()

def leapfrog(graph,X,context,R,T,eps):
    V = copy.deepcopy(X)
    g_X = grad_u(graph,V,context)
    r_mid = R - 0.5 * eps * g_X
    x = torch.tensor([context[v] for v in V]).cuda()
    x = x.unsqueeze(0) if x.dim() == 0 else x 
    for _ in range(1,T-1):
        x += eps * r_mid
        sample_context = copy.deepcopy(context)
        [sample_context.update(v=x[i]) for i,v in enumerate(V)]
        g_X = grad_u(graph,V,sample_context)
        r_mid -= eps * g_X
    x += eps * r_mid
    sample_context = copy.deepcopy(context)
    [sample_context.update(v=x[i]) for i,v in enumerate(V)]
    r_mid -= eps * grad_u(graph,V,sample_context)
    return x,r_mid,sample_context


def hmc(graph,V,context,S,T,eps,M):
    samples = []
    G = graph[1]
    for _ in range(S):
        R = dist.multivariate_normal.MultivariateNormal(torch.zeros(len(V)).cuda(),M).sample().cuda()
        x,r,sample_context = leapfrog(G,V,context,R,T,eps)
        X = torch.tensor([context[v] for v in V]).cuda()
        X = X.unsqueeze(0) if X.dim() == 0 else X
        u = dist.uniform.Uniform(0,1).sample().cuda()
        if u < H(G,context,R,sample_context,r,M):
            # keep new
            graph_x = copy.deepcopy(graph)
            [graph_x[1]['Y'].update(v=x[i]) for i,v in enumerate(V)]
            graph, context = evaluate_graph(graph_x)
            G = graph[1]
            samples.append(sample_from_graph(graph_x,context).squeeze().detach().cpu())
        else:
            # keep old
            samples.append(sample_from_graph(graph,context).squeeze().detach().cpu())        
    return samples

def hmc_init(graph, S,T,eps):
    G = graph[1]
    V = list(set(G['V']) - set(G['Y'].keys()))
    M = torch.eye(len(V)).cuda()
    X, context = evaluate_graph(graph)
    return hmc(X,V,context, S, T, eps, M)


if __name__ == '__main__':

    for i in range(1,3):

        steps = 100
        eps = 0.001
        T = 100

        print(torch.cuda.is_available())

        filename = dirn + '/programs/{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(graph)

        samples = hmc_init(graph, steps, T, eps)
        utils.gen_hists("hmc", i,samples)