import os, torch, utils
from numpy import log

from pytorch_forecasting.utils import padded_stack
from daphne import daphne
from evaluation_based_sampling import evaluate_program

dirn = os.path.dirname(os.path.abspath(__file__)) 

def row_mult(t,vector):
    extra_dims = (1,)*(t.dim()-1)
    return t * vector.view(-1, *extra_dims)

def get_grad_dim(g):
    if g[0].size():
        return g[0].size()[0]
    else:
        return len(g)

def optimizer_step(context,g,T):
    g.requires_grad_()
    for i,v in enumerate(context['Q']):
        dg = context['Q'][v]
        params = dg.Parameters()
        for j,p in enumerate(params):
            try:
                p.grad = g[i][j].clone().detach() 
            except:
                p.grad = g[i].clone().detach()
        optimizer = torch.optim.Adagrad(params, lr=0.05,weight_decay=0.9)
        optimizer.step()
        optimizer.zero_grad()
    return context['Q']

def elbo_grad(G,logW):
    L = len(G[0])
    F = torch.zeros_like(G)
    for i in range(G.size()[0]):
        for j in range(G.size()[1]):
            F[i][j] =  G[i][j] * logW[j]
    cov = torch.sum((F - torch.mean(F,dim=1,keepdim=True)) * (G - torch.mean(G,dim=1,keepdim=True)),dim=1)
    b = torch.sum(cov ,dim=1)/torch.sum(torch.var(G,dim=1), dim=1)
    bG = row_mult(G,b)
    return torch.sum(F - bG, axis=1)/L

def bbvi(ast,T,L):
    
    sig = {'G': {}, 'Q': {} , 'L_v': [], 'logW': torch.tensor(0.0)}

    G = {}
    R = []
    logW = torch.zeros((T,L))

    for t in range(T):
        r_t = []
        for l in range(L):
            sig['logW'] = torch.tensor(0.0)
            r, sig = evaluate_program(ast, sig)
            r_t.append(r) 
            logW[t][l] = sig['logW'].clone().detach()
            for v in sig['Q']:
                if v not in G:
                    g_d = get_grad_dim(sig['G'][v])
                    G[v] = torch.zeros((T,L,g_d))
                G[v][t][l] = torch.stack(sig['G'][v]).clone().detach()
        g = elbo_grad(padded_stack([G[v][t] for v in G]),logW[t]).clone().detach()
        sig['Q'] = optimizer_step(sig,g,T)
        try:  
            R.append(torch.stack(r_t))
        except:
            r_l = []
            for d in range(len(r_t[0])):
                r_l.append(torch.stack([r_d[d] for r_d in r_t]))
            R.append(r_l)
    try:
        return torch.stack(R), logW
    except:
        return R, logW


if __name__ == '__main__':

    for i in range(4,5):

        L,T = 50,1000

        filename = dirn + '/programs/{}.daphne'
        ast = daphne(['desugar', '-i', filename.format(i)])
        print('\n\n\nSample from posterior of program {}:'.format(i))

        R,W  = bbvi(ast,T,L)

        utils.plot_elbo('BBVI', i ,W,L)

        if i == 4: 
            R_W_all = []
            for d in range(len(R[0])):
                R_d = torch.stack([r[d] for r in R]).squeeze()
                if R_d.dim() == 3:
                    R_d = torch.movedim(R_d, 2, 0)
                    R_W = torch.sum(W * R_d, axis=2) / torch.sum(W,axis=1)
                    R_W_all.append(torch.movedim(R_W,0,1))
                    print(torch.mean(R_W, axis=1))
                elif R_d.dim() > 3:
                    R_d = torch.movedim(R_d, (2,-1), (0,1))
                    R_W = torch.sum(W * R_d, axis=3) / torch.sum(W,axis=1)
                    R_W_all.append(torch.movedim(R_W,2,0))
                    print(torch.mean(R_W, axis=2))
                    
                else:
                    R_d = R_d.unsqueeze(0) 
            utils.plot_hists_n('BBVI', i, R_W_all)
            utils.plot_heatmaps_n('BBVI', i,R_W_all, T,L)
                
        else:

            if R.dim() > 2:
                R = torch.movedim(R, 2, 0)
            else:
                R = R.unsqueeze(0)     
            R_W = torch.sum(W * R, axis=2) / torch.sum(W,axis=1)

            utils.gen_hists('BBVI', i, R_W.squeeze().t())
            print(torch.mean(R_W, axis=1))