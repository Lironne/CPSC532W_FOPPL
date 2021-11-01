import torch,os,utils,time,copy
import torch.distributions as dist

from daphne import daphne
from hmc import hmc_init
from gibbs_sampling import gibbs
from importance_sampling import likelihood_weighting

dirn = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    for i in range(1,5):
        
        filename = dirn + '/programs/{}.daphne'
        graph = daphne(['graph','-i',filename.format(i)])
        ast = daphne(['desugar', '-i', filename.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        
        iter = 10000
        eps = 0.001
        T = 20

        print('running for {} iterations'.format(iter))

        if i == 1 or i == 2:

            start_time = time.time()
            samples, jll = hmc_init(copy.deepcopy(graph), iter, T, eps)
            samples_t = torch.stack(samples) if torch.is_tensor(samples[0]) else torch.tensor(samples)
            var, mean = torch.var_mean(samples_t, dim=0)
            runtime = time.time() - start_time 
            print(' ---- HMC - time -- %.4fs seconds --'%runtime,' avg - ',mean,'- var -  ', var)
            utils.gen_hists("HMC", i, samples)
            utils.gen_traces('HMC', i,iter,samples_t.numpy(), [sample.detach().numpy() for sample in jll])

        start_time = time.time()
        samples, weights = likelihood_weighting(ast, iter)
        samples_t = torch.stack(samples) if torch.is_tensor(samples[0]) else torch.tensor(samples)
        weights_t = weights.squeeze()
        # compute var, mean 
        avg = utils.weighted_avg(samples_t,weights_t).squeeze()
        var = utils.weighted_avg((samples_t - avg)**2, weights_t).squeeze()
        runtime = time.time() - start_time 
        print(' ----- IS - time -- %.4fs seconds --'%runtime,' avg - ',avg,'- var -  ', var)
        utils.gen_hists('IS', i, samples)
        utils.gen_traces('IS', i,iter, samples_t.numpy(),None, weighted=True, w=weights_t.numpy())      

        start_time = time.time()
        samples, jll = gibbs(copy.deepcopy(graph), iter)
        samples_t = torch.stack(samples) if torch.is_tensor(samples[0]) else torch.tensor(samples)
        var, mean = torch.var_mean(samples_t, dim=0)
        runtime = time.time() - start_time
        print(' ---- gibbs - time -- %.4fs seconds --'%runtime,' avg - ',mean,'- var -  ', var)
        utils.gen_hists('gibbs', i, samples)
        utils.gen_traces('gibbs', i,iter,samples_t.squeeze().numpy(), [sample.numpy() for sample in jll])
         