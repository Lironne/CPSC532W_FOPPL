
import os,torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("white")
dirn = os.path.dirname(os.path.abspath(__file__))

labels = { 1: "", 2: "", 3: "", 4: {0: "W_0", 1: "b_0", 2: "W_1", 3: "b_1" }}
labels_trace = {0: 'Samples taken', 1: 'Cumulative Mean', 2: 'Cumulative Variance', 3:'joint log likelihood', 4: 'Weights'}

def weighted_avg(x,w):
    return w @ x / torch.sum(w)

def get_title_hist(eval,i,dim=False,d=0):
    if dim:
        label = labels[i][d]
        return "{} Sampled Distribution for Program {} {}".format(eval, i, label)
    else:
        return "{} Sampled Distribution for Program {}".format(eval, i)

def get_fname_hist(eval,i,dim=False,d=0):
    if dim:
        return dirn + '/figures/{}_plt_program_{}_d_{}.jpg'.format(eval,i,d)
    else:
        return dirn + '/figures/{}_plt_program_{}.jpg'.format(eval,i)

def plot_num(eval,i,samples,jll,avg,var,t,weighted=False,w=[]):
    k = 4 
    data_t = {0:samples, 1: avg.T, 2: var.T, 3:w if weighted else np.array(jll).flatten()}
    fig, axs = plt.subplots(k, figsize=(20,20), dpi=100)
    for j, ax in zip(range(k), axs.flat):
        if j == 3 and weighted:
            sns.lineplot(data=w, ax=ax)
            ax.title.set_text('Weights over {} iterations'.format(t))
        else:    
            sns.lineplot(data=data_t[j], ax=ax)
            ax.title.set_text('{} over {} iterations'.format(labels_trace[j],t))   
    fname = dirn + '/figures/{}_trace_program_{}.jpg'.format(eval,i)
    title = "{} Sample trace for Program {}".format(eval, i)
    fig.suptitle(title, fontsize=24)
    plt.savefig(fname)
    plt.clf()

def plot_elbo(eval,i,elbo,L):
    plt.figure(figsize=(10,7))
    elbo = torch.mean(elbo,axis=1)
    plt.plot(elbo)
    fname = dirn + '/figures/{}_elbo_program_{}.jpg'.format(eval,i)
    title = "{} ELBO for Program {}   L = {}".format(eval, i,L)
    plt.xlabel('T')
    plt.title(title)
    plt.savefig(fname)
    plt.clf()


def plot_bool(eval,i,samples,jll,avg,var,t,weighted=False,w=[]):
    k = 4
    data_t = {0:samples, 1: avg.T, 2: var.T, 3:w if weighted else np.array(jll).flatten()}
    fig, axs = plt.subplots(k, figsize=(20,20), dpi=100)
    sns.scatterplot(data=samples, ax=axs[0])
    axs[0].title.set_text('{} over {} iterations'.format(labels_trace[0],t))
    for j in range(1,k):
        if j == 3 and weighted:
            sns.lineplot(data=w, ax=axs[j])
            axs[j].title.set_text('Weights over {} iterations'.format(t))
        else:    
            sns.lineplot(data=data_t[j], ax=axs[j])
    fname = dirn + '/figures/{}_trace_program_{}.jpg'.format(eval,i)
    title = "{} Sample trace for Program {}".format(eval, i)
    fig.suptitle(title, fontsize=24)
    plt.savefig(fname)
    plt.clf()
    return 

def get_avg_var(samples, weighted=False, w=[]):
    n = samples.shape[0]
    if weighted:
        avg = np.cumsum((samples.T * w).T, axis=0).T / np.cumsum(w,axis=0)
        var = np.cumsum((samples - avg.T)**2,axis=0).T/ np.cumsum(np.abs(w),axis=0)
    else:
        avg = np.cumsum(samples, axis=0).T / np.arange(1,n+1)
        var = np.cumsum((samples - avg.T)**2, axis=0).T / np.arange(1,n+1)
    return avg, var

def plot_traces(eval,i,samples,jll,avg,var,t,weighted=False,w=[]):
    if i == 3 or i == 4:
        plot_bool(eval,i,samples,jll,avg,var,t,weighted=weighted,w=w)
    else:
        plot_num(eval,i,samples,jll,avg,var,t,weighted=weighted,w=w)

def gen_traces(eval,i,t,samples,jll,weighted=False,w=[]):
    avg,var = get_avg_var(samples,weighted,w)
    plot_traces(eval,i,samples,jll,avg,var,t,weighted,w)

def plot_heatmap(eval,i,samples, T, L, dim=False, d=0):
    plt.figure(figsize=(10,7))
    sns.heatmap(samples)
    fname = dirn + '/figures/{}_heatmap_plt_{}_dim_{}.jpg'.format(eval,i,d)
    title = '{} heatmap for {}   L = {}'.format(eval,labels[i][d],L)
    plt.title(title)
    plt.xlabel('T')
    plt.savefig(fname)
    plt.clf()


def plot_heatmaps_n(eval, i, samples, T,L):
    for d in range(len(samples)):
        t = samples[d]
        if t.dim() < 3:
            plot_heatmap(eval,i,t.T,T,L,True, d)
        else:
            k = t.size()[2]
            fig, axs = plt.subplots(2,5, figsize=(16,9), dpi=100)
            for j, ax in zip(range(k), axs.flat):
                sns.heatmap(data=t[:,j].t(),cbar=True, ax=ax)
                ax.title.set_text('range over {}[{}]'.format(labels[i][d],j))
                ax.set_xlabel('T')
            fname = dirn + '/figures/{}_heatmap_plt_{}_dim_{}.jpg'.format(eval,i,d)
            title = '{} heatmap for {}   L = {}'.format(eval,labels[i][d],L)
            fig.suptitle(title, fontsize=24)
            plt.savefig(fname)
            plt.clf()


def plot_hists_n(eval, i, hists):
    for d in range(len(hists)):
        samples = np.array(hists[d])
        if samples.ndim < 3:
            # 2D >= random var
            plot_hist_arr(eval,i, samples,dim=True,d=d)
        else:
            # ND random var 
            _,k,_ = samples.shape 
            fig, axs = plt.subplots(2,5, figsize=(18,9), dpi=100)
            for j, ax in zip(range(k), axs.flat):
                sns.histplot(data=samples[:,j],kde=True,stat='density',cbar=True,multiple='dodge', ax=ax)
                ax.title.set_text('range over {}[{}]'.format(labels[i][d],j))
            fname = get_fname_hist(eval, i,True,d)
            title = get_title_hist(eval, i, True, d)
            fig.suptitle(title, fontsize=24)
            plt.savefig(fname)
            plt.clf()

def plot_hist_arr(eval, i, samples, dim=False, d=0):
    plt.figure(figsize=(10,7))
    nbins = int(max(samples.max()-samples.min(),8))
    sns.histplot(data=samples,kde=True,stat='density',cbar=True,multiple='dodge',bins=nbins)
    fname = get_fname_hist(eval, i,dim,d)
    title = get_title_hist(eval, i, dim, d)
    plt.title(title)
    plt.savefig(fname)
    plt.clf()

def plot_hist(eval, i, samples, bins):
    plt.figure(figsize=(10,7))
    sns.histplot(data=samples,bins=bins,kde=True, stat='density')
    fname = get_fname_hist(eval, i)
    title = get_title_hist(eval, i)
    plt.title(title)
    plt.savefig(fname)
    plt.clf()

def gen_hists(eval, i, samples):
    try:
        # list of c
        min_v, max_v = min(samples), max(samples)
        nbins = int(max(max_v-min_v,10))
        bins = np.linspace(min_v, max_v,nbins)
        plot_hist(eval, i, samples, bins)
        plt.close('all')
    except:
        try:
            # 1 rnd var
            if torch.is_tensor(samples):
                samples = samples.numpy()
            else:
                samples = torch.stack(samples).numpy()
            plot_hist_arr(eval, i, samples)
            plt.close('all')
        except:
            # n rnd vars
            hists = []
            for d in range(len(samples[0])):
                hists.append([torch.squeeze(sample[d].t()).numpy() for sample in samples])
            plot_hists_n(eval, i, hists)
            plt.close('all')

def draw_hists(eval, i,stream, n_samples):

    samples = []
    for _ in range(int(n_samples)):
        samples.append(next(stream))

    gen_hists(eval, i, samples)