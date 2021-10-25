
import os,torch,itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("white")
dirn = os.path.dirname(os.path.abspath(__file__))

labels = { 1: "", 2: "", 3: "", 4: {0: "W_0", 1: "b_0", 2: "W_1", 3: "b_1" }}

def get_title(eval,i,dim=False,d=0):

    if dim:
        label = labels[i][d]
        return "{} Sampled Distribution for Program {} {}".format(eval, i, label)
    else:
        return "{} Sampled Distribution for Program {}".format(eval, i)

def get_fname(eval,i,dim=False,d=0):
    if dim:
        return dirn + '/figures/{}_plt_program_{}_d_{}.jpg'.format(eval,i,d)
    else:
        return dirn + '/figures/{}_plt_program_{}.jpg'.format(eval,i)

def plot_hists_n(eval, i, hists):
    for d in range(len(hists)):
        samples = np.array(hists[d])
        if samples.ndim < 3:
            # 2D >= random var
            plot_hist_arr(eval,i, samples,dim=True,d=d)
        else:
            # ND random var 
            _,k,_ = samples.shape 
            fig, axs = plt.subplots(2,5, figsize=(20,10), dpi=100)
            for j, ax in zip(range(k), axs.flat):
                sns.histplot(data=samples[:][j],kde=True,stat='density',cbar=True,multiple='dodge', ax=ax)
                ax.title.set_text('range over {}[{}]'.format(labels[i][d],j))
            fname = get_fname(eval, i,True,d)
            title = get_title(eval, i, True, d)
            fig.suptitle(title, fontsize=24)
            plt.savefig(fname)
            plt.clf()

def plot_hist_arr(eval, i, samples, dim=False, d=0):
    plt.figure(figsize=(10,7))
    nbins = int(max(samples.max()-samples.min(),8))
    sns.histplot(data=samples,kde=True,stat='density',cbar=True,multiple='dodge',bins=nbins)
    fname = get_fname(eval, i,dim,d)
    title = get_title(eval, i, dim, d)
    plt.title(title)
    plt.savefig(fname)
    plt.clf()

def plot_hist(eval, i, samples, bins):
    plt.figure(figsize=(10,7))
    sns.histplot(data=samples,bins=bins,kde=True, stat='density')
    fname = get_fname(eval, i)
    title = get_title(eval, i)
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
    except:
        try:
            # 1 rnd var
            samples = torch.stack(samples).numpy()
            plot_hist_arr(eval, i, samples)
        except:
            # n rnd vars
            hists = []
            for d in range(len(samples[0])):
                hists.append([torch.squeeze(sample[d].t()).numpy() for sample in samples])
            plot_hists_n(eval, i, hists)

def draw_hists(eval, i,stream, n_samples):

    samples = []
    for _ in range(int(n_samples)):
        samples.append(next(stream))

    gen_hists(eval, i, samples)