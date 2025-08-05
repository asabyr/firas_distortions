import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import constants as const

from help_funcs import labels, legend_labels, nuts_params
from getdist import plots, MCSamples
import getdist
import emcee
import h5py
import glob
import arviz

# plot_params= {
#     'axes.grid': True,
#     'grid.alpha': 0.25,
#     'xtick.minor.visible': True,
#     'ytick.minor.visible': True,
#     'legend.frameon': False,
# }
# plt.rcParams.update(plot_params)
#change font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'serif',
    "mathtext.fontset":'stix',
    "font.serif": "Computer Modern",
})


class AnalyzeContours:
    """
    class to plot contours and y mean value summary plots 
    (comments explain the functions)
    
    Extra notes: dT is read as dTx10^4, y in x10^6, As,Ad,EM in x10^5
    """
    def __init__(self, burn_in, param_dict, priors, colors=["blue"], save_fig=True, auto_corr=True,
                 discard_walker=[], eff_sample=1000, r_hat_max=1.01, thin=1000):

        self.burn_in=burn_in #can be fraction or number of samples
        self.param_dict=param_dict #model components dictionary with model + free param
        # (e.g. {DeltaI_DeltaT: 1, DeltaI_y: 1, thermal_dust_rad: 3})
        self.priors=priors
        self.colors=colors #list of string(s) for plots
        self.save_fig=save_fig #bool
        self.auto_corr=auto_corr #use auto_corr to determine convergence
        self.prepare_contours() #prepare labels
        self.discard_walker=discard_walker #discard any chains, not used
        self.eff_sample=eff_sample #minimum eff sample for convergence
        self.r_hat_max=r_hat_max #maximum allowed r_hat for convergence
        self.thin=thin
    def prepare_contours(self):
        #creates names/labels
        self.param_labels=legend_labels(self.param_dict)
        names=labels(self.param_dict)
        self.names=np.copy(names)
        if len(self.priors)>0:
            for i in range(len(self.priors)):
                prior_keys=np.array(list(self.priors[i].keys()))
                assert np.array_equal(prior_keys, self.names)==True, "check prior names"

    def read_samples(self, fname):
        #reads either txt, h5 or .pkl file with mcmc samples, discards burn-in
        #returns cleaned samples and file name (with no path)
        if fname[-3:]=='txt': #this option is not used
            clean_name=fname.rsplit('/',1)[1]
            clean_name=clean_name.replace(".txt",'')
            samples=np.loadtxt(fname)
            if self.burn_in < 1:
                burn_in=self.burn_in*len(samples[:,0])
            clean_samples=samples[int(burn_in):, :]

        elif fname[-2:]=='h5':
            #emcee samples are thinned to increase speed of reading/plotting samples
            clean_name=fname.rsplit('/',1)[1]
            clean_name=clean_name.replace(".h5",'')

            reader = emcee.backends.HDFBackend(fname, read_only=True)
            samples = reader.get_chain()
            # print(np.shape(samples))
            if self.burn_in < 1:
                burn_in=int(self.burn_in*len(samples[:,0,0]))

            if self.auto_corr==True:
                #check convergence: 5 x max_auto_corr is less than 30% of samples
                #this is an extra check 
                auto_corr=reader.get_autocorr_time(discard=burn_in)
                max_autocorr=np.amax(auto_corr)
                if max_autocorr*5.0<burn_in:
                    # thin=1000
                    #thin=1
                    #clean_samples=reader.get_chain(flat=True, discard=burn_in, thin=thin)
                    clean_samples_all=reader.get_chain(discard=burn_in, thin=self.thin)
                    all_steps, all_walkers, all_params=np.shape(clean_samples_all)
                    left_walkers=int(all_walkers-len(self.discard_walker))
                    clean_samples_new=np.delete(clean_samples_all,self.discard_walker, axis=1)
                    clean_samples=np.reshape(clean_samples_new, (all_steps*left_walkers, all_params))
                    #thin=1000
                    #clean_samples=reader.get_chain(flat=True, discard=burn_in)
                else:
                    sys.exit("Converged but 5 x autocorr is less than 30% of samples. Check the posteriors.")
            else:
                # thin=1000
                clean_samples=reader.get_chain(flat=True, discard=burn_in, thin=self.thin)
        
        elif fname[-3:]=='pkl':
            
            clean_name=fname.rsplit('/',1)[1]
            clean_name=clean_name.replace(".pkl",'')

            self.nuts_parameters=nuts_params(self.param_dict)
            samples=np.load(fname, allow_pickle=True)
            a_summary=arviz.summary(samples)
            # print(a_summary)
            #check that NUTs converged using effective sample size & r_hat
            if np.any(a_summary['ess_bulk'].values<self.eff_sample)==False and np.any(a_summary['r_hat'].values>self.r_hat_max)==False:

                n_params=len(self.nuts_parameters)
                chains, steps=np.shape(samples[self.nuts_parameters[0]])

                burn_in=int(self.burn_in*steps)
                n_samples=int(chains*(steps-burn_in))
                clean_steps=int(steps-burn_in)
                # n_samples=int(chains*steps)
                # np_nuts_samples=np.zeros((n_samples, n_params))
                np_nuts_samples=np.zeros((chains,clean_steps, n_params))
                

                for i in range(len(self.nuts_parameters)):
        
                    if self.nuts_parameters[i]=='DeltaT_amp' or self.nuts_parameters[i]=='y_amp':
                        # oneparam=samples[self.nuts_parameters[i]]
                        np_nuts_samples[:,:,i]=samples[self.nuts_parameters[i]][:,burn_in:]/const.dT_factor
                    else:
                        np_nuts_samples[:,:,i]=samples[self.nuts_parameters[i]][:,burn_in:]
                
                # burn_in=int(self.burn_in*len(np_nuts_samples[:,0]))
                # clean_samples=np_nuts_samples[burn_in:, :]
                clean_samples=np.reshape(np_nuts_samples, (n_samples, n_params))
            else:
                sys.exit("NUTs possibly didn't converge...")

        #scale sample values for nicer plots
        clean_samples[:,0]=clean_samples[:,0]*1e4 #dT parameter
        y_ind=np.where(self.names=='y')
        Ad_ind=np.where(self.names=='A_d')
        # print(Ad_ind)
        As_ind=np.where(self.names=='A_s')
        EM_ind=np.where(self.names=='EM')
        clean_samples[:,y_ind]=clean_samples[:,y_ind]*1e6
        
        if len(Ad_ind)!=0:
            clean_samples[:,Ad_ind]=clean_samples[:,Ad_ind]/1.e5
        
        if len(As_ind)!=0:
            clean_samples[:,As_ind]=clean_samples[:,As_ind]/1.e5
        
        if len(EM_ind)!=0:
            clean_samples[:,EM_ind]=clean_samples[:,EM_ind]/1.e5

        return clean_samples, clean_name

    def plot_contour_getdist_onesample(self, fname, read_param='y', return_values=True):
        
        #plots one file of mcmc samples
        clean_samples, clean_name=self.read_samples(fname)
        if len(self.priors)>0:
            samples1 = MCSamples(samples=clean_samples, names = self.names, labels = self.param_labels, ranges=self.priors[0])
        else:
            print("not setting priors")
            samples1 = MCSamples(samples=clean_samples, names = self.names, labels = self.param_labels)
        if return_values==True and self.auto_corr==True:
            stats = samples1.getMargeStats()
            if len(read_param)==1:
                lims1 = stats.parWithName(read_param).limits
                param_mean=stats.parWithName(read_param).mean
            else:
                ind_y=np.where(read_param=='y')[0]
                lims1 = stats.parWithName(read_param[ind_y][0]).limits
                param_mean=stats.parWithName(read_param[ind_y][0]).mean
            
            # print("compare to pure mean")
            # print(param_mean)
            # print(np.mean(clean_samples, axis=0)[1]) #get min, lower limit and upper limit for 68% confidence
            # print(param_mean/np.mean(clean_samples, axis=0)[1])
            #get errors (negative and positive)
            st_dev_min=lims1[0].lower-param_mean#hard code that y is the second parameter
            st_dev_plus=lims1[0].upper-param_mean
            
        #plot samples
        plt.ion()
        g = plots.get_subplot_plotter()
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.4
        g.settings.title_limit_fontsize = 14
        
        for i in range(len(read_param)):
            print(read_param[i])
            print("68%: ")
            print(samples1.getInlineLatex(read_param[i],limit=1)) #print 68% confidence
            print("95%: ")
            print(samples1.getInlineLatex(read_param[i],limit=2)) #print 95% confidence

        
        fig=g.triangle_plot([samples1],
        filled=True,
        legend_loc='upper right',
        contour_colors=self.colors,
        title_limit=1)

        if self.save_fig==True:
            if fname[-3:]=='txt':
                clean_name=fname.replace('.txt','')
            elif fname[-2:]=='h5':
                clean_name=fname.replace('.h5','')
            elif fname[-3:]=='pkl':
                clean_name=fname.replace('.pkl','')
            plt.tight_layout()
            plt.savefig(f"{clean_name}_post_run_contours.pdf")

        plt.close(fig)

        if return_values==True:
            return param_mean, st_dev_min, st_dev_plus
    
    def plot_contour_getdist_multiple(self, fname, legend_labels, read_param='y',
                                      clean_name="", legend_fontsize=20,
                                      axes_labelsize=20, axes_fontsize=15, legend_loc='upper right', 
                                      width=0, *args):
        
    #1 file with samples
        if len(args)==0:

            clean_samples_1, clean_name_1=self.read_samples(fname)
            
            if len(clean_name)==0:
                clean_name=clean_name_1

            if len(self.priors)>0:
                samples1 = MCSamples(samples=clean_samples_1, names = self.names, labels = self.param_labels, label=legend_labels[0], ranges=self.priors[0])
            else:
                samples1 = MCSamples(samples=clean_samples_1, names = self.names, labels = self.param_labels, label=legend_labels[0])

            for i in range(len(read_param)):
                print("reading params")
                print(read_param[i])
                print("68%: ")
                print(samples1.getInlineLatex(read_param[i],limit=1, err_sig_figs=5)) #print 68% confidence
                print("95%: ")
                print(samples1.getInlineLatex(read_param[i],limit=2, err_sig_figs=5)) #print 95% confidence
            
    #2 files
        elif len(args)==1:

            clean_samples_1, clean_name_1=self.read_samples(fname)
            clean_samples_2, clean_name_2=self.read_samples(args[0])
            
            if len(clean_name)==0:
                clean_name=clean_name_1+"_"+clean_name_2

            if len(self.priors)>0:
                samples1 = MCSamples(samples=clean_samples_1, names = self.names, labels = self.param_labels, label=legend_labels[0], ranges=self.priors[0])
                samples2 = MCSamples(samples=clean_samples_2, names = self.names, labels = self.param_labels, label=legend_labels[1], ranges=self.priors[1])
            else:
                samples1 = MCSamples(samples=clean_samples_1, names = self.names, labels = self.param_labels, label=legend_labels[0])
                samples2 = MCSamples(samples=clean_samples_2, names = self.names, labels = self.param_labels, label=legend_labels[1])

            for i in range(len(read_param)):
                print("printing params")
                print(read_param[i])
                print("68%: ")
                print(samples1.getInlineLatex(read_param[i],limit=1, err_sig_figs=5))
                print(samples2.getInlineLatex(read_param[i],limit=1, err_sig_figs=5))
                print("95%: ")
                print(samples1.getInlineLatex(read_param[i],limit=2, err_sig_figs=5))
                print(samples2.getInlineLatex(read_param[i],limit=2, err_sig_figs=5))

    #3 files
        elif len(args)==2:

            clean_samples_1, clean_name_1=self.read_samples(fname)
            clean_samples_2, clean_name_2=self.read_samples(args[0])
            clean_samples_3, clean_name_3=self.read_samples(args[1])
            
            if len(clean_name)==0:
                clean_name=clean_name_1+"_"+clean_name_2+"_"+clean_name_3

            if len(self.priors)>0:
                samples1 = MCSamples(samples=clean_samples_1, names = self.names, labels = self.param_labels, label=legend_labels[0], ranges=self.priors[0])
                samples2 = MCSamples(samples=clean_samples_2, names = self.names, labels = self.param_labels, label=legend_labels[1], ranges=self.priors[1])
                samples3 = MCSamples(samples=clean_samples_3, names = self.names, labels = self.param_labels, label=legend_labels[2], ranges=self.priors[2])
            else:
                samples1 = MCSamples(samples=clean_samples_1, names = self.names, labels = self.param_labels, label=legend_labels[0])
                samples2 = MCSamples(samples=clean_samples_2, names = self.names, labels = self.param_labels, label=legend_labels[1])
                samples3 = MCSamples(samples=clean_samples_3, names = self.names, labels = self.param_labels, label=legend_labels[2])
            
                
            for i in range(len(read_param)):
                print(read_param[i])
                print("68%: ")
                print(samples1.getInlineLatex(read_param[i],limit=1, err_sig_figs=5))
                print(samples2.getInlineLatex(read_param[i],limit=1, err_sig_figs=5))
                print(samples3.getInlineLatex(read_param[i],limit=1, err_sig_figs=5))
                print("95%:")
                print(samples1.getInlineLatex(read_param[i],limit=2, err_sig_figs=5))
                print(samples2.getInlineLatex(read_param[i],limit=2, err_sig_figs=5))
                print(samples3.getInlineLatex(read_param[i],limit=2, err_sig_figs=5))
            
    #plot samples
        plt.ion()

        if width>0:
            g = plots.get_single_plotter(width_inch=width)
        else:
            g = plots.get_subplot_plotter()

        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.6
        g.settings.title_limit_fontsize = 14
        
        g.settings.axes_fontsize=axes_fontsize
        g.settings.axes_labelsize=axes_labelsize
        g.settings.legend_fontsize=legend_fontsize

        if len(args)==0:
            fig=g.triangle_plot([samples1],
            filled=True,
            legend_loc=legend_loc,legend_fontsize=legend_fontsize,
            contour_colors=self.colors)

        elif len(args)==1:
            fig=g.triangle_plot([samples1, samples2],
            filled=True,
            legend_loc=legend_loc,legend_fontsize=legend_fontsize,
            contour_colors=self.colors)

        elif len(args)==2:
            fig=g.triangle_plot([samples1, samples2, samples3],
            filled=True,
            legend_loc=legend_loc,legend_fontsize=legend_fontsize,
            contour_colors=self.colors)

        plt.tight_layout()
        if self.save_fig==True:
                plt.savefig(f"../figs/{clean_name}_post_run_contours.pdf")
        plt.close(fig)

    def plot_contours_dir(self, file_dir, file_ext_1='.h5', 
                          file_ext_2='.txt', rerun_all=False, return_values=False):
        #plot contours for all mcmc samples files in a specified directory,
        #save mean values, lower and upper bound (68% confidence) into sigmas.txt
        #save file names into file_names.txt

        file_names=[]
        ys=[]
        sigmas_low=[]
        sigmas_high=[]

        #gather all existing files
        allfiles_pdf=glob.glob(file_dir+"*pdf")
        allfiles_1=glob.glob(file_dir+"*"+file_ext_1)
        allfiles_2=glob.glob(file_dir+"*"+file_ext_2)
        allfiles=np.concatenate((allfiles_1, allfiles_2))

        sigmas_file=file_dir+"sigmas.txt"
        fname_file=file_dir+"file_names.txt"
        
        #remove sigmas.txt and file_names.txt from array of existing files (i.e. don't count them)
        if sigmas_file in allfiles:
            sigma_arr=np.array([sigmas_file])
            allfiles = np.setdiff1d(allfiles,sigma_arr)
        if fname_file in allfiles:
            fname_arr=np.array([fname_file])
            allfiles = np.setdiff1d(allfiles,fname_arr)

        if rerun_all==True:
            allfiles_pdf=np.array([])
        
        #if sigmas.txt and file_names.txt don't exist, make them
        if os.path.exists(sigmas_file)==False or rerun_all==True:
            np.savetxt(sigmas_file, ['#file name, lower bound, mean_y, upper bound'], fmt='%s')
        if os.path.exists(fname_file)==False or rerun_all==True:
            np.savetxt(fname_file, ['#file name'], fmt='%s')

        with open(file_dir+"file_names.txt", 'a') as fname_file:
            with open(file_dir+"sigmas.txt", 'a') as sigmas_file:

                for i in range(len(allfiles)):

                    #check if contours have already been plotted
                    one_file=allfiles[i]

                    if one_file[-2:]=='h5':
                        fig_name=one_file.replace('.h5','_post_run_contours.pdf')
                    if one_file[-3:]=='pkl':
                        fig_name=one_file.replace('.pkl','_post_run_contours.pdf')
                    elif one_file[-3:]=='txt':
                        fig_name=one_file.replace('.txt','_post_run_contours.pdf')

                    if fig_name not in allfiles_pdf:
                        print(one_file)
                        #calculate mean y, lower and upper values and save contour figure
                        ymean, sigma_min, sigma_plus=self.plot_contour_getdist_onesample(one_file)

                        sigmas_low.append(sigma_min)
                        sigmas_high.append(sigma_plus)
                        ys.append(ymean)

                        file_names.append(fig_name.replace(file_dir, '').replace('_post_run_contours.pdf',''))


                data=np.column_stack((np.array(file_names),np.array(sigmas_low),np.array(ys), np.array(sigmas_high)))
                np.savetxt(sigmas_file, data, fmt=("%s"))
            np.savetxt(fname_file, file_names, fmt=("%s"))

        if return_values==True:
            return file_names, sigmas_low, sigmas_high

    def prepare_plot_y(self, y_txt_file, file_order=[], labels=[]):
        
        #plot y mean values (+1 sigma)
        #y_txt_file--file where y values are stored (i.e. from running plot_contours_dir())
        #file_order--array of file names exactly in file_names.txt but in order of your preference
        #labels--legend labels (make sure they match file_order, this function doesn't check that!!)
        #color--plot marker color
        #outputs a dictionary of all y\pm 1sigma values

        legend_labels_orig, min_sigma_orig, y_mean_orig, plus_sigma_orig=np.genfromtxt(y_txt_file,  dtype='U100,f8,f8,f8', unpack=True)

        #if file order is specified, we use it to sort the file data,
        # otherwise we use default numpy sorting
        if len(file_order)>1:
            ind_sort=np.where(legend_labels_orig==file_order[:,None])[1]

        else:
            ind_sort=np.argsort(legend_labels_orig)

        # legend_labels=legend_labels_orig[ind_sort]
        # min_sigma=min_sigma_orig[ind_sort]
        # y_mean=y_mean_orig[ind_sort]
        # plus_sigma=plus_sigma_orig[ind_sort]

        y_dict={}
        y_dict['legend_labels']=legend_labels_orig[ind_sort]
        y_dict['min_sigma']=min_sigma_orig[ind_sort]
        y_dict['y_mean']=y_mean_orig[ind_sort]
        y_dict['plus_sigma']=plus_sigma_orig[ind_sort]

        #if labels are specified, they are the legend labels
        if len(labels)>1:
            y_dict['legend_labels']=np.copy(labels)

        return y_dict

    #this function was only used in appendix 
    def plot_y(self, main_title, titles, y_txt_files,
            file_order=[], plot_labels=[], colors='orange', 
            num_plots=1, figsize=(20,8), mean_y=0.0, xlim=[-30,30]):
        
        #make horizontal y plots using values in txt files saved from above
        #this function makes panel plots and single plots
        
        if len(num_plots)==1:
            num_plots=1
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(num_plots[0], num_plots[1], figsize=figsize)
        #not used
        if num_plots==1:
            i=0
            k=0
            y_dict=self.prepare_plot_y(y_txt_files[i], file_order=file_order[i], labels=plot_labels)
            legend_labels=y_dict['legend_labels']
            ax.set_title(titles[i], fontsize=20)
            if i==0:
                locs, labels = plt.yticks()  # Get the current locations and labels.
                ax.set_yticks(np.arange(len(legend_labels), 0, -1), legend_labels)

            for j in range(len(legend_labels)):
                x_err=np.zeros((2,1))
                # y_dict['min_sigma']
                x_err[0,0]=np.abs(y_dict['min_sigma'][j])
                x_err[1,0]=y_dict['plus_sigma'][j]

                ax.errorbar(y_dict['y_mean'][j], len(legend_labels)-j, xerr=x_err, fmt='o', color=colors[i])

            ax.axvline(mean_y, -1, len(legend_labels)+1, color='black', ls='--') #plot mean y zero
            ax.set_ylim([0.5,len(legend_labels)+1])
            ax.set_xlabel(r'$\langle y \rangle \times 10^{-6}$', fontsize=20)
            ax.set_xlim(xlim)

        elif num_plots[0]==1:

            for i in range(num_plots[1]):
                
                y_dict=self.prepare_plot_y(y_txt_files[i], file_order=file_order[i], labels=plot_labels)
                legend_labels=y_dict['legend_labels']
                ax[i].set_title(titles[i], fontsize=20)
                #ax[i].grid(which='both')

                if i==0:
                    locs, labels = plt.yticks()  # Get the current locations and labels.
                    ax[i].set_yticks(np.arange(len(legend_labels), 0, -1), legend_labels)
                    # ax[i].tick_params(axis='both', which='both')
                    # ax[i].minorticks_on()
                else:
                    locs, labels = plt.yticks()  # Get the current locations and labels.
                    ax[i].set_yticks(np.arange(len(legend_labels), 0, -1), [])
                    # ax[i].set(yticklabels=[])
                    # ax[i].xaxis.grid(False, which='minor')
                    # ax[i].tick_params(axis='both', which='both')

                for j in range(len(legend_labels)):
                    x_err=np.zeros((2,1))
                    #y_dict['min_sigma']
                    # print(j)
                    x_err[0,0]=np.abs(y_dict['min_sigma'][j])
                    x_err[1,0]=y_dict['plus_sigma'][j]

                    ax[i].errorbar(y_dict['y_mean'][j], len(legend_labels)-j, xerr=x_err, fmt='o', color=colors[i])

                ax[i].axvline(mean_y[i], -1, len(legend_labels)+1, color='black', ls='--') #plot mean y zero
                ax[i].set_ylim([0.5,len(legend_labels)+0.5])
                ax[i].set_xlabel(r'$\langle y \rangle \times 10^{-6}$', fontsize=20)
                ax[i].set_xlim(xlim[i])
                
        #not used
        else:

            counter=0
            for i in range(num_plots[0]):
                for k in range(num_plots[1]):


                    y_dict=self.prepare_plot_y(y_txt_files[counter], file_order=file_order[counter], labels=plot_labels)
                    legend_labels=y_dict['legend_labels']

                    ax[i,k].set_title(titles[counter], fontsize=20)
                    #ax[i,k].grid(which='both')

                    if k==0:
                        ax[i,k].set_yticks(np.arange(len(legend_labels), 0, -1), legend_labels)
                        # ax[i,k].tick_params(axis='both', which='both')
                        # ax[i,k].minorticks_on()

                    else:
                        locs, labels = plt.yticks()  # Get the current locations and labels.
                        ax[i,k].set_yticks(np.arange(len(legend_labels), 0, -1), [])
                        

                    for j in range(len(legend_labels)):

                        x_err=np.zeros((2,1))
                        y_dict['min_sigma']
                        x_err[0,0]=np.abs(y_dict['min_sigma'][j])
                        x_err[1,0]=y_dict['plus_sigma'][j]

                        ax[i,k].errorbar(y_dict['y_mean'][j], len(legend_labels)-j, xerr=x_err, fmt='o', color=colors[counter])

                    ax[i,k].axvline(mean_y[counter], -1, len(legend_labels)+1, color='black', ls='--') #plot mean y zero
                    ax[i,k].set_ylim([0.5,len(legend_labels)+0.5])
                    

                    if i==num_plots[0]-1:

                        ax[i,k].set_xlabel(r'$\langle y \rangle \times 10^{-6}$', fontsize=20)

                    counter+=1
                    ax[i,k].set_xlim(xlim[i,k])
                    
        plt.tight_layout()
        plt.savefig('../figs/'+main_title+'.pdf')
