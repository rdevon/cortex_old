'''
Module for AOD analysis functions

'''

import numpy as np
import os
from os import path
from scipy.stats import (kendalltau, linregress, mannwhitneyu, ttest_1samp,
                         ttest_ind, ttest_rel)
import statsmodels.api as sm
from tabulate import tabulate
import theano

from .analyzer import Analyzer


def rename_rois(labels):
    for i in xrange(len(labels)):
        label = labels[i]
        label = label.replace('Left', 'L')
        label = label.replace('Right', 'R')
        label = label.replace('Inferior', 'Inf')
        label = label.replace('Middle', 'Mid')
        label = label.replace('Anterior', 'Ant')
        label = label.replace('Gyrus', 'Gy')
        label = label.replace('Superior', 'Sup')
        label = label.replace('Medial', 'Med')
        label = label.replace('Orbitalis', 'Orb')
        labels[i] = label[:35]
    return labels

def get_betas(tcs, stim, idx):
    betas = []
    stim = sm.add_constant(stim)
    for c in xrange(tcs.shape[2]):
        c_betas = []
        for s in idx:
            model = sm.OLS(tcs[:, s, c], stim)
            results = model.fit()
            c_betas.append(results.params[1])
        betas.append(c_betas)
    betas = np.array(betas)
    return betas

def get_task_relatedness(tcs, stim, idx=None):
    if idx is None: idx = range(tcs.shape[1])
    betas = get_betas(tcs, stim, idx)
    t, p = ttest_1samp(betas[:, idx], 0, axis=1)
    return t, p

def get_task_difference(tcs, stim, idx1, idx2, use_mw=True):
    idx = range(tcs.shape[1])
    betas = get_betas(tcs, stim, idx)

    if use_mw:
        utests = np.array([mannwhitneyu(np.array(betas)[i, idx1],
                                        np.array(betas)[i, idx2],
                                        alternative='two-sided')
                           for i in range(len(betas))])
        t = utests[:, 0]
        p = utests[:, 1]
    else:
        t, p = ttest_ind(betas[:, idx1], betas[:, idx2], axis=1)
    return t, p

def do_fdr_correct(p, has_dependence=False, sig=0.01):
    if p.ndim > 1:
        shape = p.shape
        p = p.reshape((reduce(lambda x, y: x * y, shape),))
    else:
        shape = None
    m = p.shape[0]
    p_sorted = sorted(p)
    p_argsort = np.argsort(p)
    if has_dependence:
        c = np.array([1 / float(i + 1) for i in xrange(m,)]).sum()
    else:
        c = 1.
    fdr = np.array([sig * (i + 1) / (m * c) for i in xrange(m)])
    w = np.where(p_sorted < fdr)[0]
    try:
        k = w.max() + 1
    except ValueError:
        k = 0
    mask = np.zeros_like(p)
    mask[p_argsort[:k]] = 1.
    if shape is not None: mask = mask.reshape(shape)
    return k, mask
        

class AODAnalyzer(Analyzer):
    def __init__(self, spatial_map_key, time_course_key, run=False, **kwargs):
        self.spatial_map_key = spatial_map_key
        self.time_course_key = time_course_key
        self.roi_dict = None
        self.stats = {}
        super(AODAnalyzer, self).__init__(**kwargs)
        if run: self.run()
        
    def build(self):
        super(AODAnalyzer, self).build()
        nifti_out_path = path.join(self.session.manager.out_path, 'niftis')
        if not path.isdir(nifti_out_path):
            os.mkdir(nifti_out_path)
        self.visualizer.add('data.make_images',
                            self.spatial_map_key,
                            set_global_norm=True,
                            out_path=nifti_out_path)
        
        self.visualizer.add('data.viz', maps='ica_viz.maps',
                            time_courses='ica_viz.tcs',
                            time_course_scales='ica_viz.tc_scales', t_limit=50,
                            y=12)
        
        self.f_tcs = theano.function(self.session.inputs,
                                     self.tensors[self.time_course_key],
                                     updates=self.session.updates,
                                     on_unused_input='ignore')
        self.targets = self.session.manager.datasets['data'][self.mode].extras['targets']
        self.novels = self.session.manager.datasets['data'][self.mode].extras['novels']
        self.Y = self.session.manager.datasets['data'][self.mode].Y[:, 0]
        
    def make_roi_dict(self, inputs=None):
        self.logger.info('Making ROI dictionary')
        if inputs is None: inputs = self.get_data()
        _, _, roi_dict = self.visualizer.run(-2, inputs=inputs, data_mode=self.mode)
        self.roi_dict = roi_dict
        
    def set_labels(self):
        labels = [self.roi_dict[k]['top_clust']['rois'][0]
                  if len(self.roi_dict[k]['top_clust']['rois']) > 0 else 'UNK'
                  for k in self.roi_dict.keys()]
                
        return labels
                
    def set_features(self):
        self.logger.info('Setting features')
        labels = self.set_labels()
        for i, l in enumerate(labels):
            self.features[i] = dict(name=l, on=True)
        
    def run(self):
        self.logger.info('Running analysis')
        inputs = self.get_data()
        self.make_roi_dict(inputs)
        self.set_features()
        
        self.logger.info('Getting time courses')
        tcs = self.f_tcs(*inputs)
        
        if isinstance(tcs, np.ndarray):
            tcs = {self.time_course_key: tcs}
        elif isinstance(tcs, list):
            tcs = dict((k, v) for k, v in zip(self.time_course_key, tcs))
        
        hc = np.where(self.Y[:, 0] == 1)[0].tolist()
        sz = np.where(self.Y[:, 1] == 1)[0].tolist()
        
        self.logger.info('T-tests')
        for k, v in tcs.items():
            self.stats[k] = {}
            for sname, stim in zip(
                ['targets', 'novels'], [self.targets, self.novels]):
                t, p = get_task_relatedness(v, stim)
                u, pu = get_task_difference(v, stim, hc, sz)
                self.stats[k][sname] = dict(p=p, t=t, u=u, pu=pu)
                
                for j in xrange(t.shape[0]):
                    self.features[j].update(
                        **{'{}_{}_t'.format(k, sname): t[j],
                           '{}_{}_p'.format(k, sname): p[j],
                           '{}_{}_u'.format(k, sname): u[j],
                           '{}_{}_pu'.format(k, sname): pu[j]})
        
    def make_table(self, task_sig=0.01, min_p=10e-7, tablefmt='plain'):
        fdr_dict = dict()
        diff_dict = dict()
        
        for tc_name in self.stats.keys():
            stats = self.stats[tc_name]
            keys = stats.keys()
            fdr = dict()
            diff = dict()
            
            for sname in keys:
                p = stats[sname]['p']
                K, mask = do_fdr_correct(p, sig=task_sig)
                asp = np.argsort(p)[:K]
                fdr[sname] = asp
                
                pu = stats[sname]['pu']
                K, mask = do_fdr_correct(pu, sig=task_sig)
                asp = np.argsort(pu)[:K]
                diff[sname] = asp
                
            fdr_dict[tc_name] = fdr
            diff_dict[tc_name] = diff
    
        td = ['', '']
        for tc_name in self.stats.keys():
            td += [tc_name, '', 'diff_{}'.format(tc_name), '']
        table = [td, ['ID', 'Label'] + ['Targets', 'Novels'] * (2 * len(self.stats))]
        for i, feature in self.features.items():
            if feature['on']:
                td = [i, feature['name']]
                for tc_name in self.stats.keys():
                    stats = self.stats[tc_name]
                    
                    for sname in stats.keys():
                        asp = fdr_dict[tc_name][sname]
                        stat = stats[sname]['p'][i]
                        td.append('%.1e' % stat if (i in asp and stat < min_p)
                                  else '')
                        
                    for sname in stats.keys():
                        asp = diff_dict[tc_name][sname]
                        stat = stats[sname]['pu'][i]
                        td.append('%.1e' % stat if (i in asp and stat < min_p)
                                  else '')
                    
            if not all([t == '' for t in td[2:]]):
                table.append(td)
        
        print tabulate(table, headers='firstrow', tablefmt=tablefmt)
        
    def save_niftis(self):
        pass
        
    def save_maps(self, task_sig=0.01, min_p=10e-7):
        self.logger.info('Saving maps')
        inputs = self.get_data()
        
        for tc_name in self.stats.keys():
            stats = self.stats[tc_name]
            keys = stats.keys()
            asp = set()
            
            for sname in keys:
                p = stats[sname]['p']
                K_ = next((i for i, p_ in enumerate(np.sort(p)[::-1]) if p_ <= min_p), None)
                if K_ is None:
                    K_ = 0
                else:
                    K_ = p.shape[0] - K_
                K, mask = do_fdr_correct(p, sig=task_sig)
                K = min(K, K_)
                idx = np.argsort(p)[:K]
                asp.update(set(idx))
            
            if len(asp) > 0:
                order = [i for i in np.argsort(stats['targets']['p']) if i in asp]
                self.visualizer.run(-1, inputs=inputs, data_mode=self.mode,
                                    name='{}_map'.format(tc_name), order=order,
                                    stats=dict(p_t=stats['targets']['p'],
                                               p_n=stats['novels']['p']))
            
            asp = set()
            for sname in keys:
                p = stats[sname]['pu']
                K_ = next((i for i, p_ in enumerate(np.sort(p)[::-1]) if p_ <= min_p), None)
                if K_ is None:
                    K_ = 0
                else:
                    K_ = p.shape[0] - K_
                K, mask = do_fdr_correct(p, sig=task_sig)
                K = min(K, K_)
                idx = np.argsort(p)[:K]
                asp.update(set(idx))
                
            if len(asp) > 0:
                order = [i for i in np.argsort(stats['targets']['pu']) if i in asp]
                self.visualizer.run(-1, inputs=inputs, data_mode=self.mode,
                                    name='{}_map_diff'.format(tc_name), order=order,
                                    stats=dict(p_t=stats['targets']['pu'],
                                               p_n=stats['novels']['pu']))
            
            
            
    