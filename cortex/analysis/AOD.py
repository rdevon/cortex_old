'''
Module for AOD analysis functions

'''

import numpy as np
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

def get_task_relatedness(tcs, stim, idx=None):
    if idx is None: idx = range(tcs.shape[1])
    betas = []
    stim = sm.add_constant(stim)
    for c in xrange(tcs.shape[2]):
        c_betas = []
        for s in range(tcs.shape[1]):
            model = sm.OLS(tcs[:, s, c], stim)
            results = model.fit()
            c_betas.append(results.params[1])
        betas.append(c_betas)
    betas = np.array(betas)
    t, p = ttest_1samp(betas[:, idx], 0, axis=1)
    return t, p

def do_fdr_correct(p, has_dependence=False, sig=0.05):
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
    def __init__(self, spatial_map_key, time_course_key, **kwargs):
        self.spatial_map_key = spatial_map_key
        self.time_course_key = time_course_key
        self.roi_dict = None
        self.stats = {}
        super(AODAnalyzer, self).__init__(**kwargs)
        
    def build(self):
        super(AODAnalyzer, self).build()
        self.visualizer.add('data.make_images',
                            self.spatial_map_key,
                            set_global_norm=True)
        
        self.visualizer.add('data.viz', maps='ica_viz.maps',
                            time_courses='ica_viz.tcs',
                            time_course_scales='ica_viz.tc_scales', t_limit=50,
                            y=12)
        
        self.f_tcs = theano.function(self.session.inputs,
                                     self.tensors[self.time_course_key],
                                     updates=self.session.updates,
                                     on_unused_input='ignore')
        self.targets = self.session.manager.datasets['data']['dummy'].extras['targets']
        self.novels = self.session.manager.datasets['data']['dummy'].extras['novels']
        
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
            self.features[i] = dict(name=l)
        
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
        
        self.logger.info('T-tests')
        for k, v in tcs.items():
            self.stats[k] = {}
            for sname, stim in zip(
                ['targets', 'novels'], [self.targets, self.novels]):
                t, p = get_task_relatedness(v, stim)
                self.stats[k][sname] = dict(p=p, t=t)
                
                for j in xrange(t.shape[0]):
                    self.features[j].update(
                        **{'{}_{}_t'.format(k, sname): t[j],
                           '{}_{}_p'.format(k, sname): p[j]})
        
    def make_table(self, task_sig=0.05, min_p=10e-7, tablefmt='plain'):
        fdr_dict = dict()
        
        for tc_name in self.stats.keys():
            stats = self.stats[tc_name]
            keys = stats.keys()
            fdr = dict()
            
            for sname in keys:
                p = stats[sname]['p']
                K, mask = do_fdr_correct(p, sig=task_sig)
                asp = np.argsort(p)[:K]
                fdr[sname] = asp
            fdr_dict[tc_name] = fdr
    
        td = ['', '']
        for tc_name in self.stats.keys():
            td += [tc_name, '']
        table = [td, ['ID', 'Label'] + ['Targets', 'Novels'] * len(self.stats)]
        for i, feature in self.features.items():
            td = [i, feature['name']]
            for tc_name in self.stats.keys():
                stats = self.stats[tc_name]
                for sname in stats.keys():
                    ask = fdr_dict[tc_name][sname]
                    stat = stats[sname]['p'][i]
                    td.append('%.1e' % stat if (i in asp and stat < min_p)
                              else '')
            if not all([t == '' for t in td[2:]]):
                table.append(td)
        
        print tabulate(table, headers='firstrow', tablefmt=tablefmt)
        
    def save_maps(self, task_sig=0.05, min_p=10e-7):
        self.logger.info('Saving maps')
        inputs = self.get_data()
        
        for tc_name in self.stats.keys():
            stats = self.stats[tc_name]
            keys = stats.keys()
            asp = set()
            
            for sname in keys:
                p = stats[sname]['p']
                K, mask = do_fdr_correct(p, sig=task_sig)
                asp.update(set(np.argsort(p)[:K]))
                
            self.visualizer.run(-1, inputs=inputs, data_mode=self.mode, name='{}_map'.format(tc_name), order=asp)
            
            
            
    