'''
    Test SNP
    '''

from cortex.datasets.neuroimaging.snp import SNP

def test_snp_data(source, batch_size,true_num_subject, true_snp_lenght):
    train = SNP(source, batch_size=batch_size)

    # test batching
    print train.X.shape == (true_num_subject,true_snp_lenght)
    print train.Y.shape == (true_num_subject, 2)

    # test next
    rval = train.next(10)
    next_10 = rval['snp'], rval['label']
    print next_10[0].shape == (10, true_snp_lenght)
    print next_10[1].shape == (10, 2)

    #print next10.shape
    train.shuffle = False
    train.reset()
    rval = train.next(5)
    next_5 = rval['snp'], rval['label']
    print next_5[0].shape == (5, true_snp_lenght)
    print next_5[1].shape == (5, 2)

    train.shuffle = True
    train.reset()
    rval = train.next(5)
    next_5_nd = rval['snp'], rval['label']
    #test reset
    print next_5[0]
    print '\n'
    print next_5_nd[0]

    return train

if __name__ == '__main__':

    #Variable is not entered at all
    source = {'snp': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05.mat',
             'label': 'SNP_large_NOTprocessed_byp_05/pheno_large.mat',
             'chrom_index': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05_chrom_index.mat'
              }

    train = test_snp_data(source, batch_size=6, true_num_subject=4737, true_snp_lenght=14136)

    #Set to some other thing
    source = {'snp': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05.mat',
             'label': 'SNP_large_NOTprocessed_byp_05/pheno_large.mat',
             'chrom_index': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05_chrom_index.mat',
              'chromosomes': 'Nonell'
              }

    train = test_snp_data(source, batch_size=6, true_num_subject=4737, true_snp_lenght=14136)

    #Set to None
    source = {'snp': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05.mat',
             'label': 'SNP_large_NOTprocessed_byp_05/pheno_large.mat',
             'chrom_index': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05_chrom_index.mat',
              'chromosomes': None
              }

    train = test_snp_data(source, batch_size=6, true_num_subject=4737, true_snp_lenght=14136)


    #case not provided
    source = {'snp': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05.mat',
             'label': 'SNP_large_NOTprocessed_byp_05/pheno_large.mat',
             'chrom_index': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05_chrom_index.mat',
             'chromosomes': ''
              }

    train = test_snp_data(source, batch_size=6, true_num_subject=4737, true_snp_lenght=14136)

    #case single chromosomes provided as integer
    source = {'snp': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05.mat',
             'label': 'SNP_large_NOTprocessed_byp_05/pheno_large.mat',
             'chrom_index': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05_chrom_index.mat',
             'chromosomes': 1
              }

    train = test_snp_data(source, batch_size=6, true_num_subject=4737, true_snp_lenght=1106)
    # case single provived as list
    source = {'snp': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05.mat',
             'label': 'SNP_large_NOTprocessed_byp_05/pheno_large.mat',
             'chrom_index': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05_chrom_index.mat',
             'chromosomes': [1]
              }

    train = test_snp_data(source, batch_size=6, true_num_subject=4737, true_snp_lenght=1106)

    #case list provided
    source = {'snp': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05.mat',
             'label': 'SNP_large_NOTprocessed_byp_05/pheno_large.mat',
             'chrom_index': 'SNP_large_NOTprocessed_byp_05/SNP_large_NOTprocessed_byp_05_chrom_index.mat',
             'chromosomes': [1, 5, 10]
              }

    train = test_snp_data(source, batch_size=6, true_num_subject=4737, true_snp_lenght=2753)
