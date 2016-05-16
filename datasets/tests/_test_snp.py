'''
    Test SNP
    '''

from datasets.snp import SNP

def test_snp_data(source_path = {'snp': 'SNP_processed_pgc108.mat' ,'labels': 'pheno.mat' }, batch_size=20):
    train = SNP(source = source_path, batch_size=batch_size)
    # test batching
    print train.Y.shape == (249, 1)
    print train.X.shape == (249, 4475)
    #import ipdb
    #ipdb.set_trace()

    # test next
    rval = train.next(10)
    next_10 = rval['snp'], rval['labels']
    print next_10[0].shape == (10, 4475)
    print next_10[1].shape == (10, 1)

    #print next10.shape
    train.shuffle = False
    train.reset()
    rval = train.next(5)
    next_5 = rval['snp'], rval['labels']
    print next_5[0].shape == (5, 4475)
    print next_5[1].shape == (5, 1)

    train.shuffle = True
    train.reset()
    rval = train.next(5)
    next_5_nd = rval['snp'], rval['labels']
    #test reset
    print next_5[0]
    print next_5_nd[0]

    return train
