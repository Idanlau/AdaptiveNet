import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import numpy as np
import os
from os.path import join
from os import remove
import h5py
from math import ceil



def train(opt, model, encoder_dim, device, dataset, criterion, optimizer, train_set, whole_train_set, whole_training_data_loader, epoch, writer):
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging


    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        start_indices = np.arange(0, len(train_set), opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = [start_indices[i:i+1] for i in range(len(start_indices))]
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]


    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    if os.path.exists(train_set.cache):
        h5_prev = h5py.File(train_set.cache, mode='r') # h5_prev stores the last run of the model over all training data
    else:
        tensor_shape = [len(train_set), opt.seqL, opt.outDims]
        h5_prev = torch.zeros(tensor_shape, dtype=torch.float32) #NOTE: If I initalize 0s will attention layer ever learn?

    for subIter in range(subsetN):
    
        print('====> Building Cache')

        model.eval()
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = encoder_dim
            if opt.pooling.lower() == 'seqnet':
                pool_size = opt.outDims
            h5feat = h5.create_dataset("features", [len(whole_train_set), pool_size], dtype=np.float32) # empty creation of a dataset
            with torch.no_grad():
                for iteration, (input, indices) in tqdm(enumerate(whole_training_data_loader, 1),total=len(whole_training_data_loader)-1, leave=False):
                    image_encoding = (input).float().to(device)
                    for ind in indices:            
                        seq_encoding = model.pool(prev_seq=h5_prev[ind-1],query=h5_prev[ind],img_ft=image_encoding[ind]) #TODO: fix this call.
                        h5feat[ind] = seq_encoding.detach().cpu().numpy()
                        del input, image_encoding, seq_encoding

        subset_indices = np.array(subsetIdx[subIter])
        sub_train_set = Subset(dataset=train_set, indices=subset_indices)


        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=not opt.nocuda)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_reserved())

        model.train()
        for iteration, (query, positives, negatives, 
                negCounts, indices) in tqdm(enumerate(training_data_loader, startIter),total=len(training_data_loader),leave=False):
            loss = 0

            if query is None:
                continue # in case we get an empty batch

            B = query.shape[0]
            nNeg = torch.sum(negCounts)

            input = torch.cat([query,positives,negatives]).float()
            input = input.to(device)
            seq_encoding = model.pool(input)
            seqQ, seqP, seqN = torch.split(seq_encoding, [B, B, nNeg])

            optimizer.zero_grad()
            
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(seqQ[i:i+1], seqP[i:i+1], seqN[negIx:negIx+1])

            loss /= nNeg.float().to(device) # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, seq_encoding, seqQ, seqP, seqN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                    nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

