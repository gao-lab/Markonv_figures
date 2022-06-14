"""
Bonito CRF basecalling
"""

import pdb
import torch
import numpy as np
from koi.decode import beam_search, to_str

from bonito.multiprocessing import thread_iter
from bonito.util import chunk, stitch, batchify, unbatchify, half_supported


def stitch_results(results, length, size, overlap, stride, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: stitch_results(v, length, size, overlap, stride, reverse=reverse)
            for k, v in results.items()
        }
    return stitch(results, size, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False, discrete=0, half=True):
    """
    Compute scores for model.
    """
    with torch.inference_mode():
        device = next(model.parameters()).device
        dtype = torch.float16 if (half_supported() and half) else torch.float32
        if discrete:
            old_data_ = batch
            bin_num = 20
            bin_min = -2.5
            bin_max = 2.5
            bins = np.linspace(bin_min, bin_max, bin_num-1)
            old_data_ = np.digitize(old_data_, bins)
            batch = torch.zeros((old_data_.shape[0],bin_num,old_data_.shape[2]),dtype=dtype)
            for i in range(old_data_.shape[0]):
                for j in range(old_data_.shape[2]):
                    batch[i, old_data_[i,0,j],j] = 1
        scores = model(batch.to(dtype).to(device))
        if reverse:
            scores = model.seqdist.reverse_complement(scores)
        sequence, qstring, moves = beam_search(
            scores.contiguous().to(torch.float16), beam_width=beam_width, beam_cut=beam_cut,
            scale=scale, offset=offset, blank_score=blank_score
        )

        return {
            'moves': moves,
            'qstring': qstring,
            'sequence': sequence,
        }


def fmt(stride, attrs):
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'qstring': to_str(attrs['qstring']),
        'sequence': to_str(attrs['sequence']),
    }


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32, reverse=False, discrete=0, half=True):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    scores = thread_iter(
        (read, compute_scores(model, batch, reverse=reverse, discrete=discrete, half=half)) for read, batch in batches
    )

    results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return thread_iter(
        (read, fmt(model.stride, attrs))
        for read, attrs in results
    )
