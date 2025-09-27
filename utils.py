# -*- coding: utf-8 -*-
"""
@author: Liaoyh
@contact: liaoyh21@mails.tsinghua.edu.cn
@file: utils.py
@time: 2025/4/16 14:40
"""
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def worker(func, arg):
    result = func(arg)
    return result


def para_process(ipts, func, num_workers):
    with Pool(num_workers if num_workers is not None else cpu_count()) as pool:
        results = []
        with tqdm(total=len(ipts)) as pbar:
            for result in pool.imap_unordered(partial(worker, func), ipts):
                pbar.update()
                results.append(result)
        return results


def process_in_batch(func, iterator, para=False, num_workers=16):
    if para:
        outs = para_process(iterator, func, num_workers)
    else:
        outs = [func(ipt) for ipt in tqdm(iterator, total=len(iterator))]
    return outs
