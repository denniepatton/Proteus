#!/usr/bin/env python3
tss_length = 2000
signal_ksize = 3
base_len = tss_length - signal_ksize + 1

n_reductions = 3
interaction_ksize = 3
dil_small = 2
dil_large = 7

out = base_len
for _ in range(n_reductions):
    out = out - dil_small * (interaction_ksize - 1)
    out = out // 2
small_len = out

out = base_len
for _ in range(n_reductions):
    out = out - dil_large * (15 - 1)
    out = out // 2
large_len = out

initial_reduction_channels = 64
out_channels = max(initial_reduction_channels // (4 ** (n_reductions - 1)), 1)

print(f'base_len={base_len}')
print(f'small_len={small_len} x {out_channels}ch = {small_len * out_channels} flat')
print(f'large_len={large_len} x {out_channels}ch = {large_len * out_channels} flat')
total = small_len * out_channels + large_len * out_channels
print(f'signal_encoder input: {total} -> 1024 -> 256 -> 64')
sig_params = total*1024 + 1024 + 1024*256 + 256 + 256*64 + 64
print(f'signal_encoder params: ~{sig_params:,}')
print(f'Model total is 2,209,312 params')
print(f'signal_encoder is {sig_params/2209312*100:.1f}% of total')
