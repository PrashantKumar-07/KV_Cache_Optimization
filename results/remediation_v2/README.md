# Rejected exploratory run

Do not use these artifacts for quality, latency, tables, or figures. This run
temporarily moved LRU access stamps from GPU to CPU. CPU and GPU unstable-sort
tie ordering selected different rows and changed predictions. The change was
reverted; `results/remediation_v3/` reruns the matrix with LRU stamps on GPU
and only promotion-streak metadata on CPU. This directory is retained solely
to document the rejected optimization.
