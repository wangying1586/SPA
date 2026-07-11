# Upstream dependencies

This package is an overlay for the following research repositories and datasets:

- PAFA: https://github.com/wa976/pafa
- PAFA paper: https://arxiv.org/abs/2505.23834
- BEATs: https://github.com/microsoft/unilm/tree/master/beats
- ICBHI 2017 Respiratory Sound Database: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- SPA code reference: https://github.com/wangying1586/SPA

The installer does not replace PAFA's `main.py`, dataset implementation, BEATs wrapper, or PAFA loss. It adds a separate SPA training entry point and auxiliary scripts. PAFA and BEATs remain subject to their respective upstream terms. The ICBHI dataset must be downloaded separately and used under the dataset provider's terms.
