# Mutation Set Search in Boolean Network Ensembles

This repo benchmarks nested search algorithms (NMCS, LNMCS, NRPA, BILNMCS) for identifying mutation sets that achieve the good phenotype probabilities(>0.8) in ensembles of asynchronous Boolean networks.

---

## Installation

```bash
git clone https://github.com/Cazou21/Monte-Carlo_Search_for_Boolean_Networks.git
pip install -r requirements.txt
```

## Quick Start
 Run all search algorithms (quick test)
 ```
python3 julia_exp.py \
  --algos NMCS LNMCS NRPA \
  --depths 2 3 4 \
  --timeouts 60 \
  --nmcs_level 2 \
  --lnmcs_level 2 \
  --nrpa_level 50 \
  --chunk_id 0 \
  --num_chunks 1
```
� Runs nested search algorithms on all the ensemble.

## See the uniform simulation bound 
```
python3 mean_plot.py \
  --exp_name TumourInvasion-WT \ #This is the experiment by default
  --timeout 60 \
  --nb_chunk 1
```

## Project Structure
```
Monte-Carlo_Search_for_Boolean_Networks/
├── src/            # Core simulation + algorithms
├── data/           # Boolean networks and bundles
├── results/        # main results
└── README.md
```

## Methods Included
> **NMCS** — Nested Monte Carlo Search  
> **LNMCS** — Lazy NMCS

> **NRPA** — Policy adaptation nested rollout  
> **BILNMCS** — LNMCS with possible restriction on the ensemble size




