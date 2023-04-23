# SDRO_code
> Implementation and experiments based on the paper [Sinkhorn Distributionally Robust Optimization
](https://arxiv.org/abs/2109.11926)

> Citation:
```
@article{wang2022sinkhorn,
  title={Sinkhorn distributionally robust optimization},
  author={Wang, Jie and Gao, Rui and Xie, Yao},
  journal={arXiv preprint arXiv:2109.11926},
  year={2022},
  month=dec
}
```

- See [SDRO_optimizer.py](https://github.com/WalterBabyRudin/SDRO_code/blob/main/SDRO_optimizer.py) regarding how to implement main optimization algorithm for SDRO.  
    Although the provided code only contains numerical study for least squares problem, you can edit line 119-120, 191-92, or 268-269 to modify this code for other numerical studies.  
    For nonsmooth loss function, it is recommended to use SG estimator. For smooth loss function, it is recommended to use RT-MLMC/MLMC estimators (Personally, I prefer RT-MLMC method because it has theoretically smaller storage cost.)
- See [relative_error_plot_0510.ipynb](https://github.com/WalterBabyRudin/SDRO_code/blob/main/results/relative_error_plot_0510.ipynb) regarding how to generate plots in our manuscript in Appendix EC.2.1

