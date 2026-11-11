Per-rollout Pearson corr(c_t, v_t): mean -0.204, median -0.203, 16% positive; by correctness {'False': -0.21702595243592704, 'True': -0.19330854131755792}
Cov_t(c,v) = Z_tok − Z: mean -0.0022, 16% positive; by correctness {'False': -0.002490101543481325, 'True': -0.0018841457482549724}
Top-1 agreement between Z and Z_tok rankings (held-out questions): 67.0%

|                         |    1 |    2 |    4 |    8 |   16 |   32 |
|:------------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| Best VisConf            | 53.6 | 55.4 | 56.3 | 56.3 | 56.8 | 57.8 |
| Best tokenwise VisConf  | 53.6 | 55.4 | 55.8 | 55.7 | 55.7 | 56.2 |
| VisConf + RWC           | 53.6 | 55.4 | 57.7 | 60.9 | 61.9 | 63.1 |
| Tokenwise VisConf + RWC | 53.6 | 55.4 | 57.5 | 61   | 61.9 | 62.7 |