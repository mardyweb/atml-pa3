| policy                    | setting      | accuracy         | mean tokens            |   mean rollouts | stops @4/8/16/32   |
|:--------------------------|:-------------|:-----------------|:-----------------------|----------------:|:-------------------|
| fixed-4 self-consistency  | —            | 57.8             | 1112                   |             4   | —                  |
| fixed-8 self-consistency  | —            | 59.9             | 2230                   |             8   | —                  |
| fixed-16 self-consistency | —            | 60.9             | 4462                   |            16   | —                  |
| fixed-32 self-consistency | —            | 61.8             | 8898                   |            32   | —                  |
| adaptive (agreement)      | τ=0.5        | 61.3 [58.4,64.1] | 3323 (37% of fixed-32) |             9.9 | 71%/7%/2%/19%      |
| adaptive (quality)        | τ=0.5, γ=0.0 | 61.6 [58.7,64.4] | 3638 (41% of fixed-32) |            11   | 65%/10%/3%/22%     |

agreement: early stops 81% of runs; wrong early stops 27% of early stops, of which fixed-32 would have been correct in 313 runs; pool fraction correct by stop count {'4': 0.6527990797546013, '8': 0.4855567226890756, '16': 0.45125786163522014, '32': 0.13554510058403635}
quality: early stops 78% of runs; wrong early stops 25% of early stops, of which fixed-32 would have been correct in 251 runs; pool fraction correct by stop count {'4': 0.6772807773310358, '8': 0.48691765091863515, '16': 0.46847943722943725, '32': 0.15530409192825112}