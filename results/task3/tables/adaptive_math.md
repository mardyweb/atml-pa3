| policy                    | setting      | accuracy         | mean tokens             |   mean rollouts | stops @4/8/16/32   |
|:--------------------------|:-------------|:-----------------|:------------------------|----------------:|:-------------------|
| fixed-4 self-consistency  | —            | 68.8             | 2680                    |             4   | —                  |
| fixed-8 self-consistency  | —            | 71.4             | 5344                    |             8   | —                  |
| fixed-16 self-consistency | —            | 72.7             | 10678                   |            16   | —                  |
| fixed-32 self-consistency | —            | 73.4             | 21347                   |            32   | —                  |
| adaptive (agreement)      | τ=0.625      | 73.3 [70.3,76.1] | 10137 (47% of fixed-32) |            11.8 | 66%/6%/1%/27%      |
| adaptive (quality)        | τ=0.5, γ=0.0 | 72.8 [69.9,75.6] | 7663 (36% of fixed-32)  |             9   | 76%/6%/1%/16%      |

agreement: early stops 73% of runs; wrong early stops 12% of early stops, of which fixed-32 would have been correct in 51 runs; pool fraction correct by stop count {'4': 0.8387547125353441, '8': 0.5246781115879828, '16': 0.5175233644859814, '32': 0.17461416116870876}
quality: early stops 84% of runs; wrong early stops 18% of early stops, of which fixed-32 would have been correct in 131 runs; pool fraction correct by stop count {'4': 0.7708452692748404, '8': 0.4383736559139785, '16': 0.34172453703703703, '32': 0.1291018588770865}