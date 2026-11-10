| condition   | comparison                            |   max |Δlogit| |   mean |Δlogit| |   top-1 agreement |   mean KL |   positions |
|:------------|:--------------------------------------|---------------:|----------------:|------------------:|----------:|------------:|
| lora4       | merged_vs_unmerged                    |          0.116 |          0.0075 |            1      |  2.45e-05 |         186 |
| lora16      | merged_vs_unmerged                    |          0.136 |          0.0071 |            1      |  2.59e-05 |         186 |
| oplora16    | projected_adapter_vs_training_forward |          0.141 |          0.0066 |            0.9896 |  2.47e-05 |         186 |
| oplora16    | merged_vs_training_forward            |          0.117 |          0.0075 |            1      |  3.06e-05 |         186 |
| oplora16    | merged_vs_projected_adapter           |          0.156 |          0.0074 |            0.9896 |  2.27e-05 |         186 |