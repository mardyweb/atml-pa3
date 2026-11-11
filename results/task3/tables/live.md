| strategy   | accuracy     |   output tok/question |   input tok/question | allowance used   |   runtime s/question | truncation   | extraction   |
|:-----------|:-------------|----------------------:|---------------------:|:-----------------|---------------------:|:-------------|:-------------|
| single     | 66.0 [52,78] |                   736 |                  108 | 24%              |                 31.3 | 2%           | 98%          |
| parallel   | 56.0 [42,70] |                  1866 |                  325 | 61%              |                 79.4 | 16%          | 84%          |
| sequential | 56.0 [42,70] |                  1184 |                  873 | 39%              |                 50.5 | 7%           | 93%          |

Revision transitions (relative to the shared initial answer, acc 52%): wrong→correct 2, correct→wrong 0, correct→correct 26, wrong→wrong 22, answer changed 8/50. Parallel oracle (any of 3 correct): 68%. GPU NVIDIA L4, peak memory 6.5 GB.