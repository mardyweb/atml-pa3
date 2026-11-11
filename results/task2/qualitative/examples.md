# Qualitative complementary-pair examples

Selection rule (stated in the report): within each suite, in pair-ID order, take the first `n_sensitive` pairs whose
correct-image predictions are both right AND change under the complementary swap (image-sensitive), and the first
`n_insensitive` pairs whose two correct-image predictions are identical (image-insensitive).


## ground_test


### Image-sensitive and correct


**test_counting_11** — question: *How many cakes can you see?*  (type: counting)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000242060.jpg | 4 | `4` ✓ | `1` | `1` | `1` |
| COCO_val2014_000000048050.jpg | 1 | `1` ✓ | `4` | `1` ✓ | `1` ✓ |

**test_counting_13** — question: *How many toothbrushes can you see?*  (type: counting)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000390190.jpg | 1 | `1` ✓ | `4` | `1` ✓ | `1` ✓ |
| COCO_val2014_000000495419.jpg | 4 | `4` ✓ | `1` | `1` | `1` |

**test_counting_16** — question: *How many bowls can you see?*  (type: counting)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000405691.jpg | 2 | `2` ✓ | `1` | `1` | `1` |
| COCO_val2014_000000498368.jpg | 1 | `1` ✓ | `2` | `1` ✓ | `1` ✓ |

### Image-insensitive (identical predictions across the pair)


**test_counting_0** — question: *How many handbags can you see?*  (type: counting)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000554459.jpg | 3 | `3` ✓ | `3` ✓ | `1` | `1` |
| COCO_val2014_000000361586.jpg | 4 | `3` | `3` | `1` | `1` |

**test_counting_1** — question: *How many bottles can you see?*  (type: counting)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000233376.jpg | 3 | `3` ✓ | `3` ✓ | `4` | `1` |
| COCO_val2014_000000479067.jpg | 1 | `3` | `3` | `1` ✓ | `1` ✓ |

## vqa_test


### Image-sensitive and correct


**vqa_111819003_251816004** — question: *What kind of animal is saying hello?*  (type: other)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000111819.jpg | elephant (elephant, elephant, elephant, elephant, elephant, elephant, elephant, elephant, elephant, elephant) | `elephant` ✓ | `zebra` | `dog` | `dog` |
| COCO_val2014_000000251816.jpg | zebra (zebra, zebra, zebra, zebra, zebra, zebra, zebra, zebra, zebra, zebra) | `zebra` ✓ | `elephant` | `dog` | `dog` |

**vqa_140490000_190676011** — question: *Is this a beach?*  (type: yes/no)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000140490.jpg | yes (yes, yes, yes, yes, yes, yes, yes, yes, yes, yes) | `yes` ✓ | `no` | `no` | `no` |
| COCO_val2014_000000190676.jpg | no (no, no, no, no, no, no, no, no, no, no) | `no` ✓ | `yes` | `no` ✓ | `no` ✓ |

**vqa_171335000_562419000** — question: *Is a man's reflection seen when looking at the clock?*  (type: yes/no)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000171335.jpg | yes (yes, yes, no, yes, yes, yes, yes, yes, yes, yes) | `yes` ✓ | `no` | `no` | `no` |
| COCO_val2014_000000562419.jpg | no (no, no, no, no, yes, no, no, no, no, no) | `no` ✓ | `yes` | `no` ✓ | `no` ✓ |

### Image-insensitive (identical predictions across the pair)


**vqa_112160002_359149001** — question: *Is there anyone else in the picture?*  (type: yes/no)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000112160.jpg | no (yes, no, no, no, no, no, no, no, yes, no) | `no` ✓ | `no` ✓ | `no` ✓ | `no` ✓ |
| COCO_val2014_000000359149.jpg | yes (yes, yes, yes, yes, yes, yes, yes, no, yes, yes) | `no` | `no` | `yes` ✓ | `no` |

**vqa_113945001_6484013** — question: *What shape is the cake?*  (type: other)

| image | reference(s) | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| COCO_val2014_000000113945.jpg | heart (heart, heart, heart shaped, heart, heart, heart, heart, heart, heart, heart) | `circle` | `circle` | `circle` | `circle` |
| COCO_val2014_000000006484.jpg | round (round, round, round, round, round, round, round, round, round, circle) | `circle` | `circle` | `round` ✓ | `circle` |