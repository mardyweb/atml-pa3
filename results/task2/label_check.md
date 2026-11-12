# Task 2 – label sanity check (annotation-derived grounding test pairs)

Rule: the first 8 presence pairs and first 8 counting pairs of `results/task2/data/ground_test.json` (file order). Open each image link, decide whether the COCO-derived answer is right, and tick one box per question.

| pair | question | image | expected | label OK? | ambiguous? | note |
|---|---|---|---|---|---|---|
| test_presence_0 | Does this image contain a clock? | [64150](http://images.cocodataset.org/val2014/COCO_val2014_000000064150.jpg) | **yes** | yes | no | |
| test_presence_0 | Does this image contain a clock? | [478857](http://images.cocodataset.org/val2014/COCO_val2014_000000478857.jpg) | **no** | yes | no | |
| test_presence_1 | Does this image contain a potted plant? | [456254](http://images.cocodataset.org/val2014/COCO_val2014_000000456254.jpg) | **yes** | yes | no | potted plant is present but partly visible |
| test_presence_1 | Does this image contain a potted plant? | [424443](http://images.cocodataset.org/val2014/COCO_val2014_000000424443.jpg) | **no** | yes | no | |
| test_presence_2 | Does this image contain a traffic light? | [411845](http://images.cocodataset.org/val2014/COCO_val2014_000000411845.jpg) | **yes** | yes | no | |
| test_presence_2 | Does this image contain a traffic light? | [54533](http://images.cocodataset.org/val2014/COCO_val2014_000000054533.jpg) | **no** | yes | no | |
| test_presence_3 | Does this image contain a fire hydrant? | [246612](http://images.cocodataset.org/val2014/COCO_val2014_000000246612.jpg) | **yes** | yes | no | fire hydrant present but partly visible (person is more prominent) |
| test_presence_3 | Does this image contain a fire hydrant? | [234677](http://images.cocodataset.org/val2014/COCO_val2014_000000234677.jpg) | **no** | yes | no | |
| test_presence_4 | Does this image contain a sports ball? | [383384](http://images.cocodataset.org/val2014/COCO_val2014_000000383384.jpg) | **yes** | yes | yes | partly ambiguous since the sports ball is in motion and barely visible |
| test_presence_4 | Does this image contain a sports ball? | [467580](http://images.cocodataset.org/val2014/COCO_val2014_000000467580.jpg) | **no** | yes | no | |
| test_presence_5 | Does this image contain a kite? | [348982](http://images.cocodataset.org/val2014/COCO_val2014_000000348982.jpg) | **yes** | yes | yes | unusual looking box kite |
| test_presence_5 | Does this image contain a kite? | [424247](http://images.cocodataset.org/val2014/COCO_val2014_000000424247.jpg) | **no** | yes | no | |
| test_presence_6 | Does this image contain a horse? | [522175](http://images.cocodataset.org/val2014/COCO_val2014_000000522175.jpg) | **yes** | yes | no | |
| test_presence_6 | Does this image contain a horse? | [549462](http://images.cocodataset.org/val2014/COCO_val2014_000000549462.jpg) | **no** | yes | no | |
| test_presence_7 | Does this image contain a cow? | [570693](http://images.cocodataset.org/val2014/COCO_val2014_000000570693.jpg) | **yes** | yes | no | |
| test_presence_7 | Does this image contain a cow? | [168334](http://images.cocodataset.org/val2014/COCO_val2014_000000168334.jpg) | **no** | yes | no | |
| test_counting_0 | How many handbags can you see? | [554459](http://images.cocodataset.org/val2014/COCO_val2014_000000554459.jpg) | **3** | yes | no | |
| test_counting_0 | How many handbags can you see? | [361586](http://images.cocodataset.org/val2014/COCO_val2014_000000361586.jpg) | **4** | yes | yes | partly ambiguous as it's hard to discern if people further into the photo are wearing a handbag |
| test_counting_1 | How many bottles can you see? | [233376](http://images.cocodataset.org/val2014/COCO_val2014_000000233376.jpg) | **3** | yes | no | |
| test_counting_1 | How many bottles can you see? | [479067](http://images.cocodataset.org/val2014/COCO_val2014_000000479067.jpg) | **1** | yes | no | |
| test_counting_2 | How many keyboards can you see? | [255165](http://images.cocodataset.org/val2014/COCO_val2014_000000255165.jpg) | **3** | unsure | yes | I count four keyboards, not three; the one on top of the cabinet might be missed by the model |
| test_counting_2 | How many keyboards can you see? | [531784](http://images.cocodataset.org/val2014/COCO_val2014_000000531784.jpg) | **2** | unsure | yes | 1 keyboard is clearly visible, the other potential keyboard's edge is only slightly visible in the picture |
| test_counting_3 | How many bicycles can you see? | [504091](http://images.cocodataset.org/val2014/COCO_val2014_000000504091.jpg) | **3** | unsure | yes | I count 2 bicycles clearly; can't see the third as the person is obstructing the view of the bicycles |
| test_counting_3 | How many bicycles can you see? | [44045](http://images.cocodataset.org/val2014/COCO_val2014_000000044045.jpg) | **1** | yes | no | |
| test_counting_4 | How many microwaves can you see? | [242543](http://images.cocodataset.org/val2014/COCO_val2014_000000242543.jpg) | **1** | yes | no | |
| test_counting_4 | How many microwaves can you see? | [54959](http://images.cocodataset.org/val2014/COCO_val2014_000000054959.jpg) | **3** | yes | no | |
| test_counting_5 | How many surfboards can you see? | [116173](http://images.cocodataset.org/val2014/COCO_val2014_000000116173.jpg) | **3** | yes | yes | a bit ambiguous as there are four people but only three surfboards visible, it is possible the fourth is obstructed from view |
| test_counting_5 | How many surfboards can you see? | [558717](http://images.cocodataset.org/val2014/COCO_val2014_000000558717.jpg) | **2** | yes | no | |
| test_counting_6 | How many couches can you see? | [10707](http://images.cocodataset.org/val2014/COCO_val2014_000000010707.jpg) | **2** | yes | no | |
| test_counting_6 | How many couches can you see? | [579648](http://images.cocodataset.org/val2014/COCO_val2014_000000579648.jpg) | **4** | yes | yes | fourth one is barely visible |
| test_counting_7 | How many giraffes can you see? | [261461](http://images.cocodataset.org/val2014/COCO_val2014_000000261461.jpg) | **2** | yes | yes | there might be a third giraffe lying down |
| test_counting_7 | How many giraffes can you see? | [553081](http://images.cocodataset.org/val2014/COCO_val2014_000000553081.jpg) | **4** | yes | no | |

**Summary** (fill in): presence labels checked 16 / 16, wrong 0, ambiguous 2; counting labels checked 16 / 16, wrong 0, ambiguous 7.

Typical COCO issues to look for: unannotated objects (a 'no' image that actually contains the object, often small or in the background); crowd scenes or partially visible objects making counts ambiguous; category confusions (e.g. 'couch' vs 'chair', 'cup' vs 'bowl').
