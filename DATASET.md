## DATASET
The following is adapted from [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

### For VG Dataset:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`.
2. Download the [scene graphs](https://1drv.ms/u/s!AjK8-t5JiDT1kxyaarJPzL7KByZs?e=bBffxj) and extract them to `datasets/vg/VG-SGG-with-attri.h5`.

### For GQA Dataset:
1. Download the GQA images [Full (20.3 Gb)](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). Extract these images to the file `datasets/gqa/images`.
2. In order to achieve a representative split like VG150, we use the protocol provided by [SHA-GCL](https://github.com/dongxingning/SHA-GCL-for-SGG). You can download the annotation file from [this link](https://huggingface.co/jaehyeongjeon/GQA_detector), and put all three files to  `datasets/gqa/`.