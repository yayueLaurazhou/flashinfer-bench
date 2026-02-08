# 9.7.18. Video Instructions

### 9.7.18. [Video Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#video-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#video-instructions "Permalink to this headline")

All video instructions operate on 32-bit register operands. However, the video instructions may be
classified as either scalar or SIMD based on whether their core operation applies to one or multiple
values.

The video instructions are:

* `vadd`, `vadd2`, `vadd4`
* `vsub`, `vsub2`, `vsub4`
* `vmad`
* `vavrg2`, `vavrg4`
* `vabsdiff`, `vabsdiff2`, `vabsdiff4`
* `vmin`, `vmin2`, `vmin4`
* `vmax`, `vmax2`, `vmax4`
* `vshl`
* `vshr`
* `vset`, `vset2`, `vset4`