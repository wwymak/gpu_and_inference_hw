============================================================================
Part 3: Short Writeup
============================================================================
Answer these after you generate `results/roofline.png` and inspect the points.

Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
Why does performance rise as arithmetic intensity increases even though the
measured runtime changes only a little?

In this regime, the performance is bottlenecked by how fast the data flows between the 
cpu and gpu. For the compiled operation, data movement is fixed

Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
`128 ops` compiled element-wise operation. Give one or two reasons why that can
happen on a large GPU like an H100.

Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
did for smaller operations. What does that suggest about what resource is
becoming the bottleneck?

Q4. Why do the eager `ops-K` points look so different from the compiled ones?
In the eager mode, _all_ the points sit at ~ the same point. this is because in pytorch
eager mode, a kernel is launched for each op