Why: Converting other representations to a point cloud is easy.

Challenge: No-regularity.

[Canonical capsules: Self-supervised capsules in canonical pose](https://proceedings.neurips.cc/paper/2021/file/d1ee59e20ad01cedc15f5118a7626099-Paper.pdf) Transformation invariance.

[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) 

$f(x_1,\dots,x_n)=\gamma\circ g(h(x_1),\dots h(x_n))$ is symmetric if $g()$ is symmetric. 

In Vanilla PointNet, they choose function $g()$ to be max-pool, $\gamma()$ and $h()$ to be MLP.

To address transformation invariance, they use data dependent transformation for automatic alignment. Spatial Transformer Networks. 

Datasets: classification -- ModelNet, segmentation -- ShapeNet, Stanford 2D-3D S,

Limitations of PointNet: No hierarchical feature learning, No transformation invariance, 



[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)

How can we incorporate the concept of learning shift-invariant convolution kernels from CNNs into PointNet?

Recursively apply shared PointNet at local regions.

Apply FPS to sample key points and use radius-based ball query to group.

sensitive to Non-Uniform Sampling Density.

Multi-scale grouping. However, we need to run PointNet more times.

Multi-resolution grouping. 

Can generalize to Non-Euclidean Metric space, like using geodesic distance.



[Dynamic Graph CNN for Learning on Point Clouds](https://dl.acm.org/doi/pdf/10.1145/3326362)

A small PointNet in each layer of PointNet++ is:
$$
x_i'=\max_{j:(i,j)\in E}h(x_j)
$$
A generalization of PointNet is:
$$
x' = g\left(\bigoplus_{j:(i,j)\in E} f(x_i, x_j)\right)
$$
But, what's the choice of $\oplus,f$ and $g$ ?

[[1803.10091\] Point Convolutional Neural Networks by Extension Operators](https://arxiv.org/abs/1803.10091) they choose $\oplus$ as a weighted sum operation with Gaussian kernel distance as the weight.

Dynamic Graph CNN apply $x'=\displaystyle\max_{j:(i,j)\in E} \sigma(h'(x_i)+h(x_j-x_i))$, use the learned features to redefine the point proximity.

Drawback: they need to find neighbor points in higher dimensional space, so it's very slow. Higher accuracy but much slower.

And we don't know how the neural network learn the distance metric.

