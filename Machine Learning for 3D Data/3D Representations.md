3D Grids or Voxels

The emergency of the neural network techniques actually came from the invention of convolutional neural network. 

Memory && Complexity ?

The most important reason for not using this kind of representation in most of the cases is that the data we concerned are basically surface information not the volume information. We are interested in the shell of the objects.

Many boxes are empty. Inefficiency. Huge waste of computation.

Some application need volume information: Medical Imaging. 

[[1712.01537\] O-CNN](https://arxiv.org/abs/1712.01537) [[1704.01047\] OctNetFusion](https://arxiv.org/abs/1704.01047) Architectures using adaptive data structure. More efficient, but increase complexity in the implementations.

[2018 CVPR SparseConvNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.html) compute convolutions only in the active area. However, still takes lots of memory and time in training.



Multi-View Images

[Multi-View CNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf) train CNN on ImageNet1k and finetune it on multi-view 3D dataset for 3D object classification task. 

[CVPR 2017 3D Shape Segmentation](https://openaccess.thecvf.com/content_cvpr_2017/html/Kalogerakis_3D_Shape_Segmentation_CVPR_2017_paper.html) predict information for each of the points not for the entire shape.

The property of equivalence in rotation or transformations is a good question.

(+) good for processing appearance information like color, texture, and material.

(-) requires lots of images thus takes lots of time and memory

(-) may not be able to capture geometric details



Point Cloud

irregular structure

More popular and easy to implement. But no surface/topology information. Need to be converted to other representation for downstream applications.

Weak approximation power, requires many points for the details.



Polygon Mesh

Most popular. Graph-like structure but not the same.

[[1809.05910\] MeshCNN](https://arxiv.org/abs/1809.05910) Pooling operation for polygon mesh through the process of iterative edge contraction.

[2020 NIPS Primal-Dual Mesh Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/0a656cc19f3f5b41530182a9e03982a4-Abstract.html)

[ICCV 2015 Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w22/html/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.html)

[CVPR 2024 Single Mesh Diffusion Models with Field Latents for Texture Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Mitchel_Single_Mesh_Diffusion_Models_with_Field_Latents_for_Texture_Generation_CVPR_2024_paper.html)

[2020 ICML PolyGen](https://proceedings.mlr.press/v119/nash20a.html) An AR architecture generates vertices and faces sequentially. However, a minor error could lead to significantly poor results.



CAD Representations

[CVPR 2021 BRepNet](https://openaccess.thecvf.com/content/CVPR2021/html/Lambourne_BRepNet_A_Topological_Message_Passing_System_for_Solid_Models_CVPR_2021_paper.html) [CVPR 2021 UV-Net](https://openaccess.thecvf.com/content/CVPR2021/html/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.html) [ICCV 2021 CSG-Stump](https://openaccess.thecvf.com/content/ICCV2021/html/Ren_CSG-Stump_A_Learning_Friendly_CSG-Like_Representation_for_Interpretable_Shape_Parsing_ICCV_2021_paper.html) very few networks process them, but more networks produce them.



Implicit Representation

[CVPR 2019 DeepSDF](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html) [CVPR 2019 Occupancy Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Mescheder_Occupancy_Networks_Learning_3D_Reconstruction_in_Function_Space_CVPR_2019_paper.html)



Conversion Across Representations

Voxel -> Mesh

1987 Marching Cubes. 

Weakness: no sharp features and the basic versions do not provide topological guarantees.

Also, there are Marching Tetrahedra, in 1994, which avoid ambiguous cases in the Marching Cubes.

[NIPS 2021 Deep Marching Tetrahedra](https://proceedings.neurips.cc/paper/2021/hash/30a237d18c50f563cba4531f1db44acf-Abstract.html) Represents a 3D shape with a signed distance function encoded within a deformable tetrahedral grid. An example of a hybrid representation.



Point Cloud -> Implicit Function

Here surface normal is essential information for recovering surface and volume data.

Surface Normal Estimation: Plan Fitting + Direction Consistent Alignment.

[[2003.10826\] DeepFit: 3D Surface Fitting via Neural Network Weighted Least Squares](https://arxiv.org/abs/2003.10826): A neural network can be used to learn pointwise weights.

Moving Least Squares. Local Kernel Regression: approximation only with a local region, and are sensitive to the weight function.

Poisson Surface Reconstruction:

We have the gradient of the implicit function, which is the same with the surface normal. We'll find the function $f$ with the gradient field $V=\nabla f$. 

Finding a function $\hat{f}$ minimizing the mean squared error:

$$
\hat{f}=\arg\min_f\int_\Omega (\nabla f(x)-V(x))^2dx,
$$

which is a variational calculus.

For example, in 1D-case, $\hat{f}=\arg\min_f\int_\Omega (f'(x)-V(x))^2dx$, we introduce Euler-Lagrange equation:

$$
\frac{\partial L}{\partial f}-\frac{d}{dx}\frac{\partial L}{\partial f'}=0.
$$

Here, $L=(f'(x)-V(x))^2$, so $f''(x)=V'(x)$.

In a higher dimension, this becomes a Poisson's Equation: $\Delta f=\nabla\cdot V$.

And, we can utilize the information that the given points' distance to the surface is zero.

$$
\hat{f}=\arg\min_f\int_\Omega (f'(x)-g''(x))^2dx+\sum_{x\in\chi} w(x)f^2(x)
$$

[[2021] Shape As Points: A Differentiable Poisson Solver](https://proceedings.neurips.cc/paper/2021/hash/6cd9313ed34ef58bad3fdd504355e72c-Abstract.html) A spectral method solving the Poisson Equation, which is highly optimized on GPUs, utilizing Forward Fourier Transform. 

Neural Conversion, point -> implicit without normals information.

[[2007.10453\] Points2Surf: Learning Implicit Surfaces from Point Cloud Patches](https://arxiv.org/abs/2007.10453)

[SAL: Sign Agnostic Learning of Shapes From Raw Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Atzmon_SAL_Sign_Agnostic_Learning_of_Shapes_From_Raw_Data_CVPR_2020_paper.pdf)

