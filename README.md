# [Reproduce] Neural Supersampling for Real-time Rendering with Pytorch

Paper Homepage:
https://research.fb.com/blog/2020/07/introducing-neural-supersampling-for-real-time-rendering/

Template from:
https://github.com/victoresque/pytorch-template

Some codes modified from the repository:
https://github.com/IMAC-projects/NSRR-PyTorch

I have realized the whole process, although the efficiency is not great.

The dataset is very large, so please download from Baidu Netdisk:

链接：https://pan.baidu.com/s/1IHcOfJXa5VyVdzDbrSno4A 
提取码：l06e 

## Some Results:

![image1](https://github.com/SuikaSibyl/ReproduceNSRR/blob/master/figs/1.png)

Low resolution 2x2 by Photoshop ( psnr = 36.6739), NSRR 2x2( psnr = 40.1767), Groundtruth
Tested in the same scene as the training dataset.

![image1](https://github.com/SuikaSibyl/ReproduceNSRR/blob/master/figs/2.png)

Low resolution 2x2 by Photoshop ( psnr = 38.3879), NSRR 2x2( psnr = 42.6493), Groundtruth
Tested in the same scene as the training dataset.

![image1](https://github.com/SuikaSibyl/ReproduceNSRR/blob/master/figs/3.png)

Low resolution 2x2 by Photoshop ( psnr = 48.4654), NSRR 2x2( psnr = 50.2523), Groundtruth
Tested in a different scene, the sofa is actually moving, so robust for moving objects & different scene

## Time performance:
One run costs around 220ms on my Nvidia 2070Super, which seems to be 10 times larger than the original paper.
