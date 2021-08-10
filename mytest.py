from model.loss import nsrr_loss
from cg_manipulation.cg_utils import zero_upsampling, optical_flow_to_motion, backward_warping
import matplotlib.pyplot as plt
import torch

def zero_upsampling_test():
    img = plt.imread('cg_manipulation/002.jpg')
    init_img = torch.from_numpy(img).cuda().float()
    init_img.requires_grad_(True)
    img = init_img.unsqueeze(0)
    img = img.permute(0,3,2,1)
    img = zero_upsampling(img, (2,2))

    truth = plt.imread('cg_manipulation/001.jpg')
    truth = torch.from_numpy(truth).cuda()
    truth = truth.unsqueeze(0)
    truth = truth.float()/255
    truth = truth.permute(0,3,2,1)

    loss = nsrr_loss(img,truth,0.1)
    print(loss)
    loss.requires_grad_(True)
    loss.backward()

    print(init_img.grad)


def motion_vector_test():
    for i in range(10,290):
        img = plt.imread('data/View-1/'+str(i)+'.png')
        img = torch.from_numpy(img).cuda().float()
        img = img.unsqueeze(0)
        img = img.permute(0,3,1,2)

        img1 = plt.imread('data/View-0/'+str(i)+'.png')

        optical_flow = plt.imread('data/Motion-0/'+str(i)+'.png')
        optical_flow = torch.from_numpy(optical_flow).permute(2,0,1).unsqueeze(0).float()

        motion = optical_flow_to_motion(optical_flow, 5.37)
        motion = motion.cuda()

        my_warped = backward_warping(img, motion)

        plt.figure()
        plt.imshow(img1)
        plt.figure()
        plt.imshow(img.cpu()[0].permute(1,2,0))
        plt.figure()
        plt.imshow(my_warped.cpu()[0].permute(1,2,0))
        plt.show()

motion_vector_test()