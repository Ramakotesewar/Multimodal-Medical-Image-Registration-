import numpy as np
from numpy import matlib
import nrrd
import os
import cv2
from Global_Vars import Global_Vars
from GOA import GOA
from GRO import GRO
from GTO import GTO
from Model_ATRUNet import Model_ATRUNet
from Model_CNN import Model_CNN
from Model_DRL import Model_DRL
from Model_Unet import Model_UNET
from Objfun import Objective_Function
from PROPOSED import PROPOSED
from Pred_PlotResults import plot_results_Pred, plot_results_conv
from WOA import WOA

# Read the dataset 1
an = 0
if an == 1:
    CT_Images = []
    MR_Images = []
    Dataset_path = './Dataset/Dataset_1/'
    path = os.listdir(Dataset_path)
    for i in range(len(path)):
        CT_MR = Dataset_path + path[i]
        names = CT_MR.split('/')
        if names[3] == 'CT':
            ct_im = []
            seg_ct_im = []
            CT_path = os.listdir(CT_MR)
            for j in range(len(CT_path)):
                seg_in_exhaled = []
                in_exhaled = []
                In_Exhaled = CT_MR + '/' + CT_path[j]
                In_Exhaled_path = os.listdir(In_Exhaled)
                for k in range(len(In_Exhaled_path)):
                    try:
                        Image_file_1 = In_Exhaled + '/' + In_Exhaled_path[k]
                        img_file = []
                        readdata_1, header = nrrd.read(Image_file_1)
                        reshaped_images_1 = np.transpose(readdata_1, (2, 1, 0))
                        for m in range(len(reshaped_images_1)):
                            print('CT_IMAGES', j, len(CT_path), k, len(In_Exhaled_path), m, len(reshaped_images_1))
                            Ct_Images = cv2.resize(reshaped_images_1[m], (256, 256))
                            CT_Images.append(Ct_Images)
                            img_file.append(Ct_Images)
                    except:
                        continue
                    seg_ct_im.append(img_file[0])
        else:
            seg_mr_im = []
            mr_im = []
            MR_path = os.listdir(CT_MR)
            for j in range(len(MR_path)):
                seg_in_exhaled_2 = []
                in_exhaled_2 = []
                In_Exhaled = CT_MR + '/' + MR_path[j]
                In_Exhaled_path = os.listdir(In_Exhaled)
                for k in range(len(In_Exhaled_path)):
                    try:
                        Image_file_2 = In_Exhaled + '/' + In_Exhaled_path[k]
                        img_file_2 = []
                        readdata_2, header = nrrd.read(Image_file_2)
                        reshaped_images_2 = np.transpose(readdata_2, (2, 1, 0))
                        for m in range(len(reshaped_images_2)):
                            print('MRI_IMAGES', i, len(path), j, len(MR_path), k, len(In_Exhaled_path), m,
                                  len(reshaped_images_2))
                            Mr_Images = cv2.resize(reshaped_images_2[m], (256, 256))
                            MR_Images.append(Mr_Images)
                            img_file_2.append(Mr_Images)

                    except:
                        continue
                    seg_mr_im.append(img_file_2[-1])

    np.save('CT_GT_Image.npy', np.asarray(seg_ct_im))
    np.save('MR_GT_Image.npy', np.asarray(seg_mr_im))
    np.save('CT_Images.npy', np.asarray(CT_Images))
    np.save('MR_Images.npy', np.asarray(MR_Images))

# optimization for Segmentation
an = 0
if an == 1:
    CT = np.load('CT_Images.npy', allow_pickle=True)
    MR = np.load('MR_Images.npy', allow_pickle=True)
    Global_Vars.CT = CT
    Global_Vars.MR = MR
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat(np.asarray([5, 5, 100]), Npop, 1)
    xmax = matlib.repmat(np.asarray([55, 50, 500]), Npop, 1)
    fname = Objective_Function
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("GOA...")
    [bestfit1, fitness1, bestsol1, time1] = GOA(initsol, fname, xmin, xmax, Max_iter)

    print("WOA...")
    [bestfit2, fitness2, bestsol2, time2] = WOA(initsol, fname, xmin, xmax, Max_iter)

    print("GTO...")
    [bestfit4, fitness4, bestsol4, time3] = GTO(initsol, fname, xmin, xmax, Max_iter)

    print("GRO...")
    [bestfit3, fitness3, bestsol3, time4] = GRO(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    bestfit = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', bestfit)
    np.save('BestSol_CLS.npy', BestSol_CLS)

# Image Registration
an = 1
if an == 1:
    Image1 = np.load('CT_Images.npy', allow_pickle=True)
    Image2 = np.load('MR_Images.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Sol = np.load('BestSol_CLS.npy', allow_pickle=True)
    Activation_function = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax']
    Eval_all = []
    for i in range(len(Activation_function)):
        Eval = np.zeros((10, 5))
        for j in range(len(Sol)):
            Eval[j, :] = Model_ATRUNet(Image1, Image2, Target, Sol[j].astype('int'))
        Eval[5, :] = Model_DRL(Image1, Image2, Target)
        Eval[6, :] = Model_CNN(Image1, Image2, Target)
        Eval[7, :] = Model_UNET(Image1, Image2, Target)
        Eval[8, :] = Model_ATRUNet(Image1, Image2, Target)
        Eval[9, :] = Eval[4, :]
        Eval_all.append(Eval)
    np.save('Eval_all.npy', Eval_all)

plot_results_Pred()
plot_results_conv()
