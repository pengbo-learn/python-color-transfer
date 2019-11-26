# -*- coding: utf-8 -*-
""" python implementation of N-Dimensional PDF Transfer and regraining post-process.

Paper: 
    Automated colour grading using colour distribution transfer. (2007) 
Author's Matlab Implementation:
    https://github.com/frcs/colour-transfer
"""
import os
import time

import cv2
import numpy as np

from python_color_transfer.src.utils import Rotations

class ColorTransfer:
    """ Methods for color transfer of images. """

    def __init__(self, n=300, eps=1e-6, m=0):
        """ Hyper parameters. 
        
        Attributes:
            n: discretization num of distribution of image's pixels.
            m: num of random orthogonal rotation matrices.
            eps: prevents from zero dividing.
        """
        self.n = n
        self.eps = eps
        if m > 0:
            self.rotation_matrices = Rotations.random_rotations(m)
        else:
            self.rotation_matrices = Rotations.optimal_rotations()
    def mean_transfer(self, img_arr_in=None, img_arr_ref=None):
        """ Adapt img_arr_in's mean to img_arr_ref's mean.

        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """
        mean_in = np.mean(img_arr_in, axis=(0, 1), keepdims=True)
        mean_ref = np.mean(img_arr_ref, axis=(0, 1), keepdims=True)
        img_arr_out = img_arr_in - mean_in + mean_ref
        img_arr_out[img_arr_out < 0] = 0
        img_arr_out[img_arr_out > 255] = 255
        return img_arr_out.astype('uint8')
    def pdf_tranfer(self, img_arr_in=None, img_arr_ref=None):
        """ Apply probability density function transfer.

        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """

        # reshape (h, w, c) to (c, h*w)
        [h, w, c] = img_arr_in.shape
        assert c == 3
        reshape_arr_in = img_arr_in.reshape(-1, c).transpose() / 255.
        reshape_arr_ref = img_arr_ref.reshape(-1, c).transpose() / 255.
        # n times of 1d-pdf-transfer
        for rotation_matrix in self.rotation_matrices:
            rot_arr_in = np.matmul(rotation_matrix, reshape_arr_in)
            rot_arr_ref = np.matmul(rotation_matrix, reshape_arr_ref)
            rot_arr_out = np.zeros(rot_arr_in.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = self._pdf_transfer_1d(rot_arr_in[i],
                                                      rot_arr_ref[i])
            rot_delta_arr = rot_arr_out - rot_arr_in
            delta_arr = np.matmul(rotation_matrix.transpose(), rot_delta_arr) #np.linalg.solve(rotation_matrix, rot_delta_arr)
            reshape_arr_in = delta_arr + reshape_arr_in
        # reshape (c, h*w) to (h, w, c)
        reshape_arr_in[reshape_arr_in < 0] = 0
        reshape_arr_in[reshape_arr_in > 1] = 1
        reshape_arr_out = (255. * reshape_arr_in).astype('uint8')
        img_arr_out = reshape_arr_out.transpose().reshape(h, w, c)
        return img_arr_out
    def _pdf_transfer_1d(self, arr_in=None, arr_ref=None):
        """ Apply 1-dim probability density function transfer.

        Args:
            arr_in: 1d numpy input array.
            arr_ref: 1d numpy reference array.
        Returns:
            arr_out: transfered input array.
        """   

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        min_v = arr.min() - self.eps
        max_v = arr.max() + self.eps
        xs = np.array([min_v + (max_v-min_v)*i/self.n for i in range(self.n+1)])
        hist_in, _ = np.histogram(arr_in, xs)
        hist_ref, _ = np.histogram(arr_ref, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in)
        cum_ref = np.cumsum(hist_ref)
        d_in = cum_in / cum_in[-1]
        d_ref = cum_ref / cum_ref[-1]
        # tranfer
        t_d_in = np.interp(d_in, d_ref, xs)
        t_d_in[d_in<=d_ref[0]] = min_v
        t_d_in[d_in>=d_ref[-1]] = max_v
        arr_out = np.interp(arr_in, xs, t_d_in)
        return arr_out

class Regrain:
    def __init__(self):
        self.nbits = [4, 16, 32, 64, 64, 64]
        self.smoothness = 1
        self.level = 0
    def regrain(self, img_arr_in=None, img_arr_col=None):
        '''keep gradient of img_arr_in and color of img_arr_col. '''

        img_arr_in = img_arr_in / 255.
        img_arr_col = img_arr_col / 255.
        img_arr_out = np.array(img_arr_in)
        img_arr_out = self.regrain_rec(img_arr_out, img_arr_in, img_arr_col, self.nbits, self.level)
        img_arr_out[img_arr_out < 0] = 0
        img_arr_out[img_arr_out > 1] = 1
        img_arr_out = (255. * img_arr_out).astype('uint8')
        return img_arr_out
    def regrain_rec(self, img_arr_out, img_arr_in, img_arr_col, nbits, level):
        '''direct translation of matlab code. '''

        [h, w, _] = img_arr_in.shape
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        if len(nbits) > 1 and h2 > 20 and w2 > 20:
            resize_arr_in = cv2.resize(img_arr_in, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_col = cv2.resize(img_arr_col, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_out = cv2.resize(img_arr_out, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_out = self.regrain_rec(resize_arr_out, resize_arr_in, resize_arr_col, nbits[1:], level+1)
            img_arr_out = cv2.resize(resize_arr_out, (w, h), interpolation=cv2.INTER_LINEAR)
        img_arr_out = self.solve(img_arr_out, img_arr_in, img_arr_col, nbits[0], level)
        return img_arr_out
    def solve(self, img_arr_out, img_arr_in, img_arr_col, nbit, level, eps=1e-6):
        '''direct translation of matlab code. '''

        [width, height, c] = img_arr_in.shape
        first_pad_0 = lambda arr : np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
        first_pad_1 = lambda arr : np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
        last_pad_0 = lambda arr : np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
        last_pad_1 = lambda arr : np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)

        delta_x= last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
        delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
        delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

        psi = 256*delta/5
        psi[psi > 1] = 1
        phi = 30 * 2**(-level) / (1 + 10*delta/self.smoothness)

        phi1 = (last_pad_1(phi) + phi) / 2
        phi2 = (last_pad_0(phi) + phi) / 2
        phi3 = (first_pad_1(phi) + phi) / 2
        phi4 = (first_pad_0(phi) + phi) / 2

        rho = 1/5.
        for i in range(nbit):
            den =  psi + phi1 + phi2 + phi3 + phi4
            num = (np.tile(psi, [1, 1, c])*img_arr_col
                   + np.tile(phi1, [1, 1, c])*(last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in) 
                   + np.tile(phi2, [1, 1, c])*(last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in)
                   + np.tile(phi3, [1, 1, c])*(first_pad_1(img_arr_out) - first_pad_1(img_arr_in) + img_arr_in)
                   + np.tile(phi4, [1, 1, c])*(first_pad_0(img_arr_out) - first_pad_0(img_arr_in) + img_arr_in))
            img_arr_out = num/np.tile(den + eps, [1, 1, c])*(1-rho) + rho*img_arr_out
        return img_arr_out

def demo():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    img_folder = os.path.join(cur_dir, 'imgs')
    img_names = ['scotland_house.png', 'house.jpeg', 'fallingwater.png']
    ref_names = ['scotland_plain.png', 'hats.png', 'autumn.jpg']
    out_names = ['scotland_display.png', 'house_display.png', 'fallingwater_display.png']
    img_paths = [os.path.join(img_folder, x) for x in img_names]
    ref_paths = [os.path.join(img_folder, x) for x in ref_names]
    out_paths = [os.path.join(img_folder, x) for x in out_names]
    # cls init
    PT = ColorTransfer()
    RG = Regrain()

    for img_path, ref_path, out_path in zip(img_paths, ref_paths, out_paths):
        # read input img
        img_arr_in = cv2.imread(img_path)
        [h, w, c] = img_arr_in.shape
        print('{}: {}x{}x{}'.format(img_path, h, w, c))
        # read reference img
        img_arr_ref = cv2.imread(ref_path)
        [h, w, c] = img_arr_ref.shape
        print('{}: {}x{}x{}'.format(ref_path, h, w, c))
        # pdf transfer
        t0 = time.time()    
        img_arr_col = PT.pdf_tranfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
        print('pdf transfer time: {:.2f}s'.format(time.time() - t0))
        # regrain
        t0 = time.time()    
        img_arr_reg = RG.regrain(img_arr_in=img_arr_in, img_arr_col=img_arr_col)
        print('regrain time: {:.2f}s'.format(time.time() - t0))
        # mean transfer
        img_arr_mt = PT.mean_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
        # display
        img_arr_out = np.concatenate((img_arr_in, img_arr_ref, img_arr_mt, img_arr_reg), axis=1)
        cv2.imwrite(out_path, img_arr_out)
        print('save to {}'.format(out_path))




if __name__ == '__main__':
    demo()


