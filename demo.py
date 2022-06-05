# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np

from python_color_transfer.color_transfer import ColorTransfer, Regrain

def demo():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    img_folder = os.path.join(cur_dir, 'imgs')
    img_names = ['scotland_house.png', 'house.jpeg', 'fallingwater.png', 'tower.jpeg']
    ref_names = ['scotland_plain.png', 'hats.png', 'autumn.jpg', 'sunset.jpg']
    out_names = ['scotland_display.png', 'house_display.png', 'fallingwater_display.png', 'tower_display.png']
    img_paths = [os.path.join(img_folder, x) for x in img_names]
    ref_paths = [os.path.join(img_folder, x) for x in ref_names]
    out_paths = [os.path.join(img_folder, x) for x in out_names]
    # cls init
    PT = ColorTransfer(n=300)
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
        t0 = time.time()    
        img_arr_mt = PT.mean_std_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
        print('mean std transfer time: {:.2f}s'.format(time.time() - t0))
        # lab transfer
        t0 = time.time()    
        img_arr_lt = PT.lab_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
        print('lab mean std transfer time: {:.2f}s'.format(time.time() - t0))
        # display
        img_arr_out = np.concatenate((img_arr_in, img_arr_ref, img_arr_mt, img_arr_lt, img_arr_reg), axis=1)
        cv2.imwrite(out_path, img_arr_out)
        print('save to {}\n'.format(out_path))




if __name__ == '__main__':
    demo()


