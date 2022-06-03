# Copyright 2017 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import cv2
import numpy as np
import math
import random
import struct
import copy
import sys
from struct import unpack
import os

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break
# main 
total_width, total_height = 255, 255

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-f", "--file", required=True, help="Path to the listfile")
args = vars(ap.parse_args())

# load file list
with open(args["file"]) as f:
        linecount = sum(1 for _ in f)
        f.close()

file_list = open(args["file"], 'r')

#
tf0 = open("test_stroke.txt", 'w')
tf1 = open("valid_stroke.txt", 'w')
tf2 = open("train_stroke.txt", 'w')
#

save_dir = '/home/ubuntu/sketch_helper/imgData'

n=1
continue_trig = False
for k in range(linecount):
    file_name = file_list.readline()
    print(file_name)

    image_counter = 0 

    
    for num_file in range(25000):

        if num_file < 2500: 
            tf0.write(save_dir + os.sep + str(k)+'/' + str(image_counter)+'\n')

        elif num_file < 5000:
            tf1.write(save_dir + os.sep + str(k)+'/' + str(image_counter)+'\n')

        else:
            tf2.write(save_dir + os.sep + str(k)+'/' + str(image_counter)+'\n')

        image_counter = image_counter +1

tf0.close()
tf1.close()
tf2.close()
file_list.close()
