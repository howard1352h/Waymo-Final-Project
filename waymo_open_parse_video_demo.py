#!/usr/bin/env python3

import argparse
import datetime
import imp
import os
import time

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# os.environ['PYTHONPATH']='/env/python:.'
m=imp.find_module('waymo_open_dataset', ['.'])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

tf.enable_eager_execution()




def select_tfrecord_file(folder):
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    Tk().withdraw() # don't show a root window
    return askopenfilename(initialdir=folder)


def draw_stats(image, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.
    font_color = (255,255,255)
    line_type = 4

    lines = []

    dt = datetime.datetime.fromtimestamp(frame.timestamp_micros / 1_000_000)
    lines.append(f'{dt.isoformat(sep=" ")}')

    lines.append(f'{frame.context.stats.location}, {frame.context.stats.time_of_day}, {frame.context.stats.weather}')

    for i, cam_objs in enumerate(frame.context.stats.camera_object_counts):
        lines.append(f'{open_dataset.waymo__open__dataset_dot_label__pb2.Label.Type.Name(cam_objs.type)[5:]}={cam_objs.count}')

    x_start = int(10 * font_scale)
    y_start = int(40 * font_scale)
    y_step = int(50 * font_scale)

    for i, l in enumerate(lines):
        cv2.putText(image, l, (x_start, y_start+(y_step*i)), font, font_scale, font_color, line_type)
    return image


def draw_boxes(image, labels, thickness=4):
    # Mappint of Label.Box.Type to cv2 color (these are BGR)
    l2c_ = {
        0: (0,0,0),         # TYPE_UNKNOWN
        1: (0,255,0),       # TYPE_VEHICLE
        2: (0,0,255),       # TYPE_PEDESTRIAN
        3: (255,255,255),   # TYPE_SIGN
        4: (255,0,0),       # TYPE_CYCLIST
    }

    # Label -> Color
    l2c = lambda l: l2c_[l.type]
    # Label -> Name
    l2n = lambda l: open_dataset.waymo__open__dataset_dot_label__pb2.Label.Type.Name(l.type)[5:]

    for label in labels:
        h, w = label.box.width, label.box.length
        x, y = label.box.center_x - (w/2), label.box.center_y - (h/2)
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), l2c(label), thickness)
    return image


def play_dataset(args, file):
    print(f'Processing {file}')

    # Image Frame resolutions in the data are like:
    # - FRONT         (1280, 1920, 3)
    # - FRONT_LEFT    (1280, 1920, 3)
    # - FRONT_RIGHT   (886, 1920, 3)
    # - SIDE_LEFT     (1280, 1920, 3)
    # - SIDE_RIGHT    (886, 1920, 3)
    # Pretend all resolutions are (1280, 1920, 3) and resize to fit.

    width_per_image = args.resolution[0] // 3
    height_per_image = args.resolution[1] // 3
    placeholder_image = np.zeros((height_per_image, width_per_image, 3), dtype=np.uint8)
    start_time = 0

    dataset = tf.data.TFRecordDataset(file, compression_type='')
    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        if len(frame.images) == 0:
            print(f'WARNING: No images in frame {i}')
            continue

        image_stats = np.copy(placeholder_image)
        image_stats = draw_stats(image_stats, frame)

        images = {
            'FRONT': placeholder_image,
            'FRONT_LEFT': placeholder_image,
            'FRONT_RIGHT': placeholder_image,
            'SIDE_LEFT': placeholder_image,
            'SIDE_RIGHT': placeholder_image,
        }
        for j, camera_image in enumerate(frame.images):
            image_np = tf.image.decode_image(camera_image.image).numpy()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            

            # if len(frame.camera_labels) > 0 and len(frame.camera_labels[j].labels) > 0:
            #     image_np = draw_boxes(image_np, frame.camera_labels[j].labels)

            image_np = cv2.resize(image_np, (width_per_image, height_per_image))
            images[open_dataset.CameraName.Name.Name(camera_image.name)] = image_np
            
        combined_image = np.concatenate((
            np.concatenate((images['FRONT_LEFT'], images['FRONT'], images['FRONT_RIGHT']), axis=1),
            np.concatenate((images['SIDE_LEFT'], image_stats, images['SIDE_RIGHT']), axis=1)
        ) , axis=0)

        elapsed_time = time.time() - start_time
        wait_time = 0.1 - elapsed_time
        if wait_time > 0:
            time.sleep(wait_time)
        fps = 20 # 1秒20楨
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        save_name = "./combined.mp4"
        video_writer = cv2.VideoWriter(save_name, fourcc, fps, (852, 1920))
        video_writer.write(combined_image)

        # cv2.imshow(file, combined_image)
        

        # if cv2.waitKey(1) & 0xff == 27: # ESC
        #     cv2.destroyAllWindows()
            # break

        start_time = time.time()
    

    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--initialdir', help='The starting directory when selecting tfrecord files')
    #parser.add_argument('-r', '--repeat', action='store_true', help='Loop indefinitely')
    args = parser.parse_args()

    # TODO: desired display resolution per image; the combination will be 3x with and 3x height
    args.resolution = (1920, 1280)

    return args


def main():
    args = parse_args()

    tfrecord_file = select_tfrecord_file(args.initialdir)
    if not tfrecord_file.endswith('.tfrecord'):
        print(f'WARNING: Select a tfrecord file instead of {tfrecord_file}')
        return

    play_dataset(args, tfrecord_file)


if __name__ == '__main__':
    main()
