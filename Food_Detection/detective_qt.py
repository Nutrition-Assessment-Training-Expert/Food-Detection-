#-*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np

import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QStringListModel
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap

from multiprocessing import Process, Queue
import pandas as pd

form_window = uic.loadUiType('./main.ui')[0]
form_widget = uic.loadUiType('./sub.ui')[0]
change_len = pd.read_excel('./jaeryo/재료영어_한글.xlsx', index_col=0)
eng_list = list(change_len['eng'])
kor_list = list(change_len['kor'])

recipes = pd.read_excel('./jaeryo/recipe_jaeryo.xlsx', header=None, index_col=0)
file_path = []

Path_0 = './recipes/'
Path_1s = os.listdir(Path_0)
for Path_1 in Path_1s:
    Path_2s = (os.listdir(Path_0 + Path_1))
    for Path_2 in Path_2s:
        file_path += (glob.glob(Path_0 + Path_1 + '/' + Path_2 + '/*.txt'))

# producer
def detect(q1, q2, weights='yolov5s.pt', source='inference/images', conf=0.4 , output='inference/output',
                   img=640, iou=0.5, device='', view='store_true', save='store_true',
                   classes='+', agnostic='store_true', augment='store_true', update='store_true', save_img=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=f'{weights}', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=f'{source}', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=f'{output}', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=img, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=conf, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=iou, help='IOU threshold for NMS')
    parser.add_argument('--device', default=f'{device}', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action=f'{view}', help='display results')
    parser.add_argument('--save-txt', action=f'{save}', help='save results to *.txt')
    parser.add_argument('--classes', nargs=f'{classes}', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action=f'{agnostic}', help='class-agnostic NMS')
    parser.add_argument('--augment', action=f'{augment}', help='augmented inference')
    parser.add_argument('--update', action=f'{update}', help='update all models')
    opt = parser.parse_args()

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    source = opt.source
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    s = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        #여기까지 타임 재는 곳

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, im0 = path[i], im0s[i].copy()
            else:
                p, im0 = path, im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if names[int(c)] not in s:
                        s.append(names[int(c)]) # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (imqage with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
            q1.put(" ".join(s))
            q2.put(im0)


    if save_txt or save_img:
        print('Results savqed to %s' % Path(out))

    # print('Done. (%.3fs)' % (time.time() - t0))

class consumer1(QThread):
    renewal = pyqtSignal(str)
    def __init__(self, q1):
        super().__init__()
        self.q1 = q1

    def run(self):
        while True:
            if not self.q1.empty():
                data1 = q1.get()
                # 라벨에 표시되기까지 시간
                time.sleep(0.1)
                self.renewal.emit(data1)

class consumer2(QThread):
    changePixmap = pyqtSignal(QImage)
    def __init__(self, q2):
        super().__init__()
        self.q2 = q2

    def run(self):
        while True:
            if not self.q2.empty():
                rgbImage = cv2.cvtColor(q2.get(), cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                data2 = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(data2)

class Second(QWidget, form_widget):
    def __init__(self, parent=None):
        super(Second, self).__init__(parent)
        self.setupUi(self)
        self.init_UI()
        self.cmb_recipe_2.currentIndexChanged.connect(self.update_cmb_recipe_3)
        self.cmb_recipe_3.currentIndexChanged.connect(self.update_cmb_recipe)
        self.cmb_recipe.currentIndexChanged.connect(self.update_recipe)
        self.setFixedSize(1121,751)

    def init_UI(self):
        self.cmb_recipe_2.addItem("테마")
        self.path_1_list = []
        self.path_2_list = []
        for recipe in recipe_list:
            for file in file_path:
                path_0 = file
                temp_path = path_0[path_0.find('/', 2) + 1:]
                temp_path_2 = temp_path[temp_path.find('/', 1) + 1:]
                path_1 = temp_path[:temp_path.find(temp_path_2) - 1]
                path_3 = temp_path_2[temp_path_2.find('\\', 1) + 1:-4]
                path_2 = temp_path_2[:temp_path_2.find(path_3) - 1]
                if path_3 == recipe:
                    if path_1 not in self.path_1_list:
                        self.path_1_list.append(path_1)
                    if path_2 not in self.path_2_list:
                        self.path_2_list.append(path_2)

        for path_1_name in self.path_1_list:
            self.cmb_recipe_2.addItem(path_1_name)

    def update_cmb_recipe_3(self):
        if self.cmb_recipe_2.currentText() != "테마":
            if self.cmb_recipe_3.count() >= 1:
                for i in range(self.cmb_recipe_3.count() - 1):
                    self.cmb_recipe_3.removeItem(1)
                self.cmb_recipe_3.removeItem(0)
            self.cmb_recipe_3.addItem("카테고리")
            if self.cmb_recipe.count() >= 1:
                for i in range(self.cmb_recipe.count() - 1):
                    self.cmb_recipe.removeItem(1)
                self.cmb_recipe.removeItem(0)
            self.cmb_recipe.addItem("레시피를 선택해주세요.")
            for path_2_cand in os.listdir(f'./recipes/{self.cmb_recipe_2.currentText()}'):
                for path_2_name in self.path_2_list:
                    if path_2_name == path_2_cand:
                        self.cmb_recipe_3.addItem(path_2_name)

    def update_cmb_recipe(self):
        if self.cmb_recipe_3.currentText() != "카테고리":
            if self.cmb_recipe.count() >= 1:
                for i in range(self.cmb_recipe.count() - 1):
                    self.cmb_recipe.removeItem(1)
                self.cmb_recipe.removeItem(0)
            self.cmb_recipe.addItem("레시피를 선택해주세요.")
            for path_3 in recipe_list:
                try:
                    with open(f'./recipes/{self.cmb_recipe_2.currentText()}/{self.cmb_recipe_3.currentText()}/{path_3}.txt', 'r', encoding='cp949') as _:
                        _ = _
                    self.cmb_recipe.addItem(path_3)
                except:
                    pass

    def update_recipe(self):
        if self.cmb_recipe.currentText() != "" and self.cmb_recipe_3.currentText() != "카테고리" and self.cmb_recipe.currentText() != "레시피를 선택해주세요." and self.cmb_recipe_3.currentText() != "":
            try:
                with open(f'./recipes/{self.cmb_recipe_2.currentText()}/{self.cmb_recipe_3.currentText()}/{self.cmb_recipe.currentText()}.txt', 'r', encoding='cp949') as f:
                    data = f.read()
                    self.tb_recipe.setPlainText('가지고 있는 재료 : '+ ", ".join(interdict[self.cmb_recipe.currentText()])+'\n'+'\n'+
                                                '부족한 재료 : '+ ", ".join(deferdict[self.cmb_recipe.currentText()])+'\n'+'\n'+
                                                '*****조리법*****'+'\n'+'\n'+
                                                data)
                try:
                    plt.imread(f'./recipes/{self.cmb_recipe_2.currentText()}/{self.cmb_recipe_3.currentText()}/{self.cmb_recipe.currentText()}.jpg')
                    self.lbl_image.setPixmap(QPixmap(f'./recipes/{self.cmb_recipe_2.currentText()}/{self.cmb_recipe_3.currentText()}/{self.cmb_recipe.currentText()}.jpg'))
                except FileNotFoundError:
                    self.lbl_image.setPixmap(QPixmap(f'./recipes/{self.cmb_recipe_2.currentText()}/{self.cmb_recipe_3.currentText()}/{self.cmb_recipe.currentText()}.png'))
                if len(self.cmb_recipe.currentText()) < 18:
                    self.lbl_title.setText(self.cmb_recipe.currentText())
                    self.lbl_title_2.setText('')
                else:
                    x = self.cmb_recipe.currentText()
                    reverse_x = x[::-1]
                    order_x_1 = reverse_x[-18:][reverse_x[-18:].find(' '):][::-1][:-1]
                    order_x_2 = x.replace(order_x_1 + ' ', '')
                    self.lbl_title.setText(order_x_1)
                    self.lbl_title_2.setText(order_x_2)
            except:
                pass

class Exam(QWidget, form_window):
    def __init__(self, q1, q2):
        super().__init__()
        self.setupUi(self)
        self.init_UI()
        self.consumer1 = consumer1(q1)
        self.consumer1.renewal.connect(self.add_checkbox)
        self.consumer1.start()
        self.consumer2 = consumer2(q2)
        self.consumer2.changePixmap.connect(self.setImage)
        self.consumer2.start()
        self.btn_select.clicked.connect(self.select_jaeryo)
        self.btn_add.clicked.connect(self.add_jaeryo)
        self.btn_recipe.clicked.connect(self.search_recipes)
        self.setFixedSize(1141,751)


    def init_UI(self):
        self.jaeryo_list = []
        self.selected_list = []
        self.checked_list = []
        jaeryo_names = []
        for i in range(len(recipes)):
            rec = list(recipes.iloc[i])
            jaeryo_names = jaeryo_names + rec
        jaeryo_names = list(set(jaeryo_names) | set(kor_list))
        if np.nan in jaeryo_names:
            nan_idx = jaeryo_names.index(np.nan)
            jaeryo_names.pop(nan_idx)
        jaeryo_names = sorted(jaeryo_names)
        for jaeryo in jaeryo_names:
            self.cb_jaeryo.addItem(jaeryo)

    def Second_Window(self):
        self.second = Second()
        self.second.show()

    @pyqtSlot(str)
    def add_checkbox(self, data1):
        jaeryos = data1.split(" ")
        if jaeryos != ['']:
            for jaeryo in jaeryos:
                idx = eng_list.index(jaeryo)
                jaeryo = kor_list[idx]
                if jaeryo not in self.jaeryo_list:
                    self.jaeryo_list.append(jaeryo)
                    globals()[jaeryo]=QCheckBox(jaeryo)
                    self.formLayout.addWidget(globals()[jaeryo])

    @pyqtSlot(QImage)
    def setImage(self, data2):
        self.lbl_image.setPixmap(QPixmap.fromImage(data2))

    def select_jaeryo(self):
        self.checked_list = []
        for jaeryo in self.jaeryo_list:
            check = globals()[jaeryo].isChecked()
            if check:
                self.checked_list.append(globals()[jaeryo].text())
        self.lbl_out.setText(", ".join(self.checked_list))

    def add_jaeryo(self):
        add = self.cb_jaeryo.currentText()
        self.jaeryo_list.append(add)
        globals()[add] = QCheckBox(add)
        self.formLayout.addWidget(globals()[add])

    def search_recipes(self):
        global recipe_list
        global interdict
        global deferdict
        recipe_list = []
        dict_recipe = {}
        interdict = {}
        deferdict = {}
        for i in range(len(recipes)):
            rec = list(recipes.iloc[i])
            intersection = (set(rec) & set(self.checked_list))
            len_intersection = len(intersection)
            if len_intersection >= 1:
                dict_recipe[recipes.iloc[i].name] = len_intersection
                interdict[recipes.iloc[i].name] = list(intersection)
                defferent = list(set(rec) - set(self.checked_list))
                if np.nan in defferent:
                    nan_idx = defferent.index(np.nan)
                    defferent.pop(nan_idx)
                deferdict[recipes.iloc[i].name] = defferent

        sorted_rec = sorted(dict_recipe.items(), key=lambda x: x[1], reverse=True)

        for i in range(len(sorted_rec)):
            recipe_list.append(sorted_rec[i][0])

        if recipe_list != []:
            self.Second_Window()

if __name__ == "__main__":
    q1 = Queue()
    q2 = Queue()

    # name: 프로세스 식별하려고 쓰는 이름
    # target : 프로세스
    # args : 프로세스가 받는 인자
    # daemon : 부모 프로세스가 꺼지면 자식도 꺼지게 할 것인지 여부
    p = Process(name="producer", target=detect, args=(q1, q2, './weights/best.pt', 0, 0.7, ), daemon=True)
    p.start()

    # source = 0, weights = , conf = 0.1


    # Main process
    app = QApplication(sys.argv)
    MainWindow = Exam(q1, q2)
    MainWindow.show()
    sys.exit(app.exec_())