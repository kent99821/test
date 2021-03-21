import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
import time
import requests
import time
import json

# seat
#---------------------
#seat=[[0,0],[1307,480],[952,479],[561,480],[229,482],[1188,256],[943,268],[611,269],[350,262],[1104,24],[881,124],[630,126],[439,34],[977,37],[818,39],[649,44],[498,48]]
seat=[[1307,480],[952,479],[561,480],[229,482],[1188,256],[943,268],[611,269],[350,262],[1104,24],[881,124],[630,126],[439,34],[977,37],[818,39],[649,44],[498,48]]
seat_taken=[]
maybe_seat_taken=[]
#seat_total=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
#seat_total=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
seat_total=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
seat_total_n=['00','01','02','03','10','11','12','13','20','21','22','23','30','31','32','33']

#print(seat[0])
# print(seat[1])
# print(seat[2])
# seat1=[1307,480]
# seat2=[952,479]
# seat3=[561,480]
# seat4=[229,482]
# seat5=[1188,256]
# seat6=[943,268]
# seat7=[611,269]
# seat8=[350,262]
# seat9=[1104,124]
# seat10=[881,124]
# seat11=[630,126]
# seat12=[439,134]
# seat13=[977,37]
# seat14=[818,39]
# seat15=[649,44]
# seat16=[498,48]

#---------------------

class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file + '\n')    # -------- #
        self.saver = tf.train.Saver()
        #-------#
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./data/weights'))

        #______________________
        self.leave=[]



    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            # cv2.putText(
            #     img, result[i][0] + ' : %.2f' % result[i][5],
            #     (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #     (0, 0, 0), 1, lineType)
            cv2.putText(
                img, result[i][0],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 1, lineType)

        return img

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)
        
        global maybe_seat_taken
        global seat_taken
        maybe_seat_taken.clear()
        seat_taken.clear()
        self.leave.clear()
        # __________________________#
        for i in range(len(result)):  #第几个类
            for j in range(len(result[i])):  #类的第几个元素
                    #print('类: ' + str(result[i][0]) + 'x轴： ' + str(result[i][1]) + '  ' + 'y轴： ' + str(result[i][2]))
                    zuobiao=[result[i][1],result[i][2]]   #类的中心坐标
                    min_distance=999999
                    zuobiao=np.array(zuobiao)
                    #print(zuobiao)
                    final_seat=0
                    for k in range(16):  #遍历16个座位，得出类所在的座位
                        every_seat=seat[k]   #每一次
                        #print(every_seat)
                        every_seat=np.array(every_seat)
                        every_distance=np.linalg.norm(zuobiao - every_seat)
                        #print(k)
                        if every_distance < min_distance:
                            min_distance=every_distance
                            #final_seat=k+1  #第几个座位的意思
                            final_seat=k
                            #print(k+1)
            #print("类： " + str(result[i][0]) + "在座位： " + str(final_seat)+'\n' )
            if str(result[i][0]) == "seat_taken":
                #print("***************座位： " + str(final_seat) +'  '+"已经有人坐***************\n")
                seat_taken.append(final_seat)
            else:
                maybe_seat_taken.append(final_seat)
        print('最小距离为： '+str(min_distance))

        # 变换座位编号
        def convert_seat(seat):
            if 0 in seat:
                seat[seat.index(0)] = '00'
            if 1 in seat:
                seat[seat.index(1)] = '01'
            if 2 in seat:
                seat[seat.index(2)] = '10'
            if 3 in seat:
                seat[seat.index(3)] = '11'
            if 4 in seat:
                seat[seat.index(4)] = '02'
            if 5 in seat:
                seat[seat.index(5)] = '03'
            if 6 in seat:
                seat[seat.index(6)] = '12'
            if 7 in seat:
                seat[seat.index(7)] = '13'
            if 8 in seat:
                seat[seat.index(8)] = '20'
            if 9 in seat:
                seat[seat.index(9)] = '21'
            if 10 in seat:
                seat[seat.index(10)] = '30'
            if 11 in seat:
                seat[seat.index(11)] = '31'
            if 12 in seat:
                seat[seat.index(12)] = '22'
            if 13 in seat:
                seat[seat.index(13)] = '23'
            if 14 in seat:
                seat[seat.index(14)] = '32'
            if 15 in seat:
                seat[seat.index(15)] = '33'
            return seat

        maybe_seat_taken = convert_seat(maybe_seat_taken)
        seat_taken.sort(key=int)
        seat_taken = convert_seat(seat_taken)
        print('已经有人坐的座位： '+ str(seat_taken))
        #print(maybe_seat_taken)
        self.leave=list(set(maybe_seat_taken).difference(set(seat_taken)))  #取差集，在maybe_seat_taken列表但不在seat_taken
        self.leave.sort(key=int)
        self.leave = convert_seat(self.leave)
        print('离开的座位： ' + str(self.leave))

        self.seat_taken_and_leave=list(set(seat_taken).union(set(self.leave)))  #取并集，有人的 + 离开的
        self.seat_taken_and_leave.sort(key=int)
        #print('有人+离开的座位： ' + str(self.seat_taken_and_leave))

        self.empty_seat=list(set(seat_total_n).difference(set(self.seat_taken_and_leave)))  #空座位 总的 - 求和的
        self.empty_seat.sort(key=int)
        self.empty_seat = convert_seat(self.empty_seat)
        print('空座位： ' + str(self.empty_seat))

        data_send={
            "seat_taken":seat_taken,
            "leave":self.leave,
            "empty_seat":self.empty_seat
        }
        print('data_send:  ' + str(data_send) + '\n\n')
        
        ### send to server start ###
        def sendFuck(data):
            import requests
            import json
            url = 'http://193.112.63.186:3000/api/rec'
            j = json.dumps(data)
            requests.post(url,
                        json=[j])
        # sendFuck(data_send)    # send data
        ### send to server end ###
        
        #print(result)
        #print(np.array(result).shape)  #result的维度形状
        return result
        # __________________________result是一个列表：类 x轴 y轴  宽 高 置信度



    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            #new_frame = np.rot90(frame, -1)
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            '''
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))
                '''
            if result is not None:
            	print(result)

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imnames, svnames, wait=0):
        detect_timer = Timer()

        for i in range(0,568):                #--------------------------迭代次数

            imname = imnames[i]
            #svname = svnames[i]

            image = cv2.imread(imname)

            detect_timer.tic()
            result = self.detect(image)
            detect_timer.toc()
            #print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))
            #print('平均fps: {:.3f}fps\n\n'.format(1/detect_timer.average_time))

            
            ### show image start ###
            img_test = self.draw_result(image, result)
            ih, iw = img_test.shape[0:2]
            ii = 0.7   #缩放系数
            img_test = cv2.resize(img_test, (int(ii*iw), int(ii*ih)))
            cv2.imshow('TESTING', img_test)
            if cv2.waitKey(1) & 0xFF == ord('q'): #按q退出
                cv2.destroyAllWindows()
                break
            ### show image end ###
            

            #time.sleep(0.05)
            #cv2.imwrite(svname, image)
            #print('OK:' + svname)
        #self.send_data.close()


def main():

    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weights', default="yolo-35000.data-00000-of-00001", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    detector = Detector(yolo, weight_file)

    # detect from camera
    #cm = 'http://admin:admin@192.168.1.11:8081'
    #cap = cv2.VideoCapture(1)
    #detector.camera_detector(cap)

    # detect from image file
    path = 'data/library/JPEGImages'  
    files = os.listdir(path)

    imnames = []
    svnames = []

    for img in files:
        i = 'data/library/JPEGImages'
        s = 'test'
        i = os.path.join(i, img)
        s = os.path.join(s, img)
        imnames.append(i)
        svnames.append(s)
		
    detector.image_detector(imnames, svnames)


if __name__ == '__main__':
    main()
