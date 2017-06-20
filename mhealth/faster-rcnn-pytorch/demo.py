import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
import os
import pandas as pd
import sys

idssss = []
iter_ = int(sys.argv[1])
def test(img_id):
    import os
    # im_file = 'demo/11228.jpg'
    # im_file = '../VOCdevkit/VOC2007/JPEGImages/' + str(img_id)
    im_file = './demo_dc/'+str(img_id)
    # im_file = './demo_dc/'+str(img_id)
    # im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    # im_file = '/media/longc/Data/data/2DMOT2015/test/ETH-Crossing/img1/000100.jpg'
    image = cv2.imread(im_file)

    # model_file = '../VGGnet_fast_rcnn_iter_70000.h5'
    model_file = 'models/saved_model_EX10_A_Hash/faster_rcnn_10000.h5'
    

    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = detector.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))
	
    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    #cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show)
    if len(dets) != 0:
        # cv2.imshow('demo', im2show)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join('demo',img_id),im2show)
        idssss.append(img_id)

#from os import listdir
if __name__ == '__main__':
    # list = ['1810']
    folder = './demo_dc/'
    list_id = os.listdir(folder)
    #ids = pd.read_csv("./csvlist/Positive31.csv", header=None)
    #idss = ids[[1]]
    #idss = ids.iloc[:,0]
    #list_id = idss.values.flatten()
    start_ix = 10000+280*iter_
    stop_ix = 10000+280*(iter_+1)
    if stop_ix > len(list_id):
        stop_ix = len(list_id)
    for i, id in enumerate(list_id[start_ix:stop_ix]):
    #for i, id in enumerate(list_id[:5]):
        print id, i
        try:
            test(id)
        except:
            pass
    idsss = pd.Series(idssss)
    idsss.to_csv("./csvlist/Positive"+str(iter_)+".csv")
