import cv2
import time
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np



def model_restore_from_pb(pb_path,node_dict):
    config = tf.ConfigProto(log_device_placement=True,#印出目前的運算是使用CPU或GPU
                            allow_soft_placement=True,#當設備不存在時允許tf選擇一个存在且可用的設備來繼續執行程式
                            )
    config.gpu_options.allow_growth = True#依照程式執行所需要的資料來自動調整
    #config.gpu_options.per_process_gpu_memory_fraction = 0.6#手動限制GPU資源的使用
    sess = tf.Session(config=config)
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 匯入計算圖
    sess.run(tf.global_variables_initializer())
    for key,value in node_dict.items():
        node = sess.graph.get_tensor_by_name(value)
        node_dict[key] = node
    return sess,node_dict

def video_init(is_2_write=False,save_path=None):
    writer = None
    cap = cv2.VideoCapture(0)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#default 640x480
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # width = 480
    # height = 640
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    '''
    ref:https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
    FourCC is a 4-byte code used to specify the video codec. 
    The list of available codes can be found in fourcc.org. 
    It is platform dependent. The following codecs work fine for me.
    In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
    In Windows: DIVX (More to be tested and added)
    In OSX: MJPG (.mp4), DIVX (.avi), X264 (.mkv).
    FourCC code is passed as `cv.VideoWriter_fourcc('M','J','P','G')or cv.VideoWriter_fourcc(*'MJPG')` for MJPG.
    '''

    if is_2_write is True:
        #fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')
        #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if save_path is None:
            save_path = 'demo.avi'
        writer = cv2.VideoWriter(save_path, fourcc, 20, (int(width), int(height)))

    return cap,height,width,writer

def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    '''
    generate anchors.
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    '''
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes

def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    '''
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]
    :param anchors: numpy array with shape [batch, num_anchors, 4]
    :param raw_outputs: numpy array with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
    '''
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick]

def mask_detection(is_2_write=False,save_path=None):
    #----var
    pb_path = "face_mask_detection.pb"
    node_dict = {'input':'data_1:0',
                 'detection_bboxes':'loc_branch_concat_1/concat:0',
                 'detection_scores':'cls_branch_concat_1/concat:0'}
    conf_thresh = 0.5
    iou_thresh = 0.4
    frame_count = 0
    FPS = "0"
    #====anchors config
    feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    anchor_ratios = [[1, 0.62, 0.42]] * 5
    id2class = {0: 'Mask', 1: 'NoMask'}

    #----video streaming init
    cap, height, width, writer = video_init(is_2_write=is_2_write,save_path=save_path)

    #----model init

    #====generate anchors
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
    # for inference , the batch size is 1, the model output shape is [1, N, 4],
    # so we expand dim for anchors to [1, anchor_num, 4]
    anchors_exp = np.expand_dims(anchors, axis=0)

    #====model restore from pb file
    sess,node_dict = model_restore_from_pb(pb_path, node_dict)
    tf_input = node_dict['input']
    model_shape = tf_input.shape#[N,H,W,C]
    print("model_shape = ", model_shape)
    detection_bboxes = node_dict['detection_bboxes']
    detection_scores = node_dict['detection_scores']

    while (cap.isOpened()):

        #----get image
        ret, img = cap.read()

        if ret:
            #----image processing
            img_resized = cv2.resize(img, (model_shape[2], model_shape[1]))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_resized = img_resized.astype('float32')
            img_resized /= 255

            #----mask detection
            y_bboxes_output, y_cls_output = sess.run([detection_bboxes, detection_scores],
                                                     feed_dict={tf_input: np.expand_dims(img_resized, axis=0)})

            #remove the batch dimension, for batch is always 1 for inference.
            y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
            y_cls = y_cls_output[0]
            # To speed up, do single class NMS, not multiple classes NMS.
            bbox_max_scores = np.max(y_cls, axis=1)
            bbox_max_score_classes = np.argmax(y_cls, axis=1)

            # keep_idx is the alive bounding box after nms.
            keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                         bbox_max_scores,
                                                         conf_thresh=conf_thresh,
                                                         iou_thresh=iou_thresh,
                                                         )
            #====draw bounding box
            for idx in keep_idxs:
                conf = float(bbox_max_scores[idx])
                class_id = bbox_max_score_classes[idx]
                bbox = y_bboxes[idx]
                # clip the coordinate, avoid the value exceed the image boundary.
                xmin = max(0, int(bbox[0] * width))
                ymin = max(0, int(bbox[1] * height))
                xmax = min(int(bbox[2] * width), width)
                ymax = min(int(bbox[3] * height), height)

                if class_id == 0:
                    color = (0, 255, 0)  # (B,G,R)
                else:
                    color = (0, 0, 255)  # (B,G,R)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

            #----FPS count
            if frame_count == 0:
                t_start = time.time()
            frame_count += 1
            if frame_count >= 10:
                FPS = "FPS=%1f" % (10 / (time.time() - t_start))
                frame_count = 0
            #cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            #----image display
            cv2.imshow("demo by JohnnyAI", img)

            #----image writing
            if writer is not None:
                writer.write(img)

            #----'q' key pressed?
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("get image failed")
            break

    #----release
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    save_path = r"E:\demo.avi"
    mask_detection(is_2_write=False, save_path=save_path)