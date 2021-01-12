from mmdet.apis import init_detector, inference_detector, save_result_pyplot
import mmcv

# config_file = '../configs/retinanet/retinanet_r50_fpn_1x_coco_analysis.py'
config_file = '../configs/retinanet/retinanet_r50_fpn_1x_coco_analysis_scale2.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'


model = init_detector(config_file, checkpoint_file, device='cuda:0')


# test a single image
img = '../../data/coco/train2017/000000000009.jpg'
result = inference_detector(model, img)

# show the results
save_result_pyplot(model, img, result, file_path='analysis_images/tmp_scale2.png')
