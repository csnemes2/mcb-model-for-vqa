training_img_num = 4000
validation_img_num = 500
word_num = 1000
ans_candi_num = 1000

# Caffe model : ResNet
res_model = '/home/csn/caffe/models/resnet/ResNet-101-model.caffemodel'
res_deploy = '/home/csn/caffe/models/resnet/ResNet-101-deploy.prototxt'

###############  INPUT ####################
annotations_path = 'train2014_json/mscoco_train2014_annotations.json'
questions_path = 'train2014_json/OpenEnded_mscoco_train2014_questions.json'
image_path = 'train2014/COCO_train2014_'

###############  OUTPUT ####################
worddic_path = 'train2014_vqa_mybuild1/'

layer_set = {
        'default' : {'layers' : 'pool5', 'layer_size' : [2048], 'feat_path' : worddic_path+'/features/train_res_feat.npy'},
        '4b' : {'layers' : 'res4b22_branch2c', 'layer_size' : [1024, 14, 14], 'feat_path' : worddic_path+'/features/train_res4b_feat.npy'}
        }
annotations_result_path = worddic_path+'/train_annotations_result'
imgix2featix_path       = worddic_path+'/img2feat'