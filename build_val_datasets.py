import json, operator, re, time, pickle, os
from cnn import *


#from train2014_vqa_mybuild1.valconf import *
from train2014_vqa_mybuild2.valconf import *


def create_annotations_result():
    """
    annotations['annotations']
        answers : list of 10 answers (answer, answer_confidence, answer_id)
        image_id
        question_id
    questions['questions']
        image_id
        question
        question_id
    annotations_result(train only answers with confidence 'yes')
        image_id_list : list of image ids (with original index)
        question_list : list of questions (with sentence)
        question_id_list : list of question ids (with index)
    """
    annotations = json.load(open(annotations_path, 'rb'))['annotations'][:validation_img_num*3]
    questions = json.load(open(questions_path, 'rb'))['questions'][:validation_img_num*3]
    json.dump({'annotations':annotations}, open(selected_annotations_path, 'wb'))
    json.dump({'questions':questions,
		'info' : '', 'task_type' : '', 'data_type' : '',
		'data_subtype' : '', 'license' : ''},
		open(selected_questions_path, 'wb'))
    #return

    image_id_list, question_list, question_id_list, answer_list = [], [], [], []

    a_word2ix = pickle.load(open(worddic_path + 'a_word2ix', 'rb'))

    # (1) create annotations_result
    q_dic = {}
    for dic in questions:
        q_dic[(dic['image_id'], dic['question_id'])] = dic['question']

    for dic in annotations:
        q = q_dic[(dic['image_id'], dic['question_id'])]
        image_id_list.append(dic['image_id'])
        question_list.append(q)
        question_id_list.append(dic['question_id'])

    print "All (img, question, answer) pairs are %d"%(len(image_id_list))
    pickle.dump({
        'image_ids' : image_id_list,
        'questions' : question_list,
	'question_ids' : question_id_list},
        open(annotations_result_path, 'wb'))
    print "Success to save Annotation results"


    # (2) Create image features
    # If you run this seperatly, load image_id_list
    # image_id_list = pickle.load(open(annotations_result_path, 'rb'))['image_ids']
    unique_image_ids = list(set(image_id_list))
    unique_images = map(lambda x: \
            os.path.join(image_path+("%12s"%str(x)).replace(" ","0")+".jpg"),
	    unique_image_ids)
    print "Unique images are %d" %(len(unique_images))
    imgix2featix = {}
    for i in range(len(unique_images)):
        imgix2featix[unique_image_ids[i]] = i
    pickle.dump(imgix2featix, open(imgix2featix_path, 'wb'))

    cnn = CNN(model=res_model, deploy=res_deploy, width=224, height=224)


    print unique_images

    for dic in layer_set.values():

        layers = dic['layers']
        layer_size = dic['layer_size']
        feat_path = dic['feat_path']
        if not os.path.exists(feat_path):
            feats = cnn.get_features(unique_images,
                layers=layers, layer_sizes=layer_size)
            np.save(feat_path, feats)
    print "Success to save features"


if __name__ == '__main__' :
    create_annotations_result()

