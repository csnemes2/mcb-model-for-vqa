import json, operator, re, time, pickle, os
import numpy as np
from IPython import embed
from cnn import *

from train2014_vqa_mybuild2.trainconf import *


def create_annotations_result():
    """
    annotations['annotations']
        answers : list of 10 answers (answer, answer_confidence, answer_id)
        image_id
        question_id
    questions['questions']
        image_id
        question
        question_id
    annotations_result(train only answers with confidence 'yes')
        image_id_list : list of image ids (with original image_id)
        question_list : list of questions (with sentence)
        answer_list : list of answers (with index)
    q_wordtoix
    q_ixtoword
    a_wordtoix
    a_ixtoword
    """

    annotations = json.load(open(annotations_path, 'rb'))['annotations'][:3*training_img_num]
    questions = json.load(open(questions_path, 'rb'))['questions'][:3*training_img_num]
    image_id_list, question_list, answer_list = [], [], []

    q_dic, q_word2ix, q_ix2word = {}, {}, {}
    a_dic, a_word2ix, a_ix2word = {}, {}, {}




    # (1) create wordtoix, ixtoword for answers
    answer_type_dic = {}
    for dic in annotations:
        for a in dic['answers']:
            if a['answer_confidence'] == 'yes':
                ans = a['answer']
                if ans in a_dic: a_dic[ans] += 1
                else: a_dic[ans] = 1
                ans_type = dic['answer_type']
                if ans_type in answer_type_dic:
                    answer_type_dic[ans_type] += 1
                else:
                    answer_type_dic[ans_type] = 1
    a_dic = sorted(a_dic.items(), key=operator.itemgetter(1))
    a_dic.reverse()
    print "Answer Type Dic"
    print answer_type_dic
    print "The number of words in answers is %d. Select only %d words."%(len(a_dic), ans_candi_num)
    print "Top 20 frequent answers : "
    for i in range(20):
        print a_dic[i][0], a_dic[i][1]
    for i in range(ans_candi_num):
        a_word2ix[a_dic[i][0]] = i
        a_ix2word[i] = a_dic[i][0]
    print "Answer word2ix, ix2word created. Threshold is %d"%(a_dic[ans_candi_num][1])

    # (2) create wordtoix, ixtoword for questions
    p = re.compile('[\w]+')
    q_len_dic, q_freq_dic = {}, {}
    for dic in questions:
        q_dic[(dic['image_id'], dic['question_id'])] = dic['question']
        q_words = p.findall(dic['question'].lower())
        for qw in q_words:
            if qw in q_freq_dic : q_freq_dic[qw] += 1
            else: q_freq_dic[qw] = 1
        q_len_key = 10*int(len(q_words)/10)
        if q_len_key in q_len_dic : q_len_dic[q_len_key] += 1
        else : q_len_dic[q_len_key] = 1
    print "Length of questions"
    for q_len_key in q_len_dic:
        print "%d ~ %d\t: %d" %(q_len_key, q_len_key+10, q_len_dic[q_len_key])
    print "Total\t: %d" %(sum(q_len_dic.values()))
    q_freq_dic = sorted(q_freq_dic.items(), key=operator.itemgetter(1))
    q_freq_dic.reverse()
    print "The number of words in questions is %d. Select only %d words."%(len(q_freq_dic), word_num)
    q_word2ix['?'] = 0
    q_ix2word[0] = '?'
    for i in range(1, word_num):
        q_word2ix[q_freq_dic[i-1][0]] = i
        q_ix2word[i] = q_freq_dic[i-1][0]
    print "Question word2ix, ix2word created. Threshold is %d"%(q_freq_dic[word_num][1])

    # (3) create annotations_result
    num = 0
    answer_freq_dic = {}
    for dic in annotations:
        q = q_dic[(dic['image_id'], dic['question_id'])]
        i = 0
	for a in dic['answers']:
            if a['answer_confidence'] == 'yes' and a['answer'] in a_word2ix:
                image_id_list.append(dic['image_id'])
                question_list.append(q)
                answer_list.append(a_word2ix[a['answer']])
		i += 1
	if i==0: num+=1
    print "All (img, question, answer) pairs are %d"%(len(image_id_list))
    pickle.dump({'image_ids' : image_id_list,
        'questions' : question_list,
        'answers' : answer_list},
        open(annotations_result_path, 'wb'))
    print "Success to save Annotation results"

    pickle.dump(q_word2ix, open(worddic_path+'q_word2ix', 'wb'))
    pickle.dump(q_ix2word, open(worddic_path+'q_ix2word', 'wb'))
    pickle.dump(a_word2ix, open(worddic_path+'a_word2ix', 'wb'))
    pickle.dump(a_ix2word, open(worddic_path+'a_ix2word', 'wb'))
    print "Success to save Worddics"

    # (4) Create image features
    # If you run this seperatly, load image_id_list
    #image_id_list = pickle.load(open(annotations_result_path, 'rb'))['image_ids']
    unique_image_ids = list(set(image_id_list))

    unique_images = map(lambda x: \
            os.path.join(image_path+("%12s"%str(x)).replace(" ","0")+".jpg"),
		unique_image_ids)
    imgix2featix = {}
    for i in range(len(unique_images)):
        imgix2featix[unique_image_ids[i]] = i
    pickle.dump(imgix2featix, open(imgix2featix_path, 'wb'))

    cnn = CNN(model=res_model, deploy=res_deploy, width=224, height=224)
    print "CNN class was created"
    print len(unique_images)

    for dic in layer_set.values():
        layers = dic['layers']
        layer_size = dic['layer_size']
        feat_path = dic['feat_path']
        if not os.path.exists(feat_path):
            feats = cnn.get_features(unique_images, layers=layers, layer_sizes=layer_size)
            print feats.shape
            #import ipdb;ipdb.set_trace()
            np.save(feat_path, feats)
    print "Success to save features"

if __name__ == '__main__':
    create_annotations_result()
