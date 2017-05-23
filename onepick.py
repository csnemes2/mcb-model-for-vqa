import json, operator, re, time, pickle
import tensorflow as tf, numpy as np
from tensorflow.python.framework import ops
from config import Config
from keras.preprocessing import sequence
from optparse import OptionParser
import webbrowser
import os



#
# backend configs
#
# onepick img link
#   show image
#   forever
#       prompt question
#       answer question - several backends possible


def answer(question,sess):
    ans = "yeah"

    return ans

def onepick(imglink):
    p = re.compile('[\w]+')
    webbrowser.open(imglink)
    os.system('python beforepick.py %s'%args[0])
    feats = np.load('mytest.npy')

    curr_image_feat = feats

    config = Config(config_name = 'mcb')
    config.val_batch_size = 1
    config.validation_num = 1

    q_word2ix = pickle.load(open(config.worddic_path + 'q_word2ix', 'rb'))
    a_ix2word = pickle.load(open(config.worddic_path + 'a_ix2word', 'rb'))

    from_idx = range(0, config.validation_num, config.val_batch_size)
    to_idx = range(config.val_batch_size, config.val_batch_size+config.validation_num, config.val_batch_size)

    epoch =9
    print    "*** Loading model for Epoch %d ***" % (epoch)
    sess = tf.Session()
    model = config.vqamodel(
        batch_size=config.val_batch_size,
        feature_dim=config.feature_dim,
        proj_dim=config.proj_dim,
        word_num=config.word_num,
        embed_dim=config.embed_dim,
        ans_candi_num=config.ans_candi_num,
        n_lstm_steps=config.n_lstm_steps)

    sess.run(tf.initialize_all_variables())
    image_feat, question, max_prob_words = model.solver()
    saver = tf.train.Saver(max_to_keep=50)
    saver.restore(sess, config.model_path + 'model-%d' % (epoch))


    while True:
        myq = raw_input("Give a question! >> ")

        curr_question = [myq]
        curr_question = map(lambda ques:
                            [q_word2ix[word] for word in p.findall(ques.lower())
                             if word in q_word2ix],
                            curr_question)
        print curr_question[0]

        curr_question = np.array(sequence.pad_sequences(
            curr_question, padding='post', maxlen=config.n_lstm_steps))

        answer_ids = sess.run(max_prob_words,
                              feed_dict={image_feat: curr_image_feat,
                                         question: curr_question})

        answers = map(lambda ix: a_ix2word[ix], answer_ids)

        print "answer is >> " + str(answers[0])

    sess.close()

if __name__ == '__main__':

    parser = OptionParser(usage="usage: %prog [options] web_link_to_an_image",
                          version="%prog 1.0")

    options, args = parser.parse_args()

    if len(args) != 1:
        parser.error("wrong number of arguments")

    onepick(args[0])

