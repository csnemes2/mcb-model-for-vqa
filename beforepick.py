from optparse import OptionParser
from cnn import *
import urllib2

# Caffe model : ResNet
res_model = '/home/csn/caffe/models/resnet/ResNet-101-model.caffemodel'
res_deploy = '/home/csn/caffe/models/resnet/ResNet-101-deploy.prototxt'

layer_set = {
        'default' : {'layers' : 'pool5', 'layer_size' : [2048], 'feat_path' : 'mytest.npy'}#,
        #'4b' : {'layers' : 'res4b22_branch2c', 'layer_size' : [1024, 14, 14], 'feat_path' : worddic_path+'/features/val_res4b_feat.npy'}
        }

if __name__ == '__main__':

    parser = OptionParser(usage="usage: %prog [options] web_link_to_an_image",
                          version="%prog 1.0")

    options, args = parser.parse_args()

    if len(args) != 1:
        parser.error("wrong number of arguments")



    cnn = CNN(model=res_model, deploy=res_deploy, width=224, height=224)

    mylink=(args[0])
    myfile="mytest.jpg"

    print "loading started"

    resp = urllib2.urlopen(mylink)
    with open(myfile, 'w') as f:
        f.write(resp.read())
    print "loading finished"

    unique_images=["mytest.jpg"]

    for dic in layer_set.values():

        layers = dic['layers']
        layer_size = dic['layer_size']
        feat_path = dic['feat_path']
        print unique_images
        feats = cnn.get_features(unique_images,
            layers=layers, layer_sizes=layer_size)
        print np.shape(feats)
        np.save(feat_path, feats)
    print "Success to save features"
