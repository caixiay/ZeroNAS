import classifier2
import pickle
import util
import argparse
import torch
from torch.autograd import Variable
import model_retrain

import numpy as np
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import torch
import pickle
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle, islice
import scipy.io as sio
import util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/home/ycx/cvpr18xian/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=60, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='/home/ycx/cvpr18xian/darts_gan/trained_models/cub_nas.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')


opt = parser.parse_args()
print(opt)

def plot_embedding_new(data, label_color, classes, title):
    colors = np.array(
        list(islice(cycle(['red', 'blue', 'green', 'coral', 'gray', 'orange', 'purple', 'pink', 'yellow', 'cyan',
                           'greenyellow', 'fuchsia', 'dodgerblue', 'plum', 'deeppink', 'palegreen', 'lime', 'olive',
                           'black',
                           'teal']), int(np.unique(label_color).shape[0]))))
    mapped_label = torch.LongTensor(label_color.size())
    for i in range(classes.size(0)):
        mapped_label[label_color == classes[i]] = i
    fig = plt.figure()
    plt.axis('off')
    plt.scatter(data[:, 0], data[:, 1], marker='o', s=20, color=colors[mapped_label])

    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label_text[i]),
    #              color=colors[label_color[i]],
    #              fontdict={'weight': 'bold', 'size': 9})
    return fig

def generate_syn_feature(netG, classes, attribute, num):
    print('classes', classes)
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
#cub
# genotype_G = [('fc_relu', 0), ('fc_lrelu', 2), ('fc_relu', 0), ('fc_relu', 3), ('fc_relu', 0), ('fc_relu', 4), ('fc_relu', 0), ('fc_relu', 5), ('fc_relu', 5), ('fc_relu', 6)]
# genotype_D = [('fc_relu', 0), ('fc_lrelu', 2), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_relu', 3), ('fc_relu', 4)]
#flo
# genotype_G = [('fc_lrelu', 2), ('fc_lrelu', 0), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1)]
# genotype_D = [('fc_lrelu', 2), ('fc_lrelu', 0), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_relu', 4), ('fc_relu', 1), ('fc_lrelu', 5), ('fc_relu', 4), ('fc_lrelu', 1), ('fc_relu', 3)]
#awa
genotype_G = [('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_relu', 6), ('fc_relu', 5)]
genotype_D = [('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 3), ('fc_lrelu', 2), ('fc_lrelu', 4), ('fc_lrelu', 3), ('fc_lrelu', 4), ('fc_lrelu', 5), ('fc_lrelu', 6), ('fc_lrelu', 4)]
opt.netG = '%s/%s_nas.pth' % ('/home/ycx/cvpr18xian/darts_gan/trained_models', opt.dataset)
netG = model_retrain.NetworkRetrain(opt, 'g', 5, genotype_G)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
if opt.cuda:
    netG.cuda()
unseen_classes = data.unseenclasses
syn_feature, syn_label = generate_syn_feature(netG, unseen_classes, data.attribute, opt.syn_num)
cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
acc = cls.acc
print('unseen class accuracy= ', acc)

with open('/home/ycx/cvpr18xian/data/%s/%s_syn_feature_nas1.pickle'%(opt.dataset, opt.dataset), 'wb') as f:
    pickle.dump(syn_feature, f)
with open('/home/ycx/cvpr18xian/data/%s/%s_syn_label_nas1.pickle'%(opt.dataset, opt.dataset), 'wb') as f:
    pickle.dump(syn_label, f)

# with open('/home/ycx/cvpr18xian/data/%s/%s_syn_feature_nas.pickle'%(opt.dataset, opt.dataset), 'rb') as f:
#     syn_feature = pickle.load(f, encoding='bytes')
# with open('/home/ycx/cvpr18xian/data/%s/%s_syn_label_nas.pickle'%(opt.dataset, opt.dataset), 'rb') as f:
#     syn_label = pickle.load(f, encoding='bytes')

# print('Computing t-SNE embedding')
# tsne = TSNE(n_components=2, init='pca', random_state=0)
# result = tsne.fit_transform(syn_feature)
# fig = plot_embedding_new(result, syn_label, unseen_classes, 'Unseen')
# fig.savefig('/home/ycx/cvpr18xian/darts_gan/trained_models/'+opt.dataset+'_nas.pdf')
# plt.show(fig)


