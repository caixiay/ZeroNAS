from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import classifier
import classifier2
import sys
import model
import torch.utils.data as Data
from plot import plot_genotype
import datetime
import dateutil
from dateutil import tz
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/home/ycx/cvpr18xian/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
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
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
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
parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                    help='weight decay for alpha')
parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--figure_dir', default='./output/net_plot', help='FLO')


opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
train_data = Data.TensorDataset(data.train_feature, data.train_label)
n_train = data.ntrain
split = n_train // 2
indices = list(range(n_train))
random.shuffle(indices)
train_index = indices[:split]
valid_index = indices[split:]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_index)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_index)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=opt.batch_size,
                                           sampler=train_sampler,
                                           num_workers=opt.workers,
                                           pin_memory=True)
valid_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=opt.batch_size,
                                           sampler=valid_sampler,
                                           num_workers=opt.workers,
                                           pin_memory=True)

# initialize generator and discriminator
# netG = model.MLP_G(opt)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
#     netG.load_state_dict(torch.load(opt.netG))
# print(netG)
netG_search = model.MLP_search(opt, 'g')
netD_search = model.MLP_search(opt, 'd')
print('netG_search', netG_search)
print('netD_search', netD_search)
# netD = model.MLP_CRITIC(opt)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    # netD.cuda()
    # netG.cuda()
    netG_search.cuda()
    netD_search.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
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
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def sample_darts(index):
    batch_feature, batch_label, batch_att = data.next_batch_darts(index, opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

# Optimizers
alpha_optimizer_G = torch.optim.Adam(
    netG_search.arch_parameters(),
    lr=opt.alpha_lr,
    weight_decay=opt.alpha_weight_decay
)

weight_optimizer_G = torch.optim.Adam(
    netG_search.parameters(),
    lr=opt.lr,
    weight_decay=opt.weight_decay
)

weight_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
    weight_optimizer_G,
    float(opt.epochs),
    eta_min=1e-3 * opt.lr
)

alpha_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
    alpha_optimizer_G,
    float(opt.epochs),
    eta_min=1e-3 * opt.alpha_lr
)

alpha_optimizer_D = torch.optim.Adam(
    netD_search.arch_parameters(),
    lr=opt.alpha_lr,
    weight_decay=opt.alpha_weight_decay
)

weight_optimizer_D = torch.optim.Adam(
    netD_search.parameters(),
    lr=opt.lr,
    weight_decay=opt.weight_decay
)

weight_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
    weight_optimizer_D,
    float(opt.epochs),
    eta_min=1e-3 * opt.lr
)

alpha_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
    alpha_optimizer_D,
    float(opt.epochs),
    eta_min=1e-3 * opt.alpha_lr
)
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
best_acc = 0
# Best genotype
best_genotype = None
accuracy_curve_list = []
print('start_time', timestamp)
for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    weight_scheduler_G.step()
    alpha_scheduler_G.step()
    weight_scheduler_D.step()
    alpha_scheduler_D.step()
    # print('netG_search.parameters', netG_search.edge_weights(), netG_search.operation_weights())
    # print('netD_search.parameters', netD_search.edge_weights(), netD_search.operation_weights())
    for i in range(0, data.ntrain//2, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD_search.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            # sample()
            sample_darts(valid_index)
            alpha_optimizer_D.zero_grad()
            # train with realG
            # sample a mini-batch
            input_resv_val = Variable(input_res)
            input_attv_val = Variable(input_att)
            input_label_val = input_label
            criticD_real = netD_search(input_resv_val, input_attv_val)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG_search(noisev, input_attv_val)
            # fake_norm = fake.data[0].norm()
            # sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD_search(fake.detach(), input_attv_val)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD_search, input_res, fake.data, input_att)
            gradient_penalty.backward()

            alpha_optimizer_D.step()

            sample_darts(train_index)
            weight_optimizer_D.zero_grad()
            # train with realG
            # sample a mini-batch
            input_resv_train = Variable(input_res)
            input_attv_train = Variable(input_att)
            input_label_train = input_label
            criticD_real = netD_search(input_resv_train, input_attv_train)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG_search(noisev, input_attv_train)
            # fake_norm = fake.data[0].norm()
            # sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD_search(fake.detach(), input_attv_train)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD_search, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            # nn.utils.clip_grad_norm(netD_search.parameters(), 5)
            weight_optimizer_D.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD_search.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        alpha_optimizer_G.zero_grad()
        # input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG_search(noisev, input_attv_val)
        criticG_fake = netD_search(fake, input_attv_val)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label_val))
        errG = G_cost + opt.cls_weight * c_errG
        errG.backward()
        alpha_optimizer_G.step()

        weight_optimizer_G.zero_grad()
        # input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG_search(noisev, input_attv_train)
        criticG_fake = netD_search(fake, input_attv_train)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label_train))
        errG = G_cost + opt.cls_weight*c_errG
        errG.backward()
        weight_optimizer_G.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, opt.nepoch, D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0], c_errG.data[0]))

    # evaluate the model, set G to evaluation mode
    netG_search.eval()
    # Generalized zero-shot learning
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG_search, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG_search, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls.acc
        accuracy_curve_list.append(acc)
        print('unseen class accuracy= ', acc)

    cur_genotype_G = netG_search.get_cur_genotype()
    cur_genotype_D = netD_search.get_cur_genotype()
    if best_acc < acc:
        best_acc = acc
        # best_genotype = cur_genotype_G + cur_genotype_D
        print('Best')
        print('generator', cur_genotype_G)
        print('discriminator', cur_genotype_D)
        # Plot the architecture picture
        if opt.figure_dir:
            plot_genotype('g',
                cur_genotype_G,
                file_name='test_G_' + str(epoch),
                figure_dir=opt.figure_dir,
                 #          '%s_%s_%s' % \
                 # (opt.figure_dir, opt.dataset, timestamp),
                save_figure=True
            )
            plot_genotype('d',
                cur_genotype_D,
                file_name='test_D_' + str(epoch),
                figure_dir=opt.figure_dir,
                 #          '%s_%s_%s' % \
                 # (opt.figure_dir, opt.dataset, timestamp),
                save_figure=True
            )
            print('Figure saved.')
    # reset G to training mode
    netG_search.train()
print('accuracy_curve_list', accuracy_curve_list)
print('best_acc', best_acc)
now1 = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp1 = now1.strftime('%Y_%m_%d_%H_%M_%S')
print('end_time', timestamp1)
