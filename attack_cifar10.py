import torch
import os

from models.googlenet import GoogLeNet
from models.vgg import VGG
from models.mobilenetv2 import MobileNetV2
from models.resnet import ResNet18,ResNet50
from models.senet import SENet18
from utils import AverageMeter,accuracy,clamp
from torchvision import transforms,datasets,models
from Bag_trick.attack_method import *
from tqdm import tqdm
import torch.autograd
import torch.cuda
from ema import EMA

lower_limit,upper_limit=0,1
mean=(0.4914, 0.4822, 0.4465)
std=(0.2023, 0.1994, 0.2010)
normalize = transforms.Normalize(mean=mean,std=std)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean,std=std)
])

def loadModels(modelnames):
    models=[]
    for name in modelnames:
        if name=='VGG16':
            model=eval("VGG('VGG16')")
        else:
            model=eval(name+'()')
        ckpt = torch.load('./model_data/'+name+'_ckpt.t7')
        print(name+'\' acc :'+str(ckpt['acc']))
        model.load_state_dict(ckpt['net'])
        model = model.cuda()#.half()
        model.eval()
        models.append(nn.DataParallel(model))
    return models





def attack_pgd(models, X, y, epsilon=8/255, attack_iters=20):
    alpha = epsilon / attack_iters
    diversity_prob =0.7
    resize_rate= [1.25, 1.5, 1.75, 2]
    momentum = torch.zeros_like(X).detach().cuda()
    # ImageNet
    gaussian_kernel = torch.from_numpy(kernel_generation(len_kernel=args.len_kernel)).cuda()

    decay = 1
    # early stop pgd counter for each x
    # initialize perturbation
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta = torch.autograd.Variable(delta, requires_grad=True)
    # craft adversarial examples
    delta_lower_limit = lower_limit-X
    delta_uppper_limit = upper_limit-X
    ema = EMA(0.98,epsilon,delta_lower_limit,delta_uppper_limit)
    ema.register(delta)
    for _ in range(attack_iters):
        delta = torch.autograd.Variable(delta, requires_grad=True)
        loss_list=[]
        if args.DI:
            for j in range(len(resize_rate)):
                # with autocast():
                for model in models:
                    """DI"""
                    if args.NI:
                        output = model(normalize(input_diversity(X + delta+momentum*alpha, resize_rate=resize_rate[j],diversity_prob=diversity_prob)))
                    else :
                        output = model(normalize(input_diversity(X + delta, resize_rate=resize_rate[j], diversity_prob=diversity_prob)))
                    # output = model(normalize(X + delta))
                    #             # if use early stop pgd
                    loss = F.cross_entropy(output, y)
                    loss_list.append(loss)
        else:
            # with autocast():
            for model in models:
                if args.NI:
                    output = model(normalize(X + delta+momentum*alpha))
                else:
                    output = model(normalize(X + delta))
                #             # if use early stop pgd
                loss = F.cross_entropy(output, y)
                loss_list.append(loss)
        loss=torch.mean(torch.stack(loss_list))
        loss.backward()
        # scaler.scale(loss).backward()
        grad = delta.grad.detach()


        """TI"""
        if args.TI:
            grad = F.conv2d(grad, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            # grad = conv2d_same_padding(grad, gaussian_kernel, stride=1, padding=1, groups=3)
        """MI"""
        if args.MI or args.NI:
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * decay
            momentum = grad
        delta = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        ema.update(delta)
    adv_image = X+delta
    ema.apply_shadow(delta)
    adv_image_ema = X + delta
    ema.restore(delta)

    return adv_image,adv_image_ema




import json
from torch import nn
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MI',action='store_true', default=False)
    parser.add_argument('--TI',action='store_true', default=False)
    parser.add_argument('--DI',action='store_true', default=False)
    parser.add_argument('--NI',action='store_true', default=False)
    parser.add_argument('--SI',action='store_true', default=False)
    parser.add_argument('--PI',action='store_true', default=False)
    parser.add_argument('--EM',action='store_true', default=False,help=' Ensemble models')
    parser.add_argument('--len_kernel', default=3, type=int)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=str, default='0,2')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("GPU :"+args.gpu)
    # scaler = torch.cuda.amp.GradScaler()
    #     # autocast = torch.cuda.amp.autocast


    modelnames = ['GoogLeNet', 'MobileNetV2', 'ResNet18', 'SENet18','ResNet50'] #
    models = loadModels(modelnames)

    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    train_batches = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    test_batches = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=4)

    if not args.EM:
        for index in range(len(modelnames)):
            attack_model = models[index]
            accs = dict()
            for name in modelnames:
                accs[name] = AverageMeter()
        # accs = [AverageMeter()]*len(modelnames)
            for input,target in tqdm(train_batches):
                input,target= input.cuda(),target.cuda()
                # break
                adv_image,adv_ema = attack_pgd([attack_model],input,target,epsilon=8/255,attack_iters=10)
                with torch.no_grad():
                    for name,model in zip(modelnames,models):
                        output = model(normalize(adv_image))
                        acc = accuracy(output,target)
                        accs[name].update(acc[0].item()),len(target)
                        print(name,"test acc:%5.4f",accs[name].avg)
            result= dict()
            for name,acc in accs.items():
                result[name]= '%3.2f'%acc.avg
                print(name," test acc:%5.4f"%(acc.avg))
            os.makedirs('./user_data/transfer_result_%s/'%(modelnames[index]),exist_ok=True)
            params = ''+'_MI' if args.MI else ''+'_DI' if args.DI else '' +f'_TI_{args.len_kernel}' if args.TI else '' +'_NI' if args.NI else '' +'_SI' if args.SI else ''
            with open('./user_data/transfer_result_%s/pgd8_step10_%s.json'%(modelnames[index],params),'w+') as f:
                json.dump(result,f)

