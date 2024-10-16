import argparse
import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *
from copy import deepcopy



parser = argparse.ArgumentParser(description='Stealthy Multi-task/Auxiliary Attack: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--network', default='mtan', type=str, help='split, mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--with_noise', action='store_true', help='with noise prediction task')
parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')

parser.add_argument('--epsilon', default=(2/255), type=float, help='epsilon for attack')
parser.add_argument('--attack_method', default='ifsgm', type=str, help='pgdl2, pgdli, ifsgm')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--attack_target', default=1, type=int, help='1,2,3 for attack 1st, 2nd, 3rd task only, 0 for do not do this')
parser.add_argument('--train', default=0, type=int, help='train flag')
parser.add_argument('--save_img', default=0, type=int, help='save image flag')


opt = parser.parse_args()

epsilon = opt.epsilon

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
if opt.with_noise:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=True)
else:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=False)

pri_tasks = create_task_flags(opt.task, opt.dataset, with_noise=False)

K = 3
beta = 30
alpha= 0.25*epsilon
alpha2 = 0.2*epsilon
total_epoch = 1

penalty1 = 0.12
penalty2 = 0.2

task_W = torch.ones(K) / K

train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
print('Dataset: {} | Training Task: {} | Primary Task: {} in Stealthy Multi-task Attack Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(opt.weight.title(), opt.grad_method.upper()))

if opt.network == 'split':
    model = MTLDeepLabv3(train_tasks).to(device)
elif opt.network == 'mtan':
    model = MTANDeepLabv3(train_tasks).to(device)


T = 2.0  # temperature used in dwa
lambda_weight = np.ones([total_epoch, len(train_tasks)])
params = model.parameters()

optimizer = optim.SGD(params, lr=0.1, weight_decay=1e-4, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

# define dataset
if opt.dataset == 'nyuv2':
    dataset_path = '../data/nyuv2'
    train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 4

elif opt.dataset == 'cityscapes':
    dataset_path = '../data/cityscapes'
    train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
    test_set = CityScapes(root=dataset_path, train=False)
    batch_size = 4

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

# apply gradient methods
if opt.grad_method != 'none':
    rng = np.random.default_rng()
    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), len(train_tasks)).to(device)


# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset, include_mtl=True)


for index in range(total_epoch):

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[index, :] = 1.0
        else:
            w = []
            for i, t in enumerate(train_tasks):
                w += [train_metric.metric[t][index - 1, 0] / train_metric.metric[t][index - 2, 0]]
            w = torch.softmax(torch.tensor(w) / T, dim=0)
            lambda_weight[index] = len(train_tasks) * w.numpy()

    # iteration for all batches
    # model.train()
    train_dataset = iter(train_loader)

    if opt.train:
        for k in range(train_batch):
            train_data, train_target = next(train_dataset)
            train_data = train_data.to(device)
            train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}


            # update multi-task network parameters with task weights
            optimizer.zero_grad()
            
            train_pred = model(train_data)
            train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
            train_loss_tmp = [0] * len(train_tasks)
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]

            loss = sum(train_loss_tmp)

            if opt.grad_method == 'none':
                loss.backward()
                optimizer.step()

            # gradient-based methods applied here:
            elif opt.grad_method == "graddrop":
                for i in range(len(train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(model, g, grad_dims, len(train_tasks))
                optimizer.step()

            elif opt.grad_method == "pcgrad":
                for i in range(len(train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = pcgrad(grads, rng, len(train_tasks))
                overwrite_grad(model, g, grad_dims, len(train_tasks))
                optimizer.step()

            elif opt.grad_method == "cagrad":
                for i in range(len(train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = cagrad(grads, len(train_tasks), 0.4, rescale=1)
                overwrite_grad(model, g, grad_dims, len(train_tasks))
                optimizer.step()

            train_metric.update_metric(train_pred, train_target, train_loss)

        train_str = train_metric.compute_metric()
        train_metric.reset()

    if opt.dataset == 'nyuv2':
        pretrained_model = "../data/NYUv2_autol_model.pth"
    elif opt.dataset == 'cityscapes':
        pretrained_model = "../data/Cityscapes_model.pth"

    model.load_state_dict(torch.load(pretrained_model))

    # evaluating test data
    model.eval()
    test_dataset = iter(test_loader)

    for k in range(test_batch):
        test_data, test_target = next(test_dataset)
        num_image = np.shape(test_data)[0]
        test_data = test_data.to(device)
        ori_test_data = deepcopy(test_data)

        # Set requires_grad attribute of tensor. Important for Attack
        test_data.requires_grad = True
        test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

        test_pred = model(test_data)
        test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]


        if opt.save_img:
            if not os.path.exists('img/test_ori/'):
                os.system('mkdir -p img/test_ori/')
            for bt in range(len(test_data)):
                image_save(test_data[bt], f'img/test_ori/{k*len(test_data)+bt}.png')
                for i, task_id in enumerate(train_tasks):
                    image_save_segmentation(test_pred[i][bt], test_data[bt], f'img/test_ori/{k*len(test_data)+bt}_{task_id}.png')
                    image_save_segmentation(test_target[task_id][bt], test_data[bt], f'img/test_ori/{k*len(test_data)+bt}_{task_id}_label.png')

        ori_images = test_data.clone().detach()

        task_W = torch.ones(K) / K

        for _ in range(50):

            test_data.requires_grad = True
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

            if _ == 0:
                test_loss_record0 = test_loss[0].detach()
                test_loss_record1 = test_loss[1].detach()
                test_loss_record2 = test_loss[2].detach()

            L1 = (test_loss[0]-test_loss_record0)/test_loss_record0
            L2 = (test_loss[1]-test_loss_record1)/test_loss_record1
            L3 = (test_loss[2]-test_loss_record2)/test_loss_record2

            if opt.attack_target == 1:
                total_loss = L1 - penalty1*L2 - penalty2*L3

            elif opt.attack_target == 2:
                total_loss = -penalty1*L1 + L2 - penalty2*L3

            elif opt.attack_target == 3:
                total_loss = -penalty1*L1 - penalty2*L2 + L3
            
            else:
                total_loss = task_W[0]*L1 + task_W[1]*L2 + task_W[2]*L3            
            
            model.zero_grad()

            total_loss.backward()

            test_data_grad = test_data.grad.data

            sign_data_grad = test_data_grad.sign()
        

            if opt.attack_method == 'ifsgm':

                adv_images = test_data + alpha * sign_data_grad
                a = torch.clamp(ori_images - epsilon, min=-1)
                b = (adv_images >= a).float() * adv_images + (adv_images < a).float() * a  # nopep8
                c = (b > ori_images + epsilon).float() * (ori_images + epsilon) + (b <= ori_images + epsilon).float() * b  # nopep8
                test_data = torch.clamp(c, max=1).detach()

            elif opt.attack_method == 'pgdl2':

                grad_norms = (torch.norm(test_data_grad.view(num_image, -1), p=2, dim=1) + 1e-10)  # nopep8
                test_data_grad = test_data_grad / grad_norms.view(num_image, 1, 1, 1)
                adv_images = test_data + alpha2 * test_data_grad

                delta = adv_images - ori_images
                delta_norms = torch.norm(delta.view(num_image, -1), p=2, dim=1)
                factor = epsilon / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)

                test_data = torch.clamp(ori_images + delta, min=-1, max=1).detach()

            elif opt.attack_method == 'pgdli':
                adv_images = test_data + alpha * sign_data_grad
                delta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
                test_data = torch.clamp(ori_images + delta, min=-1, max=1).detach()
        


        # Re-classify the perturbed image
        test_pred = model(test_data)
        test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        test_metric.update_metric(test_pred, test_target, test_loss)
        
        if opt.save_img:
            # Save attacked image
            if not os.path.exists('img/test_atk/'):
                os.system('mkdir -p img/test_atk/')
                with open(f'img/test_atk/attack_info.txt', 'w') as f:
                    f.writelines('')
            
            for bt in range(len(test_data)):
                image_save(test_data[bt], f'img/test_atk/{k*len(test_data)+bt}.png')
                for i, task_id in enumerate(train_tasks):
                    image_save_segmentation(test_pred[i][bt], test_data[bt], f'img/test_atk/{k*len(test_data)+bt}_{task_id}.png')

                # calculate PSNR and L2 Distance
                with open(f'img/test_atk/attack_info.txt', 'a') as f:
                    f.writelines(f'{k*len(test_data)+bt} {calculate_psnr(ori_test_data[bt], test_data[bt])} \
                                {calculate_l2_distance(ori_test_data[bt], test_data[bt])} \
                                {calculate_diff_pixel(ori_test_data[bt], test_data[bt])}\n')


    test_str = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric, 'weight': lambda_weight}

    print(get_weight_str(lambda_weight[index], train_tasks))

    np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.npy'
            .format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed), dict)

# torch.save(model.state_dict(), '../data/NYUv2_autol_model.pth')
