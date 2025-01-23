import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='SMTA + AT')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--network', default='mtan', type=str, help='split, mtan')
parser.add_argument('--dataset', default='cityscapes', type=str, help='nyuv2, cityscapes')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: autol, dwa, uncert, equal')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--with_noise', action='store_true', help='with noise prediction task')
parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda')
parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')

parser.add_argument('--epoch', default=1, type=int, help='total epoch')
parser.add_argument('--epsilon', default=(8/255), type=float, help='epsilon for attack')
parser.add_argument('--attack_method', default='ifgsm', type=str, help='pgdl2, pgdli, ifgsm')
parser.add_argument('--attack_weight', default='minma', type=str, help='minmax, normalize')
parser.add_argument('--gamma', default=0, type=int, help='gamma for minmax')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--attack_target', default=3, type=int, help='1,2,3 for attack 1st, 2nd, 3rd task only, 0 for do not do this')
parser.add_argument('--w_reset', default=1, type=int, help='1 for reset weight, 0 for not reset')

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

total_epoch = opt.epoch

K = 3
beta = 30
gamma = opt.gamma
alpha= 0.25*epsilon
alpha2 = 0.2*epsilon

task_W = torch.ones(K) / K

train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(opt.weight.title(), opt.grad_method.upper()))

if opt.network == 'split':
    model = MTLDeepLabv3(train_tasks).to(device)
elif opt.network == 'mtan':
    model = MTANDeepLabv3(train_tasks).to(device)


# choose task weighting here
if opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
    params = list(model.parameters()) + [logsigma]
    logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

if opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([total_epoch, len(train_tasks)])
    params = model.parameters()

if opt.weight == 'autol':
    params = model.parameters()
    autol = AutoLambda(model, device, train_tasks, pri_tasks, opt.autol_init)
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=opt.autol_lr)

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
    num_workers = 4
)

# a copy of train_loader with different data order, used for Auto-Lambda meta-update
if opt.weight == 'autol':
    val_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers = 4
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
    #model.train()
    train_dataset = iter(train_loader)
    if opt.weight == 'autol':
        val_dataset = iter(val_loader)

    for k in range(train_batch):
        train_data, train_target = next(train_dataset)
        train_data = train_data.to(device)
        train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}

        # update meta-weights with Auto-Lambda
        if opt.weight == 'autol':
            val_data, val_target = next(val_dataset)
            val_data = val_data.to(device)
            val_target = {task_id: val_target[task_id].to(device) for task_id in train_tasks.keys()}

            meta_optimizer.zero_grad()
            autol.unrolled_backward(train_data, train_target, val_data, val_target,
                                    scheduler.get_last_lr()[0], optimizer)
            meta_optimizer.step()

        # update multi-task network parameters with task weights
        optimizer.zero_grad()
        
        train_pred = model(train_data)
    
        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        train_loss_tmp = [0] * len(train_tasks)

        if opt.weight in ['equal', 'dwa']:
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]

        if opt.weight == 'uncert':
            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

        if opt.weight == 'autol':
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(autol.meta_weights)]

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

    # model.load_state_dict(torch.load(pretrained_model))

    load_path = f'../data/MTA/{opt.dataset}_model_epsilon_0.0078.pth'  # 0.0078, 0.0157, 0.0314
    model.load_state_dict(torch.load(load_path))

    # evaluating test data
    model.eval()
    test_dataset = iter(test_loader)

    for k in range(test_batch):
        test_data, test_target = next(test_dataset)
        num_image = np.shape(test_data)[0]
        test_data = test_data.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        test_data.requires_grad = True
        test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

        test_pred = model(test_data)
        test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        # total_loss = test_loss[0] - test_loss[1] - test_loss[2]

        # model.zero_grad()

        # test_loss[2].backward()
        # total_loss.backward()

        # test_data_grad = test_data.grad.data

        # Restore the data to its original scale
        # test_data_denorm = denorm(test_data)

        # test_data = denorm(test_data)

        # Call FGSM Attack
        # perturbed_data = fgsm_attack(test_data, epsilon, test_data_grad)

        # perturbed_data = ifgsm_attack(test_data, epsilon, test_data_grad, alpha=0.001, steps=10)

        ori_images = test_data.clone().detach()

        if opt.w_reset == 1:
            task_W = torch.ones(K) / K

        penalty1 = 0.7
        penalty2 = 0.85

        for _ in range(50):

            test_data.requires_grad = True
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

            if _ == 0:
                test_loss_record0 = test_loss[0].detach()
                test_loss_record1 = test_loss[1].detach()
                test_loss_record2 = test_loss[2].detach()
                # print(test_loss_record0)

            L1 = (test_loss[0]-test_loss_record0)/test_loss_record0
            L2 = (test_loss[1]-test_loss_record1)/test_loss_record1
            L3 = (test_loss[2]-test_loss_record2)/test_loss_record2

            if opt.attack_target == 1:
                attack_loss_record = train_loss*1
                total_loss = L1 - penalty1*L2 - penalty2*L3

            elif opt.attack_target == 2:
                attack_loss_record = train_loss*1
                total_loss = -penalty1*L1 + L2 - penalty2*L3

            elif opt.attack_target == 3:
                attack_loss_record = train_loss*1
                total_loss = -penalty1*L1 - penalty2*L2 + L3
            
            else:
                attack_loss_record = train_loss*1
                attack_loss_record[0] =  -L1
                attack_loss_record[1] =  -L2
                attack_loss_record[2] =  -L3

                total_loss = task_W[0]*L1 + task_W[1]*L2 + task_W[2]*L3            

            ###total_loss = test_loss[2] - test_loss[1] - test_loss[0]
            

            # T2_attack_loss_norm = -torch.abs(L1)+L2-torch.abs(L3)
            # T3_attack_loss_norm = -torch.abs(L1)-torch.abs(L2)+L3

            # T1_attack_loss = test_loss[0]-torch.abs(test_loss[1])-torch.abs(test_loss[2])
            # T2_attack_loss = -torch.abs(test_loss[0])+test_loss[1]-torch.abs(test_loss[2])
            # T3_attack_loss = -torch.abs(test_loss[0])-torch.abs(test_loss[1])+test_loss[2]
   
            model.zero_grad()

            # test_loss[1].backward()
            total_loss.backward()

            test_data_grad = test_data.grad.data

            sign_data_grad = test_data_grad.sign()

            if opt.attack_method == 'ifgsm':

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
            
            para = 0
            stride1 = 0
            stride2 = 0

            if opt.attack_target == 1:
                if L2.detach()>0:
                    penalty1 += stride1*L2.detach()
                else:
                    penalty1 += para*stride1*L2.detach()
                if L3.detach()>0:
                    penalty2 += stride2*L3.detach()
                else:
                    penalty2 += para*stride2*L3.detach()

                # penalty1 += L1.detach()
                # penalty2 += L2.detach()

            elif opt.attack_target == 2:
                if L1.detach()>0:
                    penalty1 += stride1*L1.detach()
                else:
                    penalty1 += para*stride1*L1.detach()
                if L3.detach()>0:
                    penalty2 += stride2*L3.detach()
                else:
                    penalty2 += para*stride2*L3.detach()

            elif opt.attack_target == 3:
                if L1.detach()>0:
                    penalty1 += stride1*L1.detach()
                else:
                    penalty1 += para*stride1*L1.detach()
                if L2.detach()>0:
                    penalty2 += stride2*L2.detach()
                else:
                    penalty2 += para*stride2*L2.detach()

            # print('Penalty1:', penalty1)
            # print('Penalty2:', penalty2)
        

            if opt.attack_weight == 'minmax':
                with torch.no_grad():
                    F = torch.stack(attack_loss_record)
                    task_W = task_W.to(device)
                    G = F - gamma * (task_W - 1/K)
                    task_W += 1.0 / beta * G
                    task_W = project_simplex(task_W)
                        
                task_W = task_W.cpu()

            # print(task_W)

        # Reapply normalization
        #perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # perturbed_data= transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        test_pred = model(test_data)
        test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        test_metric.update_metric(test_pred, test_target, test_loss)

    test_str = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    if opt.weight == 'autol':
        meta_weight_ls[index] = autol.meta_weights.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': meta_weight_ls}

        print(get_weight_str(meta_weight_ls[index], train_tasks))

    if opt.weight in ['dwa', 'equal']:
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': lambda_weight}

        print(get_weight_str(lambda_weight[index], train_tasks))

    if opt.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': logsigma_ls}

        print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

    np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.npy'
            .format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed), dict)

# torch.save(model.state_dict(), '../data/MTA/NYUv2_autol_model.pth')
    

