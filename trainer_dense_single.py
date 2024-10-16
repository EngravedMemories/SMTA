import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from create_network import *
from create_dataset import *
from utils import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(description='Single-task Attack: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--network', default='mtan', type=str, help='split, mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
parser.add_argument('--task', default='depth', type=str, help='choose single tasks: `seg, depth, normal` for NYUv2, `seg, part_seg, disp` for CityScapes')
parser.add_argument('--epsilon', default=(4/255), type=float, help='epsilon for attack')
parser.add_argument('--attack_method', default='pgdl2', type=str, help='pgdl2, pgdli, ifsgm')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--seed', default=0, type=int, help='gpu ID')

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
train_tasks = create_task_flags(opt.task, opt.dataset)

print('Training Task: {} - {} in Single Task Attack Mode with {}'
      .format(opt.dataset.title(), opt.task.title(), opt.network.upper()))

if opt.network == 'split':
    model = MTLDeepLabv3(train_tasks).to(device)
elif opt.network == 'mtan':
    model = MTANDeepLabv3(train_tasks).to(device)


total_epoch = 1
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
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
    num_workers=4   # Change to 0 if printing images.
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4   # Change to 0 if printing images.
)


# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset)
for index in range(total_epoch):

    # evaluating train data
    model.train()
    train_dataset = iter(train_loader)
    for k in range(train_batch):
        train_data, train_target = next(train_dataset)

        train_data = train_data.to(device)
        train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}

        train_pred = model(train_data)
        optimizer.zero_grad()

        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
        train_loss[0].backward()
        optimizer.step()

        train_metric.update_metric(train_pred, train_target, train_loss)

    train_str = train_metric.compute_metric()
    train_metric.reset()

    # file_name = f"../data/Cityscapes_autol_{opt.task}.pth"
    if opt.dataset == 'nyuv2':
        if opt.task == "seg":
            pretrained_model = "../data/MTA/NYUv2_autol_seg.pth"
        elif opt.task == "depth":
            pretrained_model = "../data/MTA/NYUv2_autol_depth.pth"
        elif opt.task == "normal":
            pretrained_model = "../data/MTA/NYUv2_autol_normal.pth"
    
    elif opt.dataset == 'cityscapes':
        if opt.task == "seg":
            pretrained_model = "../data/MTA/Cityscapes_autol_seg.pth"
        elif opt.task == "part_seg":
            pretrained_model = "../data/MTA/Cityscapes_autol_part_seg.pth"
        elif opt.task == "disp":
            pretrained_model = "../data/MTA/Cityscapes_autol_disp.pth"
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load(pretrained_model, map_location=device)

    model.load_state_dict(torch.load(pretrained_model))

    # evaluating test data
    model.eval()

    test_dataset = iter(test_loader)
    for k in range(test_batch):
        test_data, test_target = next(test_dataset)

        if k == 0:
            image = np.transpose(test_data[0], (1, 2, 0))
            image = (image +1)/2
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        test_data = test_data.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        test_data.requires_grad = True
        test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

        test_pred = model(test_data)
        test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        alpha=0.25*epsilon
        alpha2 = 0.2*epsilon

        ori_images = test_data.clone().detach()

        for _ in range(10):


            test_data.requires_grad = True
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}
            num_image = np.shape(test_data)[0]

            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

            model.zero_grad()

            test_loss[0].backward()
        
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

    test_str = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    task_dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric}
    np.save('logging/stl_{}_{}_{}_{}.npy'.format(opt.network, opt.dataset, opt.task, opt.seed), task_dict)

# if opt.dataset == 'nyuv2':
#     file_name = f"../data/MTA/NYUv2_autol_{opt.task}.pth"
# elif opt.dataset == 'cityscapes':
#     file_name = f"../data/MTA/Cityscapes_autol_{opt.task}.pth"

# torch.save(model.state_dict(), file_name)

