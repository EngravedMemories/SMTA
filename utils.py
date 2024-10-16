from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt

"""
Define task metrics, loss functions and model trainer here.
"""


class ConfMatrix(object):
    """
    For mIoU and other pixel-level classification tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def reset(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()


def create_task_flags(task, dataset, with_noise=False):
    """
    Record task and its prediction dimension.
    Noise prediction is only applied in auxiliary learning.
    """
    nyu_tasks = {'seg': 13, 'depth': 1, 'normal': 3}
    cityscapes_tasks = {'seg': 19, 'part_seg': 10, 'disp': 1}

    tasks = {}
    if task != 'all':
        if dataset == 'nyuv2':
            tasks[task] = nyu_tasks[task]
        elif dataset == 'cityscapes':
            tasks[task] = cityscapes_tasks[task]
    else:
        if dataset == 'nyuv2':
            tasks = nyu_tasks
        elif dataset == 'cityscapes':
            tasks = cityscapes_tasks

    if with_noise:
        tasks['noise'] = 1
    return tasks


def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str


def get_weight_str_ranked(weight, tasks, rank_num):
    """
    Record top-k ranked task weighting.
    """
    rank_idx = np.argsort(weight)

    if type(tasks) == dict:
        tasks = list(tasks.keys())

    top_str = 'Top {}: '.format(rank_num)
    bot_str = 'Bottom {}: '.format(rank_num)
    for i in range(rank_num):
        top_str += '{} {:.02f} '.format(tasks[rank_idx[-i-1]].title(), weight[rank_idx[-i-1]])
        bot_str += '{} {:.02f} '.format(tasks[rank_idx[i]].title(), weight[rank_idx[i]])

    return 'Task Weighting | {}| {}'.format(top_str, bot_str)


def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """
    if task_id in ['seg', 'part_seg'] or 'class' in task_id:
        # Cross Entropy Loss with Ignored Index (values are -1)
        loss = F.cross_entropy(pred, gt, ignore_index=-1)

    if task_id in ['normal', 'depth', 'disp', 'noise']:
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 if task_id == 'disp' else 0
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
    return loss


class TaskMetric:
    def __init__(self, train_tasks, pri_tasks, batch_size, epochs, dataset, include_mtl=False):
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.batch_size = batch_size
        self.dataset = dataset
        self.include_mtl = include_mtl
        self.metric = {key: np.zeros([epochs, 2]) for key in train_tasks.keys()}  # record loss & task-specific metric
        self.data_counter = 0
        self.epoch_counter = 0
        self.conf_mtx = {}

        if include_mtl:  # include multi-task performance (relative averaged task improvement)
            self.metric['all'] = np.zeros(epochs)
        for task in self.train_tasks:
            if task in ['seg', 'part_seg']:
                self.conf_mtx[task] = ConfMatrix(self.train_tasks[task])

    def reset(self):
        """
        Reset data counter and confusion matrices.
        """
        self.epoch_counter += 1
        self.data_counter = 0

        if len(self.conf_mtx) > 0:
            for i in self.conf_mtx:
                self.conf_mtx[i].reset()

    def update_metric(self, task_pred, task_gt, task_loss):
        """
        Update batch-wise metric for each task.
            :param task_pred: [TASK_PRED1, TASK_PRED2, ...]
            :param task_gt: {'TASK_ID1': TASK_GT1, 'TASK_ID2': TASK_GT2, ...}
            :param task_loss: [TASK_LOSS1, TASK_LOSS2, ...]
        """
        curr_bs = task_pred[0].shape[0]
        r = self.data_counter / (self.data_counter + curr_bs / self.batch_size)
        e = self.epoch_counter
        self.data_counter += 1

        with torch.no_grad():
            for loss, pred, (task_id, gt) in zip(task_loss, task_pred, task_gt.items()):
                self.metric[task_id][e, 0] = r * self.metric[task_id][e, 0] + (1 - r) * loss.item()

                if task_id in ['seg', 'part_seg']:
                    # update confusion matrix (metric will be computed directly in the Confusion Matrix)
                    self.conf_mtx[task_id].update(pred.argmax(1).flatten(), gt.flatten())

                if 'class' in task_id:
                    # Accuracy for image classification tasks
                    pred_label = pred.data.max(1)[1]
                    acc = pred_label.eq(gt).sum().item() / pred_label.shape[0]
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * acc

                if task_id in ['depth', 'disp', 'noise']:
                    # Abs. Err.
                    invalid_idx = -1 if task_id == 'disp' else 0
                    valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
                    abs_err = torch.mean(torch.abs(pred - gt).masked_select(valid_mask)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * abs_err

                if task_id in ['normal']:
                    # Mean Degree Err.
                    valid_mask = (torch.sum(gt, dim=1) != 0).to(pred.device)
                    degree_error = torch.acos(torch.clamp(torch.sum(pred * gt, dim=1).masked_select(valid_mask), -1, 1))
                    mean_error = torch.mean(torch.rad2deg(degree_error)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * mean_error

    def compute_metric(self, only_pri=False):
        metric_str = ''
        e = self.epoch_counter
        tasks = self.pri_tasks if only_pri else self.train_tasks  # only print primary tasks performance in evaluation

        for task_id in tasks:
            if task_id in ['seg', 'part_seg']:  # mIoU for segmentation
                self.metric[task_id][e, 1] = self.conf_mtx[task_id].get_metrics()

            metric_str += ' {} {:.4f} {:.4f}'\
                .format(task_id.capitalize(), self.metric[task_id][e, 0], self.metric[task_id][e, 1])

        if self.include_mtl:
            # Pre-computed single task learning performance using trainer_dense_single.py
            if self.dataset == 'nyuv2':
                stl = {'seg': 0.4337, 'depth': 0.5224, 'normal': 22.40}
            elif self.dataset == 'cityscapes':
                stl = {'seg': 0.5620, 'part_seg': 0.5274, 'disp': 0.84}
            elif self.dataset == 'cifar100':
                stl = {'class_0': 0.6865, 'class_1': 0.8100, 'class_2': 0.8234, 'class_3': 0.8371, 'class_4': 0.8910,
                       'class_5': 0.8872, 'class_6': 0.8475, 'class_7': 0.8588, 'class_8': 0.8707, 'class_9': 0.9015,
                       'class_10': 0.8976, 'class_11': 0.8488, 'class_12': 0.9033, 'class_13': 0.8441, 'class_14': 0.5537,
                       'class_15': 0.7584, 'class_16': 0.7279, 'class_17': 0.7537, 'class_18': 0.9148, 'class_19': 0.9469}

            delta_mtl = 0
            for task_id in self.train_tasks:
                if task_id in ['seg', 'part_seg'] or 'class' in task_id:  # higher better
                    delta_mtl += (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]
                elif task_id in ['depth', 'normal', 'disp']:
                    delta_mtl -= (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]

            self.metric['all'][e] = delta_mtl / len(stl)
            metric_str += ' | All {:.4f}'.format(self.metric['all'][e])
        return metric_str

    def get_best_performance(self, task):
        e = self.epoch_counter
        if task in ['seg', 'part_seg'] or 'class' in task:  # higher better
            return max(self.metric[task][:e, 1])
        if task in ['depth', 'normal', 'disp']:  # lower better
            return min(self.metric[task][:e, 1])
        if task in ['all']:  # higher better
            return max(self.metric[task][:e])


"""
Define Gradient-based frameworks here. 
Based on https://github.com/Cranial-XIX/CAGrad/blob/main/cityscapes/utils.py
"""


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)   # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims, num_tasks):
    newgrad = newgrad * num_tasks  # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image

def ifgsm_attack(image, epsilon, data_grad, alpha, steps):
    for _ in range(steps):

        # test_data_grad = image.grad.data

        sign_data_grad = data_grad.sign()

        adv_images = image + alpha * sign_data_grad
        a = torch.clamp(image - epsilon, min=-1)
        b = (adv_images >= a).float() * adv_images + (adv_images < a).float() * a  # nopep8
        c = (b > image + epsilon).float() * (image + epsilon) + (b <= image + epsilon).float() * b  # nopep8
        image = torch.clamp(c, max=1).detach()

    return image

def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)
        
def image_save(test_data, save_path):
    """
    Saves an image tensor to the specified file path.
    
    Args:
        test_data (torch.Tensor): The input image tensor to be saved.
        save_path (str): The file path to save the image.
    
    Returns:
        None
    """
    if test_data.ndim == 3:
        tensor_np = test_data.cpu().detach().permute(1, 2, 0).numpy()
    else:
        tensor_np = test_data.cpu().detach().numpy()
    tensor_np2 = 2 * (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min()) - 1
    tensor_np_img = (tensor_np2 + 1) / 2
    plt.imsave(save_path, tensor_np_img)

def image_save_segmentation(test_data, original_image, save_path, alpha=0.8):
    """
    Visualizes and saves a semantic segmentation result overlaid on the original image.

    Args:
        test_data (torch.Tensor): The input tensor of shape (C, H, W) representing C class segmentation results,
                                  or (H, W) representing predicted classes in range [0, C-1],
                                  or (1, H, W) representing a depth map.
        original_image (torch.Tensor): The original image tensor of shape (3, H, W).
        save_path (str): The file path to save the visualized image.
        alpha (float): The transparency of the segmentation overlay. Default is 0.8.

    Returns:
        None
    """

    # Ensure the inputs are on CPU and convert to numpy
    test_data = test_data.cpu().detach().numpy()
    original_image = original_image.cpu().detach().numpy()

    # Check if test_data is a depth map
    if len(test_data.shape) == 3 and test_data.shape[0] == 1:
        # Normalize depth map to [0, 1] range
        depth_map = (test_data[0] - np.min(test_data)) / (np.max(test_data) - np.min(test_data))
        # depth_map = test_data[0]
        # Create a heatmap for depth visualization
        cmap = plt.get_cmap('jet')
        seg_image = cmap(depth_map)[:, :, :3]
    else:
        # Get the predicted class for each pixel
        if len(test_data.shape) == 3:
            num_classes = test_data.shape[0]
            predicted_classes = np.argmax(test_data, axis=0)
        elif len(test_data.shape) == 2:
            predicted_classes = test_data
            num_classes = int(np.max(predicted_classes)) + 1
        else:
            raise ValueError(f"Unsupported shape of test_data: {test_data.shape}")

        # Create a colormap
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]

        # Create an RGB image for segmentation
        height, width = predicted_classes.shape
        seg_image = np.zeros((height, width, 3), dtype=np.float32)

        # Fill the RGB image with colors based on predicted classes
        for class_idx in range(num_classes):
            mask = predicted_classes == class_idx
            seg_image[mask] = colors[class_idx][:3]

    # Transpose original image from (3, H, W) to (H, W, 3)
    original_image = np.transpose(original_image, (1, 2, 0))

    # Normalize original image to [0, 1] range if needed
    if original_image.max() > 1:
        original_image = original_image / 255.0

    # Blend the segmentation with the original image
    blended_image = (1 - alpha) * original_image + alpha * seg_image

    # Clip values to ensure they are in the valid range [0, 1]
    blended_image = np.clip(blended_image, 0, 1)

    # Save the visualization
    plt.imsave(save_path, blended_image)


# calculate distance
def calculate_psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two image tensors.
    
    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
    
    Returns:
        float: The PSNR value between the two input images. If the Mean Squared Error (MSE) between the images is 0, the function returns positive infinity.
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0  # Assuming the pixel values are in range [0, 1]
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr

def calculate_l2_distance(tensor1, tensor2):
    """
    Calculates the L2 distance between two tensors.
    
    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.
    
    Returns:
        float: The L2 distance between the two input tensors.
    """
    return torch.norm(tensor1 - tensor2)
 
def calculate_diff_pixel(tensor1, tensor2):
    """
    Calculates the absolute difference between two input tensors and sums the differences across all pixels and channels to get the total pixel difference.
    
    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.
    
    Returns:
        float: The total pixel difference between the two input tensors.
    """
    # Calculate the absolute difference
    difference = torch.abs(tensor1 - tensor2)
    # Sum the differences across all pixels and channels to get the total pixel difference
    return torch.sum(difference)
