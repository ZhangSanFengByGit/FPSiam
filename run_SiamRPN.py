
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import time


from utils import get_subwindow_tracking


def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1 #19
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = False

    # by zzc
    pos_th = 0.5
    low_th = 0.1
    lamda = 10.

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1


def tracker_train(net, x_crop, target_pos, target_sz, scale_z, p, gt_pos, gt_sz):

    delta, score = net(x_crop)
    delta = delta.permute(1,2,3,0).contiguous().view(4,-1)  # 4 * (5*19*19)
    score = F.softmax(score.permute(1,2,3,0).contiguous().view(2,-1), dim = 0).float()  # 2 * (5*19*19)


    delta_np = delta.data.cpu().numpy()
    #print('delta_raw x      max:{}, min:{}, mean:{}'.format(np.amax(delta_np[0,:]), np.amin(delta_np[0,:]), np.mean(delta_np[0,:])))
    #print('delta_raw y      max:{}, min:{}, mean:{}'.format(np.amax(delta_np[1,:]), np.amin(delta_np[1,:]), np.mean(delta_np[1,:])))
    #print('delta_raw width  max:{}, min:{}, mean:{}'.format(np.amax(delta_np[2,:]), np.amin(delta_np[2,:]), np.mean(delta_np[2,:])))
    #print('delta_raw height max:{}, min:{}, mean:{}'.format(np.amax(delta_np[3,:]), np.amin(delta_np[3,:]), np.mean(delta_np[3,:])))


    proposals = np.zeros(delta_np.shape)
    proposals[0,:] = delta_np[0,:] * p.anchor[:,2] + p.anchor[:,0]
    proposals[1,:] = delta_np[1,:] * p.anchor[:,3] + p.anchor[:,1]
    proposals[2,:] = np.exp(delta_np[2,:]) * p.anchor[:,2]
    proposals[3,:] = np.exp(delta_np[3,:]) * p.anchor[:,3]
    np.clip(proposals[2,:], 1, 2000)
    np.clip(proposals[3,:], 1, 2000)
    #assert np.amax(proposals[2,:])<2000, 'the value is : {}!!!!!!!'.format(np.amax(proposals[2,:]))
    #assert np.amax(proposals[3,:])<2000, 'the value is : {}!!!!!!!'.format(np.amax(proposals[3,:]))
    #assert np.amin(proposals[2,:])>=1, 'the value is : {}!!!!!!!'.format(np.amin(proposals[2,:]))
    #assert np.amin(proposals[3,:])>=1, 'the value is : {}!!!!!!!'.format(np.amin(proposals[3,:]))
    assert np.amax(proposals)<float('inf'), 'the value is : {}!!!!!!!'.format(np.amax(proposals))
    assert np.amin(proposals)>-float('inf'), 'the value is : {}!!!!!!!'.format(np.amin(proposals))

    shift = [(gt_pos[0] - target_pos[0])*scale_z, (gt_pos[1] - target_pos[1])*scale_z]
    boxB = [shift[0]-gt_sz[0]/2, shift[1]-gt_sz[1]/2, shift[0]+gt_sz[0]/2, shift[1]+gt_sz[1]/2]
    print('boxB width:{}, height:{}'.format(gt_sz[0], gt_sz[1]))

    proposals_box = np.zeros(proposals.shape)
    proposals_box[0,:] = proposals[0,:] - proposals[2,:]/2.
    proposals_box[1,:] = proposals[1,:] - proposals[3,:]/2.
    proposals_box[2,:] = proposals[0,:] + proposals[2,:]/2.
    proposals_box[3,:] = proposals[1,:] + proposals[3,:]/2.

    iou = bb_intersection_over_union_parallel(proposals_box, boxB)
    print('max iou:{}, min iou:{}, mean iou:{}'.format(np.amax(iou), np.amin(iou), np.mean(iou)))
    max_pos = np.argmax(iou)
    positive = np.greater_equal(iou, p.pos_th)
    negative = np.less(iou, p.pos_th)
    positive_pos = np.argwhere(positive==1)
    assert positive_pos.shape[1]==1

    score_gt = np.zeros([2, delta_np.shape[1]])
    score_gt[1,:] = np.copy(positive*1)
    score_gt[0,:] = np.copy(negative*1)
    score_gt[1,max_pos] = 1
    score_gt[0,max_pos] = 0
    np.ascontiguousarray(score_gt, dtype=np.float32)
    assert score_gt.flags['C_CONTIGUOUS']==True
    score_gt = torch.from_numpy(score_gt[1]).cuda().float()

    if positive_pos.shape[0]>=1:
        delta_positive = torch.empty([4, positive_pos.shape[0]]).cuda()
        for i in range(positive_pos.shape[0]):
            delta_positive[:, i] = delta[:, positive_pos[i, 0] ]

        delta_np_pos = np.zeros(delta_np.shape)
        delta_np_pos[0,:] = (shift[0] - p.anchor[:,0])/p.anchor[:,2]
        delta_np_pos[1,:] = (shift[1] - p.anchor[:,1])/p.anchor[:,3]
        delta_np_pos[2,:] = np.log(gt_sz[0]/p.anchor[:,2])
        delta_np_pos[3,:] = np.log(gt_sz[1]/p.anchor[:,3])

        #assert positive_pos.shape[0]>=1
        delta_gt = np.zeros([4, positive_pos.shape[0]])
        for i in range(positive_pos.shape[0]):
            delta_gt[:, i] = np.copy(delta_np_pos[:, positive_pos[i, 0]])
        np.ascontiguousarray(delta_gt, dtype=np.float32)
        assert delta_gt.flags['C_CONTIGUOUS']==True

    else:
        delta_positive = torch.empty([4, 1]).cuda()
        delta_positive[:,0] = delta[:,max_pos]

        delta_np_pos = np.zeros(delta_np.shape)
        delta_np_pos[0,:] = (shift[0] - p.anchor[:,0])/p.anchor[:,2]
        delta_np_pos[1,:] = (shift[1] - p.anchor[:,1])/p.anchor[:,3]
        delta_np_pos[2,:] = np.log(gt_sz[0]/p.anchor[:,2])
        delta_np_pos[3,:] = np.log(gt_sz[1]/p.anchor[:,3])

        delta_gt = np.zeros([4,1])
        delta_gt[:,0] = np.copy(delta_np_pos[:,max_pos])
        np.ascontiguousarray(delta_gt, dtype = np.float32)
        assert delta_gt.flags['C_CONTIGUOUS']==True

    #delta_gt = delta_np_pos*np.tile(positive, (4,1)) + delta_np*np.tile(negative, (4,1))
    delta_gt = torch.from_numpy(delta_gt).cuda().float()
    '''
    delta_np_pos = np.zeros(delta_np.shape)
    delta_np_pos[0,:] = (shift[0] - p.anchor[:,0])/p.anchor[:,2]
    delta_np_pos[1,:] = (shift[1] - p.anchor[:,1])/p.anchor[:,3]
    delta_np_pos[2,:] = np.log(gt_sz[0]/p.anchor[:,2])
    delta_np_pos[3,:] = np.log(gt_sz[1]/p.anchor[:,3])
    delta_gt = torch.from_numpy(delta_np_pos).cuda().float()
    '''
    #compute loss:
    cls_loss = class_balanced_cross_entropy_loss(score[1], score_gt)
    #box_loss = F.mse_loss(delta, delta_gt, reduction='mean')
    box_loss = _smooth_l1_loss(delta_positive, delta_gt)

    return cls_loss, box_loss


def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    '''#test print
    for i in range(delta.shape[1]):
        print('delta_w:{}, delta_h:{}'.format(delta[2,i], delta[3,i]))
    print('delta_max_w:{}, delta_max_h:{}'.format(np.amax(delta[2,:]), np.amax(delta[3,:])))
    exit('finish')
    #test print'''

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

    p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1
    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    z_crop_large = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)

    z = z_crop.unsqueeze(0)         # removed the Variable interface
    z_large = z_crop_large.unsqueeze(0)
    net.temple(z.cuda(), z_large.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z  #the ratio between the in-model sizes and the real sizes
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0).cuda()   # removed the Variable interface

    target_pos, target_sz, score = tracker_eval(net, x_crop, target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state


def SiamRPN_train(state, im, old_pos, old_sz, gt_pos, gt_sz):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = old_pos  #atually is the old ground truth of the last frame
    target_sz = old_sz

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z  #the ratio between the in-model sizes and the real sizes
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0).cuda()

    cls_loss, box_loss = tracker_train(net, x_crop, target_pos, target_sz* scale_z, scale_z, p, gt_pos, gt_sz* scale_z)
    return cls_loss, box_loss


def SiamRPN_set_source(state, im, source_pos, source_sz):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    target_pos = source_pos
    target_sz = source_sz

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z  #the ratio between the in-model sizes and the real sizes
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0).cuda() 
    # x_crop removed the torch.Variable interface, due to the deprecated Variable in torch 0.4.0
    net(x_crop, set_source = True)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bb_intersection_over_union_parallel(proposals_box, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    inter = np.zeros(proposals_box.shape)

    inter[0,:] = np.maximum(proposals_box[0,:], boxB[0])
    inter[1,:] = np.maximum(proposals_box[1,:], boxB[1])
    inter[2,:] = np.minimum(proposals_box[2,:], boxB[2])
    inter[3,:] = np.minimum(proposals_box[3,:], boxB[3])
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    '''
    # compute the area of intersection rectangle
    interArea = np.maximum(0, inter[2,:] - inter[0,:] + 1) * np.maximum(0, inter[3,:] - inter[1,:] + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (proposals_box[2,:] - proposals_box[0,:] + 1) * (proposals_box[3,:] - proposals_box[1,:] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    assert interArea.shape == boxAArea.shape

    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

'''
def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= int(np.prod(label.size()))
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss
'''
def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=False, sigma=1e-4):
    #label size [batch_size, 2, score_size], either contain 0 or 1
    batch_size = label.size(0)
    num_labels_pos = torch.sum(label).item()
    num_labels_neg = torch.sum(1 - label).item()
    num_total = num_labels_neg + num_labels_pos
    print('pos_num = {}, neg_num = {}'.format((num_labels_pos), (num_labels_neg)))

    output_pos_raw = torch.clamp((output+sigma), min=0, max=1)
    output_neg_raw = torch.clamp((output-sigma), min=0, max=1)
    output_pos = torch.mul(output_pos_raw, label) + (1 - label)
    output_neg = torch.mul(output_neg_raw, (1-label))
    loss_pos_each = -torch.log(output_pos)
    loss_neg_each = -torch.log(1 - output_neg)
    loss_pos = torch.sum(loss_pos_each)
    loss_neg = torch.sum(loss_neg_each)

    final_loss = num_labels_neg/num_total*loss_pos + num_labels_pos/num_total*loss_neg

    if size_average:
        final_loss /= int(np.prod(label.size()))
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights=1, bbox_outside_weights=1, sigma=0.5):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box.sum()

    return loss_box


# V3 NEW================================================================================================================================================

def SiamRPN_init_batch(exemplar_list, exemplar_cxy_list, net):
    train_config = dict()

    batch_size = len(exemplar_list)
    p = TrackerConfig()
    p.update(net.cfg)

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    avg_chans_list = [None for i in range(batch_size)]
    z_list = [None for i in range(batch_size)]
    z_large_list = [None for i in range(batch_size)]
    
    for batch in range(batch_size):
        target_pos = exemplar_cxy_list[batch][0]
        target_sz = exemplar_cxy_list[batch][1]

        avg_chans = np.mean(exemplar_list[batch], axis=(0, 1))
        avg_chans_list[batch] = avg_chans

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        # initialize the exemplar
        z_crop = get_subwindow_tracking(exemplar_list[batch], target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop_large = get_subwindow_tracking(exemplar_list[batch], target_pos, p.instance_size, round(s_x), avg_chans)
        z = z_crop.unsqueeze(0)
        z_large = z_crop_large.unsqueeze(0)
        z_list[batch] = z
        z_large_list[batch] = z_large

    z_batch = z_list[0]
    z_large_batch = z_large_list[0]
    for idx in range(1, batch_size):
        z_batch = torch.cat((z_batch, z_list[idx]), dim=0)
        z_large_batch = torch.cat((z_large_batch, z_large_list[idx]), dim=0)

    assert z_batch.size(0)==batch_size
    assert z_large_batch.size(0)==batch_size

    net.temple(z_batch.cuda(), z_large_batch.cuda())

    train_config['avg_chans_list'] = avg_chans_list
    train_config['p'] = p
    train_config['net'] = net

    return train_config


def SiamRPN_set_source_batch(train_config, source_list, source_cxy_list):
    p = train_config['p']
    net = train_config['net']
    avg_chans_list = train_config['avg_chans_list']
    batch_size = len(source_list)
    x_batch = None

    for batch in range(batch_size):
        target_pos = source_cxy_list[batch][0]
        target_sz = source_cxy_list[batch][1]

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z  #the ratio between the in-model sizes and the real sizes
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = get_subwindow_tracking(source_list[batch], target_pos, p.instance_size, round(s_x), avg_chans_list[batch]).unsqueeze(0)
        if type(x_batch)!=torch.Tensor:
            x_batch = x_crop
        else:
            x_batch = torch.cat((x_batch, x_crop), dim=0)

    assert x_batch.size(0)==batch_size, '{}'.format(x_batch.size())

    net(x_batch.cuda(), set_source = True)


def SiamRPN_train_batch(train_config, instance_list, source_cxy_list, instance_cxy_list):
    p = train_config['p']
    net = train_config['net']
    avg_chans_list = train_config['avg_chans_list']
    batch_size = len(instance_list)
    x_batch = None
    shift = np.zeros([batch_size, 2])
    gt_sz_list = np.zeros([batch_size, 2])
    boxB = np.zeros([batch_size, 4])

    for batch in range(batch_size):
        target_pos = source_cxy_list[batch][0]
        target_sz = source_cxy_list[batch][1]
        gt_pos = instance_cxy_list[batch][0]
        gt_sz = instance_cxy_list[batch][1]

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        #scale_z transfer
        scale_z = p.exemplar_size / s_z  #the ratio between the in-model sizes and the real sizes
        gt_sz = gt_sz*scale_z
        gt_sz_list[batch,:] = gt_sz
        target_sz = target_sz*scale_z

        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        # extract scaled crops for search region x at previous target position
        x_crop = get_subwindow_tracking(instance_list[batch], target_pos, p.instance_size, round(s_x), avg_chans_list[batch]).unsqueeze(0)
        if type(x_batch)!=torch.Tensor:
            x_batch = x_crop
        else:
            x_batch = torch.cat((x_batch, x_crop), dim=0)

        shift[batch, :] = np.asarray([(gt_pos[0] - target_pos[0])*scale_z, (gt_pos[1] - target_pos[1])*scale_z], dtype=np.float32)
        boxB[batch, :] = np.asarray([shift[batch, 0]-gt_sz[0]/2, shift[batch, 1]-gt_sz[1]/2, \
                            shift[batch, 0]+gt_sz[0]/2, shift[batch, 1]+gt_sz[1]/2], dtype=np.float32)

    assert x_batch.size(0)==batch_size

    cls_loss, box_loss = tracker_train_batch(net, x_batch.cuda(), shift, boxB, gt_sz_list, p)
    return cls_loss, box_loss



def tracker_train_batch(net, x_batch, shift, boxB, gt_sz_list, p):
    batch_size = x_batch.size(0)
    delta, score = net(x_batch)

############################################################
        ##      transfer feature map to proposals
############################################################
    delta = delta.view(batch_size, 4, -1)  # batch * 4 * (5*19*19)
    score = F.softmax(score.view(batch_size, 2, -1), dim = 1).float()  # batch * 2 * (5*19*19)
    score_size = score.size(2)
    assert torch.max(delta)<= float('inf')
    assert torch.max(delta)>= -float('inf')

    delta_np = delta.data.cpu().numpy()
    proposals = np.zeros(delta_np.shape)

    #####encode
    proposals[:,0,:] = delta_np[:,0,:] * p.anchor[:,2] + p.anchor[:,0]
    proposals[:,1,:] = delta_np[:,1,:] * p.anchor[:,3] + p.anchor[:,1]
    proposals[:,2,:] = np.exp(delta_np[:,2,:]) * p.anchor[:,2]
    proposals[:,3,:] = np.exp(delta_np[:,3,:]) * p.anchor[:,3]
    #####encode


    #####check
    #proposals[:,0,:] = np.clip(proposals[:,0,:], -1000, 1000)
    #proposals[:,1,:] = np.clip(proposals[:,1,:], -1000, 1000)
    #proposals[:,2,:] = np.clip(proposals[:,2,:], 1, 2000)
    #proposals[:,3,:] = np.clip(proposals[:,3,:], 1, 2000)
    assert np.amax(proposals)<=float('inf'), 'the value is : {}!!!!!!!'.format(np.amax(proposals))
    assert np.amin(proposals)>=-float('inf'), 'the value is : {}!!!!!!!'.format(np.amin(proposals))
    #####

    proposals_box = np.zeros(proposals.shape)
    proposals_box[:,0,:] = proposals[:,0,:] - proposals[:,2,:]/2.
    proposals_box[:,1,:] = proposals[:,1,:] - proposals[:,3,:]/2.
    proposals_box[:,2,:] = proposals[:,0,:] + proposals[:,2,:]/2.
    proposals_box[:,3,:] = proposals[:,1,:] + proposals[:,3,:]/2.


############################################################
        ##           compute IOU
############################################################
    iou = bb_intersection_over_union_parallel_batch(proposals_box, boxB, batch_size, score_size)
    print('max iou:{}, min iou:{}, mean iou:{}'.format(np.amax(iou), np.amin(iou), np.mean(iou)))
    score_gt = np.zeros([batch_size, 2, score_size])
    max_pos = np.zeros([batch_size])
    positive = np.greater_equal(iou, p.pos_th)
    negative = np.less(iou, p.pos_th)
    num_positive = np.sum(positive)
    print('number of positive example:{}, negative example:{}'.format(np.sum(positive), np.sum(negative)))
    positive_pos = [None for i in range(batch_size)]



############################################################
        ##           class loss
############################################################

    for batch in range(batch_size):
        max_pos[batch] = np.argmax(iou[batch])
        positive_pos[batch] = np.argwhere(positive[batch]==1)
        score_gt[batch, 1, :] = positive[batch]*1
        score_gt[batch, 0, :] = negative[batch]*1
        score_gt[batch, 1, int(max_pos[batch])] = 1
        score_gt[batch, 0, int(max_pos[batch])] = 0

    score_gt = torch.from_numpy(score_gt).cuda().float()
    #compute class loss:
    cls_loss,counted_anchor,real_count = _cross_entropy_loss(score, score_gt, num_positive, proposals_box)
    print("real count in class loss is {}".format(real_count))

############################################################
        ##           box loss
############################################################
    box_loss = None
    delta_gt_all = np.zeros([batch_size, 4, 5*19*19])
    for batch in range(batch_size):
        '''
        if positive_pos[batch].shape[0]>=1:
            assert positive_pos[batch].shape[1]==1
            batch_postive_num = positive_pos[batch].shape[0]

            #pick the positive part delta
            delta_positive = torch.empty([4, batch_postive_num]).cuda()
            for i in range(batch_postive_num):
                delta_positive[:, i] = delta[batch, :, positive_pos[batch][i, 0]]

            #all the delta gt, encode
            delta_np_pos = np.zeros([4, 5*19*19])
            delta_np_pos[0,:] = (shift[batch, 0] - p.anchor[:,0])/p.anchor[:,2]
            delta_np_pos[1,:] = (shift[batch, 1] - p.anchor[:,1])/p.anchor[:,3]
            delta_np_pos[2,:] = np.log(gt_sz_list[batch, 0]/p.anchor[:,2])
            delta_np_pos[3,:] = np.log(gt_sz_list[batch, 1]/p.anchor[:,3])

            #pick the positive part delta gt
            delta_gt = np.zeros([4, batch_postive_num])
            for i in range(batch_postive_num):
                delta_gt[:, i] = delta_np_pos[:, positive_pos[batch][i, 0]]
        else:
            delta_positive = torch.empty([4, 1]).cuda()
            delta_positive[:,0] = delta[batch, :,int(max_pos[batch])]

            #all the delta gt
            delta_np_pos = np.zeros([4, 5*19*19])
            delta_np_pos[0,:] = (shift[batch, 0] - p.anchor[:,0])/p.anchor[:,2]
            delta_np_pos[1,:] = (shift[batch, 1] - p.anchor[:,1])/p.anchor[:,3]
            delta_np_pos[2,:] = np.log(gt_sz_list[batch, 0]/p.anchor[:,2])
            delta_np_pos[3,:] = np.log(gt_sz_list[batch, 1]/p.anchor[:,3])

            #pick the max_pos gt
            delta_gt = np.zeros([4,1])
            delta_gt[:,0] = delta_np_pos[:,int(max_pos[batch])]
        '''

        ############encode ground_truth
        delta_gt_all[batch, 0, :] = (shift[batch, 0] - p.anchor[:,0]) / p.anchor[:,2]
        delta_gt_all[batch, 1, :] = (shift[batch, 1] - p.anchor[:,1]) / p.anchor[:,3]
        delta_gt_all[batch, 2, :] = np.log(gt_sz_list[batch, 0] / p.anchor[:,2])
        delta_gt_all[batch, 3, :] = np.log(gt_sz_list[batch, 1] / p.anchor[:,3])
        ############encode ground_truth


    delta_gt = torch.from_numpy(delta_gt_all).cuda().float()

    #compute the box loss
    #box_loss = F.mse_loss(delta, delta_gt, reduction='mean')
    box_loss = _smooth_l1(delta, delta_gt, counted_anchor)


    #mean box loss by counted anchor number
    #box_loss = box_loss/real_count

    return cls_loss, box_loss



def bb_intersection_over_union_parallel_batch(proposals_box, boxB, batch_size, score_size):
    # determine the (x, y)-coordinates of the intersection rectangle

    inter = np.zeros(proposals_box.shape)
    iou = np.zeros([batch_size, score_size])

    for batch in range(batch_size):
        inter[batch,0,:] = np.maximum(proposals_box[batch,0,:], boxB[batch,0])
        inter[batch,1,:] = np.maximum(proposals_box[batch,1,:], boxB[batch,1])
        inter[batch,2,:] = np.minimum(proposals_box[batch,2,:], boxB[batch,2])
        inter[batch,3,:] = np.minimum(proposals_box[batch,3,:], boxB[batch,3])

        # compute the area of intersection rectangle
        interArea = np.maximum(0, inter[batch,2,:] - inter[batch,0,:] + 1) * np.maximum(0, inter[batch,3,:] - inter[batch,1,:] + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (proposals_box[batch,2,:] - proposals_box[batch,0,:] + 1) * (proposals_box[batch,3,:] - proposals_box[batch,1,:] + 1)
        boxBArea = (boxB[batch,2] - boxB[batch,0] + 1) * (boxB[batch,3] - boxB[batch,1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        assert interArea.shape == boxAArea.shape

        cur_iou = interArea / (boxAArea + boxBArea - interArea)

        iou[batch, :] = cur_iou
    # return the intersection over union value
    return iou


def _cross_entropy_loss(output, label, num_positive, proposals_box, size_average=False, batch_average=False, sigma=0):
    #label size [batch_size, 2, score_size], either contain 0 or 1
    batch_size = output.size()[0]
    num_total = np.prod(output.size())
    num_count = min( num_positive*4 , num_total ) #positive:negative = 1:3
    visited = np.zeros(output.size())

    #0output_raw = torch.clamp((output+sigma), min=0, max=1)
    assert not torch.isnan(label).any()
    assert not torch.isnan(output).any()
    output_loss = -(torch.log(output + sigma) * label) + (-torch.log(1 - output - sigma))*(1 - label)
    assert not torch.isnan(output_loss).any()
    loss_np = output_loss.data.cpu().numpy()
    final_loss = torch.Tensor([0]).cuda()
    counted_anchor = []

    loop, real_count = 0, 0
    while loop<num_total:
        loop += 1

        #####compute index
        cur_pos = np.argmax(loss_np)
        b_idx = cur_pos / (2* (5*19*19)) #second part is the size of one batch
        c_idx = (cur_pos - b_idx*(2* (5*19*19))) / 1805
        d_idx = (cur_pos - b_idx*(2* (5*19*19)) - c_idx*(1805))
        cur_pos = [b_idx, c_idx, d_idx]
        #####compute index

        loss_np[cur_pos[0], cur_pos[1], cur_pos[2]] = 0
        if visited[cur_pos[0], cur_pos[1], cur_pos[2]] == 1: #[batch_size, 2, 5*19*19]
            continue

        final_loss = final_loss + output_loss[cur_pos[0], cur_pos[1], cur_pos[2]]
        counted_anchor.append(cur_pos)
        visited[cur_pos[0], cur_pos[1], cur_pos[2]] = 1
        nms(cur_pos, proposals_box, visited, loss_np)
        real_count += 1

        if real_count>num_count:
            break

    #final_loss = torch.sum(output_loss)

    if size_average:
        final_loss /= real_count
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss, counted_anchor, real_count



# original F1 smooth loss from rcnn
def _smooth_l1( predicts, targets, counted_anchor, sigma=3.0):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
        inside_weights  = 1
        outside_weights = 1/num_examples
    '''

    #predicts = predicts.view(-1)
    #targets  = targets.view(-1)
    #weights  = weights.view(-1)
    sigma2 = sigma * sigma
    diffs  =  predicts - targets
    smooth_l1_signs = torch.abs(diffs) <  (1.0 / sigma2)
    smooth_l1_signs = smooth_l1_signs.type(torch.cuda.FloatTensor)

    smooth_l1_option1 = 0.5 * diffs* diffs *  sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5  / sigma2
    loss = smooth_l1_option1*smooth_l1_signs + smooth_l1_option2*(1-smooth_l1_signs)
    
    final_loss = torch.Tensor([0]).cuda()
    for ii in range(len(counted_anchor)):
        b, idx = counted_anchor[ii][0], counted_anchor[ii][2]
        final_loss = final_loss + loss[b, :, idx].sum()

    #loss = weights*(smooth_l1_option1*smooth_l1_signs + smooth_l1_option2*(1-smooth_l1_signs))
    #loss.sum()
    #loss = loss.sum()/(weights.sum()+1e-12)

    return final_loss



def nms(cur_pos, proposals_box, visited, loss_np):
    b_idx, c_idx, d_idx = cur_pos[0], cur_pos[1], cur_pos[2]
    batch_size, score_size = proposals_box.shape[0], 5*19*19
    chosen_box = proposals_box[b_idx, :, d_idx]

    for ii in range(score_size):
        if ii== d_idx:
            continue

        cur_box = proposals_box[b_idx, :, ii]
        iou = bb_intersection_over_union( chosen_box, cur_box )
        if iou >= 0.7: ###############iou threshold ################ 
            visited[b_idx, c_idx, d_idx] = 1
            loss_np[b_idx, c_idx, d_idx] = 0



