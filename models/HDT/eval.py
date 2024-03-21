import torch
from tqdm import tqdm
import time
import numpy as np

from core.runner_utils import index_to_time, calculate_iou, calculate_iou_accuracy, cal_statistics
from core.runner_utils import extract_index

def eval_test(model,
              data_loader,
              device,
              mode='test',
              epoch=None,
              shuffle_video_frame=False):
    ious = []
    preds, durations = [], []
    with torch.no_grad():
        during_time = 0
        for idx, batch_data in tqdm(enumerate(data_loader),
                                    total=len(data_loader),
                                    desc='evaluate {}'.format(mode)):
            data, annos = batch_data
            batch_word_vectors = data['batch_word_vectors'].to(device)
            batch_pos_tags = data['batch_pos_tags'].to(device)
            batch_txt_mask = data['batch_txt_mask'].squeeze(2).to(device)
            batch_vis_feats = data['batch_vis_feats'].to(device)
            batch_vis_mask = data['batch_vis_mask'].squeeze(2).to(device)
            batch_extend_pre = data['batch_extend_pre'].to(device)
            batch_extend_suf = data['batch_extend_suf'].to(device)
            batch_ent_weight = data["batch_entity_weight"].to(device)
            batch_mot_weight = data["batch_motion_weight"].to(device)

            if shuffle_video_frame:
                B = batch_vis_feats.shape[0]
                for i in range(B):
                    T = batch_vis_mask[i].sum().int().item()
                    pre = batch_extend_pre[i].item()
                    new_T = torch.randperm(T)
                    batch_vis_feats[i, torch.arange(T) +
                                    pre] = batch_vis_feats[i, new_T + pre]
            # compute predicted results
            with torch.cuda.amp.autocast():
                start_time = time.time()
                output = model(batch_word_vectors, batch_pos_tags,
                               batch_txt_mask, batch_vis_feats,
                               batch_vis_mask, batch_ent_weight,
                               batch_mot_weight)
                end_time = time.time()
                during_time = during_time + end_time - start_time
            start_logits, end_logits = output[0], output[1]
            start_indices, end_indices = extract_index(
                start_logits, end_logits)
            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()
            batch_vis_mask = batch_vis_mask.cpu().numpy()
            batch_extend_pre = batch_extend_pre.cpu().numpy()
            batch_extend_suf = batch_extend_suf.cpu().numpy()

            for vis_mask, start_index, end_index, extend_pre, extend_suf, anno in zip(
                    batch_vis_mask, start_indices, end_indices,
                    batch_extend_pre, batch_extend_suf, annos):
                start_time, end_time = index_to_time(
                    start_index, end_index, vis_mask.sum(), extend_pre,
                    extend_suf, anno["duration"])
                # print(start_index, end_index, vis_mask.sum(), start_time,
                #       end_time, anno["duration"])
                iou = calculate_iou(i0=[start_time, end_time],
                                    i1=anno['times'])
                ious.append(iou)
                preds.append((start_time, end_time))
                durations.append(anno["duration"])
    statistics_str = cal_statistics(preds, durations)
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    # write the scores
    score_str = "Epoch {}\n".format(epoch)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)
    print('during time:', during_time)
    return r1i3, r1i5, r1i7, mi, score_str, statistics_str