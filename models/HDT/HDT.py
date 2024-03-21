import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import time

from core.config import config
from core.runner_utils import index_to_time, calculate_iou, calculate_iou_accuracy, cal_statistics

from . import attention
from . import fusion
from .layers import Projection, PositionalEmbedding, Prediction
from .operation import mask_logits


class HDT(nn.Module):
    def __init__(self):
        super(HDT, self).__init__()
        configs = config.MODEL.PARAMS
        self.debug_print = configs.DEBUG
        self.video_affine = Projection(in_dim=configs.video_feature_dim,
                                       dim=configs.dim,
                                       drop_rate=configs.drop_rate)

        self.query_affine = Projection(in_dim=configs.query_feature_dim,
                                       dim=configs.dim,
                                       drop_rate=configs.drop_rate)
        self.query_position = configs.query_position
        self.video_position = configs.video_position
        if self.query_position or self.video_position:
            self.v_pos_embedding = PositionalEmbedding(configs.dim, 500)
            self.q_pos_embedding = PositionalEmbedding(configs.dim, 30)

        query_attention_layer = getattr(attention,
                                        configs.query_attention)(configs)

        video_attention_layer = getattr(attention,
                                        configs.video_attention)(configs)

        # self.query_encoder = nn.Sequential(*[
        #     copy.deepcopy(query_attention_layer)
        #     for _ in range(configs.query_attention_layers)
        # ])

        self.entity_query_encoder = nn.Sequential(*[
            copy.deepcopy(query_attention_layer)
            for _ in range(configs.query_attention_layers)
        ])

        self.motion_query_encoder = nn.Sequential(*[
            copy.deepcopy(query_attention_layer)
            for _ in range(configs.query_attention_layers)
        ])

        self.video_encoder = nn.Sequential(*[
            copy.deepcopy(video_attention_layer)
            for _ in range(configs.video_attention_layers)
        ])

        # ------------------------------------------------ early stage ------------------------------------------------
        early_attention_layer = getattr(attention,
                                        configs.early_attention)(configs)

        self.early_encoder = nn.Sequential(*[
            copy.deepcopy(early_attention_layer)
            for _ in range(configs.early_attention_layers)
        ])

        # early fusion
        self.early_fusion_layer = getattr(fusion,
                                          configs.fusion_module)(configs.dim)


        self.fg_prediction_layer = Prediction(in_dim=configs.dim,
                                              hidden_dim=configs.dim // 2,
                                              out_dim=1,
                                              drop_rate=configs.drop_rate)

        # ------------------------------------------------ latter stage ------------------------------------------------
        # latter fusion
        self.fusion_layer = getattr(fusion, configs.fusion_module)(configs.dim)
        
        post_attention_layer = getattr(attention,
                                       configs.post_attention)(configs)
        self.post_attention_layer = nn.Sequential(*[
            copy.deepcopy(post_attention_layer)
            for _ in range(configs.post_attention_layers)
        ])

        # ------------------------------------------------ bridge ------------------------------------------------
        self.entity_prompt = torch.nn.Parameter(torch.randn(configs.bridge_num, configs.dim))
        self.motion_prompt = torch.nn.Parameter(torch.randn(configs.bridge_num, configs.dim))


        self.video_encoder2 = nn.Sequential(*[
            copy.deepcopy(post_attention_layer)
            for _ in range(configs.video_attention_layers)
        ])
        self.starting = Prediction(in_dim=configs.dim,
                                   hidden_dim=configs.dim // 2,
                                   out_dim=1,
                                   drop_rate=configs.drop_rate)
        self.ending = Prediction(in_dim=configs.dim,
                                 hidden_dim=configs.dim // 2,
                                 out_dim=1,
                                 drop_rate=configs.drop_rate)

        self.intering = Prediction(in_dim=configs.dim,
                                   hidden_dim=configs.dim // 2,
                                   out_dim=1,
                                   drop_rate=configs.drop_rate)
        
        self.save_count = 0
        


    def forward(self, batch_word_vectors, batch_pos_tags, batch_txt_mask,
                batch_vis_feats, batch_vis_mask, batch_ent_weight,
                batch_mot_weight):

        # bridge 
        entity_prompt = self.entity_prompt.repeat(batch_word_vectors.shape[0], 1, 1)
        motion_bridge = self.motion_prompt.repeat(batch_word_vectors.shape[0], 1, 1)

        # weight
        zeros = batch_pos_tags.new_zeros(batch_pos_tags.shape)
        ones = batch_pos_tags.new_ones(batch_pos_tags.shape)

        entity_prob = torch.where(
            torch.abs(batch_pos_tags - 2) < 1e-10, zeros, ones)
        entity_weight = torch.mul(entity_prob, batch_ent_weight) # [B, T]
        # print(entity_weight)
        # entity_weight = torch.where(entity_weight == 0, zeros, 1/(1-entity_weight))
        # print(entity_weight)
        
        action_prob = torch.where(
            torch.abs(batch_pos_tags - 1) < 1e-10, zeros, ones)
        motion_weight = torch.mul(action_prob, batch_mot_weight) # [B, T]
        # motion_weight = torch.where(motion_weight == 0, zeros, 1/(1-motion_weight))

        # query encode
        batch_word_vectors = self.query_affine(batch_word_vectors)
        if self.query_position:
            batch_word_vectors = batch_word_vectors + self.q_pos_embedding(
                batch_word_vectors)
        batch_word_vectors = batch_word_vectors * batch_txt_mask.unsqueeze(2)

        for i, module in enumerate(self.entity_query_encoder):
            if i == 0:
                entity_word_features, entity_prompt, _= module(batch_word_vectors, 
                                                                    p1=entity_prompt,
                                                                    p2=None,
                                                                    prior_1=entity_weight,
                                                                    prior_2=None,
                                                                    mask=batch_txt_mask)
                                                                    
            else:
                entity_word_features, entity_prompt, _= module(word_features, 
                                                                    p1=entity_prompt,
                                                                    p2=None,
                                                                    prior_1=entity_weight,
                                                                    prior_2=None,
                                                                    mask=batch_txt_mask)
                
        for i, module in enumerate(self.motion_query_encoder):
            if i == 0:
                motion_word_features, motion_bridge, _= module(batch_word_vectors, 
                                                                    p1=motion_bridge,
                                                                    p2=None,
                                                                    prior_1=motion_weight,
                                                                    prior_2=None,
                                                                    mask=batch_txt_mask)
                                                                    
            else:
                motion_word_features, motion_bridge, _= module(word_features, 
                                                                    p1=motion_bridge,
                                                                    p2=None,
                                                                    prior_1=motion_weight,
                                                                    prior_2=None,
                                                                    mask=batch_txt_mask)                

        word_features = (entity_word_features + motion_word_features)/2

        self.save_count += 1

        # zeros = batch_pos_tags.new_zeros(batch_pos_tags.shape)
        # ones = batch_pos_tags.new_ones(batch_pos_tags.shape)
        entity_features = word_features * entity_weight.unsqueeze(2)
        action_features = word_features * motion_weight.unsqueeze(2)

        entity_features = word_features + entity_features
        action_features = word_features + action_features

        action_sum_feature = F.softmax(motion_bridge.sum(1), dim=-1)
        # action_sum_feature = action_features.sum(1) / torch.sum(action_prob)

        # early && latter stage video encode
        batch_vis_feats = self.video_affine(batch_vis_feats)
        if self.video_position:
            batch_vis_feats = batch_vis_feats + self.v_pos_embedding(batch_vis_feats)
        batch_vis_feats = batch_vis_feats * batch_vis_mask.unsqueeze(2)
        
        # early encoder
        video_features = torch.cat([entity_prompt, batch_vis_feats], dim=1)
        entity_prompt_mask = torch.ones((batch_word_vectors.shape[0], entity_prompt.shape[1])).to(batch_vis_mask.device)
        entity_prompt_mask = torch.cat([entity_prompt_mask, batch_vis_mask], dim=-1)

        for i, module in enumerate(self.video_encoder):
            video_features = module(video_features, entity_prompt_mask)
        video_features = video_features[:, entity_prompt.shape[1]:, :]
        entity_bridge = video_features[:, :entity_prompt.shape[1], :]

        # later
        for i, module in enumerate(self.video_encoder2):
            if i == 0:
                video_features2 = module(batch_vis_feats, batch_vis_mask)
            else:
                video_features2 = module(video_features2, batch_vis_mask)


        # Early stage
        entity_video_fused = self.early_fusion_layer(video_features,
                                                     entity_features,
                                                     entity_bridge,
                                                     batch_vis_mask,
                                                     batch_txt_mask)
        for i, module in enumerate(self.early_encoder):
            entity_video_fused = module(entity_video_fused, batch_vis_mask)

        fg_prob = self.fg_prediction_layer(entity_video_fused)
        if not self.training and self.debug_print:
            print('fg_prob', torch.sigmoid(fg_prob))
        fg_vis_feature = (video_features2 +
                          video_features) * torch.sigmoid(fg_prob)

        # Latter stage
        fused_action_feature = self.fusion_layer(fg_vis_feature,
                                                 action_features,
                                                 motion_bridge,
                                                 batch_vis_mask,
                                                 batch_txt_mask)

        for i, module in enumerate(self.post_attention_layer):
            fused_action_feature = module(fused_action_feature, batch_vis_mask)

        pred_start = self.starting(fused_action_feature).squeeze(2)
        pred_start = mask_logits(pred_start, batch_vis_mask)

        pred_end = self.ending(fused_action_feature).squeeze(2)
        pred_end = mask_logits(pred_end, batch_vis_mask)

        pred_inter = self.intering(fused_action_feature).squeeze(2)

        return pred_start, pred_end, pred_inter, word_features, video_features2, fg_prob.squeeze(
            2), video_features, batch_word_vectors, batch_vis_feats, action_sum_feature

    def compute_loss(self, pred_start, pred_end, pred_inter, start_labels,
                     end_labels, inter_label, mask):

        # start_regularity = self.regularization_score(pred_start)
        # end_regularity = self.regularization_score(pred_end)

        start_loss = self.compute_boundary_loss(pred_start, start_labels)
        end_loss = self.compute_boundary_loss(pred_end, end_labels)
        inter_loss = self.compute_location_loss(pred_inter, inter_label, mask)

        return start_loss + end_loss, inter_loss

    def compute_boundary_loss(self, pred, targets):
        return F.cross_entropy(pred, targets.long())
        # return self.regression_loss(pred, targets)

    def compute_location_loss(self, pred, targets, mask):
        weights_per_location = torch.where(targets == 0.0, targets + 1.0,
                                           2.0 * targets)
        loss_per_location = nn.BCEWithLogitsLoss(reduction='none')(pred,
                                                                   targets)
        loss_per_location = loss_per_location * weights_per_location
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask,
                         dim=1) / (torch.sum(mask, dim=1) + 1e-13)
        return loss.mean()

    def early_pred_loss(self, video_features, pred, targets, mask):
        return self.compute_location_loss(pred, targets, mask)

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2),
                             end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0],
                                   dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0],
                                 dim=1)  # (batch_size, )
        return start_index, end_index
