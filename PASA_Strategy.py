import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from sklearn.metrics import f1_score
import copy
import os
import json
import matplotlib.pyplot as plt
import time
import hashlib
from typing import Dict, List, Tuple, Optional
import csv
import io


class AntiMemorizationEnhancer:
    """防记忆化增强器 - 通过噪声注入防止样本ID记忆化学习"""

    def __init__(self, noise_scale=0.02, state_noise_scale=0.01, debug_mode=True):
        self.noise_scale = noise_scale  # 状态噪声强度
        self.state_noise_scale = state_noise_scale  # 状态表示噪声强度
        self.debug_mode = debug_mode
        self.noise_injection_counter = 0

        print(f"🔄 防记忆化增强器初始化")
        print(f"   - 状态噪声强度: {self.noise_scale}")
        print(f"   - 状态表示噪声强度: {self.state_noise_scale}")

    def add_state_noise(self, state):
        """为状态添加噪声，防止对特定样本的记忆化"""
        if isinstance(state, torch.Tensor):
            noise = torch.randn_like(state) * self.noise_scale
            noisy_state = state + noise
        else:  # numpy array
            noise = np.random.randn(*state.shape) * self.noise_scale
            noisy_state = state + noise

        return noisy_state

    def add_sample_index_noise(self, sample_indices, batch_size):
        """为样本索引添加轻微扰动，防止严格的ID记忆"""
        # 方法1: 随机打乱部分样本索引
        indices_array = np.array(sample_indices)

        # 随机选择10-20%的样本进行索引扰动
        num_to_perturb = max(1, int(0.15 * len(indices_array)))
        perturb_mask = np.random.choice(len(indices_array), num_to_perturb, replace=False)

        # 在这些位置上添加小的随机偏移
        for idx in perturb_mask:
            # 添加小的随机偏移（但仍保持在合理范围内）
            offset = np.random.randint(-batch_size // 4, batch_size // 4 + 1)
            indices_array[idx] = max(0, indices_array[idx] + offset)

        self.noise_injection_counter += 1

        if self.debug_mode and self.noise_injection_counter % 50 == 0:
            print(f"🔄 已对 {num_to_perturb}/{len(indices_array)} 个样本索引添加扰动")

        return indices_array.tolist()

    def create_augmented_sample_representation(self, sample_indices, features):
        """创建增强的样本表示，基于特征而非ID"""
        # 方法2: 基于特征创建虚拟样本表示
        augmented_indices = []

        for i, sample_id in enumerate(sample_indices):
            if hasattr(features, 'shape') and len(features.shape) > 1:
                # 使用特征的hash作为新的"伪ID"
                feature_hash = hash(tuple(features[i].flatten()[:10].tolist())) % 100000
                augmented_id = feature_hash
            else:
                # 如果没有特征，添加随机偏移
                augmented_id = sample_id + np.random.randint(-1000, 1000)

            augmented_indices.append(augmented_id)

        return augmented_indices


def modify_sac_strategy_for_anti_memorization(sac_strategy):
    """修改现有的SAC策略，添加防记忆化机制"""

    # 添加防记忆化增强器
    sac_strategy.anti_memorization = AntiMemorizationEnhancer()

    # 保存原始的train_step方法
    original_train_step = sac_strategy.train_step

    def train_step_with_anti_memorization(model, criterion, x, y, epoch, total_epochs, sample_indices=None):
        """带防记忆化的训练步骤"""

        # 🔄 核心修复1: 为样本索引添加噪声
        if sample_indices is not None and sac_strategy.current_phase == 2:
            # 只在第二阶段应用防记忆化
            noisy_sample_indices = sac_strategy.anti_memorization.add_sample_index_noise(
                sample_indices, x.shape[0])
        else:
            noisy_sample_indices = sample_indices

        # 🔄 核心修复2: 为输入状态添加噪声
        if sac_strategy.current_phase == 2:
            # 为输入特征添加轻微噪声
            x_noisy = x + torch.randn_like(x) * 0.01
        else:
            x_noisy = x

        # 调用原始的train_step，但使用噪声化的输入
        return original_train_step(model, criterion, x_noisy, y, epoch, total_epochs, noisy_sample_indices)

    # 替换train_step方法
    sac_strategy.train_step = train_step_with_anti_memorization

    # 保存原始的get_state方法
    original_get_state = sac_strategy.get_state

    def get_state_with_noise(model, x, y):
        """带噪声的状态获取"""
        state = original_get_state(model, x, y)

        # 🔄 核心修复3: 为状态表示添加噪声
        if sac_strategy.current_phase == 2:
            noisy_state = sac_strategy.anti_memorization.add_state_noise(state)
            return noisy_state
        else:
            return state

    # 替换get_state方法
    sac_strategy.get_state = get_state_with_noise

    print(f"🔄 SAC策略已增强防记忆化机制")
    print(f"   - 保持原有alpha值设置")
    print(f"   - 添加样本索引噪声")
    print(f"   - 添加状态表示噪声")

    return sac_strategy


# ============================================================================
# 更精细的防记忆化方案
# ============================================================================

class AdvancedAntiMemorizationEnhancer:
    """高级防记忆化增强器 - 更精细的防记忆化策略"""

    def __init__(self, feature_noise_scale=0.015, temporal_shuffle_prob=0.2, debug_mode=True):
        self.feature_noise_scale = feature_noise_scale
        self.temporal_shuffle_prob = temporal_shuffle_prob
        self.debug_mode = debug_mode
        self.sample_feature_cache = {}  # 缓存样本特征用于一致性检查

    def create_feature_based_pseudo_id(self, sample_indices, features):
        """基于特征创建伪ID，而不是直接使用样本ID"""
        pseudo_ids = []

        for i, original_id in enumerate(sample_indices):
            # 提取特征的主要成分
            if isinstance(features, torch.Tensor):
                feature_vector = features[i].flatten()
                # 使用特征的统计信息创建伪ID
                mean_val = torch.mean(feature_vector).item()
                std_val = torch.std(feature_vector).item()
                max_val = torch.max(feature_vector).item()

                # 基于特征统计创建哈希
                feature_signature = hash((
                    round(mean_val, 3),
                    round(std_val, 3),
                    round(max_val, 3)
                )) % 1000000

                # 添加少量随机性，但保持特征相关性
                pseudo_id = feature_signature + np.random.randint(-100, 100)

            else:
                # 如果不是tensor，回退到原始ID + 随机扰动
                pseudo_id = original_id + np.random.randint(-500, 500)

            pseudo_ids.append(pseudo_id)

        return pseudo_ids

    def temporal_index_shuffle(self, sample_indices, shuffle_prob=None):
        """时序索引打乱，防止时序记忆化"""
        if shuffle_prob is None:
            shuffle_prob = self.temporal_shuffle_prob

        indices = np.array(sample_indices)

        if np.random.random() < shuffle_prob:
            # 随机打乱部分索引
            num_to_shuffle = max(1, len(indices) // 3)
            shuffle_positions = np.random.choice(len(indices), num_to_shuffle, replace=False)
            shuffled_values = indices[shuffle_positions]
            np.random.shuffle(shuffled_values)
            indices[shuffle_positions] = shuffled_values

            if self.debug_mode:
                print(f"🔄 对 {num_to_shuffle}/{len(indices)} 个样本进行时序打乱")

        return indices.tolist()


def apply_simple_anti_memorization_fix(sac_strategy):
    """应用简单的防记忆化修复"""

    # 保存原始方法
    original_update_sample_confidence = sac_strategy.update_sample_confidence
    original_select_action_phase2 = sac_strategy.select_action_phase2_adaptive

    # 创建防记忆化增强器
    anti_memo = AntiMemorizationEnhancer(noise_scale=0.01, debug_mode=sac_strategy.debug_mode)

    def update_sample_confidence_with_noise(model, batch_x, batch_y, sample_indices):
        """带噪声的样本置信度更新"""
        # 为特征添加噪声
        noisy_x = batch_x + torch.randn_like(batch_x) * 0.008

        # 为样本索引添加扰动
        noisy_indices = anti_memo.add_sample_index_noise(sample_indices, batch_x.shape[0])

        # 调用原始方法
        return original_update_sample_confidence(model, noisy_x, batch_y, noisy_indices)

    def select_action_phase2_with_noise(state, sample_indices, epoch):
        """带噪声的第二阶段动作选择"""
        # 为状态添加噪声
        noisy_state = anti_memo.add_state_noise(state)

        # 为样本索引添加扰动
        noisy_indices = anti_memo.add_sample_index_noise(sample_indices, len(sample_indices))

        # 调用原始方法
        return original_select_action_phase2(noisy_state, noisy_indices, epoch)

    # 替换方法
    sac_strategy.update_sample_confidence = update_sample_confidence_with_noise
    sac_strategy.select_action_phase2_adaptive = select_action_phase2_with_noise

    return sac_strategy

def calculate_balanced_accuracy(y_true, y_pred, n_classes):
    """计算平衡准确率"""
    from sklearn.metrics import confusion_matrix
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    recalls = []
    for i in range(n_classes):
        tp = cm[i, i]
        total_true_samples = cm[i, :].sum()
        recall = tp / total_true_samples if total_true_samples > 0 else 0.0
        recalls.append(recall)

    return np.mean(recalls)


class SampleConfidenceTracker:
    """修正后的样本置信度跟踪器 - 只统计预测正确时的置信度"""

    def __init__(self, momentum=0.9, debug_mode=True):
        self.momentum = momentum
        self.debug_mode = debug_mode
        self.sample_stats = {}
        self.update_count = 0
        self.last_debug_print = 0

        # 新增：分别统计正确和错误预测
        self.correct_predictions = 0
        self.total_predictions = 0

        print(f"📊 修正版样本置信度跟踪器初始化完成，momentum={momentum}")

    def update_sample_stats(self, sample_indices: List[int], predictions: torch.Tensor,
                            true_labels: torch.Tensor):
        """
        修正后的样本统计更新 - 只有预测正确时才更新置信度统计

        Args:
            sample_indices: 样本索引列表
            predictions: 模型输出的logits [batch_size, num_classes]
            true_labels: 真实标签 [batch_size]
        """
        if not sample_indices or len(sample_indices) == 0:
            return

        batch_size = min(len(sample_indices), predictions.shape[0], true_labels.shape[0])

        for i in range(batch_size):
            try:
                sample_idx = sample_indices[i]

                # 获取当前样本的预测和真实标签
                if predictions.dim() == 1:
                    pred_logits = predictions
                else:
                    pred_logits = predictions[i]

                true_label = true_labels[i].item()

                # 计算softmax概率
                probs = F.softmax(pred_logits, dim=0)

                # 获取预测类别和对应概率
                max_prob, predicted_class = torch.max(probs, dim=0)
                predicted_class = predicted_class.item()
                max_prob = max_prob.item()

                # 统计总体预测准确性
                self.total_predictions += 1
                is_correct = (predicted_class == true_label)
                if is_correct:
                    self.correct_predictions += 1

                # 关键修正：只有预测正确时才更新置信度统计
                if is_correct:
                    # 计算置信度（预测正确时的概率）
                    confidence = max_prob

                    # 计算top1-top2差值
                    if len(probs) >= 2:
                        top2_probs, top2_indices = torch.topk(probs, min(2, len(probs)))
                        if len(top2_probs) == 2:
                            top1_top2_diff = (top2_probs[0] - top2_probs[1]).item()
                        else:
                            top1_top2_diff = confidence
                    else:
                        top1_top2_diff = confidence

                    # 更新或创建样本统计（只对正确预测的样本）
                    if sample_idx not in self.sample_stats:
                        self.sample_stats[sample_idx] = {
                            'confidence': confidence,
                            'top1_top2_diff': top1_top2_diff,
                            'update_count': 1,
                            'first_seen': self.update_count,
                            'correct_predictions': 1,
                            'total_attempts': 1,
                            'accuracy': 1.0
                        }
                    else:
                        # 指数移动平均更新（只有正确预测时才更新置信度）
                        old_stats = self.sample_stats[sample_idx]
                        old_stats['total_attempts'] += 1
                        old_stats['correct_predictions'] += 1
                        old_stats['accuracy'] = old_stats['correct_predictions'] / old_stats['total_attempts']

                        self.sample_stats[sample_idx] = {
                            'confidence': self.momentum * old_stats['confidence'] + (1 - self.momentum) * confidence,
                            'top1_top2_diff': self.momentum * old_stats['top1_top2_diff'] + (
                                    1 - self.momentum) * top1_top2_diff,
                            'update_count': old_stats['update_count'] + 1,
                            'first_seen': old_stats['first_seen'],
                            'correct_predictions': old_stats['correct_predictions'],
                            'total_attempts': old_stats['total_attempts'],
                            'accuracy': old_stats['accuracy']
                        }
                else:
                    # 预测错误时，只更新准确率统计，不更新置信度
                    if sample_idx in self.sample_stats:
                        old_stats = self.sample_stats[sample_idx]
                        old_stats['total_attempts'] += 1
                        old_stats['accuracy'] = old_stats['correct_predictions'] / old_stats['total_attempts']
                    else:
                        # 如果是第一次见到且预测错误，创建基础统计但不设置置信度
                        self.sample_stats[sample_idx] = {
                            'confidence': 0.5,  # 默认中等置信度
                            'top1_top2_diff': 0.1,  # 默认小差值
                            'update_count': 0,  # 因为没有正确预测，所以update_count保持为0
                            'first_seen': self.update_count,
                            'correct_predictions': 0,
                            'total_attempts': 1,
                            'accuracy': 0.0
                        }

                self.update_count += 1

            except Exception as e:
                if self.debug_mode:
                    print(f"样本{sample_idx}统计更新失败: {e}")
                continue

        # 定期打印调试信息
        if self.debug_mode and self.update_count - self.last_debug_print >= 100:
            self.print_debug_info()
            self.last_debug_print = self.update_count

    def get_sample_difficulty(self, sample_idx: int) -> str:
        """
        修正后的难度评估 - 基于预测正确时的置信度和样本准确率
        """
        if sample_idx not in self.sample_stats:
            return 'medium'  # 默认中等难度

        stats = self.sample_stats[sample_idx]
        confidence = stats['confidence']
        diff = stats['top1_top2_diff']
        accuracy = stats.get('accuracy', 0.5)
        correct_count = stats.get('correct_predictions', 0)

        # 修正的分类逻辑：
        # 1. 如果样本从未被正确预测过，认为是困难样本
        # 2. 如果准确率很低，也认为是困难样本
        # 3. 只有在预测正确且置信度高时才认为是易增强样本

        if correct_count == 0 or accuracy < 0.3:
            return 'hard'  # 从未正确预测或准确率很低 -> 困难
        elif confidence > 0.8 and diff > 0.3 and accuracy > 0.7:
            return 'easy'  # 高置信度、大差值、高准确率 -> 易增强
        else:
            return 'medium'  # 中等难度

    def get_all_difficulties(self) -> Dict[str, int]:
        """获取所有样本的难度分布"""
        difficulties = {'easy': 0, 'medium': 0, 'hard': 0}
        for sample_idx in self.sample_stats:
            difficulty = self.get_sample_difficulty(sample_idx)
            difficulties[difficulty] += 1
        return difficulties

    def print_debug_info(self):
        """打印调试信息"""
        total_samples = len(self.sample_stats)
        difficulties = self.get_all_difficulties()

        overall_accuracy = self.correct_predictions / max(1, self.total_predictions)

        print(f"📊 修正版置信度跟踪器状态:")
        print(f"   总样本数: {total_samples}")
        print(f"   更新次数: {self.update_count}")
        print(f"   总体准确率: {overall_accuracy:.4f} ({self.correct_predictions}/{self.total_predictions})")
        print(f"   难度分布: {difficulties}")

        if total_samples > 0:
            # 只统计有正确预测的样本的置信度
            samples_with_correct_preds = [stats for stats in self.sample_stats.values()
                                          if stats.get('correct_predictions', 0) > 0]

            if samples_with_correct_preds:
                avg_confidence = np.mean([stats['confidence'] for stats in samples_with_correct_preds])
                avg_diff = np.mean([stats['top1_top2_diff'] for stats in samples_with_correct_preds])
                avg_accuracy = np.mean([stats['accuracy'] for stats in samples_with_correct_preds])

                print(f"   有正确预测的样本数: {len(samples_with_correct_preds)}")
                print(f"   平均置信度: {avg_confidence:.4f}")
                print(f"   平均top1-top2差值: {avg_diff:.4f}")
                print(f"   平均样本准确率: {avg_accuracy:.4f}")
            else:
                print(f"   ⚠️ 暂无样本有正确预测！")

    def get_confidence_statistics(self) -> Dict[str, float]:
        """获取置信度统计信息"""
        if not self.sample_stats:
            return {
                'total_samples': 0,
                'samples_with_correct_predictions': 0,
                'overall_accuracy': 0.0,
                'avg_confidence': 0.5,
                'avg_accuracy': 0.0
            }

        samples_with_correct = [stats for stats in self.sample_stats.values()
                                if stats.get('correct_predictions', 0) > 0]

        return {
            'total_samples': len(self.sample_stats),
            'samples_with_correct_predictions': len(samples_with_correct),
            'overall_accuracy': self.correct_predictions / max(1, self.total_predictions),
            'avg_confidence': np.mean([s['confidence'] for s in samples_with_correct]) if samples_with_correct else 0.5,
            'avg_accuracy': np.mean([s['accuracy'] for s in self.sample_stats.values()]),
            'confidence_std': np.std([s['confidence'] for s in samples_with_correct]) if samples_with_correct else 0.0
        }


class AdaptiveAlphaStrategy:
    """自适应Alpha策略 - 为不同难度样本使用不同的探索强度"""

    def __init__(self, device, alpha_lr=3e-4, debug_mode=True):
        self.device = device
        self.debug_mode = debug_mode

        # 修正：为不同难度设置更合理的初始alpha值
        self.difficulty_log_alphas = {
            'easy': torch.tensor(np.log(1.0), requires_grad=True, device=device),  # 很高探索 - 易增强样本全范围搜索
            'medium': torch.tensor(np.log(0.3), requires_grad=True, device=device),  # 中等探索 - 平衡探索利用
            # 'hard': 困难样本不需要alpha，直接固定no_aug
        }

        # 为每个难度创建独立的优化器（困难样本不需要）
        self.alpha_optimizers = {}
        for difficulty in ['easy', 'medium']:
            self.alpha_optimizers[difficulty] = optim.Adam(
                [self.difficulty_log_alphas[difficulty]],
                lr=alpha_lr
            )

        # 目标熵 - 为每个难度设置不同的目标熵
        self.target_entropies = {
            'easy': -np.log(1.0 / 9) * 1.2,  # 很高目标熵，强烈鼓励探索
            'medium': -np.log(1.0 / 9) * 0.6,  # 中等目标熵
            # 'hard': 困难样本固定策略，不需要熵
        }

        # 统计信息
        self.alpha_update_counts = defaultdict(int)
        self.alpha_losses = defaultdict(list)

        print(f"🎯 修正版自适应Alpha策略初始化完成")
        for diff, alpha in self.difficulty_log_alphas.items():
            print(f"   {diff}: alpha={alpha.exp().item():.4f}, target_entropy={self.target_entropies[diff]:.4f}")
        print(f"   hard: 固定选择no_augmentation，不参与SAC学习")

    def get_alpha_for_difficulty(self, difficulty: str) -> torch.Tensor:
        """获取指定难度的alpha值"""
        # 修正：困难样本不需要alpha，因为固定选择no_aug
        if difficulty == 'hard':
            return torch.tensor(0.0, device=self.device)  # 困难样本不使用alpha
        if difficulty not in self.difficulty_log_alphas:
            difficulty = 'medium'  # 默认使用medium
        return self.difficulty_log_alphas[difficulty].exp()

    def update_alpha_for_difficulty(self, difficulty: str, log_prob: torch.Tensor):
        """为指定难度更新alpha"""
        # 修正：困难样本固定策略，不需要更新alpha
        if difficulty == 'hard':
            return 0.0  # 困难样本不更新alpha

        if difficulty not in self.difficulty_log_alphas:
            difficulty = 'medium'

        try:
            # 计算alpha损失
            target_entropy = self.target_entropies[difficulty]
            alpha_loss = -(self.difficulty_log_alphas[difficulty] *
                           (log_prob + target_entropy).detach()).mean()

            # 更新alpha
            self.alpha_optimizers[difficulty].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizers[difficulty].step()

            # 记录统计
            self.alpha_update_counts[difficulty] += 1
            self.alpha_losses[difficulty].append(alpha_loss.item())

            return alpha_loss.item()

        except Exception as e:
            if self.debug_mode:
                print(f"更新{difficulty}难度alpha失败: {e}")
            return 0.0

    def get_all_alphas(self) -> Dict[str, float]:
        """获取所有难度的当前alpha值"""
        alphas = {}
        for diff, log_alpha in self.difficulty_log_alphas.items():
            alphas[diff] = log_alpha.exp().item()
        alphas['hard'] = 0.0  # 困难样本固定策略
        return alphas

    def print_alpha_statistics(self):
        """打印alpha统计信息"""
        if self.debug_mode:
            print(f"🎯 Alpha策略统计:")
            for difficulty in ['easy', 'medium']:
                alpha_val = self.get_alpha_for_difficulty(difficulty).item()
                update_count = self.alpha_update_counts[difficulty]
                avg_loss = np.mean(self.alpha_losses[difficulty][-10:]) if self.alpha_losses[difficulty] else 0.0
                print(f"   {difficulty}: alpha={alpha_val:.4f}, updates={update_count}, avg_loss={avg_loss:.6f}")
            print(f"   hard: 固定no_augmentation策略，无需alpha更新")


class ExplorationScheduler:
    """探索性调度器 - 通过噪声注入维持探索性，避免激进的网络重置"""

    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.exploration_decay = 0.98  # 稍微调慢衰减
        self.min_exploration = 0.02
        self.noise_injection_interval = 15  # 每15个epoch注入一次探索噪声

        print(f"🔍 轻量级探索性调度器初始化完成")
        print(f"   - 探索噪声注入间隔: {self.noise_injection_interval}个epoch")
        print(f"   - 最小探索强度: {self.min_exploration}")

    def get_exploration_boost(self, epoch: int, total_epochs: int) -> float:
        """根据训练进度计算探索性增强"""
        progress = epoch / total_epochs

        # 后期适当提高探索性，防止固化
        if progress > 0.8:  # 最后20%阶段
            return 0.05  # 轻微提升探索性
        elif progress > 0.6:  # 中后期
            return 0.02  # 保持基础探索性
        else:
            return 0.0  # 前期正常训练

    def add_exploration_noise(self, action_logits: torch.Tensor, epoch: int) -> torch.Tensor:
        """为动作logits添加适度的探索噪声"""
        if epoch % self.noise_injection_interval == 0:  # 定期注入探索噪声
            noise_scale = max(self.min_exploration,
                              0.1 * (self.exploration_decay ** (epoch // self.noise_injection_interval)))
            exploration_noise = torch.randn_like(action_logits) * noise_scale

            if self.debug_mode and epoch % (self.noise_injection_interval * 5) == 0:
                print(f"🔍 Epoch {epoch}: 注入探索噪声，强度={noise_scale:.4f}")

            return action_logits + exploration_noise
        return action_logits

    def should_encourage_exploration(self, epoch: int) -> bool:
        """判断是否需要额外鼓励探索"""
        # 每50个epoch检查一次是否需要额外探索
        return (epoch % 50 == 0 and epoch > 100)

    def get_exploration_bonus(self, epoch: int, total_epochs: int) -> float:
        """获取探索奖励加成"""
        if self.should_encourage_exploration(epoch):
            progress = epoch / total_epochs
            if progress > 0.7:  # 后期给予更多探索激励
                return 0.02
            else:
                return 0.01
        return 0.0


class MetricsCollector:
    """
    全面的指标收集器：写入 CSV，并可选上传图表到 Comet
    """

    def __init__(self, log_dir, experiment=None, record_every=10, compute_grad_dot_every=50):
        """
        Args:
            log_dir: 存放 CSV 的文件夹
            experiment: Comet 实验对象，如果为 None 则不传图
            record_every: 每多少个 batch 记录一次（避免过密）
            compute_grad_dot_every: 每多少个 batch 计算一次梯度点积（耗时操作）
        """
        self.log_dir = log_dir
        self.experiment = experiment
        self.record_every = record_every
        self.compute_grad_dot_every = compute_grad_dot_every
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, 'training_metrics.csv')

        # 定义 CSV 列
        self.columns = [
            'step', 'phase', 'epoch', 'batch_idx',
            'ce_loss', 'batch_acc', 'batch_ba_improve',
            'actor_loss', 'critic1_loss', 'critic2_loss',
            'alpha_easy', 'alpha_medium', 'alpha_hard',
            'selected_op', 'selected_mag',
            'grad_cos_sim', 'grad_delta',  # 梯度点积相关
            'avg_confidence_correct', 'confidence_std'
        ]

        # 初始化 CSV 文件
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

        # 缓存最近记录的数据，用于绘图
        self.cache = {col: [] for col in self.columns}
        self.last_flush_step = 0

        # 用于梯度点积的缓存（验证集梯度的平均值）
        self.val_grad_avg = None  # 将在外部设置

    def set_validation_gradient(self, val_grad_avg):
        """设置预计算的验证集平均梯度（向量），用于计算余弦相似度"""
        self.val_grad_avg = val_grad_avg

    def compute_gradient_cos_sim(self, model, batch_x, batch_y, device):
        """
        计算 batch 的平均梯度与验证集平均梯度的余弦相似度
        简化版：只对最后一层参数计算梯度（计算量小）
        """
        if self.val_grad_avg is None:
            return 0.0
        model.eval()
        batch_x.requires_grad_(True)
        outputs = model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)
        # 只取最后一层参数的梯度（例如 classifier 的权重）
        # 这里假设 model 是 CustomEfficientNet，其 base_model._fc 是最后一层
        try:
            loss.backward(retain_graph=True)
            grad_list = []
            # 获取最后一层参数的梯度（展平）
            last_layer = model.base_model._fc
            if last_layer.weight.grad is not None:
                grad_list.append(last_layer.weight.grad.view(-1))
            if last_layer.bias is not None and last_layer.bias.grad is not None:
                grad_list.append(last_layer.bias.grad.view(-1))
            if grad_list:
                grad_batch = torch.cat(grad_list).detach()
            else:
                grad_batch = torch.zeros(1, device=device)
        except:
            grad_batch = torch.zeros(1, device=device)
        # 清除梯度
        model.zero_grad()
        batch_x.grad = None
        model.train()
        # 计算余弦相似度
        if grad_batch.norm() == 0 or self.val_grad_avg.norm() == 0:
            return 0.0
        cos_sim = F.cosine_similarity(grad_batch.unsqueeze(0), self.val_grad_avg.unsqueeze(0)).item()
        return cos_sim

    def record(self, step, phase, epoch, batch_idx, metrics_dict):
        """
        记录一批指标
        metrics_dict 应包含部分或全部与 self.columns 对应的键
        """
        # 按记录频率控制
        if step % self.record_every != 0:
            return

        # 构建一行数据
        row = {}
        for col in self.columns:
            if col in metrics_dict:
                row[col] = metrics_dict[col]
            else:
                row[col] = ''
        row['step'] = step
        row['phase'] = phase
        row['epoch'] = epoch
        row['batch_idx'] = batch_idx

        # 写入 CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)

        # 缓存用于绘图
        for key, val in row.items():
            if isinstance(val, (int, float)) and key in self.cache:
                self.cache[key].append((step, val))

        # 定期生成图表并上传到 Comet
        if self.experiment is not None and (step - self.last_flush_step) >= 100:
            self._plot_and_log(step)
            self.last_flush_step = step

    def _plot_and_log(self, current_step):
        """生成关键指标的曲线图并上传到 Comet"""
        if not self.experiment:
            return
        # 绘制 loss 曲线
        if len(self.cache['ce_loss']) > 1:
            steps, losses = zip(*self.cache['ce_loss'])
            plt.figure()
            plt.plot(steps, losses, label='CE Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('CE Loss over Training')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.experiment.log_image(buf, name='ce_loss_curve', step=current_step)
            plt.close()
        # 绘制 actor_loss
        if len(self.cache['actor_loss']) > 1:
            steps, vals = zip(*self.cache['actor_loss'])
            plt.figure()
            plt.plot(steps, vals, label='Actor Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Actor Loss')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.experiment.log_image(buf, name='actor_loss_curve', step=current_step)
            plt.close()
        # 绘制 alpha 值变化
        for alpha_name in ['alpha_easy', 'alpha_medium']:
            if len(self.cache.get(alpha_name, [])) > 1:
                steps, vals = zip(*self.cache[alpha_name])
                plt.figure()
                plt.plot(steps, vals, label=alpha_name)
                plt.xlabel('Training Step')
                plt.ylabel('Alpha')
                plt.title(f'{alpha_name} over Training')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                self.experiment.log_image(buf, name=f'{alpha_name}_curve', step=current_step)
                plt.close()
        # 梯度点积曲线（如果有）
        if len(self.cache.get('grad_cos_sim', [])) > 1:
            steps, vals = zip(*self.cache['grad_cos_sim'])
            plt.figure()
            plt.plot(steps, vals, label='Grad Cos Sim')
            plt.xlabel('Training Step')
            plt.ylabel('Cosine Similarity')
            plt.title('Gradient Alignment with Validation')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.experiment.log_image(buf, name='grad_cos_sim_curve', step=current_step)
            plt.close()

    def close(self):
        """结束时生成最终图表（如有需要）"""
        if self.experiment:
            self._plot_and_log(self.last_flush_step + 1)


class TwoPhaseSACStrategy:
    """
    修正后的二阶段SAC自适应增强策略
    - 修正奖励计算：使用Balanced Accuracy
    - 自适应Alpha策略：针对不同难度使用不同探索强度
    - 探索性维持：防止后期策略固化
    """

    def __init__(self, n_classes, feature_dim, aug_operations, device,
                 buffer_capacity=10000, gamma=0.95, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 validation_set=None, n_magnitude_levels=5,
                 phase2_trigger_patience=20, phase2_trigger_threshold=0.02,
                 debug_mode=True, **kwargs):

        self.device = device
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.aug_operations = aug_operations
        self.n_magnitude_levels = n_magnitude_levels
        self.debug_mode = debug_mode

        # SAC参数
        self.gamma = gamma
        self.tau = tau

        # 二阶段参数
        self.current_phase = 1
        self.phase2_trigger_patience = phase2_trigger_patience
        self.phase2_trigger_threshold = phase2_trigger_threshold
        self.performance_history = deque(maxlen=phase2_trigger_patience + 5)
        self.phase2_triggered = False

        # 修正：样本置信度跟踪器
        self.confidence_tracker = SampleConfidenceTracker(debug_mode=debug_mode)

        # 新增：自适应Alpha策略
        self.adaptive_alpha = AdaptiveAlphaStrategy(device, alpha_lr, debug_mode)

        # 新增：探索性调度器
        self.exploration_scheduler = ExplorationScheduler(debug_mode)

        # 状态和动作维度
        self.state_dim = feature_dim * 2 + n_classes
        self.action_dim = len(aug_operations) + 1 + n_magnitude_levels

        # 初始化幅度映射
        self.magnitude_ranges = self._init_magnitude_ranges()

        # 构建网络
        self._build_networks()

        # 优化器 - 注意：不再包括alpha优化器，由AdaptiveAlphaStrategy管理
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.validation_set = validation_set

        # 统计信息
        self.statistics = defaultdict(list)
        self.operation_stats = defaultdict(int)
        self.magnitude_stats = defaultdict(lambda: defaultdict(int))
        self.training_step = 0

        # 修复：初始化完整的phase2_stats
        self.phase2_stats = {
            'easy_samples': 0,
            'medium_samples': 0,
            'hard_samples': 0,
            'phase2_epochs': 0,
            'phase2_start_epoch': -1
        }

        # 修正：移除单一的log_alpha，由AdaptiveAlphaStrategy管理
        # self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)

        # 修正：添加样本索引管理
        self._last_phase2_epoch = -1
        self._last_validation_time = time.time()
        self._global_sample_counter = 0  # 全局样本计数器

        # 新增：指标收集器（需要传入 log_dir 和 experiment）
        self.metrics_collector = None  # 由外部设置

        print(f"🚀 修正版二阶段SAC策略初始化完成")
        print(f"   - 修正奖励：使用Balanced Accuracy")
        print(f"   - 困难样本：固定选择no_augmentation，不参与SAC学习")
        print(f"   - 易增强样本：alpha=1.0，全范围高强度搜索")
        print(f"   - 中等难度样本：alpha=0.3，适度限制幅度")
        print(f"   - 探索性维持：轻量级噪声注入，避免激进重置")
        print(f"   - 阶段1: Batch-level adaptation")
        print(f"   - 阶段2: Sample-level adaptation with fixed hard sample strategy")

    def set_metrics_collector(self, collector):
        self.metrics_collector = collector

    def _init_magnitude_ranges(self):
        """初始化幅度范围"""
        ranges = {}
        configs = {
            "time_mask": (0.05, 0.3),
            "frequency_mask": (0.05, 0.3),
            "noise_injection": (0.01, 0.1),
            "random_quantization": (0.1, 0.5),
            "spectral_contrast": (0.1, 0.4),
            "harmonic_perturbation": (0.05, 0.2),
            "breathing_cycle_stretch": (0.05, 0.25),
            "low_freq_emphasis": (0.1, 0.4)
        }

        for operation in self.aug_operations:
            if operation in configs:
                min_val, max_val = configs[operation]
            else:
                min_val, max_val = 0.1, 0.3

            magnitudes = np.linspace(min_val, max_val, self.n_magnitude_levels)
            ranges[operation] = magnitudes.tolist()

        ranges["no_augmentation"] = [0.0] * self.n_magnitude_levels
        return ranges

    def _build_networks(self):
        """构建SAC网络"""
        # Actor: 支持动态动作空间
        self.actor = AdaptiveActorNetwork(
            self.state_dim, len(self.aug_operations) + 1, self.n_magnitude_levels
        ).to(self.device)

        # Critics
        max_action_dim = (len(self.aug_operations) + 1) + self.n_magnitude_levels
        self.critic1 = CriticNetwork(self.state_dim, max_action_dim).to(self.device)
        self.critic2 = CriticNetwork(self.state_dim, max_action_dim).to(self.device)

        # Target critics
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

    @property
    def alpha(self):
        """返回默认的alpha值（兼容性）"""
        return self.adaptive_alpha.get_alpha_for_difficulty('medium')

    def generate_stable_sample_indices(self, batch_size: int, sample_indices: List[int] = None) -> List[int]:
        """
        修正：生成稳定的样本索引
        优先使用数据加载器提供的索引，否则使用全局计数器
        """
        if sample_indices is not None and len(sample_indices) == batch_size:
            return sample_indices

        # 使用全局计数器生成稳定索引
        stable_indices = []
        for i in range(batch_size):
            stable_indices.append(self._global_sample_counter + i)

        self._global_sample_counter += batch_size
        return stable_indices

    def calculate_ba_reward(self, model, original_x, augmented_x, y_true):
        """
        修正：计算基于Balanced Accuracy的奖励
        """
        try:
            model.eval()
            with torch.no_grad():
                # 计算原始数据的BA
                orig_outputs = model(original_x)
                orig_preds = torch.argmax(orig_outputs, dim=1)
                orig_ba = calculate_balanced_accuracy(
                    y_true.cpu().numpy(),
                    orig_preds.cpu().numpy(),
                    self.n_classes
                )

                # 计算增强数据的BA
                aug_outputs = model(augmented_x)
                aug_preds = torch.argmax(aug_outputs, dim=1)
                aug_ba = calculate_balanced_accuracy(
                    y_true.cpu().numpy(),
                    aug_preds.cpu().numpy(),
                    self.n_classes
                )

                # BA改善作为奖励
                ba_reward = aug_ba - orig_ba

            model.train()
            return ba_reward

        except Exception as e:
            if self.debug_mode:
                print(f"计算BA奖励失败: {e}")
            model.train()
            return 0.0

    def calculate_sample_level_ba_rewards(self, model, original_x, augmented_x, y, sample_indices):
        """
        修正：计算样本级别的BA奖励
        """
        sample_rewards = []

        try:
            # 为每个样本单独计算BA奖励
            for i in range(len(sample_indices)):
                single_orig = original_x[i:i + 1]
                single_aug = augmented_x[i:i + 1]
                single_y = y[i:i + 1]

                sample_ba_reward = self.calculate_ba_reward(model, single_orig, single_aug, single_y)
                sample_rewards.append(sample_ba_reward)

        except Exception as e:
            if self.debug_mode:
                print(f"计算样本级BA奖励失败: {e}")
            # 返回默认奖励
            sample_rewards = [0.0] * len(sample_indices)

        return sample_rewards

    def get_difficulty_action_mask(self, difficulty: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据样本难度获取动作掩码
        注意：困难样本现在直接固定选择no_aug，不使用此方法

        Returns:
            operation_mask: [n_operations] 操作掩码 (1=允许, 0=禁止)
            magnitude_mask: [n_magnitude_levels] 幅度掩码
        """
        n_operations = len(self.aug_operations) + 1  # +1 for no_augmentation
        operation_mask = torch.ones(n_operations, device=self.device)
        magnitude_mask = torch.ones(self.n_magnitude_levels, device=self.device)

        if difficulty == 'easy':
            # 易增强样本：允许所有操作和所有幅度级别，全范围搜索
            pass  # 保持全1掩码，最大搜索空间

        elif difficulty == 'medium':
            # 中等难度：允许所有操作，但限制为较轻的幅度（前3个级别）
            magnitude_mask[3:] = 0  # 禁用强幅度

        elif difficulty == 'hard':
            # 困难样本：这个分支实际上不会被调用，因为困难样本直接固定选择no_aug
            # 但为了代码完整性，保留这个逻辑
            operation_mask[:-1] = 0  # 禁用所有增强操作，只保留最后一个(no_aug)
            magnitude_mask[1:] = 0  # 只保留第一个幅度级别

        return operation_mask, magnitude_mask

    def select_action_phase1(self, state, epoch=0, deterministic=False):
        """阶段1：batch-level动作选择，添加探索性增强"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 使用完整搜索空间
        n_operations = len(self.aug_operations) + 1
        n_magnitudes = self.n_magnitude_levels

        if deterministic:
            operation_logits, magnitude_logits = self.actor(state, n_operations, n_magnitudes)

            # 添加探索性噪声（即使在确定性模式下也适当添加）
            operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)

            operation_idx = torch.argmax(F.softmax(operation_logits, dim=-1), dim=-1)
            magnitude_idx = torch.argmax(F.softmax(magnitude_logits, dim=-1), dim=-1)
        else:
            operation_logits, magnitude_logits, _ = self.actor.sample(state, n_operations, n_magnitudes)

            # 添加探索性噪声
            operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)
            magnitude_logits = self.exploration_scheduler.add_exploration_noise(magnitude_logits, epoch)

            operation_dist = torch.distributions.Categorical(logits=operation_logits)
            magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)
            operation_idx = operation_dist.sample()
            magnitude_idx = magnitude_dist.sample()

        # 构建动作向量
        action = torch.zeros(n_operations + n_magnitudes).to(self.device)
        action[operation_idx] = 1.0
        action[n_operations + magnitude_idx] = 1.0

        return action.cpu().numpy(), operation_idx.item(), magnitude_idx.item()

    def select_action_phase2_adaptive(self, state, sample_indices, epoch=0, deterministic=False):
        """
        修正：第二阶段自适应动作选择，困难样本固定选择no_aug
        """
        batch_actions = []
        batch_operation_indices = []
        batch_magnitude_indices = []
        batch_difficulties = []
        batch_log_probs = []

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        for sample_idx in sample_indices:
            # 获取样本难度
            difficulty = self.confidence_tracker.get_sample_difficulty(sample_idx)
            batch_difficulties.append(difficulty)

            # 修正：困难样本直接选择no_augmentation，不参与SAC学习
            if difficulty == 'hard':
                # 困难样本固定策略
                operation_name = 'no_augmentation'
                magnitude_idx_item = 0
                operation_idx_item = len(self.aug_operations)  # no_aug的索引

                batch_actions.append((operation_name, magnitude_idx_item))
                batch_operation_indices.append(operation_idx_item)
                batch_magnitude_indices.append(magnitude_idx_item)

                # 困难样本不需要log_prob，因为不参与SAC更新
                batch_log_probs.append(torch.tensor(0.0, device=self.device))
                continue

            # 易增强和中等难度样本通过SAC学习
            # 获取该难度的动作掩码
            operation_mask, magnitude_mask = self.get_difficulty_action_mask(difficulty)

            # 获取该难度的alpha值
            current_alpha = self.adaptive_alpha.get_alpha_for_difficulty(difficulty)

            # 使用SAC网络，但应用难度掩码和自适应alpha
            n_operations = operation_mask.shape[0]
            n_magnitudes = magnitude_mask.shape[0]

            if deterministic:
                operation_logits, magnitude_logits = self.actor(state_tensor, n_operations, n_magnitudes)

                # 应用温度调整（alpha）和掩码
                operation_logits = operation_logits / (current_alpha + 1e-8)  # 避免除零
                magnitude_logits = magnitude_logits / (current_alpha + 1e-8)

                # 添加探索性噪声（即使在确定性模式下，易增强样本也要保持探索）
                if difficulty == 'easy':
                    operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)
                    magnitude_logits = self.exploration_scheduler.add_exploration_noise(magnitude_logits, epoch)

                # 应用掩码：禁用的操作设为很小的值
                operation_logits = operation_logits + (operation_mask - 1) * 1e9
                magnitude_logits = magnitude_logits + (magnitude_mask - 1) * 1e9

                operation_idx = torch.argmax(operation_logits, dim=-1)
                magnitude_idx = torch.argmax(magnitude_logits, dim=-1)

                # 计算log_prob（用于alpha更新）
                op_dist = torch.distributions.Categorical(logits=operation_logits)
                mag_dist = torch.distributions.Categorical(logits=magnitude_logits)
                log_prob = op_dist.log_prob(operation_idx) + mag_dist.log_prob(magnitude_idx)

            else:
                operation_logits, magnitude_logits, _ = self.actor.sample(state_tensor, n_operations, n_magnitudes)

                # 应用温度调整
                operation_logits = operation_logits / (current_alpha + 1e-8)
                magnitude_logits = magnitude_logits / (current_alpha + 1e-8)

                # 易增强样本：强化探索性
                if difficulty == 'easy':
                    # 增强探索性：添加噪声+探索奖励
                    exploration_bonus = self.exploration_scheduler.get_exploration_bonus(epoch, 300)
                    operation_logits = self.exploration_scheduler.add_exploration_noise(operation_logits, epoch)
                    magnitude_logits = self.exploration_scheduler.add_exploration_noise(magnitude_logits, epoch)
                    # 为易增强样本添加额外的随机性
                    operation_logits += torch.randn_like(operation_logits) * 0.1
                    magnitude_logits += torch.randn_like(magnitude_logits) * 0.1

                # 应用掩码
                operation_logits = operation_logits + (operation_mask - 1) * 1e9
                magnitude_logits = magnitude_logits + (magnitude_mask - 1) * 1e9

                operation_dist = torch.distributions.Categorical(logits=operation_logits)
                magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)

                operation_idx = operation_dist.sample()
                magnitude_idx = magnitude_dist.sample()

                log_prob = operation_dist.log_prob(operation_idx) + magnitude_dist.log_prob(magnitude_idx)

            batch_log_probs.append(log_prob)

            # 映射到具体的增强操作
            operation_idx_item = operation_idx.item()
            magnitude_idx_item = magnitude_idx.item()

            if operation_idx_item >= len(self.aug_operations):
                operation_name = 'no_augmentation'
            else:
                operation_name = self.aug_operations[operation_idx_item]

            batch_actions.append((operation_name, magnitude_idx_item))
            batch_operation_indices.append(operation_idx_item)
            batch_magnitude_indices.append(magnitude_idx_item)

        # 返回批次平均用于向后兼容，同时返回log_probs用于alpha更新
        avg_operation_idx = int(np.mean(batch_operation_indices))
        avg_magnitude_idx = int(np.mean(batch_magnitude_indices))

        return batch_actions, avg_operation_idx, avg_magnitude_idx, batch_difficulties, batch_log_probs

    def apply_sample_level_augmentation(self, x, y, batch_actions, sample_indices):
        """
        应用样本级别的增强
        """
        augmented_x = x.clone()
        augmented_y = y.clone()

        for i, ((operation_name, magnitude_idx), sample_idx) in enumerate(zip(batch_actions, sample_indices)):
            try:
                if i >= x.shape[0]:  # 防止索引越界
                    break

                # 获取单个样本
                single_x = x[i:i + 1]
                single_y = y[i:i + 1]

                # 应用增强
                if operation_name == 'no_augmentation':
                    # 不增强，保持原样
                    continue
                else:
                    # 查找操作索引
                    if operation_name in self.aug_operations:
                        operation_idx = self.aug_operations.index(operation_name)
                        aug_x, aug_y = self.apply_augmentation(single_x, single_y, operation_idx, magnitude_idx)
                        augmented_x[i:i + 1] = aug_x
                        augmented_y[i:i + 1] = aug_y
            except Exception as e:
                if self.debug_mode:
                    print(f"样本{sample_idx}增强失败: {e}")
                continue

        return augmented_x, augmented_y

    def check_phase2_trigger(self, current_performance: float) -> bool:
        """
        检查是否触发第二阶段（修正版：基于连续无提升）
        """
        if self.phase2_triggered:
            return True

        self.performance_history.append(current_performance)

        # 必须至少有 patience + 1 个性能记录才考虑切换
        if len(self.performance_history) <= self.phase2_trigger_patience:
            if self.debug_mode and len(self.performance_history) % 5 == 0:
                print(f"📊 第二阶段触发检查 (Epoch {len(self.performance_history) - 1}):")
                print(f"   性能历史长度: {len(self.performance_history)}")
                print(f"   需要达到长度: {self.phase2_trigger_patience + 1}")
                print(f"   当前性能: {current_performance:.4f}")
            return False

        # 获取"历史"最佳性能（不包括最近patience轮的性能）
        historical_performances = list(self.performance_history)[:-self.phase2_trigger_patience]
        if len(historical_performances) == 0:
            return False

        historical_best = max(historical_performances)
        recent_performances = list(self.performance_history)[-self.phase2_trigger_patience:]

        # 计算有多少轮没有显著提升
        no_improvement_count = 0
        for perf in recent_performances:
            if perf <= historical_best + self.phase2_trigger_threshold:
                no_improvement_count += 1

        # 调试信息
        if self.debug_mode and len(self.performance_history) % 5 == 0:
            current_epoch = len(self.performance_history) - 1
            print(f"📊 第二阶段触发检查 (Epoch {current_epoch}):")
            print(f"   历史最佳: {historical_best:.4f}")
            print(f"   当前性能: {current_performance:.4f}")
            print(f"   需要超过: {historical_best + self.phase2_trigger_threshold:.4f}")
            print(f"   无提升轮数: {no_improvement_count}/{self.phase2_trigger_patience}")

        # 如果连续patience轮都没有显著提升，触发第二阶段
        if no_improvement_count >= self.phase2_trigger_patience:
            print(f"\n🔄 触发第二阶段！连续{self.phase2_trigger_patience}轮无显著提升")
            print(f"   历史最佳性能（第{len(historical_performances)}轮前）: {historical_best:.4f}")
            print(f"   最近{self.phase2_trigger_patience}轮最佳: {max(recent_performances):.4f}")
            print(f"   最近{self.phase2_trigger_patience}轮最差: {min(recent_performances):.4f}")
            print(f"   提升容忍度: {self.phase2_trigger_threshold}")
            print(f"   切换到样本级自适应增强策略（自适应Alpha架构）")

            self.current_phase = 2
            self.phase2_triggered = True
            return True

        return False

    def calculate_validation_performance(self, model, criterion):
        """
        计算验证集性能，用于第二阶段触发判断
        """
        if self.validation_set is None:
            return 0.5  # 默认值

        # 限制验证频率
        current_time = time.time()
        if hasattr(self, '_last_val_time') and current_time - self._last_val_time < 10.0:
            # 如果距离上次验证不到10秒，返回缓存值
            return getattr(self, '_cached_val_performance', 0.5)

        self._last_val_time = current_time

        was_training = model.training
        model.eval()

        all_preds = []
        all_labels = []

        try:
            with torch.no_grad():
                sample_count = 0
                for batch_data in self.validation_set:
                    if sample_count > 300:  # 限制验证样本数量，加快速度
                        break

                    # 处理不同格式的batch_data
                    if len(batch_data) == 3:
                        inputs, labels, _ = batch_data
                    else:
                        inputs, labels = batch_data

                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    sample_count += len(labels)

            if len(all_preds) > 0 and len(all_labels) > 0:
                # 计算平衡准确率作为性能指标
                val_performance = calculate_balanced_accuracy(all_labels, all_preds, self.n_classes)
            else:
                val_performance = 0.5

            self._cached_val_performance = val_performance
            return val_performance

        except Exception as e:
            if self.debug_mode:
                print(f"验证性能计算失败: {e}")
            return 0.5
        finally:
            if was_training:
                model.train()

    def update_sample_confidence(self, model, batch_x, batch_y, sample_indices):
        """
        修正后的样本置信度更新 - 传入真实标签
        """
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(batch_x)
                # 关键修正：传入真实标签而不是重复的outputs
                self.confidence_tracker.update_sample_stats(sample_indices, outputs, batch_y)
            model.train()
        except Exception as e:
            if self.debug_mode:
                print(f"更新样本置信度失败: {e}")

    def check_epoch_end_phase_transition(self, model, criterion, epoch):
        """
        在epoch结束时检查是否需要切换阶段
        """
        if not self.phase2_triggered:
            val_performance = self.calculate_validation_performance(model, criterion)

            # 记录验证性能到统计中
            if not hasattr(self.statistics, 'val_performance'):
                self.statistics['val_performance'] = []
            self.statistics['val_performance'].append(val_performance)

            phase_changed = self.check_phase2_trigger(val_performance)
            if phase_changed and self.phase2_stats['phase2_start_epoch'] == -1:
                self.phase2_stats['phase2_start_epoch'] = epoch
                if self.debug_mode:
                    print(f"🔄 阶段切换完成！Epoch {epoch} 开始第二阶段")
            return phase_changed
        else:
            # 即使已经是第二阶段，也要记录验证性能
            val_performance = self.calculate_validation_performance(model, criterion)
            if not hasattr(self.statistics, 'val_performance'):
                self.statistics['val_performance'] = []
            self.statistics['val_performance'].append(val_performance)

        return False

    def train_step(self, model, criterion, x, y, epoch, total_epochs, sample_indices=None, batch_idx=0):
        """
        修正后的训练步骤 - Sample-level增强 + Batch-level奖励
        """
        # 修正：生成稳定的样本索引
        sample_indices = self.generate_stable_sample_indices(x.shape[0], sample_indices)

        # 更新样本置信度统计
        self.update_sample_confidence(model, x, y, sample_indices)

        # 检查是否需要鼓励探索（替代激进的网络重置）
        if self.exploration_scheduler.should_encourage_exploration(epoch):
            if self.debug_mode:
                print(f"🔍 Epoch {epoch}: 鼓励探索阶段，增强噪声注入")

        # 获取状态
        initial_state = self.get_state(model, x, y)

        # 根据阶段选择动作和应用增强
        if self.current_phase == 1:
            # 第一阶段：batch-level，使用BA奖励
            action, operation_idx, magnitude_idx = self.select_action_phase1(initial_state.cpu().numpy(), epoch)
            augmented_x, augmented_y = self.apply_augmentation(x, y, operation_idx, magnitude_idx)

            # 修正：使用BA奖励
            batch_ba_reward = self.calculate_ba_reward(model, x, augmented_x, y)
            avg_reward = batch_ba_reward

        else:
            # 🔧 核心修正：第二阶段使用sample-level增强 + batch-level奖励
            batch_actions, operation_idx, magnitude_idx, batch_difficulties, batch_log_probs = self.select_action_phase2_adaptive(
                initial_state.cpu().numpy(), sample_indices, epoch)

            # 应用样本级别增强
            augmented_x, augmented_y = self.apply_sample_level_augmentation(x, y, batch_actions, sample_indices)

            # 🔧 关键修改：计算整个batch的BA奖励，而不是单个样本的奖励
            batch_ba_reward = self.calculate_ba_reward(model, x, augmented_x, y)
            avg_reward = batch_ba_reward  # 直接使用batch级别的奖励

            # 修正：只为非困难样本更新自适应alpha
            for difficulty, log_prob in zip(batch_difficulties, batch_log_probs):
                if difficulty != 'hard':  # 困难样本不参与alpha更新
                    self.adaptive_alpha.update_alpha_for_difficulty(difficulty, log_prob)

        # 获取增强后状态
        next_state = self.get_augmented_state(model, x, y, augmented_x)

        # 更新统计
        operation_name = self.aug_operations[operation_idx] if operation_idx < len(
            self.aug_operations) else "no_augmentation"
        self.operation_stats[operation_name] += 1
        self.magnitude_stats[operation_name][magnitude_idx] += 1

        # 训练模型
        model.train()
        outputs = model(augmented_x)
        loss = criterion(outputs, augmented_y)

        # 🔧 核心修正：统一的经验存储逻辑，都使用batch-level奖励
        if self.current_phase == 1:
            # 第一阶段：存储一条经验，使用批次BA奖励
            action_vector = torch.zeros(len(self.aug_operations) + 1 + self.n_magnitude_levels)
            action_vector[operation_idx] = 1.0
            action_vector[len(self.aug_operations) + 1 + magnitude_idx] = 1.0

            self.replay_buffer.push(
                initial_state.cpu().numpy(),
                action_vector.numpy(),
                batch_ba_reward,  # 使用batch-level BA奖励
                next_state.cpu().numpy(),
                False
            )
        else:
            # 🔧 关键修改：第二阶段也使用batch-level奖励，但每个样本动作不同
            for i, (operation_name, magnitude_idx_sample) in enumerate(batch_actions):
                try:
                    # 构建单样本的动作向量
                    if operation_name == 'no_augmentation':
                        op_idx = len(self.aug_operations)
                    else:
                        op_idx = self.aug_operations.index(operation_name)

                    action_vector = torch.zeros(len(self.aug_operations) + 1 + self.n_magnitude_levels)
                    action_vector[op_idx] = 1.0
                    action_vector[len(self.aug_operations) + 1 + magnitude_idx_sample] = 1.0

                    # 🔧 核心修正：所有样本都使用相同的batch-level奖励
                    self.replay_buffer.push(
                        initial_state.cpu().numpy(),
                        action_vector.numpy(),
                        batch_ba_reward,  # 关键：使用batch-level奖励而非单样本奖励
                        next_state.cpu().numpy(),
                        False
                    )
                except Exception as e:
                    if self.debug_mode:
                        print(f"存储样本{sample_indices[i]}经验失败: {e}")

        # 更新SAC网络
        sac_losses = {}
        if len(self.replay_buffer) > 256:
            sac_losses = self.update_networks()

        # 收集指标 (每 batch 都收集，但 MetricsCollector 内部会按 record_every 过滤)
        if self.metrics_collector is not None:
            # 计算 batch accuracy (增强后)
            with torch.no_grad():
                outputs_batch = model(augmented_x)
                preds = outputs_batch.argmax(1)
                batch_acc = (preds == augmented_y).float().mean().item()

            # 从 sac_losses 中获取各项损失和 alpha
            actor_loss = sac_losses.get('actor_loss', 0.0)
            critic1_loss = sac_losses.get('critic1_loss', 0.0)
            critic2_loss = sac_losses.get('critic2_loss', 0.0)
            alpha_easy = sac_losses.get('alpha_easy', 0.0)
            alpha_medium = sac_losses.get('alpha_medium', 0.0)
            alpha_hard = sac_losses.get('alpha_hard', 0.0)

            # 置信度统计（从 confidence_tracker 获取）
            conf_stats = self.confidence_tracker.get_confidence_statistics()
            avg_confidence_correct = conf_stats.get('avg_confidence', 0.5)
            confidence_std = conf_stats.get('confidence_std', 0.0)

            metrics_dict = {
                'ce_loss': loss.item(),
                'batch_acc': batch_acc,
                'batch_ba_improve': avg_reward,
                'actor_loss': actor_loss,
                'critic1_loss': critic1_loss,
                'critic2_loss': critic2_loss,
                'alpha_easy': alpha_easy,
                'alpha_medium': alpha_medium,
                'alpha_hard': alpha_hard,
                'selected_op': operation_name,
                'selected_mag': magnitude_idx,
                'avg_confidence_correct': avg_confidence_correct,
                'confidence_std': confidence_std,
            }

            # 可选：梯度点积（需要每隔 compute_grad_dot_every 步计算一次）
            if self.training_step % self.metrics_collector.compute_grad_dot_every == 0:
                # 需要传入 model, x, y (原始数据) 和验证集平均梯度
                # 这里暂时没有 val_grad_avg，可以在外部设置
                grad_cos = self.metrics_collector.compute_gradient_cos_sim(model, x, y, self.device)
                metrics_dict['grad_cos_sim'] = grad_cos
                # 增强前后的梯度点积变化需要额外计算（可选）
                # grad_cos_orig = ...
                # grad_cos_aug = ...
                # metrics_dict['grad_delta'] = grad_cos_aug - grad_cos_orig

            self.metrics_collector.record(
                step=self.training_step,
                phase=self.current_phase,
                epoch=epoch,
                batch_idx=batch_idx,  # 这里无法获得 batch_idx，可以传入参数修改 train_step 签名
                metrics_dict=metrics_dict
            )

        # 更新统计 - 修正：记录BA相关统计
        self.statistics['rewards'].append(avg_reward)

        # 🔧 修正：统一使用batch-level奖励记录
        if 'ba_improvement' not in self.statistics:
            self.statistics['ba_improvement'] = []
        self.statistics['ba_improvement'].append(batch_ba_reward)  # 统一使用batch奖励

        # 修正：移除val_performance记录，因为它现在在epoch结束时单独处理
        self.training_step += 1

        # 更新第二阶段统计
        if self.current_phase == 2:
            for difficulty in batch_difficulties:
                if difficulty == 'easy':
                    self.phase2_stats['easy_samples'] += 1
                elif difficulty == 'medium':
                    self.phase2_stats['medium_samples'] += 1
                elif difficulty == 'hard':
                    self.phase2_stats['hard_samples'] += 1

            if self._last_phase2_epoch != epoch:
                self.phase2_stats['phase2_epochs'] = epoch
                self._last_phase2_epoch = epoch

        # 定期打印Alpha统计
        if epoch % 20 == 0 and self.current_phase == 2:
            self.adaptive_alpha.print_alpha_statistics()

        return augmented_x, augmented_y, avg_reward, sac_losses

    def extract_features(self, model, x):
        """提取特征"""
        if not x.is_cuda:
            x = x.to(self.device, non_blocking=True)

        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                elif hasattr(model, 'base_model'):
                    original_fc = model.base_model._fc
                    model.base_model._fc = nn.Identity()
                    features = model.base_model(x)
                    model.base_model._fc = original_fc
                else:
                    features = model(x)
        except:
            batch_size = x.shape[0]
            features = torch.randn(batch_size, self.feature_dim).to(self.device)
        finally:
            if was_training:
                model.train()

        return features

    def get_class_distribution(self, labels):
        """计算类别分布"""
        batch_size = labels.shape[0]
        class_counts = torch.zeros(self.n_classes, device=self.device)
        for i in range(self.n_classes):
            class_counts[i] = (labels == i).float().sum()
        return class_counts / batch_size

    def get_state(self, model, batch_x, batch_y):
        """构建状态"""
        original_features = self.extract_features(model, batch_x)
        batch_original_features = torch.mean(original_features, dim=0)
        # 修正：初始状态时，增强特征应该等于原始特征
        batch_augmented_features = batch_original_features.clone()

        class_distribution = self.get_class_distribution(batch_y)

        state = torch.cat([
            batch_original_features,
            batch_augmented_features,
            class_distribution
        ])

        return state

    def get_augmented_state(self, model, batch_x, batch_y, augmented_x):
        """获取增强后状态"""
        original_features = self.extract_features(model, batch_x)
        augmented_features = self.extract_features(model, augmented_x)

        batch_original_features = torch.mean(original_features, dim=0)
        batch_augmented_features = torch.mean(augmented_features, dim=0)

        class_distribution = self.get_class_distribution(batch_y)

        state = torch.cat([
            batch_original_features,
            batch_augmented_features,
            class_distribution
        ])

        return state

    def apply_augmentation(self, x, y, operation_idx, magnitude_idx):
        """应用增强"""
        if not x.is_cuda:
            x = x.to(self.device, non_blocking=True)
        if not y.is_cuda:
            y = y.to(self.device, non_blocking=True)

        if operation_idx >= len(self.aug_operations):
            return x, y

        operation_name = self.aug_operations[operation_idx]
        magnitude = self.get_magnitude_value(operation_idx, magnitude_idx)

        try:
            if operation_name == "time_mask":
                return self._apply_time_mask(x, y, magnitude)
            elif operation_name == "frequency_mask":
                return self._apply_frequency_mask(x, y, magnitude)
            elif operation_name == "noise_injection":
                return self._apply_noise_injection(x, y, magnitude)
            elif operation_name == "random_quantization":
                return self._apply_random_quantization(x, y, magnitude)
            elif operation_name == "spectral_contrast":
                return self._apply_spectral_contrast(x, y, magnitude)
            elif operation_name == "harmonic_perturbation":
                return self._apply_harmonic_perturbation(x, y, magnitude)
            elif operation_name == "breathing_cycle_stretch":
                return self._apply_breathing_cycle_stretch(x, y, magnitude)
            elif operation_name == "low_freq_emphasis":
                return self._apply_low_freq_emphasis(x, y, magnitude)
            else:
                return x, y
        except Exception as e:
            if self.debug_mode:
                print(f"应用增强{operation_name}失败: {e}")
            return x, y

    def get_magnitude_value(self, operation_idx, magnitude_idx):
        """获取实际幅度值"""
        if operation_idx >= len(self.aug_operations):
            return 0.0

        operation_name = self.aug_operations[operation_idx]
        if operation_name in self.magnitude_ranges:
            magnitude_values = self.magnitude_ranges[operation_name]
            if magnitude_idx < len(magnitude_values):
                return magnitude_values[magnitude_idx]

        return 0.1

    # 增强操作实现
    def _apply_time_mask(self, x, y, magnitude):
        """时间掩码"""
        _, _, _, width = x.shape
        mask_width = int(width * magnitude)
        if mask_width == 0:
            return x, y
        start_pos = torch.randint(0, width - mask_width + 1, (1,)).item()
        augmented_x = x.clone()
        augmented_x[:, :, :, start_pos:start_pos + mask_width] = 0
        return augmented_x, y

    def _apply_frequency_mask(self, x, y, magnitude):
        """频率掩码"""
        _, _, height, _ = x.shape
        mask_height = int(height * magnitude)
        if mask_height == 0:
            return x, y
        start_pos = torch.randint(0, height - mask_height + 1, (1,)).item()
        augmented_x = x.clone()
        augmented_x[:, :, start_pos:start_pos + mask_height, :] = 0
        return augmented_x, y

    def _apply_noise_injection(self, x, y, magnitude):
        """噪声注入"""
        noise = torch.randn_like(x) * magnitude
        return x + noise, y

    def _apply_random_quantization(self, x, y, magnitude):
        """随机量化"""
        levels = max(2, int(256 * (1 - magnitude)))
        x_min, x_max = x.min(), x.max()
        x_scaled = (x - x_min) / (x_max - x_min + 1e-8)
        x_quantized = torch.round(x_scaled * (levels - 1)) / (levels - 1)
        return x_quantized * (x_max - x_min) + x_min, y

    def _apply_spectral_contrast(self, x, y, magnitude):
        """频谱对比度"""
        contrast_factor = 1.0 + magnitude
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - x_mean) * contrast_factor + x_mean, y

    def _apply_harmonic_perturbation(self, x, y, magnitude):
        """谐波扰动"""
        batch_size, channels, height, width = x.shape
        freq = torch.rand(1).item() * 10
        phase = torch.rand(1).item() * 2 * np.pi

        t = torch.linspace(0, 1, width).to(x.device)
        t = t.view(1, 1, 1, -1).expand(batch_size, channels, height, -1)

        harmonic = torch.sin(2 * np.pi * freq * t + phase) * magnitude
        return x + harmonic, y

    def _apply_breathing_cycle_stretch(self, x, y, magnitude):
        """呼吸周期拉伸"""
        stretch_factor = 1.0 + (torch.rand(1).item() - 0.5) * magnitude * 2

        if abs(stretch_factor - 1.0) < 0.01:
            return x, y

        try:
            augmented_x = F.interpolate(x, scale_factor=(1.0, stretch_factor), mode='bilinear', align_corners=False)

            _, _, _, new_width = augmented_x.shape
            _, _, _, orig_width = x.shape

            if new_width > orig_width:
                start = (new_width - orig_width) // 2
                augmented_x = augmented_x[:, :, :, start:start + orig_width]
            elif new_width < orig_width:
                pad_width = orig_width - new_width
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                augmented_x = F.pad(augmented_x, (pad_left, pad_right), mode='constant', value=0)

            return augmented_x, y
        except:
            return x, y

    def _apply_low_freq_emphasis(self, x, y, magnitude):
        """低频强调"""
        _, _, height, _ = x.shape
        freq_weights = torch.ones(height, device=x.device)
        low_freq_end = int(height * 0.3)
        freq_weights[:low_freq_end] = 1.0 + magnitude
        freq_weights = freq_weights.view(1, 1, -1, 1)
        return x * freq_weights, y

    def update_networks(self, batch_size=256):
        """
        修正后的SAC网络更新 - 适配自适应Alpha策略
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        try:
            # 采样经验
            batch = self.replay_buffer.sample(batch_size)
            state_batch = torch.FloatTensor(batch.state).to(self.device)
            action_batch = torch.FloatTensor(batch.action).to(self.device)
            reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
            next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
            done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

            # 获取动作空间大小
            n_operations = len(self.aug_operations) + 1
            n_magnitudes = self.n_magnitude_levels

            # 更新Critic
            with torch.no_grad():
                # 计算目标Q值
                next_operation_logits, next_magnitude_logits, next_log_prob = self.actor.sample(
                    next_state_batch, n_operations, n_magnitudes)

                next_operation_dist = torch.distributions.Categorical(logits=next_operation_logits)
                next_magnitude_dist = torch.distributions.Categorical(logits=next_magnitude_logits)

                next_operation_idx = next_operation_dist.sample()
                next_magnitude_idx = next_magnitude_dist.sample()

                # 构建下一状态的动作向量
                next_action = torch.zeros_like(action_batch)
                batch_indices = torch.arange(next_action.shape[0])
                next_action[batch_indices, next_operation_idx] = 1.0
                next_action[batch_indices, n_operations + next_magnitude_idx] = 1.0

                target_q1 = self.target_critic1(next_state_batch, next_action)
                target_q2 = self.target_critic2(next_state_batch, next_action)

                # 使用默认alpha计算目标Q值（在网络更新时使用统一alpha）
                default_alpha = self.adaptive_alpha.get_alpha_for_difficulty('medium')
                target_q = torch.min(target_q1, target_q2) - default_alpha * next_log_prob.unsqueeze(1)
                target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

            # 当前Q值
            current_q1 = self.critic1(state_batch, action_batch)
            current_q2 = self.critic2(state_batch, action_batch)

            # Critic损失
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)

            # 更新Critic1
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            self.critic1_optimizer.step()

            # 更新Critic2
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic2_optimizer.step()

            # 更新Actor
            operation_logits, magnitude_logits, log_prob = self.actor.sample(
                state_batch, n_operations, n_magnitudes)

            # 构建新动作
            new_action = torch.zeros_like(action_batch)
            operation_dist = torch.distributions.Categorical(logits=operation_logits)
            magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)

            operation_idx = operation_dist.sample()
            magnitude_idx = magnitude_dist.sample()

            batch_indices = torch.arange(new_action.shape[0])
            new_action[batch_indices, operation_idx] = 1.0
            new_action[batch_indices, n_operations + magnitude_idx] = 1.0

            # 计算Actor损失
            q1_new = self.critic1(state_batch, new_action)
            q2_new = self.critic2(state_batch, new_action)
            q_new = torch.min(q1_new, q2_new)

            # 使用默认alpha计算Actor损失
            default_alpha = self.adaptive_alpha.get_alpha_for_difficulty('medium')
            actor_loss = (default_alpha * log_prob.unsqueeze(1) - q_new).mean()

            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)

            return {
                'actor_loss': actor_loss.item(),
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'alpha': default_alpha.item(),
                'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
                'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item()
            }

        except Exception as e:
            if self.debug_mode:
                print(f"SAC网络更新失败: {e}")
            return {
                'actor_loss': 0.0,
                'critic1_loss': 0.0,
                'critic2_loss': 0.0,
                'alpha': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
                'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item()
            }

    def soft_update(self, target_network, source_network):
        """软更新"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def get_statistics(self):
        """修正后的统计信息获取"""
        if not self.statistics['rewards']:
            base_stats = {
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'avg_ba_improvement': 0.0,
                'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
                'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
                'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item(),
                'training_step': self.training_step,
                'total_episodes': 0,
                'current_phase': self.current_phase,
                'phase2_triggered': self.phase2_triggered
            }

            # 添加置信度统计
            confidence_stats = self.confidence_tracker.get_confidence_statistics()
            base_stats.update({f'confidence_{k}': v for k, v in confidence_stats.items()})

            return base_stats

        rewards = self.statistics['rewards']
        ba_improvements = self.statistics.get('ba_improvement', [0.0])

        stats = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_ba_improvement': np.mean(ba_improvements),
            'alpha_medium': self.adaptive_alpha.get_alpha_for_difficulty('medium').item(),
            'alpha_easy': self.adaptive_alpha.get_alpha_for_difficulty('easy').item(),
            'alpha_hard': self.adaptive_alpha.get_alpha_for_difficulty('hard').item(),
            'training_step': self.training_step,
            'total_episodes': len(rewards),
            'current_phase': self.current_phase,
            'phase2_triggered': self.phase2_triggered
        }

        # 添加置信度统计
        confidence_stats = self.confidence_tracker.get_confidence_statistics()
        stats.update({f'confidence_{k}': v for k, v in confidence_stats.items()})

        # 添加第二阶段统计
        if self.current_phase == 2:
            stats.update(self.phase2_stats)
            difficulties = self.confidence_tracker.get_all_difficulties()
            stats.update({f'total_{k}_samples': v for k, v in difficulties.items()})

            # 添加Alpha统计
            stats.update(self.adaptive_alpha.get_all_alphas())

        return stats

    def get_augmentation_statistics(self):
        """获取增强统计信息"""
        total_ops = sum(self.operation_stats.values())

        stats = {
            'total_operations': total_ops,
            'operation_popularity': dict(self.operation_stats),
            'operation_distribution': {op: count / max(1, total_ops) for op, count in self.operation_stats.items()},
            'training_steps': self.training_step,
            'no_augmentation_count': self.operation_stats.get('no_augmentation', 0),
            'no_augmentation_ratio': self.operation_stats.get('no_augmentation', 0) / max(1, total_ops),

            # 二阶段特有统计
            'current_phase': self.current_phase,
            'phase2_triggered': self.phase2_triggered,
            'augmentation_method': 'fixed_hard_adaptive_easy_two_phase_sac',  # 修正：反映最新设计
            'strategy_type': 'batch_to_sample_level_adaptive_alpha',

            # 修复：确保phase2_stats完整
            'phase2_stats': self.phase2_stats.copy(),

            # 修复：确保样本难度统计正确
            'sample_difficulties': self.confidence_tracker.get_all_difficulties() if self.current_phase == 2 else {},

            # 新增：Alpha统计
            'adaptive_alpha_stats': self.adaptive_alpha.get_all_alphas(),
        }

        return stats

    def validate_statistics(self):
        """验证统计信息的一致性"""
        print("\n📊 修正版统计信息验证:")

        # 验证操作统计
        total_ops = sum(self.operation_stats.values())
        total_mags = sum(sum(mag_dict.values()) for mag_dict in self.magnitude_stats.values())
        print(f"  操作统计总数: {total_ops}")
        print(f"  幅度统计总数: {total_mags}")

        if abs(total_ops - total_mags) > 1:
            print(f"  ⚠️ 警告：操作和幅度统计不一致！")
        else:
            print(f"  ✅ 操作和幅度统计一致")

        # 验证第二阶段统计
        if self.current_phase == 2:
            phase2_total = (self.phase2_stats.get('easy_samples', 0) +
                            self.phase2_stats.get('medium_samples', 0) +
                            self.phase2_stats.get('hard_samples', 0))

            print(f"  第二阶段处理样本: {phase2_total}")
            print(f"  置信度跟踪样本: {len(self.confidence_tracker.sample_stats)}")

            # 验证样本难度分布
            difficulties = self.confidence_tracker.get_all_difficulties()
            print(f"  样本难度分布: {difficulties}")

            # 验证Alpha策略（只显示参与SAC学习的难度）
            alphas = self.adaptive_alpha.get_all_alphas()
            print(f"  自适应Alpha值:")
            print(f"    easy: {alphas.get('easy', 'N/A'):.4f} (高探索，全范围搜索)")
            print(f"    medium: {alphas.get('medium', 'N/A'):.4f} (中等探索)")
            print(f"    hard: 固定no_augmentation策略")

            # 统计no_augmentation的使用情况
            no_aug_count = self.operation_stats.get('no_augmentation', 0)
            hard_sample_count = difficulties.get('hard', 0)
            print(f"  no_augmentation使用次数: {no_aug_count}")
            print(f"  困难样本处理次数: {self.phase2_stats.get('hard_samples', 0)}")

            if phase2_total > 0:
                print(f"  ✅ 第二阶段统计正常")
            else:
                print(f"  ⚠️ 第二阶段统计可能有问题")
        else:
            print(f"  当前处于第一阶段，第二阶段统计暂不适用")

        # 验证奖励机制
        if len(self.statistics['rewards']) > 0:
            print(f"  平均奖励: {np.mean(self.statistics['rewards']):.4f}")
            print(f"  奖励标准差: {np.std(self.statistics['rewards']):.4f}")
            print(f"  ✅ BA奖励机制")

        print(f"  ✅ 困难样本固定策略")
        print(f"  ✅ 易增强样本高强度搜索")
        print(f"  ✅ 轻量级探索性维持机制")

    def save_augmentation_statistics(self, save_dir, dataset_name, task_type):
        """保存统计信息"""
        os.makedirs(save_dir, exist_ok=True)

        aug_stats = self.get_augmentation_statistics()
        sac_stats = self.get_statistics()

        stats_file = os.path.join(save_dir, f'{dataset_name}_{task_type}_corrected_twophase_stats.json')
        combined_stats = {
            'augmentation_statistics': aug_stats,
            'sac_statistics': sac_stats,
            'dataset': dataset_name,
            'task_type': task_type,
            'strategy_type': 'fixed_hard_adaptive_easy_sac',  # 修正：反映固定困难样本+自适应易增强的设计
            'architecture': 'fixed_hard_sample_adaptive_alpha_sac',
            'reward_mechanism': 'balanced_accuracy_based',
            'exploration_mechanism': 'lightweight_noise_injection_adaptive_alpha',
            'hard_sample_strategy': 'fixed_no_augmentation',
            'easy_sample_strategy': 'high_exploration_full_range_search',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=4, ensure_ascii=False)

        print(f"📊 修正版二阶段统计已保存到: {stats_file}")


class AdaptiveActorNetwork(nn.Module):
    """自适应Actor网络，支持动态动作空间"""

    def __init__(self, state_dim, max_operations, max_magnitude_levels, hidden_dim=256):
        super().__init__()
        self.max_operations = max_operations
        self.max_magnitude_levels = max_magnitude_levels

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.operation_head = nn.Linear(hidden_dim, max_operations)
        self.magnitude_head = nn.Linear(hidden_dim, max_magnitude_levels)

    def forward(self, state, n_operations=None, n_magnitude_levels=None):
        x = self.shared(state)

        operation_logits = self.operation_head(x)
        magnitude_logits = self.magnitude_head(x)

        if n_operations is not None:
            operation_logits = operation_logits[:, :n_operations]
        if n_magnitude_levels is not None:
            magnitude_logits = magnitude_logits[:, :n_magnitude_levels]

        return operation_logits, magnitude_logits

    def sample(self, state, n_operations=None, n_magnitude_levels=None):
        operation_logits, magnitude_logits = self.forward(state, n_operations, n_magnitude_levels)

        operation_dist = torch.distributions.Categorical(logits=operation_logits)
        magnitude_dist = torch.distributions.Categorical(logits=magnitude_logits)

        operation_sample = operation_dist.sample()
        magnitude_sample = magnitude_dist.sample()

        operation_log_prob = operation_dist.log_prob(operation_sample)
        magnitude_log_prob = magnitude_dist.log_prob(magnitude_sample)

        total_log_prob = operation_log_prob + magnitude_log_prob

        return operation_logits, magnitude_logits, total_log_prob


class CriticNetwork(nn.Module):
    """Critic网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        from collections import namedtuple
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
        return Transition(state, action, reward, next_state, done)

    def __len__(self):
        return len(self.buffer)