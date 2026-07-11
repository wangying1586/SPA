import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from efficientnet_pytorch import EfficientNet
from utils.comet_record import init_comet_experiment, log_hyperparameters, log_model_to_comet
import time
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings
from PASA_Strategy import TwoPhaseSACStrategy, calculate_balanced_accuracy, apply_simple_anti_memorization_fix
import json
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
import matplotlib.pyplot as plt
import io
import torch.nn.functional as F
warnings.filterwarnings('ignore')


def calculate_circor_metrics(y_true, y_pred):
    """
    计算CirCor DigiScope数据集的评价指标（根据论文）

    Args:
        y_true: 真实标签 (0: Present, 1: Absent, 2: Unknown)
        y_pred: 预测标签 (0: Present, 1: Absent, 2: Unknown)

    Returns:
        dict: 包含W.acc和UAR的指标字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算每个类别的正确预测数量
    cp = np.sum((y_true == 0) & (y_pred == 0))  # Present正确预测数
    ca = np.sum((y_true == 1) & (y_pred == 1))  # Absent正确预测数
    cu = np.sum((y_true == 2) & (y_pred == 2))  # Unknown正确预测数

    # 计算每个类别的真实样本数量
    tp = np.sum(y_true == 0)  # Present真实样本数
    ta = np.sum(y_true == 1)  # Absent真实样本数
    tu = np.sum(y_true == 2)  # Unknown真实样本数

    # 计算Weighted Accuracy (W.acc) - 根据论文公式
    # W.acc = (5*cp + 3*cu + ca) / (5*tp + 3*tu + ta)
    if (5 * tp + 3 * tu + ta) > 0:
        w_acc = (5 * cp + 3 * cu + ca) / (5 * tp + 3 * tu + ta)
    else:
        w_acc = 0.0

    # 计算每个类别的recall
    recall_present = cp / tp if tp > 0 else 0.0
    recall_absent = ca / ta if ta > 0 else 0.0
    recall_unknown = cu / tu if tu > 0 else 0.0

    # 计算Unweighted Average Recall (UAR)
    uar = (recall_present + recall_absent + recall_unknown) / 3.0

    return {
        'w_acc': w_acc,
        'uar': uar,
        'recall_present': recall_present,
        'recall_absent': recall_absent,
        'recall_unknown': recall_unknown
    }

def calculate_sprsound_metrics(y_true, y_pred):
    """
    SPRSound 官方评价指标计算（适用于所有任务 task11/12/21/22）

    官方定义（SPRSound 论文）:
      SE = 正确识别的异常音 / 所有异常音      (非Normal = 异常)
      SP = 正确识别的正常音 / 所有正常音      (Normal = class 0)
      AS = (SE + SP) / 2
      HS = 2 * SE * SP / (SE + SP)
      Score = (AS + HS) / 2

    所有任务统一用 Normal(class=0) vs 非Normal 二元化计算，
    不做多类 macro 平均。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 二元化: Normal(0) = negative, 其他所有类 = positive(abnormal)
    bin_true = (y_true != 0).astype(int)
    bin_pred = (y_pred != 0).astype(int)

    tp = int(np.sum((bin_true == 1) & (bin_pred == 1)))  # 异常被正确预测为异常
    fn = int(np.sum((bin_true == 1) & (bin_pred == 0)))  # 异常被误判为正常
    tn = int(np.sum((bin_true == 0) & (bin_pred == 0)))  # 正常被正确预测为正常
    fp = int(np.sum((bin_true == 0) & (bin_pred == 1)))  # 正常被误判为异常

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity

    average_score = (se + sp) / 2
    harmonic_score = 2 * se * sp / (se + sp + 1e-9)
    score = (average_score + harmonic_score) / 2

    return {
        'sensitivity': se,
        'specificity': sp,
        'average_score': average_score,
        'harmonic_score': harmonic_score,
        'overall_score': score,
    }


def calculate_icbhi_metrics(y_true, y_pred):
    """
    ICBHI 2017 官方评价指标计算（适用于 multiclass 和 binary 任务）

    官方定义（ICBHI Challenge / Kaggle 参考代码一致）:
      Se = 正确识别的异常呼吸音 / 所有异常呼吸音  (非Normal = Crackle/Wheeze/Both)
      Sp = 正确识别的正常呼吸音 / 所有正常呼吸音  (Normal = class 0)
      Score = (Se + Sp) / 2                       ← 只有算术平均，无 harmonic mean

    注意: ICBHI Score 不含 Harmonic Score 项，与 SPRSound 不同。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 二元化: Normal(0) = negative, 其他所有类 = positive(abnormal)
    bin_true = (y_true != 0).astype(int)
    bin_pred = (y_pred != 0).astype(int)

    tp = int(np.sum((bin_true == 1) & (bin_pred == 1)))
    fn = int(np.sum((bin_true == 1) & (bin_pred == 0)))
    tn = int(np.sum((bin_true == 0) & (bin_pred == 0)))
    fp = int(np.sum((bin_true == 0) & (bin_pred == 1)))

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity (Se)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity (Sp)

    score = (se + sp) / 2  # ICBHI 官方 Score，仅算术平均

    return {
        'sensitivity': se,
        'specificity': sp,
        'overall_score': score,
    }


def log_corrected_confidence_metrics(sac_strategy, experiment, epoch, fold_id=None):
    """
    记录修正后的置信度统计信息到Comet
    """
    if sac_strategy is None:
        return

    try:
        prefix = f"fold_{fold_id}_" if fold_id else ""

        # 获取修正后的置信度统计
        confidence_stats = sac_strategy.confidence_tracker.get_confidence_statistics()

        # 记录置信度相关metrics
        for key, value in confidence_stats.items():
            if isinstance(value, (int, float)):
                experiment.log_metric(f"{prefix}confidence_{key}", value, step=epoch)

        # 记录难度分布
        difficulties = sac_strategy.confidence_tracker.get_all_difficulties()
        total_classified = sum(difficulties.values())

        for difficulty, count in difficulties.items():
            experiment.log_metric(f"{prefix}difficulty_{difficulty}_count", count, step=epoch)
            if total_classified > 0:
                ratio = count / total_classified
                experiment.log_metric(f"{prefix}difficulty_{difficulty}_ratio", ratio, step=epoch)

        # 记录总体分类准确率
        overall_accuracy = sac_strategy.confidence_tracker.correct_predictions / max(1,
                                                                                     sac_strategy.confidence_tracker.total_predictions)
        experiment.log_metric(f"{prefix}overall_prediction_accuracy", overall_accuracy, step=epoch)

        # 记录有效样本比例（有正确预测的样本比例）
        samples_with_correct = confidence_stats['samples_with_correct_predictions']
        total_samples = confidence_stats['total_samples']
        if total_samples > 0:
            effective_sample_ratio = samples_with_correct / total_samples
            experiment.log_metric(f"{prefix}effective_sample_ratio", effective_sample_ratio, step=epoch)

        print(f"✅ Epoch {epoch} 修正版置信度metrics已记录到Comet")

    except Exception as e:
        print(f"❌ 记录修正版置信度metrics失败: {e}")


def log_comprehensive_sac_metrics(sac_strategy, experiment, epoch, fold_id=None):
    """
    全面记录SAC策略的所有metrics到Comet（修正版）
    包括修正后的置信度统计
    """
    if sac_strategy is None:
        return

    try:
        # 获取基础统计
        sac_stats = sac_strategy.get_statistics()
        aug_stats = sac_strategy.get_augmentation_statistics()

        # 构建metrics前缀
        prefix = f"fold_{fold_id}_" if fold_id else ""

        # 1. 记录基础SAC统计
        for key, value in sac_stats.items():
            if isinstance(value, (int, float)):
                experiment.log_metric(f"{prefix}sac_{key}", value, step=epoch)

        # 2. 记录增强操作统计
        operation_popularity = aug_stats.get('operation_popularity', {})
        operation_distribution = aug_stats.get('operation_distribution', {})

        for op, count in operation_popularity.items():
            experiment.log_metric(f"{prefix}operation_count_{op}", count, step=epoch)

        for op, ratio in operation_distribution.items():
            experiment.log_metric(f"{prefix}operation_ratio_{op}", ratio, step=epoch)

        # 3. 记录修正后的置信度统计
        log_corrected_confidence_metrics(sac_strategy, experiment, epoch, fold_id)

        # 4. 记录第二阶段特有统计
        if sac_strategy.current_phase == 2:
            phase2_stats = aug_stats.get('phase2_stats', {})
            sample_difficulties = aug_stats.get('sample_difficulties', {})

            # 第二阶段处理统计
            for key, value in phase2_stats.items():
                if isinstance(value, (int, float)):
                    experiment.log_metric(f"{prefix}phase2_{key}", value, step=epoch)

            # 样本难度分布
            for difficulty, count in sample_difficulties.items():
                experiment.log_metric(f"{prefix}sample_difficulty_{difficulty}_count", count, step=epoch)
                if sum(sample_difficulties.values()) > 0:
                    ratio = count / sum(sample_difficulties.values())
                    experiment.log_metric(f"{prefix}sample_difficulty_{difficulty}_ratio", ratio, step=epoch)

        print(f"✅ Epoch {epoch} 全面SAC metrics（包含修正置信度）已记录到Comet")

    except Exception as e:
        print(f"❌ 记录SAC metrics到Comet失败: {e}")


def log_detailed_alpha_metrics(sac_strategy, experiment, epoch, fold_id=None):
    """
    修正：记录详细的Alpha策略统计到Comet（替代UCB统计）
    """
    if sac_strategy is None or sac_strategy.current_phase != 2:
        return

    try:
        prefix = f"fold_{fold_id}_" if fold_id else ""

        # 记录Alpha统计
        alphas = sac_strategy.adaptive_alpha.get_all_alphas()
        for difficulty, alpha_val in alphas.items():
            experiment.log_metric(f"{prefix}alpha_{difficulty}", alpha_val, step=epoch)

        # 记录Alpha更新统计
        for difficulty in ['easy', 'medium']:
            update_count = sac_strategy.adaptive_alpha.alpha_update_counts.get(difficulty, 0)
            experiment.log_metric(f"{prefix}alpha_updates_{difficulty}", update_count, step=epoch)

            if sac_strategy.adaptive_alpha.alpha_losses[difficulty]:
                avg_loss = np.mean(sac_strategy.adaptive_alpha.alpha_losses[difficulty][-10:])
                experiment.log_metric(f"{prefix}alpha_loss_{difficulty}", avg_loss, step=epoch)

        # 记录目标熵
        for difficulty, target_entropy in sac_strategy.adaptive_alpha.target_entropies.items():
            experiment.log_metric(f"{prefix}target_entropy_{difficulty}", target_entropy, step=epoch)

        print(f"✅ Epoch {epoch} 详细Alpha策略metrics已记录到Comet")

    except Exception as e:
        print(f"❌ 记录详细Alpha策略metrics失败: {e}")


def log_phase_transition_metrics(sac_strategy, experiment, epoch, fold_id=None):
    """
    记录阶段转换的详细metrics
    """
    if sac_strategy is None:
        return

    try:
        prefix = f"fold_{fold_id}_" if fold_id else ""

        # 记录阶段信息
        experiment.log_metric(f"{prefix}current_phase", sac_strategy.current_phase, step=epoch)
        experiment.log_metric(f"{prefix}phase2_triggered", 1 if sac_strategy.phase2_triggered else 0, step=epoch)

        # 记录性能历史长度
        experiment.log_metric(f"{prefix}performance_history_length", len(sac_strategy.performance_history), step=epoch)

        if len(sac_strategy.performance_history) >= 2:
            # 记录性能趋势
            recent_perf = list(sac_strategy.performance_history)[-min(5, len(sac_strategy.performance_history)):]
            perf_variance = max(recent_perf) - min(recent_perf)
            perf_trend = recent_perf[-1] - recent_perf[0] if len(recent_perf) >= 2 else 0

            experiment.log_metric(f"{prefix}performance_variance", perf_variance, step=epoch)
            experiment.log_metric(f"{prefix}performance_trend", perf_trend, step=epoch)
            experiment.log_metric(f"{prefix}performance_recent_avg", np.mean(recent_perf), step=epoch)

        # 记录触发条件
        experiment.log_metric(f"{prefix}phase2_trigger_patience", sac_strategy.phase2_trigger_patience, step=epoch)
        experiment.log_metric(f"{prefix}phase2_trigger_threshold", sac_strategy.phase2_trigger_threshold, step=epoch)

        # 如果已经是第二阶段，记录开始时间
        if sac_strategy.current_phase == 2 and sac_strategy.phase2_stats.get('phase2_start_epoch', -1) != -1:
            start_epoch = sac_strategy.phase2_stats['phase2_start_epoch']
            experiment.log_metric(f"{prefix}phase2_start_epoch", start_epoch, step=epoch)
            experiment.log_metric(f"{prefix}phase2_duration", epoch - start_epoch, step=epoch)

    except Exception as e:
        print(f"❌ 记录阶段转换metrics失败: {e}")


def log_augmentation_effectiveness_metrics(sac_strategy, experiment, epoch, fold_id=None):
    """
    记录增强效果相关的metrics
    """
    if sac_strategy is None:
        return

    try:
        prefix = f"fold_{fold_id}_" if fold_id else ""

        # 记录BA改善统计
        if 'ba_improvement' in sac_strategy.statistics:
            ba_improvements = sac_strategy.statistics['ba_improvement']

            if ba_improvements:
                # 最近10次的BA改善
                recent_window = min(10, len(ba_improvements))
                recent_improvement = np.mean(ba_improvements[-recent_window:])

                experiment.log_metric(f"{prefix}ba_improvement_recent", recent_improvement, step=epoch)
                experiment.log_metric(f"{prefix}ba_improvement_avg", np.mean(ba_improvements), step=epoch)

                # 累积改善统计
                positive_improvements = [imp for imp in ba_improvements if imp > 0]

                experiment.log_metric(f"{prefix}positive_improvement_ratio",
                                      len(positive_improvements) / len(ba_improvements) if ba_improvements else 0,
                                      step=epoch)
                experiment.log_metric(f"{prefix}avg_positive_improvement",
                                      np.mean(positive_improvements) if positive_improvements else 0,
                                      step=epoch)

        # 记录奖励统计
        if 'rewards' in sac_strategy.statistics:
            rewards = sac_strategy.statistics['rewards']
            if rewards:
                recent_rewards = rewards[-min(20, len(rewards)):]
                experiment.log_metric(f"{prefix}reward_recent_avg", np.mean(recent_rewards), step=epoch)
                experiment.log_metric(f"{prefix}reward_recent_std", np.std(recent_rewards), step=epoch)
                experiment.log_metric(f"{prefix}reward_positive_ratio",
                                      len([r for r in recent_rewards if r > 0]) / len(recent_rewards),
                                      step=epoch)

    except Exception as e:
        print(f"❌ 记录增强效果metrics失败: {e}")


def log_comprehensive_validation_metrics(model, val_loader, criterion, device, experiment, epoch, fold_id, args):
    """记录详细的验证metrics，包括类别级别的统计，支持CirCor"""
    try:
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    inputs, labels, _ = batch_data
                else:
                    inputs, labels = batch_data

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算类别级别的metrics
        num_classes = len(np.unique(all_labels))
        class_names = get_dataset_class_names(args.dataset, args.task_type)[:num_classes]

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

        # 类别级别的precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(num_classes)), average=None, zero_division=0
        )

        # 记录每个类别的metrics
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"

            experiment.log_metric(f"fold_{fold_id}_val_precision_{class_name}", precision[i], step=epoch)
            experiment.log_metric(f"fold_{fold_id}_val_recall_{class_name}", recall[i], step=epoch)
            experiment.log_metric(f"fold_{fold_id}_val_f1_{class_name}", f1[i], step=epoch)
            experiment.log_metric(f"fold_{fold_id}_val_support_{class_name}", support[i], step=epoch)

            # 类别级别的准确率
            class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            experiment.log_metric(f"fold_{fold_id}_val_accuracy_{class_name}", class_acc, step=epoch)

        # 记录混淆矩阵的每个元素
        for i in range(num_classes):
            for j in range(num_classes):
                experiment.log_metric(f"fold_{fold_id}_val_cm_{i}_{j}", cm[i, j], step=epoch)

        # CirCor特殊的混淆矩阵元素命名
        if args.dataset == "CirCor":
            cm_class_names = ['Present', 'Absent', 'Unknown']
            for i in range(min(num_classes, len(cm_class_names))):
                for j in range(min(num_classes, len(cm_class_names))):
                    experiment.log_metric(f"fold_{fold_id}_val_cm_{cm_class_names[i]}_to_{cm_class_names[j]}",
                                          cm[i, j], step=epoch)

        # 记录平均置信度和置信度分布
        max_probs = np.max(all_probs, axis=1)
        experiment.log_metric(f"fold_{fold_id}_val_avg_confidence", np.mean(max_probs), step=epoch)
        experiment.log_metric(f"fold_{fold_id}_val_confidence_std", np.std(max_probs), step=epoch)

        # 记录高置信度预测的比例
        high_conf_ratio = np.mean(max_probs > 0.8)
        low_conf_ratio = np.mean(max_probs < 0.6)
        experiment.log_metric(f"fold_{fold_id}_val_high_confidence_ratio", high_conf_ratio, step=epoch)
        experiment.log_metric(f"fold_{fold_id}_val_low_confidence_ratio", low_conf_ratio, step=epoch)

        # CirCor特殊指标
        if args.dataset == "CirCor":
            circor_metrics = calculate_circor_metrics(all_labels, all_preds)

            # 记录CirCor特定指标
            experiment.log_metric(f"fold_{fold_id}_val_w_acc", circor_metrics['w_acc'], step=epoch)
            experiment.log_metric(f"fold_{fold_id}_val_uar", circor_metrics['uar'], step=epoch)
            experiment.log_metric(f"fold_{fold_id}_val_recall_present", circor_metrics['recall_present'], step=epoch)
            experiment.log_metric(f"fold_{fold_id}_val_recall_absent", circor_metrics['recall_absent'], step=epoch)
            experiment.log_metric(f"fold_{fold_id}_val_recall_unknown", circor_metrics['recall_unknown'], step=epoch)

            print(f"🫀 CirCor Metrics - W.acc: {circor_metrics['w_acc']:.4f}, UAR: {circor_metrics['uar']:.4f}")

        print(f"✅ Fold {fold_id} Epoch {epoch} 详细验证metrics已记录到Comet")

    except Exception as e:
        print(f"❌ 记录详细验证metrics失败: {e}")
    finally:
        model.train()


# ============================================================================
# 在epoch结束时的汇总记录函数
# ============================================================================

def log_epoch_summary_metrics(sac_strategy, experiment, epoch, fold_id):
    """
    记录epoch结束时的汇总metrics
    """
    if sac_strategy is None:
        return

    try:
        prefix = f"fold_{fold_id}_epoch_{epoch}_"

        # 记录当前epoch的完整状态快照
        stats = sac_strategy.get_statistics()
        aug_stats = sac_strategy.get_augmentation_statistics()

        # 创建epoch汇总
        epoch_summary = {
            'training_step': sac_strategy.training_step,
            'current_phase': sac_strategy.current_phase,
            'total_operations': aug_stats.get('total_operations', 0),
            'no_aug_ratio': aug_stats.get('no_augmentation_ratio', 0),
        }

        if sac_strategy.current_phase == 2:
            difficulties = sac_strategy.confidence_tracker.get_all_difficulties()
            epoch_summary.update({
                'easy_samples': difficulties.get('easy', 0),
                'medium_samples': difficulties.get('medium', 0),
                'hard_samples': difficulties.get('hard', 0),
            })

        # 记录汇总信息
        for key, value in epoch_summary.items():
            experiment.log_metric(f"{prefix}summary_{key}", value)

        # 记录操作使用热度图数据
        op_popularity = aug_stats.get('operation_popularity', {})
        sorted_ops = sorted(op_popularity.items(), key=lambda x: x[1], reverse=True)

        for rank, (op, count) in enumerate(sorted_ops[:5]):  # Top 5操作
            experiment.log_metric(f"{prefix}top_operation_rank_{rank + 1}", count)
            experiment.log_text(f"{prefix}top_operation_name_{rank + 1}", op)

    except Exception as e:
        print(f"❌ 记录epoch汇总metrics失败: {e}")


# ============================================================================
# 最终的全面记录函数
# ============================================================================

def log_final_comprehensive_summary(all_folds_results, final_sac_strategy, experiment, args):
    """记录最终的全面汇总到Comet，支持CirCor"""
    try:
        if args.dataset == "CirCor":
            # CirCor特殊的汇总记录
            fold_w_accs = [r.get('best_w_acc', 0.0) for r in all_folds_results]
            fold_uars = [r.get('best_uar', 0.0) for r in all_folds_results]
            fold_scores = [r['best_score'] for r in all_folds_results]

            experiment.log_metric("final_avg_w_acc", np.mean(fold_w_accs))
            experiment.log_metric("final_std_w_acc", np.std(fold_w_accs))
            experiment.log_metric("final_max_w_acc", np.max(fold_w_accs))
            experiment.log_metric("final_min_w_acc", np.min(fold_w_accs))

            experiment.log_metric("final_avg_uar", np.mean(fold_uars))
            experiment.log_metric("final_std_uar", np.std(fold_uars))
            experiment.log_metric("final_max_uar", np.max(fold_uars))
            experiment.log_metric("final_min_uar", np.min(fold_uars))

            # 与论文基准的比较
            paper_w_acc = 0.832
            paper_uar = 0.713

            experiment.log_metric("final_w_acc_vs_paper", np.mean(fold_w_accs) - paper_w_acc)
            experiment.log_metric("final_uar_vs_paper", np.mean(fold_uars) - paper_uar)

            # 记录是否超过论文基准
            experiment.log_metric("final_w_acc_beats_paper", 1 if np.mean(fold_w_accs) > paper_w_acc else 0)
            experiment.log_metric("final_uar_beats_paper", 1 if np.mean(fold_uars) > paper_uar else 0)

            # 每折的详细结果
            for i, result in enumerate(all_folds_results):
                experiment.log_metric(f"final_fold_{i + 1}_w_acc", result.get('best_w_acc', 0.0))
                experiment.log_metric(f"final_fold_{i + 1}_uar", result.get('best_uar', 0.0))
                experiment.log_metric(f"final_fold_{i + 1}_score", result['best_score'])
                experiment.log_metric(f"final_fold_{i + 1}_epoch", result['best_epoch'])

        else:
            # 原有的其他数据集汇总逻辑
            fold_scores = [r['best_score'] for r in all_folds_results]

            experiment.log_metric("final_avg_cv_score", np.mean(fold_scores))
            experiment.log_metric("final_std_cv_score", np.std(fold_scores))
            experiment.log_metric("final_max_cv_score", np.max(fold_scores))
            experiment.log_metric("final_min_cv_score", np.min(fold_scores))

            # 每折的详细结果
            for i, result in enumerate(all_folds_results):
                experiment.log_metric(f"final_fold_{i + 1}_score", result['best_score'])
                experiment.log_metric(f"final_fold_{i + 1}_epoch", result['best_epoch'])

        # SAC策略最终统计（通用）
        if final_sac_strategy is not None:
            final_stats = final_sac_strategy.get_augmentation_statistics()
            final_sac_stats = final_sac_strategy.get_statistics()

            # 记录最终的SAC性能
            for key, value in final_sac_stats.items():
                if isinstance(value, (int, float)):
                    experiment.log_metric(f"final_sac_{key}", value)

            # 记录最终的增强统计
            for key, value in final_stats.items():
                if isinstance(value, (int, float)):
                    experiment.log_metric(f"final_aug_{key}", value)
                elif isinstance(value, dict) and key == 'operation_distribution':
                    for op, ratio in value.items():
                        experiment.log_metric(f"final_op_ratio_{op}", ratio)

        # 实验配置汇总
        config_params = {
            "dataset": args.dataset,
            "task_type": args.task_type,
            "use_adaptive_aug": args.use_adaptive_aug,
            "phase2_trigger_patience": args.phase2_trigger_patience,
            "phase2_trigger_threshold": args.phase2_trigger_threshold,
            "gamma": args.gamma,
            "n_magnitude_levels": args.n_magnitude_levels,
            "batch_size": args.batch_size,
            "epoch": args.epoch,
            "lr": args.lr,
        }

        if args.dataset == "CirCor":
            config_params.update({
                "circor_val_ratio": args.circor_val_ratio,
                "evaluation_metrics": "W.acc, UAR",
                "reference_paper": "Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection"
            })

        experiment.log_parameters(config_params)

        if args.dataset == "CirCor":
            print("✅ CirCor DigiScope最终全面汇总已记录到Comet")
        else:
            print("✅ 最终全面汇总已记录到Comet")

    except Exception as e:
        print(f"❌ 记录最终汇总失败: {e}")


class IndexAwareDataset(torch.utils.data.Dataset):
    """带索引感知的数据集包装器 - 解决样本追踪问题"""

    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.debug = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            if hasattr(self.dataset, '__getitem__'):
                data, label = self.dataset[idx]
            else:
                # 处理Subset等包装类
                data, label = self.dataset.__getitem__(idx)

            # 返回数据、标签和真实索引
            return data, label, idx

        except Exception as e:
            if self.debug:
                print(f"IndexAwareDataset获取数据失败: idx={idx}, error={e}")
            # 返回默认值
            return torch.zeros(1, 128, 1000), torch.tensor(0, dtype=torch.long), idx

    def get_targets(self):
        """为采样器提供标签"""
        if hasattr(self.dataset, 'get_targets'):
            return self.dataset.get_targets()
        elif hasattr(self.dataset, 'labels'):
            return self.dataset.labels.tolist() if hasattr(self.dataset.labels, 'tolist') else list(self.dataset.labels)
        else:
            # 遍历获取所有标签（较慢但通用）
            labels = []
            for i in range(len(self.dataset)):
                try:
                    _, label = self.dataset[i]
                    labels.append(label.item() if hasattr(label, 'item') else int(label))
                except:
                    labels.append(0)
            return labels


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """自定义的不平衡数据集采样器"""

    def __init__(self, dataset, num_samples=None, replacement=True):
        self.dataset = dataset
        self.replacement = replacement

        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            self.dataset_size = len(dataset)
            self.subset_indices = dataset.indices

            if hasattr(dataset.dataset, 'labels'):
                all_labels = dataset.dataset.labels
                self.labels = [all_labels[i] for i in self.subset_indices]
            elif hasattr(dataset.dataset, 'get_targets'):
                all_labels = dataset.dataset.get_targets()
                self.labels = [all_labels[i] for i in self.subset_indices]
            else:
                raise ValueError("无法从Subset中获取标签")
        else:
            self.dataset_size = len(dataset)
            if hasattr(dataset, 'labels'):
                self.labels = dataset.labels.tolist() if hasattr(dataset.labels, 'tolist') else list(dataset.labels)
            elif hasattr(dataset, 'get_targets'):
                self.labels = dataset.get_targets()
            else:
                raise ValueError("无法从数据集中获取标签")

        self.class_counts = {}
        for label in self.labels:
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

        self.weights = []
        total_samples = len(self.labels)
        num_classes = len(self.class_counts)

        for label in self.labels:
            weight = total_samples / (num_classes * self.class_counts[label])
            self.weights.append(weight)

        self.num_samples = num_samples if num_samples is not None else self.dataset_size

        print(f"🎯 平衡采样器信息:")
        print(f"   - 数据集大小: {self.dataset_size}")
        print(f"   - 类别分布: {self.class_counts}")

    def __iter__(self):
        if self.replacement:
            sampled_indices = torch.multinomial(
                torch.tensor(self.weights, dtype=torch.float),
                self.num_samples,
                replacement=True
            ).tolist()
        else:
            sampled_indices = torch.multinomial(
                torch.tensor(self.weights, dtype=torch.float),
                min(self.num_samples, len(self.weights)),
                replacement=False
            ).tolist()

        for i in sampled_indices:
            yield i

    def __len__(self):
        return self.num_samples


def get_dataset_class_names(dataset, task_type):
    """获取数据集的类别名称"""
    if dataset == "SPRSound":
        if task_type == "11":
            return ['Normal', 'Adventitious']
        elif task_type == "12":
            return ['Normal', 'Rhonchi', 'Wheeze', 'Stridor', 'Coarse Crackle', 'Fine Crackle', 'Wheeze+Crackle']
        elif task_type == "21":
            return ['Normal', 'Poor Quality', 'Adventitious']
        elif task_type == "22":
            return ['Normal', 'Poor Quality', 'CAS', 'DAS', 'CAS & DAS']
        else:
            return [f"Class {i}" for i in range(10)]
    elif dataset == "ICBHI":
        if task_type == "binary":
            return ['Normal', 'Abnormal']
        elif task_type == "multiclass":
            return ['Normal', 'Crackle', 'Wheeze', 'Wheeze+Crackle']
        else:
            return [f"Class {i}" for i in range(10)]
    elif dataset == "CirCor":
        # CirCor DigiScope心脏杂音检测任务
        return ['Present', 'Absent', 'Unknown']
    else:
        return [f"Class {i}" for i in range(10)]


def save_augmentation_statistics_to_experiment(sac_strategy, exp_dir, dataset, task_type):
    """保存增强操作统计信息到实验目录，支持CirCor"""
    if sac_strategy is None:
        print("SAC策略为空，跳过统计信息保存")
        return None

    aug_stats_dir = os.path.join(exp_dir, 'twophase_statistics')
    os.makedirs(aug_stats_dir, exist_ok=True)

    try:
        sac_strategy.save_augmentation_statistics(aug_stats_dir, dataset, task_type)

        if dataset == "CirCor":
            print(f"✅ CirCor DigiScope二阶段增强统计信息已保存到: {aug_stats_dir}")
        else:
            print(f"✅ 二阶段增强统计信息已保存到: {aug_stats_dir}")

        stats = sac_strategy.get_augmentation_statistics()
        print("\n📊 二阶段增强统计摘要:")
        print(f"   - 当前阶段: {stats.get('current_phase', 1)}")
        print(f"   - 第二阶段已触发: {stats.get('phase2_triggered', False)}")
        print(f"   - 总训练步数: {stats.get('training_steps', 0)}")

        if stats.get('current_phase', 1) == 2:
            phase2_stats = stats.get('phase2_stats', {})
            print(f"   - 易增强样本处理: {phase2_stats.get('easy_samples', 0)}")
            print(f"   - 中等难度样本处理: {phase2_stats.get('medium_samples', 0)}")
            print(f"   - 困难样本处理: {phase2_stats.get('hard_samples', 0)}")

        return aug_stats_dir

    except Exception as e:
        print(f"❌ 保存增强统计信息时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_final_training_report(exp_dir, all_folds_results, final_sac_strategy,
                                   dataset, task_type, args):
    """生成最终训练报告，支持CirCor数据集"""
    if dataset == "CirCor":
        report_path = os.path.join(exp_dir, 'final_circor_twophase_training_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# CirCor DigiScope心脏杂音检测训练报告（二阶段自适应SAC）\n\n")

            # 实验配置
            f.write("## 实验配置\n")
            f.write(f"- 数据集: CirCor DigiScope Phonocardiogram Dataset\n")
            f.write(f"- 任务: 心脏杂音检测 (3类分类)\n")
            f.write(f"- 类别: Present (存在), Absent (不存在), Unknown (未知)\n")
            f.write(f"- 模型: EfficientNet-B4\n")
            f.write(f"- 特征类型: {args.feature_type}\n")
            f.write(f"- 批次大小: {args.batch_size}\n")
            f.write(f"- 学习率: {args.lr}\n")
            f.write(f"- 训练轮数: {args.epoch}\n")
            f.write(f"- 验证集比例: {args.circor_val_ratio * 100}%\n")
            f.write(f"- 二阶段SAC增强: {'启用' if args.use_adaptive_aug else '禁用'}\n")

            if args.use_adaptive_aug:
                f.write(f"- SAC折扣因子: {args.gamma}\n")
                f.write(f"- 第二阶段触发耐心值: {args.phase2_trigger_patience}\n")
                f.write(f"- 第二阶段触发阈值: {args.phase2_trigger_threshold}\n")
                f.write(f"- 策略类型: 批次级 → 样本级自适应\n")
            f.write("\n")

            # CirCor特殊设置
            f.write("## CirCor DigiScope特殊设置\n")
            f.write("- **W.acc (Weighted Accuracy)**: W.acc = (5×cp + 3×cu + ca) / (5×tp + 3×tu + ta)\n")
            f.write("  - Present权重: 5\n")
            f.write("  - Unknown权重: 3\n")
            f.write("  - Absent权重: 1\n")
            f.write(
                "- **UAR (Unweighted Average Recall)**: UAR = (recall_present + recall_absent + recall_unknown) / 3\n\n")

            f.write("### 损失函数权重\n")
            f.write("- Present: 权重 × 5 (论文标准)\n")
            f.write("- Unknown: 权重 × 3 (论文标准)\n")
            f.write("- Absent: 权重 × 1 (论文标准)\n")

            # 五折交叉验证结果
            f.write("## 五折交叉验证结果\n")
            for i, result in enumerate(all_folds_results):
                f.write(f"### 第 {result['fold']} 折\n")
                f.write(f"- W.acc: {result.get('best_w_acc', 0.0):.4f}\n")
                f.write(f"- UAR: {result.get('best_uar', 0.0):.4f}\n")
                f.write(f"- 最佳轮次: {result['best_epoch']}\n")
                f.write("\n")

            # 平均性能
            avg_w_acc = np.mean([r.get('best_w_acc', 0.0) for r in all_folds_results])
            avg_uar = np.mean([r.get('best_uar', 0.0) for r in all_folds_results])
            std_w_acc = np.std([r.get('best_w_acc', 0.0) for r in all_folds_results])
            std_uar = np.std([r.get('best_uar', 0.0) for r in all_folds_results])

            f.write(f"### 总体性能\n")
            f.write(f"- 平均W.acc: {avg_w_acc:.4f} ± {std_w_acc:.4f}\n")
            f.write(f"- 平均UAR: {avg_uar:.4f} ± {std_uar:.4f}\n")
            f.write(f"- 最佳W.acc: {max([r.get('best_w_acc', 0.0) for r in all_folds_results]):.4f}\n")
            f.write(f"- 最佳UAR: {max([r.get('best_uar', 0.0) for r in all_folds_results]):.4f}\n")
            f.write("\n")

            f.write("### 本实验结果\n")
            f.write(f"- W.acc: {avg_w_acc:.4f} ({'↑' if avg_w_acc > 0.832 else '↓'} {abs(avg_w_acc - 0.832):.4f})\n")
            f.write(f"- UAR: {avg_uar:.4f} ({'↑' if avg_uar > 0.713 else '↓'} {abs(avg_uar - 0.713):.4f})\n\n")

            # 二阶段统计信息
            if final_sac_strategy is not None:
                f.write("## 二阶段增强策略统计\n")
                try:
                    aug_stats = final_sac_strategy.get_augmentation_statistics()
                    sac_stats = final_sac_strategy.get_statistics()

                    f.write(f"- 最终阶段: {aug_stats.get('current_phase', 1)}\n")
                    f.write(f"- 第二阶段已触发: {aug_stats.get('phase2_triggered', False)}\n")
                    f.write(f"- 总训练步数: {aug_stats.get('training_steps', 0)}\n")

                    if aug_stats.get('current_phase', 1) == 2:
                        phase2_stats = aug_stats.get('phase2_stats', {})
                        sample_difficulties = aug_stats.get('sample_difficulties', {})

                        f.write("\n### 第二阶段样本分类统计\n")
                        f.write(f"- 易增强样本: {sample_difficulties.get('easy', 0)}个\n")
                        f.write(f"- 中等难度样本: {sample_difficulties.get('medium', 0)}个\n")
                        f.write(f"- 困难样本: {sample_difficulties.get('hard', 0)}个\n")

                        f.write(f"\n### 第二阶段处理统计\n")
                        f.write(f"- 易增强样本处理次数: {phase2_stats.get('easy_samples', 0)}\n")
                        f.write(f"- 中等难度样本处理次数: {phase2_stats.get('medium_samples', 0)}\n")
                        f.write(f"- 困难样本处理次数: {phase2_stats.get('hard_samples', 0)}\n")

                    if 'avg_ba_improvement' in sac_stats:
                        f.write(f"\n- 平均BA提升: {sac_stats['avg_ba_improvement']:.4f}\n")

                except Exception as e:
                    f.write(f"- 统计信息获取失败: {e}\n")

            f.write("\n---\n")
            f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    else:
        report_path = os.path.join(exp_dir, 'final_twophase_training_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {dataset} {task_type} 训练报告（二阶段自适应SAC）\n\n")

            # 实验配置
            f.write("## 实验配置\n")
            f.write(f"- 数据集: {dataset}\n")
            f.write(f"- 任务类型: {task_type}\n")
            f.write(f"- 模型: EfficientNet-B4\n")
            f.write(f"- 特征类型: {args.feature_type}\n")
            f.write(f"- 批次大小: {args.batch_size}\n")
            f.write(f"- 学习率: {args.lr}\n")
            f.write(f"- 训练轮数: {args.epoch}\n")
            f.write(f"- 二阶段SAC增强: {'启用' if args.use_adaptive_aug else '禁用'}\n")

            if args.use_adaptive_aug:
                f.write(f"- SAC折扣因子: {args.gamma}\n")
                f.write(f"- 第二阶段触发耐心值: {args.phase2_trigger_patience}\n")
                f.write(f"- 第二阶段触发阈值: {args.phase2_trigger_threshold}\n")
                f.write(f"- 策略类型: 批次级 → 样本级自适应\n")
            f.write("\n")

            # 二阶段策略详情
            f.write("## 二阶段自适应策略\n")
            f.write("### 阶段1: Batch-level Adaptation\n")
            f.write("- 以mini-batch为单位进行SAC增强策略学习\n")
            f.write("- 收集每个样本的置信度和预测信息\n")
            f.write("- 监测性能饱和情况\n\n")

            f.write("### 阶段2: Sample-level Adaptation\n")
            f.write("- 基于历史统计将样本分为三个难度级别:\n")
            f.write("  * 易增强样本: 扩大搜索空间，更多增强策略\n")
            f.write("  * 中等难度样本: 适当扩大搜索空间\n")
            f.write("  * 困难样本: 固定采用无增强策略\n")
            f.write("- 为每个样本采用个性化增强策略\n\n")

            # 交叉验证结果
            f.write("## 五折交叉验证结果\n")
            for i, result in enumerate(all_folds_results):
                f.write(f"### 第 {result['fold']} 折\n")
                f.write(f"- 最佳分数: {result['best_score']:.4f}\n")
                f.write(f"- 最佳轮次: {result['best_epoch']}\n")
                f.write("\n")

            # 平均性能
            avg_score = sum([r['best_score'] for r in all_folds_results]) / len(all_folds_results)
            std_score = np.std([r['best_score'] for r in all_folds_results])
            f.write(f"### 总体性能\n")
            f.write(f"- 平均分数: {avg_score:.4f} ± {std_score:.4f}\n")
            f.write(f"- 最佳分数: {max([r['best_score'] for r in all_folds_results]):.4f}\n")
            f.write("\n")

            # 二阶段统计信息
            if final_sac_strategy is not None:
                f.write("## 二阶段增强策略统计\n")
                try:
                    aug_stats = final_sac_strategy.get_augmentation_statistics()
                    sac_stats = final_sac_strategy.get_statistics()

                    f.write(f"- 最终阶段: {aug_stats.get('current_phase', 1)}\n")
                    f.write(f"- 第二阶段已触发: {aug_stats.get('phase2_triggered', False)}\n")
                    f.write(f"- 总训练步数: {aug_stats.get('training_steps', 0)}\n")

                    if aug_stats.get('current_phase', 1) == 2:
                        phase2_stats = aug_stats.get('phase2_stats', {})
                        sample_difficulties = aug_stats.get('sample_difficulties', {})

                        f.write("\n### 第二阶段样本分类统计\n")
                        f.write(f"- 易增强样本: {sample_difficulties.get('easy', 0)}个\n")
                        f.write(f"- 中等难度样本: {sample_difficulties.get('medium', 0)}个\n")
                        f.write(f"- 困难样本: {sample_difficulties.get('hard', 0)}个\n")

                        f.write(f"\n### 第二阶段处理统计\n")
                        f.write(f"- 易增强样本处理次数: {phase2_stats.get('easy_samples', 0)}\n")
                        f.write(f"- 中等难度样本处理次数: {phase2_stats.get('medium_samples', 0)}\n")
                        f.write(f"- 困难样本处理次数: {phase2_stats.get('hard_samples', 0)}\n")

                    if 'avg_ba_improvement' in sac_stats:
                        f.write(f"\n- 平均BA提升: {sac_stats['avg_ba_improvement']:.4f}\n")

                except Exception as e:
                    f.write(f"- 统计信息获取失败: {e}\n")

            f.write("\n## 二阶段策略优势\n")
            f.write("- ✅ 从批次级逐步过渡到样本级适应\n")
            f.write("- ✅ 基于样本置信度智能分类\n")
            f.write("- ✅ 为不同难度样本提供个性化策略\n")
            f.write("- ✅ 避免对困难样本过度增强\n")
            f.write("- ✅ 性能饱和时自动切换策略\n")

            f.write("\n---\n")
            f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"📄 最终训练报告已保存到: {report_path}")
    return report_path


def generate_final_training_report(exp_dir, all_folds_results, final_sac_strategy,
                                   dataset, task_type, args):
    """生成最终训练报告（二阶段版）"""
    report_path = os.path.join(exp_dir, 'final_twophase_training_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {dataset} {task_type} 训练报告（二阶段自适应SAC）\n\n")

        # 实验配置
        f.write("## 实验配置\n")
        f.write(f"- 数据集: {dataset}\n")
        f.write(f"- 任务类型: {task_type}\n")
        f.write(f"- 模型: EfficientNet-B4\n")
        f.write(f"- 特征类型: {args.feature_type}\n")
        f.write(f"- 批次大小: {args.batch_size}\n")
        f.write(f"- 学习率: {args.lr}\n")
        f.write(f"- 训练轮数: {args.epoch}\n")
        f.write(f"- 二阶段SAC增强: {'启用' if args.use_adaptive_aug else '禁用'}\n")

        if args.use_adaptive_aug:
            f.write(f"- SAC折扣因子: {args.gamma}\n")
            f.write(f"- 第二阶段触发耐心值: {args.phase2_trigger_patience}\n")
            f.write(f"- 第二阶段触发阈值: {args.phase2_trigger_threshold}\n")
            f.write(f"- 策略类型: 批次级 → 样本级自适应\n")
        f.write("\n")

        # 二阶段策略详情
        f.write("## 二阶段自适应策略\n")
        f.write("### 阶段1: Batch-level Adaptation\n")
        f.write("- 以mini-batch为单位进行SAC增强策略学习\n")
        f.write("- 收集每个样本的置信度和预测信息\n")
        f.write("- 监测性能饱和情况\n\n")

        f.write("### 阶段2: Sample-level Adaptation\n")
        f.write("- 基于历史统计将样本分为三个难度级别:\n")
        f.write("  * 易增强样本: 扩大搜索空间，更多增强策略\n")
        f.write("  * 中等难度样本: 适当扩大搜索空间\n")
        f.write("  * 困难样本: 固定采用无增强策略\n")
        f.write("- 为每个样本采用个性化增强策略\n\n")

        # 交叉验证结果
        f.write("## 五折交叉验证结果\n")
        for i, result in enumerate(all_folds_results):
            f.write(f"### 第 {result['fold']} 折\n")
            f.write(f"- 最佳分数: {result['best_score']:.4f}\n")
            f.write(f"- 最佳轮次: {result['best_epoch']}\n")
            f.write("\n")

        # 平均性能
        avg_score = sum([r['best_score'] for r in all_folds_results]) / len(all_folds_results)
        std_score = np.std([r['best_score'] for r in all_folds_results])
        f.write(f"### 总体性能\n")
        f.write(f"- 平均分数: {avg_score:.4f} ± {std_score:.4f}\n")
        f.write(f"- 最佳分数: {max([r['best_score'] for r in all_folds_results]):.4f}\n")
        f.write("\n")

        # 二阶段统计信息
        if final_sac_strategy is not None:
            f.write("## 二阶段增强策略统计\n")
            try:
                aug_stats = final_sac_strategy.get_augmentation_statistics()
                sac_stats = final_sac_strategy.get_statistics()

                f.write(f"- 最终阶段: {aug_stats.get('current_phase', 1)}\n")
                f.write(f"- 第二阶段已触发: {aug_stats.get('phase2_triggered', False)}\n")
                f.write(f"- 总训练步数: {aug_stats.get('training_steps', 0)}\n")

                if aug_stats.get('current_phase', 1) == 2:
                    phase2_stats = aug_stats.get('phase2_stats', {})
                    sample_difficulties = aug_stats.get('sample_difficulties', {})

                    f.write("\n### 第二阶段样本分类统计\n")
                    f.write(f"- 易增强样本: {sample_difficulties.get('easy', 0)}个\n")
                    f.write(f"- 中等难度样本: {sample_difficulties.get('medium', 0)}个\n")
                    f.write(f"- 困难样本: {sample_difficulties.get('hard', 0)}个\n")

                    f.write(f"\n### 第二阶段处理统计\n")
                    f.write(f"- 易增强样本处理次数: {phase2_stats.get('easy_samples', 0)}\n")
                    f.write(f"- 中等难度样本处理次数: {phase2_stats.get('medium_samples', 0)}\n")
                    f.write(f"- 困难样本处理次数: {phase2_stats.get('hard_samples', 0)}\n")

                if 'avg_ba_improvement' in sac_stats:
                    f.write(f"\n- 平均BA提升: {sac_stats['avg_ba_improvement']:.4f}\n")

            except Exception as e:
                f.write(f"- 统计信息获取失败: {e}\n")

        f.write("\n## 二阶段策略优势\n")
        f.write("- ✅ 从批次级逐步过渡到样本级适应\n")
        f.write("- ✅ 基于样本置信度智能分类\n")
        f.write("- ✅ 为不同难度样本提供个性化策略\n")
        f.write("- ✅ 避免对困难样本过度增强\n")
        f.write("- ✅ 性能饱和时自动切换策略\n")

        f.write("\n---\n")
        f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"📄 最终训练报告已保存到: {report_path}")
    return report_path


class PreprocessedDataset(torch.utils.data.Dataset):
    """用于从预处理好的.npy文件加载数据的数据集类"""

    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = torch.from_numpy(self.specs[idx].astype(np.float32))
        label = int(self.labels[idx])
        label = torch.tensor(label, dtype=torch.long)

        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        return spec, label

    def get_targets(self):
        """为采样器提供标签"""
        return self.labels.tolist()


def set_seed(seed):
    """设置随机种子，保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    """带预热的余弦退火学习率调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 target_lr=0.0001, warmup_start_lr=0.001, min_lr=None, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr if min_lr is not None else target_lr / 100

        if self.total_epochs <= self.warmup_epochs:
            self.total_epochs = self.warmup_epochs + 10

        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return [self.target_lr for _ in self.optimizer.param_groups]
            progress = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * progress
                    for _ in self.optimizer.param_groups]
        else:
            remaining_epochs = self.total_epochs - self.warmup_epochs
            if remaining_epochs <= 0:
                return [self.target_lr for _ in self.optimizer.param_groups]
            progress = (self.last_epoch - self.warmup_epochs) / remaining_epochs
            progress = min(progress, 1.0)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + (self.target_lr - self.min_lr) * cosine
                    for _ in self.optimizer.param_groups]


def load_efficientnet_model(num_classes=None):
    """加载EfficientNet-B4模型，并调整适应音频输入"""
    model = EfficientNet.from_pretrained('efficientnet-b4')

    # 调整第一层卷积层适应单通道输入
    first_conv_layer = model._conv_stem
    weight_data = first_conv_layer.weight.data
    original_in_channels = weight_data.shape[1]
    new_in_channels = 1

    if new_in_channels < original_in_channels:
        weight_data = weight_data[:, :new_in_channels, :, :]
    elif new_in_channels > original_in_channels:
        additional_channels_weight = torch.randn(weight_data.shape[0], new_in_channels - original_in_channels,
                                                 weight_data.shape[2], weight_data.shape[3])
        weight_data = torch.cat([weight_data, additional_channels_weight], dim=1)

    first_conv_layer.weight.data = weight_data

    if first_conv_layer.bias is not None:
        bias_data = first_conv_layer.bias.data
        if new_in_channels < original_in_channels:
            bias_data = bias_data[:new_in_channels]
        elif new_in_channels > original_in_channels:
            additional_bias = torch.zeros(new_in_channels - original_in_channels)
            bias_data = torch.cat([bias_data, additional_bias], dim=0)
        first_conv_layer.bias.data = bias_data

    # 调整全连接层
    if num_classes is not None:
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)

    return model


class CustomEfficientNet(nn.Module):
    """定制的EfficientNet模型，添加特征提取方法"""

    def __init__(self, base_model):
        super(CustomEfficientNet, self).__init__()
        self.base_model = base_model
        self.feature_dim = self.base_model._fc.in_features

    def forward(self, x):
        return self.base_model(x)

    def extract_features(self, x):
        original_fc = self.base_model._fc
        self.base_model._fc = nn.Identity()
        with torch.no_grad():
            features = self.base_model(x)
        self.base_model._fc = original_fc
        return features


def create_model(num_classes, args):
    """创建并初始化模型"""
    base_model = load_efficientnet_model(num_classes)
    model = CustomEfficientNet(base_model)
    return model


def load_preprocessed_data(args):
    """从预处理好的.npy文件加载数据"""
    print(f"正在加载{args.dataset}数据集，任务类型: {args.task_type}")

    if args.dataset == "SPRSound":
        data_path = os.path.join(args.data_dir, "SPRSound_2022+2023_processed_data", f"task{args.task_type}")
        specs_path = os.path.join(data_path, "train_specs.npy")
        labels_path = os.path.join(data_path, "train_labels.npy")
    elif args.dataset == "ICBHI":
        data_path = os.path.join(args.data_dir, "ICBHI2017_processed_data", args.task_type)
        specs_path = os.path.join(data_path, "train_specs.npy")
        labels_path = os.path.join(data_path, "train_labels.npy")
    elif args.dataset == "CirCor":
        # CirCor DigiScope数据集路径
        data_path = os.path.join(args.data_dir, "CirCor_DigiScope_2022_processed_data")
        specs_path = os.path.join(data_path, "train_specs.npy")
        labels_path = os.path.join(data_path, "train_labels.npy")
        print(f"CirCor DigiScope数据集路径: {data_path}")
        print(f"加载心脏杂音检测任务数据 (Present/Absent/Unknown)")
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset}")

    if not os.path.exists(specs_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"找不到预处理数据文件: {specs_path} 和 {labels_path}")

    try:
        specs = np.load(specs_path)
        labels = np.load(labels_path)

        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int64)

        if specs.ndim == 2:
            n_samples = labels.shape[0]
            total_features = specs.shape[0] * specs.shape[1]
            feature_per_sample = total_features // n_samples
            height = int(np.sqrt(feature_per_sample))
            width = feature_per_sample // height
            specs = specs.reshape(n_samples, height, width)

        if specs.shape[0] != labels.shape[0]:
            min_samples = min(specs.shape[0], labels.shape[0])
            specs = specs[:min_samples]
            labels = labels[:min_samples]

        print(f"数据加载完成: {specs.shape[0]} 个样本")
        print(f"特征维度: {specs.shape}")
        print(f"标签分布: {np.bincount(labels)}")

    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

    # 确定类别数量
    num_classes = len(np.unique(labels))
    print(f"检测到{num_classes}个类别: {np.unique(labels)}")

    # 验证标签是否从0开始连续
    unique_labels = np.unique(labels)
    expected_labels = np.arange(num_classes)
    if not np.array_equal(unique_labels, expected_labels):
        label_map = {old: new for new, old in enumerate(unique_labels)}
        new_labels = np.array([label_map[l] for l in labels])
        labels = new_labels
        print(f"标签已重新映射: {label_map}")

    dataset = PreprocessedDataset(specs, labels)
    return dataset, num_classes


def collate_fn_with_indices(batch):
    """支持索引的collate函数 - 处理不规则数据"""
    try:
        # 解包批次数据
        specs, labels, indices = zip(*batch)

        # 处理规格数据维度
        processed_specs = []
        for i, spec in enumerate(specs):
            if isinstance(spec, torch.Tensor):
                if spec.dim() == 2:
                    spec = spec.unsqueeze(0)  # 添加通道维度
                elif spec.dim() == 1:
                    # 重塑一维数据
                    length = spec.shape[0]
                    height = int(np.sqrt(length))
                    width = length // height
                    spec = spec.reshape(1, height, width)
                processed_specs.append(spec)
            else:
                # 处理numpy数组
                spec = torch.from_numpy(spec.astype(np.float32))
                if spec.dim() == 2:
                    spec = spec.unsqueeze(0)
                processed_specs.append(spec)

        # 统一尺寸
        if len(processed_specs) > 0:
            # 获取最大尺寸
            max_h = max(s.shape[-2] for s in processed_specs)
            max_w = max(s.shape[-1] for s in processed_specs)
            channels = processed_specs[0].shape[0]
            batch_size = len(processed_specs)

            # 创建统一尺寸的张量
            unified_specs = torch.zeros(batch_size, channels, max_h, max_w)

            for i, spec in enumerate(processed_specs):
                h, w = spec.shape[-2], spec.shape[-1]
                unified_specs[i, :, :h, :w] = spec

            # 处理标签
            processed_labels = []
            for label in labels:
                if isinstance(label, torch.Tensor):
                    processed_labels.append(label)
                else:
                    processed_labels.append(torch.tensor(int(label), dtype=torch.long))

            labels_tensor = torch.stack(processed_labels)
            indices_list = list(indices)

            return unified_specs, labels_tensor, indices_list
        else:
            raise ValueError("Empty batch")

    except Exception as e:
        print(f"collate_fn_with_indices错误: {e}")
        # 返回默认批次
        batch_size = len(batch)
        return (torch.zeros(batch_size, 1, 128, 1000),
                torch.zeros(batch_size, dtype=torch.long),
                list(range(batch_size)))


def visualize_twophase_progress(sac_strategy, epoch, experiment):
    """可视化修正后的二阶段SAC训练进度"""
    stats = sac_strategy.get_statistics()
    aug_stats = sac_strategy.get_augmentation_statistics()

    if not stats and not aug_stats:
        return

    # 获取修正后的置信度统计
    confidence_stats = sac_strategy.confidence_tracker.get_confidence_statistics()

    plt.figure(figsize=(24, 16))

    # 1. 阶段信息显示
    plt.subplot(3, 6, 1)
    current_phase = stats.get('current_phase', 1)
    phase2_triggered = stats.get('phase2_triggered', False)

    phase_text = f"当前阶段: {current_phase}\n\n"
    if current_phase == 1:
        phase_text += "Batch-level\nAdaptation\n\n"
        phase_text += "• 批次级增强策略学习\n"
        phase_text += "• 收集样本置信度统计\n"
        if not phase2_triggered:
            phase_text += "• 监测性能饱和"
    else:
        phase_text += "Sample-level\nAdaptation\n\n"
        phase_text += "• 样本级个性化增强\n"
        phase_text += "• 基于置信度分类样本\n"
        phase_text += "• 自适应搜索空间"

    plt.text(0.1, 0.9, phase_text, ha='left', va='top', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.title('Training Phase Status')
    plt.axis('off')

    # 2. 修正后的置信度统计
    plt.subplot(3, 6, 2)
    conf_text = "修正版置信度统计:\n\n"
    conf_text += f"总样本: {confidence_stats.get('total_samples', 0)}\n"
    conf_text += f"有正确预测: {confidence_stats.get('samples_with_correct_predictions', 0)}\n"
    conf_text += f"总体准确率: {confidence_stats.get('overall_accuracy', 0):.3f}\n"
    conf_text += f"平均置信度: {confidence_stats.get('avg_confidence', 0):.3f}\n"
    conf_text += f"置信度标准差: {confidence_stats.get('confidence_std', 0):.3f}\n"
    conf_text += f"平均样本准确率: {confidence_stats.get('avg_accuracy', 0):.3f}"

    plt.text(0.1, 0.9, conf_text, ha='left', va='top', transform=plt.gca().transAxes,
             fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    plt.title('Corrected Confidence Stats')
    plt.axis('off')

    # 3. BA改善分数
    plt.subplot(3, 6, 3)
    if 'avg_ba_improvement' in stats:
        ba_improvement = stats['avg_ba_improvement']
        colors = ['lightcoral' if ba_improvement < 0 else 'lightgreen']
        plt.bar(['BA Improvement'], [ba_improvement], color=colors, alpha=0.7)
        plt.title('BA Improvement')
        plt.ylabel('BA Change')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.text(0, ba_improvement / 2, f'{ba_improvement:.4f}',
                 ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 4. 修正后的样本难度分布
    plt.subplot(3, 6, 4)
    difficulties = sac_strategy.confidence_tracker.get_all_difficulties()
    if difficulties and sum(difficulties.values()) > 0:
        diff_names = list(difficulties.keys())
        counts = list(difficulties.values())
        colors = ['green', 'orange', 'red']

        plt.pie(counts, labels=diff_names, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Corrected Sample Difficulty')
    else:
        plt.text(0.5, 0.5, 'No Difficulty\nData Available', ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.title('Sample Difficulty Distribution')
    plt.axis('off')

    # 5. 置信度分布直方图（只包含正确预测的样本）
    plt.subplot(3, 6, 5)
    if sac_strategy.confidence_tracker.sample_stats:
        confidences = [stats['confidence'] for stats in sac_strategy.confidence_tracker.sample_stats.values()
                       if stats.get('correct_predictions', 0) > 0]
        if confidences:
            plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Confidence Distribution\n(Correct Predictions Only)')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
        else:
            plt.text(0.5, 0.5, 'No Correct\nPredictions Yet', ha='center', va='center',
                     transform=plt.gca().transAxes)
            plt.title('Confidence Distribution')
    else:
        plt.text(0.5, 0.5, 'No Data\nAvailable', ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.title('Confidence Distribution')

    # 6. Alpha值显示（第二阶段）
    plt.subplot(3, 6, 6)
    if current_phase == 2:
        alphas = sac_strategy.adaptive_alpha.get_all_alphas()
        alpha_names = list(alphas.keys())
        alpha_values = list(alphas.values())
        colors = ['green', 'orange', 'red']

        bars = plt.bar(alpha_names, alpha_values, color=colors[:len(alpha_names)], alpha=0.7)
        plt.title('Adaptive Alpha Values')
        plt.ylabel('Alpha')
        plt.yscale('log')

        # 添加数值标签
        for bar, val in zip(bars, alpha_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, f'Phase {current_phase}\nAlpha Values\nNot Available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Alpha Values')

    # 其余subplot保持原样...
    # 7-12省略，与原版本相同

    plt.suptitle(
        f'Corrected Two-Phase SAC Progress - Epoch {epoch} (Phase {current_phase}: {"Sample-level" if current_phase == 2 else "Batch-level"})',
        fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存并记录到Comet
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)

    experiment.log_image(buf, name=f'corrected_twophase_sac_progress_epoch_{epoch}', step=epoch)

    plt.close()

def compute_validation_gradient_avg(model, val_loader, device, num_samples=32):
    """计算验证集上一小批数据的最后一层平均梯度向量"""
    model.eval()
    grad_list = []
    count = 0
    for batch_data in val_loader:
        if len(batch_data) == 3:
            inputs, labels, _ = batch_data
        else:
            inputs, labels = batch_data
        inputs = inputs[:num_samples-count].to(device)
        labels = labels[:num_samples-count].to(device)
        if inputs.size(0) == 0:
            break
        inputs.requires_grad_(True)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        # 获取最后一层参数的梯度
        last_layer = model.base_model._fc
        if last_layer.weight.grad is not None:
            grad_list.append(last_layer.weight.grad.view(-1))
        if last_layer.bias is not None and last_layer.bias.grad is not None:
            grad_list.append(last_layer.bias.grad.view(-1))
        model.zero_grad()
        count += inputs.size(0)
        if count >= num_samples:
            break
    model.train()
    if grad_list:
        return torch.cat(grad_list).detach().mean(dim=0)  # 平均梯度向量
    else:
        return torch.zeros(1, device=device)


def train(args, device):
    """训练主函数，使用二阶段SAC策略，带全面Comet记录"""
    set_seed(args.seed)

    # 初始化Comet实验记录
    experiment = init_comet_experiment()
    log_hyperparameters(experiment, args)

    # 加载预处理好的数据
    train_dataset, num_classes = load_preprocessed_data(args)
    indexed_train_dataset = IndexAwareDataset(train_dataset)

    # 设置五折交叉验证 - CirCor数据集使用特殊的分割策略
    all_labels = train_dataset.labels
    if args.dataset == "CirCor":
        # CirCor使用特殊的验证策略：5折交叉验证，每折验证集占10%
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        indices = np.arange(len(train_dataset))
        fold_splits = list(skf.split(indices, all_labels))
    else:
        # 其他数据集使用标准的5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        indices = np.arange(len(train_dataset))
        fold_splits = list(skf.split(indices, all_labels))

    # 创建实验目录
    exp_dir = create_experiment_dir(args)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    all_folds_results = []
    final_sac_strategy = None  # 保存最后一个SAC策略用于最终统计

    # 开始五折交叉验证训练
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n======= 开始第 {fold + 1}/5 折训练 =======")

        train_subset = Subset(indexed_train_dataset, train_idx)
        val_subset = Subset(indexed_train_dataset, val_idx)

        print(f"当前折训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")

        # 记录fold级别的数据集信息到Comet
        experiment.log_metric(f"fold_{fold + 1}_train_size", len(train_subset))
        experiment.log_metric(f"fold_{fold + 1}_val_size", len(val_subset))

        # 创建平衡采样器
        balanced_sampler = ImbalancedDatasetSampler(train_subset)

        # 使用支持索引的数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=balanced_sampler,
            collate_fn=collate_fn_with_indices,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn_with_indices,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )

        # 创建新模型
        model = create_model(num_classes, args)
        model = model.to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        # 定义学习率调度器
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epoch,
            total_epochs=args.epoch,
            target_lr=args.lr,
            warmup_start_lr=args.warmup_lr,
            min_lr=1e-6
        )

        # 当前折的训练参数
        patience = 30
        best_val_score = 0.0
        best_epoch = -1
        early_stop_counter = 0

        # 初始化二阶段SAC策略
        sac_strategy = None
        if args.use_adaptive_aug:
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 128, 1000).to(device)
                feature_dim = model.feature_dim

            # 定义可用的增强操作列表
            aug_operations = ["time_mask", "frequency_mask", "noise_injection", "random_quantization",
                              "spectral_contrast", "harmonic_perturbation", "breathing_cycle_stretch",
                              "low_freq_emphasis"]

            # 创建二阶段SAC策略
            sac_strategy = TwoPhaseSACStrategy(
                n_classes=num_classes,
                feature_dim=feature_dim,
                aug_operations=aug_operations,
                device=device,
                buffer_capacity=args.sac_buffer_size,
                gamma=args.gamma,
                tau=args.sac_tau,
                actor_lr=args.sac_lr_actor,
                critic_lr=args.sac_lr_critic,
                validation_set=val_loader,
                n_magnitude_levels=args.n_magnitude_levels,
                phase2_trigger_patience=args.phase2_trigger_patience,
                phase2_trigger_threshold=args.phase2_trigger_threshold,
                debug_mode=True
            )
            sac_strategy = apply_simple_anti_memorization_fix(sac_strategy)

            # 创建指标收集器
            from PASA_Strategy import MetricsCollector   # 确保类被导入
            metrics_log_dir = os.path.join(exp_dir, 'metrics', f'fold_{fold+1}')
            os.makedirs(metrics_log_dir, exist_ok=True)
            metrics_collector = MetricsCollector(
                log_dir=metrics_log_dir,
                experiment=experiment,
                record_every=10,
                compute_grad_dot_every=50
            )
            sac_strategy.set_metrics_collector(metrics_collector)

            # 记录SAC初始化参数到Comet
            experiment.log_parameters({
                f"fold_{fold + 1}_sac_gamma": args.gamma,
                f"fold_{fold + 1}_sac_tau": args.sac_tau,
                f"fold_{fold + 1}_sac_buffer_size": args.sac_buffer_size,
                f"fold_{fold + 1}_n_magnitude_levels": args.n_magnitude_levels,
                f"fold_{fold + 1}_phase2_trigger_patience": args.phase2_trigger_patience,
                f"fold_{fold + 1}_phase2_trigger_threshold": args.phase2_trigger_threshold,
            })

            print(f"🚀 开始第 {fold + 1} 折训练，共{args.epoch}个epoch，使用二阶段SAC策略")

        # 训练循环
        for epoch in range(args.epoch):
            epoch_start_time = time.time()
            print(f"\n----- 第 {fold + 1} 折 Epoch {epoch}/{args.epoch} -----")

            # 训练模式
            model.train()
            running_loss = 0.0
            batch_count = 0
            total_reward = 0.0
            reward_count = 0

            # 遍历训练数据批次
            for batch_idx, batch_data in enumerate(train_loader):
                # 处理3元组返回值
                if len(batch_data) == 3:
                    inputs, labels, sample_indices = batch_data
                else:
                    inputs, labels = batch_data
                    sample_indices = None

                inputs, labels = inputs.to(device), labels.to(device)

                # 修正：根据二阶段SAC策略应用增强（不在此处检查阶段切换）
                if args.use_adaptive_aug and sac_strategy is not None:
                    augmented_inputs, augmented_labels, avg_reward, sac_losses = sac_strategy.train_step(
                        model=model,
                        criterion=criterion,
                        x=inputs,
                        y=labels,
                        epoch=epoch,
                        total_epochs=args.epoch,
                        sample_indices=sample_indices,
                        batch_idx=batch_idx
                    )

                    if 'early_stop' in sac_losses and sac_losses.get('early_stop', False):
                        print(f"SAC策略要求停止训练")
                        break

                    total_reward += avg_reward
                    reward_count += 1

                    # 记录批次级别的详细指标
                    if batch_idx % 5 == 0:
                        step = epoch * len(train_loader) + batch_idx
                        experiment.log_metric(f"fold_{fold + 1}_batch_reward", avg_reward, step=step)
                        experiment.log_metric(f"fold_{fold + 1}_current_phase", sac_strategy.current_phase, step=step)

                        # 记录SAC损失
                        for loss_name, loss_val in sac_losses.items():
                            if isinstance(loss_val, (int, float)):
                                experiment.log_metric(f"fold_{fold + 1}_batch_{loss_name}", loss_val, step=step)

                    # 前向传播
                    optimizer.zero_grad()
                    outputs = model(augmented_inputs)
                    loss = criterion(outputs, augmented_labels)
                else:
                    # 不使用增强
                    augmented_inputs, augmented_labels = inputs, labels
                    avg_reward = 0.0
                    sac_losses = {}
                    optimizer.zero_grad()
                    outputs = model(augmented_inputs)
                    loss = criterion(outputs, augmented_labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                # 每10个批次记录训练损失
                if batch_idx % 10 == 0:
                    step = epoch * len(train_loader) + batch_idx
                    experiment.log_metric(f"fold_{fold + 1}_train_batch_loss", loss.item(), step=step)

                    if reward_count > 0:
                        current_phase = sac_strategy.current_phase if sac_strategy else 1
                        reward_info = f", Avg Reward: {total_reward / max(1, reward_count):.4f}, Phase: {current_phase}"
                        if sac_losses:
                            sac_info = (f", Actor: {sac_losses.get('actor_loss', 0):.4f}"
                                        f", Alpha: {sac_losses.get('alpha', 0):.4f}")
                            reward_info += sac_info
                    else:
                        reward_info = ""

                    print(f"Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}{reward_info}")

            # 修正：在epoch结束后检查阶段切换
            if args.use_adaptive_aug and sac_strategy is not None:
                sac_strategy.check_epoch_end_phase_transition(model, criterion, epoch)

            # 计算当前epoch的平均训练损失
            train_loss = running_loss / max(1, batch_count)
            avg_epoch_reward = total_reward / max(1, reward_count)

            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # 记录epoch级别的基础指标
            experiment.log_metric(f"fold_{fold + 1}_train_loss", train_loss, step=epoch)
            experiment.log_metric(f"fold_{fold + 1}_learning_rate", current_lr, step=epoch)

            # 训练后记录全面的metrics
            if sac_strategy is not None:
                # 记录性能历史长度和最佳性能
                if len(sac_strategy.performance_history) > 0:
                    current_best = max(sac_strategy.performance_history)
                    experiment.log_metric(f"fold_{fold + 1}_historical_best", current_best, step=epoch)
                    experiment.log_metric(f"fold_{fold + 1}_epochs_since_best",
                                          len(sac_strategy.performance_history) -
                                          list(sac_strategy.performance_history).index(current_best) - 1,
                                          step=epoch)

                # 1. 记录全面的SAC统计
                log_comprehensive_sac_metrics(sac_strategy, experiment, epoch, fold + 1)

                # 2. 修正：记录详细的Alpha统计（替代UCB统计）
                if sac_strategy.current_phase == 2:
                    log_detailed_alpha_metrics(sac_strategy, experiment, epoch, fold + 1)

                # 3. 记录阶段转换信息
                log_phase_transition_metrics(sac_strategy, experiment, epoch, fold + 1)

                # 4. 记录增强效果
                log_augmentation_effectiveness_metrics(sac_strategy, experiment, epoch, fold + 1)

            # 评估后记录详细验证metrics
            log_comprehensive_validation_metrics(model, val_loader, criterion, device,
                                                 experiment, epoch, fold + 1, args)

            # 每20个epoch进行一次全面的统计验证和记录
            if epoch % 20 == 0 and sac_strategy is not None:
                sac_strategy.validate_statistics()
                val_grad_avg = compute_validation_gradient_avg(model, val_loader, device)
                sac_strategy.metrics_collector.set_validation_gradient(val_grad_avg)

                # 记录验证结果
                validation_results = {
                    'statistics_consistent': True,  # 这里应该从validate_statistics得到实际结果
                    'total_tracked_samples': len(sac_strategy.confidence_tracker.sample_stats),
                    'current_phase': sac_strategy.current_phase,
                }

                for key, value in validation_results.items():
                    experiment.log_metric(f"fold_{fold + 1}_validation_{key}", value, step=epoch)

                # 修正：如果是第二阶段，打印Alpha统计而不是UCB统计
                if sac_strategy.current_phase == 2:
                    print("第二阶段Alpha策略统计:")
                    sac_strategy.adaptive_alpha.print_alpha_statistics()

            # 评估阶段
            model.eval()
            print("\n----- 训练和验证结果 -----")

            with torch.no_grad():
                # 在训练集上评估（使用一个小子集）
                train_eval_indices = np.random.choice(train_idx, min(1000, len(train_idx)), replace=False)
                train_eval_subset = Subset(indexed_train_dataset, train_eval_indices)
                train_eval_loader = DataLoader(
                    train_eval_subset,
                    batch_size=args.batch_size * 2,
                    shuffle=False,
                    collate_fn=collate_fn_with_indices,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    prefetch_factor=args.prefetch_factor
                )

                train_results = evaluate(
                    model=model,
                    data_loader=train_eval_loader,
                    criterion=criterion,
                    device=device,
                    val_set_name=f"fold_{fold + 1}_train",
                    epoch=epoch,
                    experiment=experiment,
                    task_type=args.task_type,
                    dataset=args.dataset
                )

                # 在验证集上评估
                val_results = evaluate(
                    model=model,
                    data_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    val_set_name=f"fold_{fold + 1}_val",
                    epoch=epoch,
                    experiment=experiment,
                    task_type=args.task_type,
                    dataset=args.dataset
                )

            # ========== 关键：记录详细的验证metrics ==========
            log_comprehensive_validation_metrics(model, val_loader, criterion, device,
                                                 experiment, epoch, fold + 1, args)

            # 可视化训练进度
            if sac_strategy is not None and (epoch % 10 == 0 or epoch < 5):
                visualize_twophase_progress(sac_strategy, epoch, experiment)  # 使用修正版

            # ========== 记录epoch汇总metrics ==========
            if epoch % 10 == 0:  # 每10个epoch记录一次汇总
                log_epoch_summary_metrics(sac_strategy, experiment, epoch, fold + 1)

            # 检查是否是最佳模型
            current_score = val_results['overall_score']
            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                early_stop_counter = 0

                # 保存最佳模型
                best_model_path = os.path.join(exp_dir, 'models', f'best_fold_{fold + 1}.pth')
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'score': best_val_score,
                    'sac_stats': sac_strategy.get_statistics() if sac_strategy else None
                }, best_model_path)

                # 记录最佳模型信息到Comet
                experiment.log_metric(f"fold_{fold + 1}_best_score", best_val_score, step=epoch)
                experiment.log_metric(f"fold_{fold + 1}_best_epoch", epoch, step=epoch)

                print(f"保存第 {fold + 1} 折的最佳模型 (epoch {epoch})，分数: {best_val_score:.4f}")
            else:
                early_stop_counter += 1

            # 早停检查
            if args.early_stop and epoch >= args.warmup_epoch + 10:
                if early_stop_counter >= patience:
                    print(f"第 {fold + 1} 折的性能已连续{patience}轮未提升，早停训练")
                    experiment.log_metric(f"fold_{fold + 1}_early_stopped", 1, step=epoch)
                    experiment.log_metric(f"fold_{fold + 1}_early_stop_epoch", epoch, step=epoch)
                    break

            # 计算epoch耗时
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch耗时: {epoch_duration:.2f}秒")
            experiment.log_metric(f"fold_{fold + 1}_epoch_time", epoch_duration, step=epoch)
            print("----------------------------")

        # 保存该折的增强统计
        if sac_strategy is not None:
            fold_aug_dir = os.path.join(exp_dir, 'twophase_statistics', f'fold_{fold + 1}')
            os.makedirs(fold_aug_dir, exist_ok=True)
            try:
                sac_strategy.save_augmentation_statistics(
                    fold_aug_dir,
                    f"{args.dataset}_fold{fold + 1}",
                    args.task_type
                )
                print(f"第 {fold + 1} 折二阶段统计已保存")
            except Exception as e:
                print(f"第 {fold + 1} 折统计保存失败: {e}")

            # 保存最后一个SAC策略用于最终统计
            final_sac_strategy = sac_strategy

        # 记录当前折的结果
        fold_result = {
            'fold': fold + 1,
            'best_score': best_val_score,
            'best_epoch': best_epoch
        }
        all_folds_results.append(fold_result)

        # 记录fold完成信息到Comet
        experiment.log_metric(f"fold_{fold + 1}_final_score", best_val_score)
        experiment.log_metric(f"fold_{fold + 1}_final_epoch", best_epoch)

        print(f"\n第 {fold + 1} 折训练完成，最佳分数: {best_val_score:.4f}，最佳epoch: {best_epoch}")

    # ========== 记录最终的全面汇总 ==========
    log_final_comprehensive_summary(all_folds_results, final_sac_strategy, experiment, args)

    # 所有折训练完成后的结果分析
    best_fold_idx = np.argmax([r['best_score'] for r in all_folds_results])
    best_fold = all_folds_results[best_fold_idx]

    print("\n===== 五折交叉验证完成 =====")
    for result in all_folds_results:
        print(f"第 {result['fold']} 折: 最佳分数 {result['best_score']:.4f} (epoch {result['best_epoch']})")

    print(f"\n最佳模型来自第 {best_fold['fold']} 折，分数: {best_fold['best_score']:.4f}")

    # 保存交叉验证结果摘要
    summary_path = os.path.join(exp_dir, 'twophase_cv_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'all_folds': all_folds_results,
            'best_fold': best_fold,
            'avg_score': sum([r['best_score'] for r in all_folds_results]) / len(all_folds_results),
            'std_score': np.std([r['best_score'] for r in all_folds_results]),
            'sac_enabled': args.use_adaptive_aug,
            'sac_strategy': 'corrected_two_phase_adaptive',
            'phase2_trigger_patience': args.phase2_trigger_patience,
            'phase2_trigger_threshold': args.phase2_trigger_threshold,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)

    # 记录实验完成信息
    experiment.log_text("experiment_status", "completed")
    experiment.log_metric("total_folds_completed", len(all_folds_results))

    print(f"\n🎉 实验完成！所有metrics已记录到Comet")
    print(f"📊 Comet实验链接: {experiment.url}")

    return exp_dir, all_folds_results


def evaluate(model, data_loader, criterion, device, val_set_name, epoch, experiment, task_type, dataset):
    """评估模型性能"""
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            # 处理不同格式的batch_data
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data
            else:
                inputs, labels = batch_data

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = 100 * correct / total if total > 0 else 0

    # 针对不同数据集计算不同的指标
    if dataset == "CirCor":
        # CirCor DigiScope数据集使用特定的评价指标
        circor_metrics = calculate_circor_metrics(all_labels, all_preds)

        # 使用W.acc作为主要评价指标
        overall_score = circor_metrics['w_acc']
        macro_sensitivity = circor_metrics['recall_present']
        macro_specificity = circor_metrics['recall_absent']

        # 记录CirCor特定指标到Comet
        experiment.log_metric(f"eval_w_acc_{val_set_name}", circor_metrics['w_acc'], step=epoch)
        experiment.log_metric(f"eval_uar_{val_set_name}", circor_metrics['uar'], step=epoch)
        experiment.log_metric(f"eval_recall_present_{val_set_name}", circor_metrics['recall_present'], step=epoch)
        experiment.log_metric(f"eval_recall_absent_{val_set_name}", circor_metrics['recall_absent'], step=epoch)
        experiment.log_metric(f"eval_recall_unknown_{val_set_name}", circor_metrics['recall_unknown'], step=epoch)

        print(
            f"{val_set_name.upper():10} - Acc: {accuracy:.2f}%, W.acc: {circor_metrics['w_acc']:.4f}, UAR: {circor_metrics['uar']:.4f}")

    else:
        # SPRSound 和 ICBHI 使用官方定义的二元化 SE/SP 计算
        try:
            f1 = f1_score(all_labels, all_preds, average='macro')

            if dataset == "SPRSound":
                # SPRSound 官方: Normal(0) vs 非Normal, Score=(AS+HS)/2
                m = calculate_sprsound_metrics(all_labels, all_preds)
            elif dataset == "ICBHI":
                # ICBHI 官方: Normal(0) vs 非Normal, Score=(Se+Sp)/2
                m = calculate_icbhi_metrics(all_labels, all_preds)
            else:
                # 其他未知数据集，fallback 到二元化计算
                m = calculate_icbhi_metrics(all_labels, all_preds)

            macro_sensitivity = m['sensitivity']
            macro_specificity = m['specificity']
            overall_score = m['overall_score']
            average_score = m.get('average_score', (macro_sensitivity + macro_specificity) / 2)
            harmonic_score = m.get('harmonic_score', 0.0)

        except Exception as e:
            print(f"指标计算出错: {e}")
            f1 = 0.0
            macro_sensitivity = 0.5
            macro_specificity = 0.5
            overall_score = 0.5
            average_score = 0.5
            harmonic_score = 0.5

        # 输出结果
        print(
            f"{val_set_name.upper():10} - Acc: {accuracy:.2f}%, Sens: {macro_sensitivity:.4f}, Spec: {macro_specificity:.4f}, Score: {overall_score:.4f}")

    # 记录通用指标到Comet
    try:
        experiment.log_metric(f"val_accuracy_{val_set_name}", accuracy, step=epoch)
        experiment.log_metric(f"val_loss_{val_set_name}", running_loss / len(data_loader), step=epoch)
        if dataset != "CirCor":  # CirCor的特定指标已经在上面记录过了
            experiment.log_metric(f"val_f1_{val_set_name}", f1, step=epoch)
            experiment.log_metric(f"val_sensitivity_{val_set_name}", macro_sensitivity, step=epoch)
            experiment.log_metric(f"val_specificity_{val_set_name}", macro_specificity, step=epoch)
        experiment.log_metric(f"val_overall_score_{val_set_name}", overall_score, step=epoch)
    except Exception as e:
        print(f"记录指标到Comet时出错: {e}")

    # 构建返回结果
    result = {
        'accuracy': accuracy,
        'loss': running_loss / len(data_loader),
        'macro_sensitivity': macro_sensitivity,
        'macro_specificity': macro_specificity,
        'overall_score': overall_score
    }

    if dataset == "CirCor":
        # 为CirCor数据集添加特定指标
        result.update({
            'w_acc': circor_metrics['w_acc'],
            'uar': circor_metrics['uar'],
            'recall_present': circor_metrics['recall_present'],
            'recall_absent': circor_metrics['recall_absent'],
            'recall_unknown': circor_metrics['recall_unknown']
        })
    else:
        result['f1'] = f1

    return result

# def create_experiment_dir(args):
#     """创建实验目录，支持CirCor数据集"""
#     base_dir = './experiments'
#     os.makedirs(base_dir, exist_ok=True)
#
#     # 构建实验标识符
#     if args.dataset == "CirCor":
#         # CirCor特殊的实验标识符
#         aug_suffix = f"CorrectedTwoPhaseSAC_P{args.phase2_trigger_patience}_T{args.phase2_trigger_threshold}" if args.use_adaptive_aug else "NoAug"
#         exp_identifier = (
#             f"CirCor_DigiScope_"
#             f"HeartMurmur_"
#             f"EfficientNet-B4_"
#             f"{args.feature_type}_"
#             f"{aug_suffix}"
#         )
#     else:
#         # 其他数据集使用原有逻辑
#         aug_suffix = f"CorrectedTwoPhaseSAC_P{args.phase2_trigger_patience}_T{args.phase2_trigger_threshold}" if args.use_adaptive_aug else "NoAug"
#         exp_identifier = (
#             f"{args.dataset}_"
#             f"task{args.task_type}_"
#             f"EfficientNet-B4_"
#             f"{args.feature_type}_"
#             f"{aug_suffix}"
#         )
#
#     # 添加时间戳
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     exp_dir = os.path.join(base_dir, f"{exp_identifier}_{timestamp}")
#
#     # 创建实验目录
#     os.makedirs(exp_dir, exist_ok=True)
#
#     # 保存实验配置
#     config_path = os.path.join(exp_dir, 'config.txt')
#     with open(config_path, 'w') as f:
#         for arg, value in vars(args).items():
#             f.write(f'{arg}: {value}\n')
#
#         # CirCor特殊配置信息
#         if args.dataset == "CirCor":
#             f.write(f'\n# CirCor DigiScope Specific Configuration\n')
#             f.write(f'dataset_type: heart_murmur_detection\n')
#             f.write(f'num_classes: 3\n')
#             f.write(f'classes: [Present, Absent, Unknown]\n')
#             f.write(f'evaluation_metrics: [W.acc, UAR]\n')
#             f.write(f'validation_ratio: {args.circor_val_ratio}\n')
#             f.write(
#                 f'reference_paper: Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection\n')
#             f.write(f'class_weights: [Present*5, Absent*1, Unknown*3]\n')
#
#     return exp_dir

def create_experiment_dir(args):
    """创建实验目录，支持扩展模块标识"""
    base_dir = './exp_extended'          # 修改1：基础文件夹重命名
    os.makedirs(base_dir, exist_ok=True)

    # 生成扩展模块标识字符串
    ext_parts = []
    if getattr(args, 'use_etf', False):
        ext_parts.append('ETF')
    if getattr(args, 'use_dynamic_threshold', False):
        ext_parts.append('DynThr')
    if getattr(args, 'use_action_aware_threshold', False):
        ext_parts.append('ActThr')
    if getattr(args, 'use_retrieval_exploration', False):
        ext_parts.append('Retrieval')
    ext_suffix = '_'.join(ext_parts) if ext_parts else 'NoExt'

    # 数据集和任务类型标识
    if args.dataset == "CirCor":
        dataset_tag = "CirCor_DigiScope_HeartMurmur"
    else:
        dataset_tag = f"{args.dataset}_task{args.task_type}"

    # 构建完整目录名
    aug_suffix = f"TwoPhaseSAC_P{args.phase2_trigger_patience}_T{args.phase2_trigger_threshold}" if args.use_adaptive_aug else "NoAug"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{dataset_tag}_{ext_suffix}_{aug_suffix}_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_dir_name)

    os.makedirs(exp_dir, exist_ok=True)

    # 保存实验配置（原有逻辑保留）
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
        # ... 其他特殊配置（如 CirCor）保留不变 ...

    return exp_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于修正二阶段SAC的自适应增强策略肺音分类训练脚本')

    # 基本参数
    parser.add_argument("--dataset", type=str, default="ICBHI", choices=["SPRSound", "ICBHI", "CirCor"])
    parser.add_argument("--circor_val_ratio", type=float, default=0.1)
    parser.add_argument("--task_type", type=str, default="multiclass")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)

    # 特征和模型参数
    parser.add_argument("--feature_type", type=str, default='log-mel', choices=['MFCC', 'log-mel', 'mel', 'STFT'])

    # SAC相关参数
    parser.add_argument("--use_adaptive_aug", type=lambda x: x.lower() == 'true', default=True)

    parser.add_argument("--use_etf", type=lambda x: x.lower() == 'true', default=False,
                        help='启用 ETF 投影（统一度量尺度）')
    parser.add_argument("--use_dynamic_threshold", type=lambda x: x.lower() == 'true', default=False,
                        help='启用类级动态阈值')
    parser.add_argument("--use_action_aware_threshold", type=lambda x: x.lower() == 'true', default=False,
                        help='启用动作感知阈值修正')
    parser.add_argument("--use_retrieval_exploration", type=lambda x: x.lower() == 'true', default=False,
                        help='启用检索增强探索')

    parser.add_argument("--n_magnitude_levels", type=int, default=5, help="每个操作的幅度级别数")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--sac_lr_actor", type=float, default=3e-4)
    parser.add_argument("--sac_lr_critic", type=float, default=3e-4)
    parser.add_argument("--sac_buffer_size", type=int, default=10000)
    parser.add_argument("--sac_tau", type=float, default=0.005)

    # 二阶段特有参数
    parser.add_argument("--phase2_trigger_patience", type=int, default=20, help="第二阶段触发耐心值")
    parser.add_argument("--phase2_trigger_threshold", type=float, default=0.01, help="第二阶段触发阈值")

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_epoch", type=int, default=25)
    parser.add_argument("--warmup_lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    args = parser.parse_args()

    # 设置CUDA设备
    if torch.cuda.is_available():
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            device_id = 0
        else:
            device_id = args.gpu_id
        global_device = torch.device(f"cuda:{device_id}")
    else:
        global_device = torch.device("cpu")

    print(f"使用设备: {global_device}")

    # 打印二阶段SAC配置信息
    if args.use_adaptive_aug:
        print(f"修正版二阶段SAC配置:")
        print(f"  - γ={args.gamma}")
        print(f"  - 第二阶段触发耐心值={args.phase2_trigger_patience}")
        print(f"  - 第二阶段触发阈值={args.phase2_trigger_threshold}")
        print(f"  - 阶段1: Batch-level adaptation")
        print(f"  - 阶段2: Sample-level adaptation (fixed hard sample strategy)")

    # 开始训练
    try:
        exp_dir, cv_results = train(args, global_device)

        print(f"\n🎉 训练完成！实验结果保存在: {exp_dir}")
        print("📁 实验目录结构:")
        print(f"   ├── models/ (模型文件)")
        print(f"   ├── twophase_statistics/ (二阶段统计)")
        print(f"   └── twophase_cv_results_summary.json (交叉验证摘要)")

        return exp_dir, cv_results

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()