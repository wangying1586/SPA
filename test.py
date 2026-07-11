"""
A²-PASA 测试评估脚本
功能:
  - 加载 5-fold 交叉验证训练的模型
  - 每个 fold 独立评估性能
  - 集成投票（概率平均）评估
  - 可视化: 混淆矩阵、ROC 曲线、PR 曲线、t-SNE 分布图
  - 兼容 ICBHI / SPRSound / CirCor 数据集
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.manifold import TSNE
import pandas as pd
from datetime import datetime
from tabulate import tabulate

from PASA_Main import (
    get_dataset_class_names,
    CustomEfficientNet,
    load_efficientnet_model,
    calculate_circor_metrics,
)


# ============================================================================
# 数据加载
# ============================================================================

class PreprocessedDataset(Dataset):
    """预处理数据集"""
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        if len(spec.shape) == 2:
            spec = np.expand_dims(spec, axis=0)
        return spec, self.labels[idx]


def collate_fn_preprocessed(batch):
    """Collate function"""
    specs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    specs = [torch.from_numpy(spec).float() if isinstance(spec, np.ndarray) else spec for spec in specs]
    labels = torch.tensor(labels, dtype=torch.long)
    stacked_specs = torch.stack(specs)
    return stacked_specs, labels


def load_preprocessed_data(args):
    """加载预处理好的测试数据"""
    print(f"正在加载{args.dataset}数据集，任务类型: {args.task_type}")

    if args.dataset == "SPRSound":
        data_path = os.path.join(args.data_dir, "SPRSound_2022+2023_processed_data", f"task{args.task_type}")
        specs_path = os.path.join(data_path, "test_specs.npy")
        labels_path = os.path.join(data_path, "test_labels.npy")
    elif args.dataset == "ICBHI":
        data_path = os.path.join(args.data_dir, "ICBHI2017_processed_data", args.task_type)
        specs_path = os.path.join(data_path, "test_specs.npy")
        labels_path = os.path.join(data_path, "test_labels.npy")
    elif args.dataset == "CirCor":
        data_path = os.path.join(args.data_dir, "CirCor_DigiScope_2022_processed_data")
        specs_path = os.path.join(data_path, "test_specs.npy")
        labels_path = os.path.join(data_path, "test_labels.npy")
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    if not os.path.exists(specs_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"找不到数据文件: {specs_path} 或 {labels_path}")

    specs = np.load(specs_path)
    labels = np.load(labels_path)

    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)

    if specs.shape[0] != labels.shape[0]:
        min_samples = min(specs.shape[0], labels.shape[0])
        specs = specs[:min_samples]
        labels = labels[:min_samples]

    num_classes = len(np.unique(labels))
    print(f"数据加载完成: {specs.shape[0]} 个样本, {num_classes} 个类别")

    unique_labels = np.unique(labels)
    expected_labels = np.arange(num_classes)
    if not np.array_equal(unique_labels, expected_labels):
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])

    dataset = PreprocessedDataset(specs, labels)
    return dataset, num_classes


def load_test_dataset(args, test_set_name):
    """加载特定测试集"""
    if args.dataset == 'SPRSound':
        task_type = int(args.task_type) if isinstance(args.task_type, str) else args.task_type
        if test_set_name == 'test_2022_intra':
            specs_path = os.path.join(args.data_dir, f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_intra_specs.npy')
            labels_path = os.path.join(args.data_dir, f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_intra_labels.npy')
        elif test_set_name == 'test_2022_inter':
            specs_path = os.path.join(args.data_dir, f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_inter_specs.npy')
            labels_path = os.path.join(args.data_dir, f'SPRSound_2022+2023_processed_data/task{task_type}/test_2022_inter_labels.npy')
        elif test_set_name == 'test_2023':
            specs_path = os.path.join(args.data_dir, f'SPRSound_2022+2023_processed_data/task{task_type}/test_2023_specs.npy')
            labels_path = os.path.join(args.data_dir, f'SPRSound_2022+2023_processed_data/task{task_type}/test_2023_labels.npy')
        else:
            raise ValueError(f"Unknown test set: {test_set_name}")

        if not os.path.exists(specs_path) or not os.path.exists(labels_path):
            print(f"Warning: {test_set_name} 数据文件不存在")
            return None, None

        specs = np.load(specs_path)
        labels = np.load(labels_path)
        dataset = PreprocessedDataset(specs, labels)
        num_classes = len(np.unique(labels))
        print(f"Loaded {test_set_name}: {len(dataset)} samples, {num_classes} classes")
        return dataset, num_classes

    else:
        dataset, num_classes = load_preprocessed_data(args)
        return dataset, num_classes


def create_combined_dataset(datasets):
    """合并多个测试集"""
    all_specs, all_labels = [], []
    for dataset in datasets:
        for i in range(len(dataset)):
            spec, label = dataset[i]
            all_specs.append(spec)
            all_labels.append(label)
    all_specs = np.array(all_specs)
    all_labels = np.array(all_labels)
    combined = PreprocessedDataset(all_specs, all_labels)
    num_classes = len(np.unique(all_labels))
    print(f"Combined dataset: {len(combined)} samples, {num_classes} classes")
    return combined, num_classes


# ============================================================================
# 模型加载
# ============================================================================

def load_model_weights(model, weights_path, device):
    """加载模型权重"""
    print(f"Loading weights: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        epoch = checkpoint.get('epoch', 0)
    else:
        state_dict = checkpoint
        epoch = 0

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded successfully (epoch={epoch})")
    return model, epoch


def get_num_classes(args):
    """根据数据集和任务类型确定类别数"""
    if args.dataset == 'SPRSound':
        task_map = {'11': 2, '12': 7, '21': 3, '22': 5,
                    11: 2, 12: 7, 21: 3, 22: 5}
        return task_map.get(args.task_type, 5)
    elif args.dataset == 'CirCor':
        return 3
    elif args.dataset == 'ICBHI':
        if args.task_type == 'binary':
            return 2
        return 4
    return 4


# ============================================================================
# 可视化函数
# ============================================================================

def set_plot_style():
    """设置统一的绘图风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 18,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """混淆矩阵可视化"""
    plt.figure(figsize=(12, 10))
    num_classes = len(class_names)
    set_plot_style()

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm_present = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, tl in enumerate(unique_labels):
        for j, pl in enumerate(unique_labels):
            if tl < num_classes and pl < num_classes:
                cm[tl, pl] = cm_present[i, j]

    row_sums = cm.sum(axis=1)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(len(row_sums)):
        if row_sums[i] != 0:
            cm_percent[i] = (cm[i] / row_sums[i]) * 100

    labels_annot = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            if row_sums[i] == 0:
                row.append("N/A")
            else:
                row.append(f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)')
        labels_annot.append(row)

    df_cm = pd.DataFrame(labels_annot, index=class_names, columns=class_names)
    cmap = sns.light_palette("#4169E1", as_cmap=True)
    annot_size = 20 if num_classes <= 4 else 16 if num_classes <= 6 else 12

    ax = sns.heatmap(cm_percent, annot=df_cm, fmt='', cmap=cmap,
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={'size': annot_size})
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.title(title, fontsize=24, pad=20)
    plt.xlabel('Predicted Label', fontsize=22)
    plt.ylabel('True Label', fontsize=22)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_prob, class_names, title, save_path):
    """ROC 曲线可视化"""
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    set_plot_style()

    ax = plt.gca()
    ax.set_facecolor('#F5F8FA')
    ax.grid(True, color='white', linestyle='-', linewidth=1.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    unique_labels = np.unique(y_true)

    if n_classes == 2:
        if 1 in unique_labels:
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[0], lw=3,
                     label=f'{class_names[1]} (AUC={roc_auc:.3f})')
    else:
        for i in range(n_classes):
            if i in unique_labels and np.sum(y_true == i) > 0:
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i], lw=3,
                         label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], '--', color='#8B0000', alpha=0.8, lw=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title(title, fontsize=22, pad=10)
    plt.legend(loc='lower right', fontsize=16, frameon=True, edgecolor='none', facecolor='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_pr_curve(y_true, y_pred_prob, class_names, title, save_path):
    """PR 曲线可视化"""
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    set_plot_style()

    ax = plt.gca()
    ax.set_facecolor('#F5F8FA')
    ax.grid(True, color='white', linestyle='-', linewidth=1.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    unique_labels = np.unique(y_true)

    if n_classes == 2:
        if 1 in unique_labels:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
            ap = average_precision_score(y_true, y_pred_prob[:, 1])
            plt.plot(recall, precision, color=colors[0], lw=3,
                     label=f'{class_names[1]} (AP={ap:.3f})')
    else:
        for i in range(n_classes):
            if i in unique_labels and np.sum(y_true == i) > 0:
                precision, recall, _ = precision_recall_curve(y_true == i, y_pred_prob[:, i])
                ap = average_precision_score(y_true == i, y_pred_prob[:, i])
                plt.plot(recall, precision, color=colors[i], lw=3,
                         label=f'{class_names[i]} (AP={ap:.3f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title(title, fontsize=22, pad=10)
    plt.legend(loc='lower left', fontsize=16, frameon=True, edgecolor='none', facecolor='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_tsne(features, labels, class_names, title, save_path, perplexity=30, n_iter=1000):
    """t-SNE 特征分布可视化"""
    print(f"  Computing t-SNE (perplexity={perplexity})...")
    set_plot_style()

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=42, learning_rate='auto', init='pca')
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i]], label=class_names[i],
                       alpha=0.6, s=30, edgecolors='none')

    plt.title(title, fontsize=22, pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=18)
    plt.ylabel('t-SNE Dimension 2', fontsize=18)
    plt.legend(loc='best', fontsize=14, frameon=True, edgecolor='none',
               facecolor='white', markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  t-SNE saved: {save_path}")


# ============================================================================
# 评估函数
# ============================================================================

def extract_features_and_predict(model, data_loader, device):
    """提取特征和预测结果"""
    model.eval()
    all_features = []
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)

            # 提取倒数第二层特征
            if hasattr(model, 'base_model'):
                feat = model.base_model.extract_features(inputs)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            else:
                feat = model.extract_features(inputs)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)

            # 前向传播获取预测
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_features.append(feat.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_features = np.vstack(all_features)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)

    return all_features, all_preds, all_labels, all_probs


def compute_metrics(all_preds, all_labels, class_names, dataset_name):
    """计算评估指标"""
    accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    metrics = {'accuracy': accuracy, 'f1': macro_f1, 'confusion_matrix': cm.tolist()}

    if dataset_name == "CirCor":
        circor = calculate_circor_metrics(all_labels, all_preds)
        metrics.update({
            'w_acc': circor['w_acc'],
            'uar': circor['uar'],
            'recall_present': circor['recall_present'],
            'recall_absent': circor['recall_absent'],
            'recall_unknown': circor['recall_unknown'],
            'overall_score': circor['w_acc'],
            'macro_sensitivity': circor['recall_present'],
            'macro_specificity': circor['recall_absent'],
        })
    else:
        sensitivities, specificities = [], []
        total_samples = np.sum(cm)
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = total_samples - (tp + fn + fp)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivities.append(sens)
            specificities.append(spec)

        macro_sens = np.mean(sensitivities)
        macro_spec = np.mean(specificities)
        avg_score = (macro_sens + macro_spec) / 2
        harm_score = 2 * macro_sens * macro_spec / (macro_sens + macro_spec + 1e-9)
        overall_score = (avg_score + harm_score) / 2

        metrics.update({
            'sensitivities': sensitivities,
            'specificities': specificities,
            'macro_sensitivity': macro_sens,
            'macro_specificity': macro_spec,
            'average_score': avg_score,
            'harmonic_score': harm_score,
            'overall_score': overall_score,
        })

        for i, cn in enumerate(class_names):
            if i < len(sensitivities):
                metrics[f'recall_{cn.lower()}'] = sensitivities[i]

    return metrics


def print_metrics(metrics, dataset_name, prefix=""):
    """打印指标"""
    print(f"{prefix}Accuracy: {metrics['accuracy']:.2f}%")
    print(f"{prefix}F1 Score: {metrics['f1']:.4f}")
    if dataset_name == "CirCor":
        print(f"{prefix}W.acc: {metrics['w_acc']:.4f}")
        print(f"{prefix}UAR: {metrics['uar']:.4f}")
        print(f"{prefix}Recall Present: {metrics['recall_present']:.4f}")
        print(f"{prefix}Recall Absent: {metrics['recall_absent']:.4f}")
        print(f"{prefix}Recall Unknown: {metrics['recall_unknown']:.4f}")
    else:
        print(f"{prefix}Sensitivity: {metrics['macro_sensitivity']:.4f}")
        print(f"{prefix}Specificity: {metrics['macro_specificity']:.4f}")
        print(f"{prefix}Average Score: {metrics.get('average_score', 0):.4f}")
        print(f"{prefix}Harmonic Score: {metrics.get('harmonic_score', 0):.4f}")
        print(f"{prefix}Overall Score: {metrics['overall_score']:.4f}")


# ============================================================================
# 结果汇总表格
# ============================================================================

def compile_results_table(all_results, test_set_names, dataset_name):
    """编译结果表格"""
    if dataset_name == "CirCor":
        headers = ["Dataset", "Model", "Acc(%)", "F1", "W.acc", "UAR",
                   "Recall_P", "Recall_A", "Recall_U", "Score"]
    else:
        headers = ["Dataset", "Model", "Acc(%)", "F1", "Sens", "Spec",
                   "Avg Score", "Harm Score", "Overall"]

    table_data = []
    for test_name in test_set_names:
        results = all_results.get(test_name, {})
        for fold in range(1, 6):
            m = results.get(f'fold_{fold}', {}).get('metrics', None)
            if m:
                if dataset_name == "CirCor":
                    table_data.append([test_name, f"Fold {fold}",
                        f"{m.get('accuracy',0):.2f}", f"{m.get('f1',0):.4f}",
                        f"{m.get('w_acc',0):.4f}", f"{m.get('uar',0):.4f}",
                        f"{m.get('recall_present',0):.4f}", f"{m.get('recall_absent',0):.4f}",
                        f"{m.get('recall_unknown',0):.4f}", f"{m.get('overall_score',0):.4f}"])
                else:
                    table_data.append([test_name, f"Fold {fold}",
                        f"{m.get('accuracy',0):.2f}", f"{m.get('f1',0):.4f}",
                        f"{m.get('macro_sensitivity',0):.4f}", f"{m.get('macro_specificity',0):.4f}",
                        f"{m.get('average_score',0):.4f}", f"{m.get('harmonic_score',0):.4f}",
                        f"{m.get('overall_score',0):.4f}"])

        # 平均
        avg = results.get('avg_metrics', None)
        if avg:
            if dataset_name == "CirCor":
                table_data.append([test_name, "Average",
                    f"{avg.get('accuracy',0):.2f}", f"{avg.get('f1',0):.4f}",
                    f"{avg.get('w_acc',0):.4f}", f"{avg.get('uar',0):.4f}",
                    f"{avg.get('recall_present',0):.4f}", f"{avg.get('recall_absent',0):.4f}",
                    f"{avg.get('recall_unknown',0):.4f}", f"{avg.get('overall_score',0):.4f}"])
            else:
                table_data.append([test_name, "Average",
                    f"{avg.get('accuracy',0):.2f}", f"{avg.get('f1',0):.4f}",
                    f"{avg.get('macro_sensitivity',0):.4f}", f"{avg.get('macro_specificity',0):.4f}",
                    f"{avg.get('average_score',0):.4f}", f"{avg.get('harmonic_score',0):.4f}",
                    f"{avg.get('overall_score',0):.4f}"])

        # 集成
        ens = results.get('ensemble_metrics', None)
        if ens:
            if dataset_name == "CirCor":
                table_data.append([test_name, "Ensemble",
                    f"{ens.get('accuracy',0):.2f}", f"{ens.get('f1',0):.4f}",
                    f"{ens.get('w_acc',0):.4f}", f"{ens.get('uar',0):.4f}",
                    f"{ens.get('recall_present',0):.4f}", f"{ens.get('recall_absent',0):.4f}",
                    f"{ens.get('recall_unknown',0):.4f}", f"{ens.get('overall_score',0):.4f}"])
            else:
                table_data.append([test_name, "Ensemble",
                    f"{ens.get('accuracy',0):.2f}", f"{ens.get('f1',0):.4f}",
                    f"{ens.get('macro_sensitivity',0):.4f}", f"{ens.get('macro_specificity',0):.4f}",
                    f"{ens.get('average_score',0):.4f}", f"{ens.get('harmonic_score',0):.4f}",
                    f"{ens.get('overall_score',0):.4f}"])

        table_data.append([''] * len(headers))

    if table_data and table_data[-1] == [''] * len(headers):
        table_data.pop()

    return headers, table_data


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='A²-PASA Test & Evaluation')

    parser.add_argument('--dataset', type=str, default='SPRSound',
                        choices=['SPRSound', 'ICBHI', 'CirCor'])
    parser.add_argument('--task_type', type=str, default='22')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录（包含 models/ 子目录）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--tsne_perplexity', type=int, default=30,
                        help='t-SNE perplexity 参数')
    parser.add_argument('--no_tsne', action='store_true',
                        help='跳过 t-SNE 可视化（加速测试）')

    args = parser.parse_args()

    # 设置随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建输出目录
    if args.dataset == 'CirCor':
        base_dir = f'test_results_extended/CirCor/heart_murmur'
    else:
        base_dir = f'test_results_extended/{args.dataset}/task_{args.task_type}'
    vis_dir = os.path.join(base_dir, timestamp)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📁 输出目录: {vis_dir}")

    # 保存测试配置
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(vis_dir, 'test_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 设备: {device}")

    # 确定测试集
    if args.dataset == 'SPRSound':
        test_set_names = ['test_2022_intra', 'test_2022_inter', 'test_2023', 'test_2022_combined']
    else:
        test_set_names = ['test']

    # 加载测试集
    test_datasets = {}
    for name in test_set_names:
        if args.dataset == 'SPRSound':
            if name != 'test_2022_combined':
                ds, nc = load_test_dataset(args, name)
                if ds:
                    test_datasets[name] = ds
        else:
            ds, nc = load_test_dataset(args, name)
            if ds:
                test_datasets[name] = ds

    # SPRSound 合并数据集
    if args.dataset == 'SPRSound' and 'test_2022_intra' in test_datasets and 'test_2022_inter' in test_datasets:
        combined, _ = create_combined_dataset([test_datasets['test_2022_intra'], test_datasets['test_2022_inter']])
        if combined:
            test_datasets['test_2022_combined'] = combined

    if not test_datasets:
        print("❌ 未找到任何测试集数据")
        return

    # 获取类别信息
    num_classes = get_num_classes(args)
    class_names = get_dataset_class_names(args.dataset, args.task_type)
    print(f"📋 类别: {class_names} ({num_classes} classes)")

    # 存储结果
    all_results = {name: {} for name in test_datasets.keys()}
    fold_probs_by_dataset = {name: [] for name in test_datasets.keys()}
    fold_features_by_dataset = {name: [] for name in test_datasets.keys()}
    fold_labels_by_dataset = {name: None for name in test_datasets.keys()}

    # ====================================================================
    # 逐 fold 测试
    # ====================================================================
    # 支持两种模型文件命名: best_model_fold{i}.pth 或 best_fold_{i}.pth
    def find_fold_model_path(exp_dir, fold):
        candidates = [
            os.path.join(exp_dir, 'models', f'best_model_fold{fold}.pth'),
            os.path.join(exp_dir, 'models', f'best_fold_{fold}.pth'),
            os.path.join(exp_dir, f'best_model_fold{fold}.pth'),
            os.path.join(exp_dir, f'best_fold_{fold}.pth'),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    for fold in range(1, 6):
        model_path = find_fold_model_path(args.exp_dir, fold)
        if model_path is None:
            print(f"⚠️ Fold {fold} 模型文件未找到，跳过")
            continue

        print(f"\n{'=' * 60}")
        print(f"📊 Testing Fold {fold}")
        print(f"{'=' * 60}")

        # 加载模型
        base_model = load_efficientnet_model(num_classes=num_classes)
        model = CustomEfficientNet(base_model)
        model, saved_epoch = load_model_weights(model, model_path, device)
        model = model.to(device)

        # 对每个测试集评估
        for test_name, test_dataset in test_datasets.items():
            print(f"\n  --- Fold {fold} on {test_name} ---")

            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn_preprocessed, num_workers=args.num_workers,
                pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor
            )

            # 提取特征和预测
            features, preds, labels, probs = extract_features_and_predict(model, test_loader, device)

            # 计算指标
            metrics = compute_metrics(preds, labels, class_names, args.dataset)
            print_metrics(metrics, args.dataset, prefix="  ")

            # 保存结果
            all_results[test_name][f'fold_{fold}'] = {
                'metrics': metrics,
                'preds': preds.tolist(),
                'labels': labels.tolist(),
            }
            fold_probs_by_dataset[test_name].append(probs)
            fold_features_by_dataset[test_name].append(features)
            fold_labels_by_dataset[test_name] = labels

            # 可视化: 混淆矩阵
            plot_confusion_matrix(
                labels, preds, class_names,
                f'Confusion Matrix - Fold {fold} - {test_name}',
                os.path.join(vis_dir, f'cm_fold{fold}_{test_name}.png')
            )

            # 可视化: ROC
            plot_roc_curve(
                labels, probs, class_names,
                f'ROC Curve - Fold {fold} - {test_name}',
                os.path.join(vis_dir, f'roc_fold{fold}_{test_name}.png')
            )

            # 可视化: PR
            plot_pr_curve(
                labels, probs, class_names,
                f'PR Curve - Fold {fold} - {test_name}',
                os.path.join(vis_dir, f'pr_fold{fold}_{test_name}.png')
            )

            # 可视化: t-SNE (每个 fold)
            if not args.no_tsne:
                plot_tsne(
                    features, labels, class_names,
                    f't-SNE - Fold {fold} - {test_name}',
                    os.path.join(vis_dir, f'tsne_fold{fold}_{test_name}.png'),
                    perplexity=args.tsne_perplexity
                )

    # ====================================================================
    # 集成投票（概率平均）
    # ====================================================================
    for test_name, fold_probs_list in fold_probs_by_dataset.items():
        if len(fold_probs_list) < 2:
            continue

        print(f"\n{'=' * 60}")
        print(f"🗳️ Ensemble Prediction for {test_name}")
        print(f"{'=' * 60}")

        labels = fold_labels_by_dataset[test_name]
        ensemble_probs = np.mean(fold_probs_list, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        # 计算集成指标
        ens_metrics = compute_metrics(ensemble_preds, labels, class_names, args.dataset)
        print_metrics(ens_metrics, args.dataset, prefix="  ")
        all_results[test_name]['ensemble_metrics'] = ens_metrics

        # 集成可视化: 混淆矩阵
        plot_confusion_matrix(
            labels, ensemble_preds, class_names,
            f'Confusion Matrix - Ensemble - {test_name}',
            os.path.join(vis_dir, f'cm_ensemble_{test_name}.png')
        )

        # 集成可视化: ROC
        plot_roc_curve(
            labels, ensemble_probs, class_names,
            f'ROC Curve - Ensemble - {test_name}',
            os.path.join(vis_dir, f'roc_ensemble_{test_name}.png')
        )

        # 集成可视化: PR
        plot_pr_curve(
            labels, ensemble_probs, class_names,
            f'PR Curve - Ensemble - {test_name}',
            os.path.join(vis_dir, f'pr_ensemble_{test_name}.png')
        )

        # 集成 t-SNE: 使用所有 fold 特征的平均
        if not args.no_tsne and len(fold_features_by_dataset[test_name]) > 0:
            avg_features = np.mean(fold_features_by_dataset[test_name], axis=0)
            plot_tsne(
                avg_features, labels, class_names,
                f't-SNE - Ensemble (Avg Features) - {test_name}',
                os.path.join(vis_dir, f'tsne_ensemble_{test_name}.png'),
                perplexity=args.tsne_perplexity
            )

    # ====================================================================
    # 计算每个测试集的平均指标
    # ====================================================================
    for test_name, results in all_results.items():
        fold_metrics_list = []
        for fold in range(1, 6):
            fk = f'fold_{fold}'
            if fk in results and 'metrics' in results[fk]:
                fold_metrics_list.append(results[fk]['metrics'])

        if fold_metrics_list:
            avg_metrics = {
                'accuracy': np.mean([m['accuracy'] for m in fold_metrics_list]),
                'f1': np.mean([m['f1'] for m in fold_metrics_list]),
                'macro_sensitivity': np.mean([m['macro_sensitivity'] for m in fold_metrics_list]),
                'macro_specificity': np.mean([m['macro_specificity'] for m in fold_metrics_list]),
                'overall_score': np.mean([m['overall_score'] for m in fold_metrics_list]),
            }

            # 标准差
            std_metrics = {
                'accuracy_std': np.std([m['accuracy'] for m in fold_metrics_list]),
                'f1_std': np.std([m['f1'] for m in fold_metrics_list]),
                'overall_score_std': np.std([m['overall_score'] for m in fold_metrics_list]),
            }
            avg_metrics.update(std_metrics)

            if args.dataset == "CirCor":
                avg_metrics.update({
                    'w_acc': np.mean([m.get('w_acc', 0) for m in fold_metrics_list]),
                    'uar': np.mean([m.get('uar', 0) for m in fold_metrics_list]),
                    'recall_present': np.mean([m.get('recall_present', 0) for m in fold_metrics_list]),
                    'recall_absent': np.mean([m.get('recall_absent', 0) for m in fold_metrics_list]),
                    'recall_unknown': np.mean([m.get('recall_unknown', 0) for m in fold_metrics_list]),
                    'w_acc_std': np.std([m.get('w_acc', 0) for m in fold_metrics_list]),
                })
            else:
                avg_metrics.update({
                    'average_score': np.mean([m.get('average_score', 0) for m in fold_metrics_list]),
                    'harmonic_score': np.mean([m.get('harmonic_score', 0) for m in fold_metrics_list]),
                })
                for cn in class_names:
                    rk = f'recall_{cn.lower()}'
                    vals = [m.get(rk, 0) for m in fold_metrics_list if rk in m]
                    if vals:
                        avg_metrics[rk] = np.mean(vals)

            results['avg_metrics'] = avg_metrics

            print(f"\n📊 Average across folds for {test_name}:")
            print(f"  Accuracy: {avg_metrics['accuracy']:.2f}% ± {std_metrics['accuracy_std']:.2f}")
            print(f"  F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
            print(f"  Overall Score: {avg_metrics['overall_score']:.4f} ± {std_metrics['overall_score_std']:.4f}")

    # ====================================================================
    # 汇总表格
    # ====================================================================
    print(f"\n\n{'=' * 80}")
    print(f"📋 SUMMARY OF ALL TEST RESULTS (A²-PASA)")
    print(f"{'=' * 80}")

    active_test_names = [n for n in test_set_names if n in test_datasets]
    headers, table_data = compile_results_table(all_results, active_test_names, args.dataset)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # 保存表格
    with open(os.path.join(vis_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"A²-PASA Test Results\n")
        f.write(f"Dataset: {args.dataset}, Task: {args.task_type}\n")
        f.write(f"Experiment: {args.exp_dir}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))

    # ====================================================================
    # 保存详细结果 JSON
    # ====================================================================
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_filename = f'{args.dataset}_task_{args.task_type}_test_results.json'
    with open(os.path.join(vis_dir, results_filename), 'w') as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)

    # 保存详细预测 CSV
    detailed_dir = os.path.join(vis_dir, 'detailed_predictions')
    os.makedirs(detailed_dir, exist_ok=True)

    for test_name, results in all_results.items():
        # 每个 fold 的预测
        for fold in range(1, 6):
            fk = f'fold_{fold}'
            if fk in results:
                df = pd.DataFrame({
                    'true_label': results[fk]['labels'],
                    'pred_label': results[fk]['preds'],
                    'correct': np.array(results[fk]['labels']) == np.array(results[fk]['preds'])
                })
                df.to_csv(os.path.join(detailed_dir, f'{test_name}_fold{fold}_predictions.csv'), index=False)

        # 集成预测
        if 'ensemble_metrics' in results and test_name in fold_probs_by_dataset:
            if len(fold_probs_by_dataset[test_name]) > 0:
                labels = fold_labels_by_dataset[test_name]
                ens_probs = np.mean(fold_probs_by_dataset[test_name], axis=0)
                ens_preds = np.argmax(ens_probs, axis=1)
                df = pd.DataFrame({
                    'true_label': labels,
                    'pred_label': ens_preds,
                    'correct': labels == ens_preds
                })
                # 添加每个类别的概率
                for i, cn in enumerate(class_names):
                    df[f'prob_{cn}'] = ens_probs[:, i]
                df.to_csv(os.path.join(detailed_dir, f'{test_name}_ensemble_predictions.csv'), index=False)

    # ====================================================================
    # 最终总结
    # ====================================================================
    print(f"\n{'=' * 60}")
    print(f"🎉 测试完成!")
    print(f"📁 所有结果保存在: {vis_dir}")
    print(f"📊 可视化文件:")
    print(f"   - 混淆矩阵: cm_*.png")
    print(f"   - ROC 曲线: roc_*.png")
    print(f"   - PR 曲线: pr_*.png")
    if not args.no_tsne:
        print(f"   - t-SNE 分布: tsne_*.png")
    print(f"📋 汇总表格: results_summary.txt")
    print(f"📄 详细结果: {results_filename}")
    print(f"📝 逐样本预测: detailed_predictions/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

