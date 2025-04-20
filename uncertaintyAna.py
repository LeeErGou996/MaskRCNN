import os
import sys

sys.path.append('/kaggle/input/fudanpenn/')
import matplotlib.pyplot as plt
from mymaskrcnn import custom_maskrcnn_resnet50_fpn
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image


from tqdm import tqdm
import random
from torchvision.ops import box_iou




def compute_average_iou(boxes1, boxes2):
    """
    计算两个边界框集合的平均 IoU。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.stack(boxes1)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.stack(boxes2)

    boxes1 = boxes1.to(device)
    boxes2 = boxes2.to(device)

    iou_matrix = box_iou(boxes1, boxes2)
    avg_iou = iou_matrix.mean().item()
    return avg_iou


def compute_average_dice(masks1, masks2, threshold=0.5):
    """
    批量计算两个掩码集合的平均 Dice 系数。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not isinstance(masks1, torch.Tensor):
        masks1 = torch.tensor(masks1)
    if not isinstance(masks2, torch.Tensor):
        masks2 = torch.tensor(masks2)

    masks1 = masks1.to(device)
    masks2 = masks2.to(device)

    if len(masks1.shape) == 3:
        masks1 = masks1.unsqueeze(1)
    if len(masks2.shape) == 3:
        masks2 = masks2.unsqueeze(1)

    masks1 = masks1[:, None, :, :, :]
    masks2 = masks2[None, :, :, :, :]

    intersection = (masks1 * masks2).sum(dim=(-2, -1))
    union = masks1.sum(dim=(-2, -1)) + masks2.sum(dim=(-2, -1))

    dice_scores = 2 * intersection / (union + 1e-6)
    avg_dice = dice_scores.mean().item()
    return avg_dice



def calculate_uncertainty_with_rare_classes(boxes_list, masks_list, scores_list, classes_list, rare_classes, threshold=0.5, ct=0.2):
    """
    计算分类、检测、分割和稀有类别的不确定性，优化为 GPU 计算。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classification_uncertainty = 1
    if classes_list:
        top_classes = [cls[0].item() for cls in classes_list if len(cls) > 0]
        if top_classes:
            mode_class = max(set(top_classes), key=top_classes.count)
            classification_matches = sum([1 if cls == mode_class else 0 for cls in top_classes])
            classification_uncertainty = 1 - (classification_matches / len(top_classes))

    detection_ious = []
    for i in range(1, len(boxes_list)):
        iou = compute_average_iou(boxes_list[0].to(device), boxes_list[i].to(device))
        detection_ious.append(iou)
    detection_uncertainty = 1 - (np.mean(detection_ious) if detection_ious else 0)

    segmentation_dices = []
    for i in range(1, len(masks_list)):
        dice = compute_average_dice(masks_list[0].to(device), masks_list[i].to(device), threshold)
        segmentation_dices.append(dice)
    segmentation_uncertainty = 1 - (np.mean(segmentation_dices) if segmentation_dices else 0)

    rare_class_uncertainty = 1
    rare_class_scores = {cls: [] for cls in rare_classes}

    for i in range(len(classes_list)):
        for cls, score in zip(classes_list[i], scores_list[i]):
            if cls.item() in rare_classes:
                rare_class_scores[cls.item()].append(score.item())

    for cls, scores in rare_class_scores.items():
        if scores:
            average_score = sum(scores) / len(scores)
            if average_score >= ct:
                rare_class_uncertainty = 0
                break

    return classification_uncertainty, detection_uncertainty, segmentation_uncertainty, rare_class_uncertainty


def multiple_inference_with_details(model, image, num_inferences=5):
    """
    使用 GPU 对输入图像进行多次推断。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    images, boxes_list, masks_list, scores_list, classes_list = [], [], [], [], []

    for _ in range(num_inferences):
        with torch.no_grad():
            predictions = model(image)

        pred_boxes = predictions[0]["boxes"].to("cpu")
        pred_masks = predictions[0]["masks"].to("cpu")
        pred_scores = predictions[0]["scores"].to("cpu")
        pred_classes = predictions[0]["labels"].to("cpu")

        boxes_list.append(pred_boxes)
        masks_list.append(pred_masks)
        scores_list.append(pred_scores)
        classes_list.append(pred_classes)

        images.append(image[0].permute(1, 2, 0).cpu().numpy())

    return images, boxes_list, masks_list, scores_list, classes_list





def visualize_top_uncertain_images(image_paths, model, rare_classes, threshold=0.3, num_inferences=10, ct=0.2, top_n=10, N_objects=None, top10uncertain=None):
    """
    可视化前 top_n 个最不确定的图像，并打印文件名和预测结果类别，同时返回最高不确定性的图像名称。

    Args:
        image_paths: 图像路径列表。
        model: 目标检测模型。
        rare_classes: 稀有类别集合。
        threshold: mask 的二值化阈值。
        num_inferences: 每张图像的推断次数。
        ct: 稀有类别的概率阈值。
        top_n: 可视化的最不确定图像数量。
        N_objects: 数据集中要处理的图像数量（限制图像总数）。
        top10uncertain: 空列表，用于存储不确定性最高的图像名称。
    """
    CLASSES = {
        'EOS': 1,
        'LYT': 2,
        'MON': 3,
        'MYO': 4,
        'NGB': 5,
        'NGS': 6,
        'EBO': 7,
        'BAS': 8
    }
    class_labels = {v: k for k, v in CLASSES.items()}

    # 限制处理的图像数量
    if N_objects is not None:
        image_paths = image_paths[:N_objects]

    uncertainties = []

    for img_path in tqdm(image_paths, desc="Analyzing Images"):
        image = Image.open(img_path).convert("RGB")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU

        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        # 多次推断，仅保留最高得分的结果
        images, boxes_list, masks_list, scores_list, classes_list = multiple_inference_with_details(
            model, image_tensor, num_inferences=num_inferences
        )

        # 计算各个不确定性指标
        classification_uncertainty, detection_uncertainty, segmentation_uncertainty, rare_class_uncertainty = calculate_uncertainty_with_rare_classes(
            boxes_list=boxes_list,
            masks_list=masks_list,
            scores_list=scores_list,
            classes_list=classes_list,
            rare_classes=rare_classes,
            threshold=threshold,
            ct=ct,
        )

        # 总不确定性 = 4 项不确定性指标之和
        total_uncertainty = (
            classification_uncertainty 
            + detection_uncertainty
            + segmentation_uncertainty
            + rare_class_uncertainty
        )

        uncertainties.append((img_path, total_uncertainty, classification_uncertainty, detection_uncertainty, segmentation_uncertainty, rare_class_uncertainty))

    # 按不确定性排序，取前 top_n
    uncertainties = sorted(uncertainties, key=lambda x: x[1], reverse=False)[:top_n]

    # 提取不确定性最高图像的名称并存入 top10uncertain
    if top10uncertain is not None:
        top10uncertain.extend([os.path.basename(u[0]) for u in uncertainties])

    # 打印文件名和预测结果类别
    from collections import Counter

    print("Top Uncertain Images:")
    for i, (img_path, total_uncertainty, cls_unc, det_unc, seg_unc, rare_cls_unc) in enumerate(uncertainties):
        print(f"{i + 1}. File: {img_path} | Total Uncertainty: {total_uncertainty:.2f}")
    
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
        # 多次推断，获取所有预测结果
        images, boxes_list, masks_list, scores_list, classes_list = multiple_inference_with_details(
            model, image_tensor, num_inferences=num_inferences
        )
    
        # 提取所有类别和置信度
        all_classes = [cls for sublist in classes_list for cls in sublist.tolist()]
        all_scores = [score for sublist in scores_list for score in sublist.tolist()]
    
        # 统计类别出现次数
        class_counts = Counter(all_classes)
    
        # 计算每个类别的平均置信度
        class_scores = {}
        for cls, score in zip(all_classes, all_scores):
            if cls not in class_scores:
                class_scores[cls] = []
            class_scores[cls].append(score)
        average_scores = {cls: sum(scores) / len(scores) for cls, scores in class_scores.items()}
    
        # 排序类别
        sorted_classes = sorted(
            class_counts.keys(),
            key=lambda cls: (-class_counts[cls], -average_scores[cls])
        )
    
        # 映射类别索引到名称
        sorted_class_names = [class_labels.get(cls, "Unknown") for cls in sorted_classes]
        print(f"   Predicted Classes (Sorted by Count and Score): {sorted_class_names}")
    
        # 打印每个类别及其置信度
        print("   Class Scores:")
        for cls in sorted_classes:
            print(f"      {class_labels.get(cls, 'Unknown')}: {average_scores[cls]:.2f}")
    
        # 如果 rare_class_uncertainty 小于 1，打印检测到的稀有类别
        if rare_cls_unc < 1:
            detected_rare_classes = [
                class_labels.get(cls, "Unknown")
                for cls in sorted_classes
                if cls in rare_classes
            ]
            print(f"   Detected Rare Classes: {detected_rare_classes}")
    
        # 可视化：绘制 box 和 mask
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
    
        if len(boxes_list) > 0 and len(classes_list) > 0:
            # 遍历所有预测结果
            for j in range(len(boxes_list[0])):
                box = boxes_list[0][j].cpu().numpy()
                cls = classes_list[0][j].item()
                mask = masks_list[0][j].squeeze().cpu().numpy()
                score = scores_list[0][j].item()
    
                # 绘制预测边界框
                x1, y1, x2, y2 = box
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
                )
    
                # 绘制 mask
                mask = (mask > threshold).astype(np.uint8)
                plt.imshow(mask, alpha=0.5, cmap="Blues")
    
                # 添加预测类别和得分
                plt.text(
                    x1,
                    y1 - 10,
                    f"{class_labels.get(cls, 'Unknown')}: {score:.2f}",
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.5),
                )
    
        plt.title(
            f"Image {i + 1} | Total Uncertainty: {total_uncertainty:.2f}\nCls: {cls_unc:.2f}, Det: {det_unc:.2f}, Seg: {seg_unc:.2f}, RareCls: {rare_cls_unc:.2f}"
        )
        plt.axis("off")
        plt.show()

    return top10uncertain



def visualize_top_uncertain_images_gpu(
    image_paths, model, rare_classes, threshold=0.3, num_inferences=10, ct=0.2, top_n=10, N_objects=None, top10uncertain=None
):
    """
    使用 GPU 加速的方式可视化前 top_n 个最不确定的图像，并打印文件名和预测结果类别，同时返回最高不确定性的图像名称。

    Args:
        image_paths: 图像路径列表。
        model: 目标检测模型。
        rare_classes: 稀有类别集合。
        threshold: mask 的二值化阈值。
        num_inferences: 每张图像的推断次数。
        ct: 稀有类别的概率阈值。
        top_n: 可视化的最不确定图像数量。
        N_objects: 数据集中要处理的图像数量（限制图像总数）。
        top10uncertain: 空列表，用于存储不确定性最高的图像名称。
    """
    CLASSES = {
        "EOS": 1,
        "LYT": 2,
        "MON": 3,
        "MYO": 4,
        "NGB": 5,
        "NGS": 6,
        "EBO": 7,
        "BAS": 8,
    }
    class_labels = {v: k for k, v in CLASSES.items()}

    # 设置 GPU 或 CPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 限制处理的图像数量
    if N_objects is not None:
        image_paths = image_paths[:N_objects]

    uncertainties = []

    for img_path in tqdm(image_paths, desc="Analyzing Images"):
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # 将图像移动到 GPU

        # 多次推断
        images, boxes_list, masks_list, scores_list, classes_list = multiple_inference_with_details(
            model, image_tensor, num_inferences=num_inferences
        )

        # 计算各个不确定性指标
        classification_uncertainty, detection_uncertainty, segmentation_uncertainty, rare_class_uncertainty = calculate_uncertainty_with_rare_classes(
            boxes_list=boxes_list,
            masks_list=masks_list,
            scores_list=scores_list,
            classes_list=classes_list,
            rare_classes=rare_classes,
            threshold=threshold,
            ct=ct,
        )

        # 总不确定性 = 4 项不确定性指标之和
        total_uncertainty = (
            classification_uncertainty
            + detection_uncertainty
            + segmentation_uncertainty
            + rare_class_uncertainty
        )

        uncertainties.append(
            (
                img_path,
                total_uncertainty,
                classification_uncertainty,
                detection_uncertainty,
                segmentation_uncertainty,
                rare_class_uncertainty,
            )
        )

    # 按不确定性排序，取前 top_n
    uncertainties = sorted(uncertainties, key=lambda x: x[1], reverse=False)[:top_n]

    # 提取不确定性最高图像的名称并存入 top10uncertain
    if top10uncertain is not None:
        top10uncertain.extend([os.path.basename(u[0]) for u in uncertainties])

    # 打印文件名和预测结果类别
    print("Top Uncertain Images:")
    for i, (img_path, total_uncertainty, cls_unc, det_unc, seg_unc, rare_cls_unc) in enumerate(uncertainties):
        print(f"{i + 1}. File: {img_path} | Total Uncertainty: {total_uncertainty:.2f}")
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        # 多次推断，获取所有预测结果
        images, boxes_list, masks_list, scores_list, classes_list = multiple_inference_with_details(
            model, image_tensor, num_inferences=num_inferences
        )

        # 提取所有类别和置信度
        all_classes = [cls for sublist in classes_list for cls in sublist.tolist()]
        all_scores = [score for sublist in scores_list for score in sublist.tolist()]

        # 映射类别索引到名称
        from collections import Counter
        class_counts = Counter(all_classes)
        class_scores = {cls: [] for cls in set(all_classes)}

        for cls, score in zip(all_classes, all_scores):
            class_scores[cls].append(score)

        average_scores = {cls: sum(scores) / len(scores) for cls, scores in class_scores.items()}
        sorted_classes = sorted(class_counts.keys(), key=lambda cls: (-class_counts[cls], -average_scores[cls]))
        sorted_class_names = [class_labels.get(cls, "Unknown") for cls in sorted_classes]

        # 打印分类和得分
        print(f"   Predicted Classes (Sorted by Count and Score): {sorted_class_names}")
        print("   Class Scores:")
        for cls in sorted_classes:
            print(f"      {class_labels.get(cls, 'Unknown')}: {average_scores[cls]:.2f}")

        # 如果稀有类别不确定性小于 1，打印检测到的稀有类别
        if rare_cls_unc < 1:
            detected_rare_classes = [class_labels.get(cls, "Unknown") for cls in sorted_classes if cls in rare_classes]
            print(f"   Detected Rare Classes: {detected_rare_classes}")

        # 可视化：绘制边界框和 mask
        plt.figure(figsize=(10, 6))
        plt.imshow(image)

        if len(boxes_list) > 0 and len(classes_list) > 0:
            for j in range(len(boxes_list[0])):
                box = boxes_list[0][j].cpu().numpy()
                cls = classes_list[0][j].item()
                mask = masks_list[0][j].squeeze().cpu().numpy()
                score = scores_list[0][j].item()

                x1, y1, x2, y2 = box
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
                )

                mask = (mask > threshold).astype(np.uint8)
                plt.imshow(mask, alpha=0.5, cmap="Blues")

                plt.text(
                    x1,
                    y1 - 10,
                    f"{class_labels.get(cls, 'Unknown')}: {score:.2f}",
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.5),
                )

        plt.title(
            f"Image {i + 1} | Total Uncertainty: {total_uncertainty:.2f}\nCls: {cls_unc:.2f}, Det: {det_unc:.2f}, Seg: {seg_unc:.2f}, RareCls: {rare_cls_unc:.2f}"
        )
        plt.axis("off")
        plt.show()

    return top10uncertain

# # 数据集路径
# data_root = '/kaggle/input/fudanpenn/Penn/Penn/test'
# rare_classes = {1, 5, 7, 8}  # 稀有类别
# image_files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(('.jpg', '.png'))]

# # 可视化前 10 个最不确定的图像
# visualize_top_uncertain_images(
#     image_paths=image_files,
#     model=model1,
#     rare_classes=rare_classes,
#     threshold=0.3,
#     num_inferences=10,
#     ct=0.2,
#     top_n=10,
#     N_objects=400,  # 数据集中选取的图像数量
# )