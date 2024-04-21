# -*- coding: utf-8 -*-
# 这段脚本实现了在单GPU上对推断出的图像和文本特征进行kNN搜索，并输出用于评估的文本到图像的预测文件。

import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch

# 定义参数解析函数
def parse_args():
    parser = argparse.ArgumentParser()
    # 添加命令行参数选项，分别指定图像特征路径、文本特征路径、top-k值、批处理大小以及输出文件路径
    parser.add_argument('--image-feats', type=str, required=True, help="指定图像特征的路径")
    parser.add_argument('--text-feats', type=str, required=True, help="指定文本特征的路径")
    parser.add_argument('--top-k', type=int, default=10, help="指定top-k预测的数量，默认为10")
    parser.add_argument('--eval-batch-size', type=int, default=32768, help="指定计算内积时图像侧的批处理大小，默认为32768")
    parser.add_argument('--output', type=str, required=True, help="指定输出预测结果的JSONL格式文件路径")
    parser.add_argument('--mode', type=str, choices=['text2image', 'image2text'], required=True, help="指定检索模式，可选'text2image'（文本检索图片）或'image2text'（图片检索文本）")


    # 解析并返回命令行参数
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 打印参数信息
    print("参数设置如下：")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    # 检查是否有可用的GPU设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载特征数据，这里根据模式不同调整加载的数据
    if args.mode == 'text2image':
        print("开始加载图像特征...")
        # 初始化图像ID列表和图像特征列表
        image_ids = []
        image_feats = []
        # 读取并解析图像特征文件
        with open(args.image_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                image_ids.append(obj['image_id'])  # 存储图像ID
                image_feats.append(obj['feature'])  # 存储图像特征
        # 将图像特征列表转换为NumPy数组
        image_feats_array = np.array(image_feats, dtype=np.float32)
        print("图像特征加载完成。")

        # 开始计算文本的top-k预测
        print(f"开始计算文本的top-{args.top_k}预测...")
        # 创建输出文件句柄并打开文本特征文件
        with open(args.output, "w") as fout, open(args.text_feats, "r") as fin:
            # 遍历文本特征文件
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                text_id = obj['text_id']  # 获取文本ID
                text_feat = obj['feature']  # 获取文本特征
                # 初始化存储图像ID和相似度得分的元组列表
                score_tuples = []
                # 将文本特征转化为CUDA张量（如果有可用GPU）
                text_feat_tensor = torch.tensor([text_feat], dtype=torch.float).to(device)  # 形状为 [1, 特征维度]
                # 按照批处理大小遍历图像特征
                idx = 0
                while idx < len(image_ids):
                    # 截取图像特征批次
                    img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + args.eval_batch_size, len(image_ids))]).to(device)  # 形状为 [批处理大小, 特征维度]
                    # 计算文本特征与图像特征批次的内积
                    batch_scores = text_feat_tensor @ img_feats_tensor.t()  # 形状为 [1, 批处理大小]
                    # 将图像ID和得分存入score_tuples
                    for image_id, score in zip(image_ids[idx : min(idx + args.eval_batch_size, len(image_ids))], batch_scores.squeeze(0).tolist()):
                        score_tuples.append((image_id, score))
                    idx += args.eval_batch_size
                # 对score_tuples按得分降序排序，并取前k个元素作为top-k预测
                top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:args.top_k]
                # 将top-k预测结果写入输出文件，格式为JSON字符串
                fout.write("{}\n".format(json.dumps({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})))
    elif args.mode == 'image2text':
        # 加载文本特征
        print("开始加载文本特征...")
        text_ids = []
        text_feats = []
        with open(args.text_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                text_ids.append(obj['text_id'])
                text_feats.append(obj['feature'])
        text_feats_array = np.array(text_feats, dtype=np.float32)
        print("文本特征加载完成。")

        # 创建输出文件句柄并打开图像特征文件
        print(f"开始计算图像到文本的top-{args.top_k}预测...")
        with open(args.output, "w") as fout, open(args.image_feats, "r") as fin:
            # 遍历图像特征文件
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                image_id = obj['image_id']  # 获取图像ID
                image_feat = obj['feature']  # 获取图像特征
                # 初始化存储文本ID和相似度得分的元组列表
                score_tuples = []
                # 将图像特征转化为CUDA张量（如果有可用GPU）
                image_feat_tensor = torch.tensor([image_feat], dtype=torch.float).to(device)  # 形状为 [1, 特征维度]
                # 按照批处理大小遍历文本特征
                idx = 0
                while idx < len(text_ids):
                    # 截取文本特征批次
                    txt_feats_tensor = torch.from_numpy(text_feats_array[idx : min(idx + args.eval_batch_size, len(text_ids))]).to(device)  # 形状为 [批处理大小, 特征维度]
                    # 计算图像特征与文本特征批次的内积
                    batch_scores = image_feat_tensor @ txt_feats_tensor.t()  # 形状为 [1, 批处理大小]
                    # 将文本ID和得分存入score_tuples
                    for text_id, score in zip(text_ids[idx : min(idx + args.eval_batch_size, len(text_ids))], batch_scores.squeeze(0).tolist()):
                        score_tuples.append((text_id, score))
                    idx += args.eval_batch_size
                # 对score_tuples按得分降序排序，并取前k个元素作为top-k预测
                top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:args.top_k]
                # 将top-k预测结果写入输出文件，格式为JSON字符串
                fout.write("{}\n".format(json.dumps({"image_id": image_id, "text_ids": [entry[0] for entry in top_k_predictions]})))
    else:
        raise ValueError(f"未知的检索模式'{args.mode}'")
    

    

    
    # 根据检索模式输出相应的完成信息
    if args.mode == 'text2image':
        print(f"已完成文本到图像的top-{args.top_k}预测，结果已保存至{args.output}")
    elif args.mode == 'image2text':
        print(f"已完成图像到文本的top-{args.top_k}预测，结果已保存至{args.output}")
    print("全部任务完成！")