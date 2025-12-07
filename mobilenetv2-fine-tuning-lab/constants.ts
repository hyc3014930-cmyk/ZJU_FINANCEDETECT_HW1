
export const GARBAGE_CATEGORIES = {
  'Hazardous (有害垃圾)': ['Battery (电池)', 'Tablet capsules (药片胶囊)', 'Fluorescent lamp (荧光灯)', 'Paint bucket (油漆桶)'],
  'Recyclable (可回收物)': ['Newspaper (报纸)', 'Glassware (玻璃制品)', 'Basketball (篮球)', 'Plastic Bottle (塑料瓶)', 'Cardboard (硬纸板)', 'Glass Bottle (玻璃瓶)', 'Metalware (金属制品)', 'Hats (帽子)', 'Cans (易拉罐)', 'Paper (纸张)'],
  'Wet (湿垃圾)': ['Vegetable Leaf (菜叶)', 'Orange Peel (橙皮)', 'Eggshell (蛋壳)', 'Banana Peel (香蕉皮)'],
  'Dry (干垃圾)': ['Seashell (贝壳)', 'Lighter (打火机)', 'Old Mirror (旧镜子)', 'Broom (扫把)', 'Ceramic Bowl (陶瓷碗)', 'Toothbrush (牙刷)', 'Disposable Chopsticks (一次性筷子)', 'Dirty Cloth (脏污衣服)']
};

export const COURSE_CONTENT = `
# MobileNetV2 垃圾分类实验 (MindSpore)

## 1. 实验简介
MindSpore 是最佳匹配 Ascend（昇腾）芯片的开源 AI 计算框架，同时也支持 CPU、GPU 平台。
在深度学习计算中，从头开始训练一个实用的模型通常非常耗时（需大量算力 + 百万级数据如 ImageNet）。
因此，本实验采用 **Fine-Tuning（微调）** 策略：基于 MobileNetV2 预训练模型，在垃圾分类数据集上进行迁移学习。

## 2. 数据集信息
本次实验涵盖 4 大类共 26 小类生活垃圾：
- **有害垃圾**: 电池, 药片胶囊, 油漆桶等
- **可回收物**: 易拉罐, 塑料瓶, 报纸等
- **湿垃圾**: 香蕉皮, 菜叶, 蛋壳等
- **干垃圾**: 贝壳, 陶瓷碗, 脏污衣服等

## 3. 关键模块解析
### 3.1 预训练与微调
- **Backbone (骨干)**: 复用 MobileNetV2 在 ImageNet 上学到的通用特征（边缘、纹理）。
- **Head (分类头)**: 替换原有的 1000 类全连接层，重置为 26 类（num_classes=26），仅训练此部分。

### 3.2 学习率策略 (build_lr)
- **Warmup**: 预热阶段，学习率线性增加，防止破坏预训练权重。
- **Decay**: 衰减阶段（Cosine/Square），学习率逐渐降低，帮助模型收敛到最优解。

### 3.3 推理 (Inference)
- **预处理**: 必须与训练保持一致 (Resize 224x224, Normalize, HWC->CHW)。
- **模式**: 必须调用 \`network.set_train(False)\`。

## 常见问题
- **MindSpore 环境**: 即使无 GPU/Ascend，MindSpore 也会自动切换到 CPU 运行（速度较慢）。
- **维度错误**: MindSpore 数据集默认为 CHW (通道优先)，而 Matplotlib 需要 HWC。
`;

export const MOCK_HEAD_ONLY_DATA = Array.from({ length: 20 }, (_, i) => ({
  epoch: i + 1,
  accuracy: 40 + (85 - 40) * (1 - Math.exp(-0.3 * i)) + (Math.random() * 2 - 1),
  phase: 'Head Training'
}));

export const MOCK_TWO_STEP_DATA = [
  ...Array.from({ length: 10 }, (_, i) => ({
    epoch: i + 1,
    accuracy: 40 + (82 - 40) * (1 - Math.exp(-0.3 * i)) + (Math.random() * 2 - 1),
    phase: 'Stage 1: Frozen Backbone'
  })),
  ...Array.from({ length: 15 }, (_, i) => ({
    epoch: i + 11,
    accuracy: 82 + (96 - 82) * (1 - Math.exp(-0.2 * i)) + (Math.random() * 1.5 - 0.75),
    phase: 'Stage 2: Fine-tuning'
  }))
];
