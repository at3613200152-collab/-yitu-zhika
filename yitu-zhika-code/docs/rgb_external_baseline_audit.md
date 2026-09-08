# 外部 RGB 基线与题设原论文审计

核查日期：2026-09-04。状态：只读核查作者/维护者发布的仓库、配置、模型代码与论文；没有下载权重、复制第三方实现或运行外部模型。本文不是实验成绩单。

## 结论与使用边界

| 候选 | 核查结论 | 本项目应如何使用 |
| --- | --- | --- |
| Oatsty/nutrition5k 默认配置 | RGB + 真实深度 + OpenSeeD 食物区域 mask，双 Swin/FPN/交叉注意力；不是普通 RGB-only 网络 | 可作为 RGB-D 外部方法研究对象。删深度分支是改编，不能声称原仓库严格复现 |
| jc-builds/CalorieCLIP | 已找到维护者模型仓库与推理实现，CLIP ViT-B/32 + 热量回归头；模型卡声明 MIT | 是真正 RGB-only 候选，但公开权重训练样本可能包含本项目测试餐盘；不能直接作为无泄漏主表基线 |
| 本项目内部 RGB-only / RGB+预测 NIR 对照 | 同骨干、同餐盘划分的消融实验 | 能检验预测 NIR 的增量作用，但不自动满足“复现指定外部开源方法”的要求 |

来源：[Oatsty 配置](https://github.com/Oatsty/nutrition5k/blob/main/config/models/base.yaml)、[Oatsty 模型](https://github.com/Oatsty/nutrition5k/blob/main/src/model.py)、[CalorieCLIP 模型卡](https://huggingface.co/jc-builds/CalorieCLIP)。

## 1. Oatsty/nutrition5k

### 可获取性与许可

仓库公开可访问。根目录展示 README、配置、源码及 OpenSeeD 目录；本次未在根目录或 README 找到主体代码的明确许可证，也未找到可直接下载的营养回归成品权重。README 中的 `models/base.pt` 是训练输出位置，不是已发布权重的证据。第三方 OpenSeeD 的许可不能推定为整个仓库的许可；在得到明确授权前，不把该仓库源码复制进本项目或作为自己的源码发布。[仓库主页与 README](https://github.com/Oatsty/nutrition5k)

版本状态：本次网页可核查 `main`，但提交 API/历史页未成功返回完整 commit SHA；故目前尚未完成不可变版本锁定，不应称严格可复现。真正开始复现前须读取并保存 HEAD SHA，所有配置、代码事实再对该 SHA 核对，不以持续变化的 `main` 为实验版本。

### 实际输入、模型与目标

默认 `cross-swin-cls` 路线采用预训练 `microsoft/swin-tiny-patch4-window7-224`。`FPNCrossSwinCLS` 继承的主干显式有 RGB 与 depth 两个 Swin 编码器，深度扩展成三通道；区域 mask 用于特征处理，随后是 FPN 和带 token 的交叉注意力，再输出 `cal / mass / fat / carb / protein`。没有由此证据支持的菜品分类结果；名称里的 CLS token 不等于训练了食物类别分类任务。[模型源码](https://github.com/Oatsty/nutrition5k/blob/main/src/model.py)

数据加载器要求每餐盘目录下有 `rgb.png`、`depth_raw.png`，默认 mask 路线还需 `mask.pt`。原始深度是测距数据，不是 NIR；已有可视化深度 JPG 也不能不经验证直接替代 `depth_raw.png`。其元数据读取顺序为热量、质量、脂肪、碳水、蛋白质，和 Nutrition5k 官方格式一致。[加载器](https://github.com/Oatsty/nutrition5k/blob/main/src/dataset/nutrition5k_dataset.py)

### 预处理、训练与划分

- RGB：ToTensor 后 ImageNet 均值/标准差归一化；深度：PILToTensor 后 `(depth - 3091) / 1307`。
- 回归标签使用固定标准化常数；按 cal/mass/fat/carb/protein 顺序，均值为 255/218/12.7/19.3/18.1，标准差为 221/163/13.4/22.3/20.2。输出必须还原单位后计算误差。原始热量是 kcal，质量及宏量营养素是 g。
- 成对旋转范围为 -180 到 180 度，并以 0.5 概率水平翻转。加载器返回 train/test 两个集合，过滤缺 RGB/原始深度的餐盘，且排除代码指定的 11 个餐盘 ID。

来源：[基础数据类](https://github.com/Oatsty/nutrition5k/blob/main/src/dataset/base_dataset.py)、[数据加载实现](https://github.com/Oatsty/nutrition5k/blob/main/src/dataset/nutrition5k_dataset.py)、[Nutrition5k 官方元数据说明](https://github.com/google-research-datasets/Nutrition5k#dataset-metadata)。

默认配置指向官方 `depth_train_ids.txt` 和 `depth_test_ids.txt`，metadata 路径仅指 cafe1；batch 8，150 epochs，seed 12345，学习率 1e-4，weight decay 1e-4，trainer `mask_lp`，loss `multi`，LP 0.8，warmup 相关字段也在配置中。评估尺寸字段为 480×640；预训练模型名称中的 224 不能当成最终训练输入尺寸。[base.yaml](https://github.com/Oatsty/nutrition5k/blob/main/config/models/base.yaml)

尚待补核：`multi` 损失的实际公式与各项权重、动态噪声移除的执行条件、训练时 resize/crop、优化器、逐轮验证和最终测试协议。本次未成功取得对应 trainer/evaluate/loss 文件，不能用配置字段名称猜测执行语义。基础数据类的 `rotate_flip` 默认为 True，构建测试集合时没有在该层显式关闭；需要继续核查评估入口是否关闭，不能仅凭这一层就断言最终评估有随机增强缺陷。未核实上述信息前不可直接汇报仓库方法成绩。

### 本机复现路径与资源门槛

作者 README 以 Python 3.10、torch 1.13.1、torchvision 0.14.1、OpenSeeD 和 detectron2 为环境示例，并要求先生成 mask，再运行 `src/main.py --cfg config/models/base.yaml`；评估入口是 `src/evaluate.py --cfg <配置>`。[作者步骤](https://github.com/Oatsty/nutrition5k)

本机是 RTX 5060 8 GB、现有 yitu 环境为较新的 PyTorch/CUDA，不应为了旧示例直接降级正在使用的环境。推断：双骨干、480×640、mask 模块比本项目的单 RGB 小模型更耗资源；作者未声明最低显存，本次未跑显存测试，所以不能保证原 batch 8 在 8 GB 可运行。若需迁移依赖、减小 batch/分辨率或改变输入，必须记录为本机适配复现，而非位级/超参数完全一致的复现。

准备顺序：明确主体许可与固定 commit → 核对完整训练/评估实现 → 准备原始 RGB/深度和 mask → 独立环境两张图的前向/反向测试 → 小规模显存测试 → 明确独立验证集后正式运行。当前没有可安全直接粘贴即获得原方法有效成绩的一键命令。

## 2. CalorieCLIP

### 一手定位、版本与权重

维护者发布位置为 `jc-builds/CalorieCLIP`，模型卡作者写为 Haplo LLC，公开说明为软件模型而非本次已定位的同行评审论文。模型卡声明 MIT；文件列表没有看到独立 LICENSE 正文。页面最新可见短提交为 `26b9bd6`，权重/配置/推理代码更新提交为 `f8a9cee`；完整 SHA 尚待 API 核验。权重 `calorie_clip.pt` 页面标约 607 MB，并标出 pickle 导入，不能将“可下载”当成“已验证安全或已复现”。本次未下载、未加载该文件。[模型卡](https://huggingface.co/jc-builds/CalorieCLIP)、[文件](https://huggingface.co/jc-builds/CalorieCLIP/tree/main)、[提交历史](https://huggingface.co/jc-builds/CalorieCLIP/commits/main)

### 结构与实际推理行为

OpenCLIP 的 OpenAI 预训练 ViT-B/32 提取 512 维图像特征，回归头依次为 512→512→256→64→1，中间含 BN/ReLU/Dropout，输出整张输入图对应的热量标量，没有重量/食物分类头。输入 224×224 RGB；特征不做 L2 归一化。图像由 OpenCLIP 官方 transform 预处理，配置写有 CLIP mean/std。[推理实现](https://huggingface.co/jc-builds/CalorieCLIP/blob/main/calorie_clip.py)、[配置](https://huggingface.co/jc-builds/CalorieCLIP/blob/main/config.json)

重要工程风险：公开 wrapper 的 `from_pretrained` 实际接收本地目录，不是自动解析 HF 仓库 ID；缺少权重时不会主动失败，可能保留随机回归头；CLIP 的 load 使用 `strict=False`，而 `encode_image` 内使用 `no_grad`。因此不可直接把这个推理 wrapper 当成训练实现，也不可重复本项目曾出现的静默随机权重问题。若以后接入，必须要求权重存在、校验哈希、审计 checkpoint 内容与所有加载键；不能把不完整加载显示成有效预测。[wrapper 源码](https://huggingface.co/jc-builds/CalorieCLIP/blob/main/calorie_clip.py)

### 训练声明与评估边界

配置声明训练 30 epochs、batch 16、AdamW、Huber loss，最后两个视觉 transformer blocks 与回归头可训练；CLIP/head 学习率分别为 1e-5/1e-3。模型卡说训练集为 Nutrition5k 与带估计热量的 Food-101 子集，合计 13,004 样本，11,053 train、1,951 validation。[配置](https://huggingface.co/jc-builds/CalorieCLIP/blob/main/config.json)、[训练数据声明](https://huggingface.co/jc-builds/CalorieCLIP#-training-data)

当前未见训练脚本、逐餐盘 split manifest、随机种子或 Food-101 热量标注生成细节。模型卡将 1,951 样本称 validation，config 又用 `test_samples`；推理类注释的旧 MAE 54.3 与当前卡的 51.4 也不一致。51.4 是维护者报告值，不是本项目的复测值；页面的速度和模型优劣声明没有在本机或相同测试集独立验证。它不能据此充当 Nutrition5k 官方测试集的无污染参考成绩。

### 建议的本机路径

优先走“固定版本的公开结构 + 通用预训练 CLIP + 本项目训练集重新拟合”：用已经校验物理单位的 Nutrition5k 标签；官方测试餐盘完全隔离；在官方训练餐盘内部按分组划分验证集；单独实现可训练模块，不套用有 `no_grad` 的推理封装；保留 CLIP 预处理和回归结构，并保存 checkpoint/预处理/split 哈希。先 batch 2 冒烟，再测 batch 8/16 显存。其 224 RGB 与局部微调对 8 GB GPU 比 Oatsty 全流程更可行，这是待显存实测的工程判断，不是已完成训练。

这条路线应命名“CalorieCLIP 结构的独立受控再训练”；它改变原训练数据、划分及缺失的训练细节，不能声称重现 51.4 MAE。公开成品权重若另做 Demo 推理或探索评估，必须标注“训练污染未知”，与无泄漏主实验表隔离。Food-101 类别估计热量不能混入实测 Nutrition5k 监督而不区分来源。

## 3. 题名原论文与课程改编的区别

Ki-Seung Lee 的 *Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images*，Foods 2024，13(4):551，DOI 10.3390/foods13040551，实验为作者采集的 101 种食物、多波长成像数据；数据可向通讯作者索取。它不是在 HSIFoodIngr-64 上训练、Nutrition5k 上评估的原始实验。论文以改进 U-Net 做 RGB→UV/NIR 预测，再分别评估识别与热量；本项目可以按照课程题设改编，但必须写清数据域/结构/划分/任务和原论文不同。[原论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10887625/)

因此，Capsicum 的 PSNR/SSIM 只说明农场域转换效果，不能换算成餐盘热量精度；HSIFoodIngr-64→Nutrition5k 同样存在跨域问题。最终有意义的结论来自固定餐盘测试集上的 RGB-only 与 RGB+预测 NIR 配对实验，以及明确输入和训练数据条件的外部 RGB 方法，而不是拿不同论文/数据集的数值直接相减。

## 4. 执行前门禁

- 固定源 commit、通用预训练权重哈希与依赖；记录明确许可。
- 输入来自 RGB、真实深度或预测 NIR 必须分别命名，禁止混淆。
- 保存 dish/group 级 train/val/test manifest；全部帧和同餐盘增量扫描不跨集合。
- 标签单位和反标准化先用人工可核对样本做测试；绝不恢复旧 cal/mass 对调。
- 缺权重、结构不匹配、未知输出名必须失败，不继续输出随机结果。
- 不在测试集选 epoch、调损失/阈值；主表报告 MAE/RMSE 和明示分母规则的相对误差，并包含样本数。
- 同时保留“严格复现缺口”“本机适配”“独立实现”标记；未真正运行的候选不写成已完成基线。
