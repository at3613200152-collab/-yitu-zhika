# HSIFoodIngr-64 数据来源与生成器输入审计

审计日期：2026-09-05。作者论文和制造商资料为事实来源；本地文件观测由主任务读取原始 ENVI 头文件后提供；训练建议是本项目设计。

## 作者论文能确认什么

原始数据 DOI 为 `10.7910/DVN/E7WDNQ`，发布在 Harvard Dataverse。共有 3,389 对图像、21 种菜品、64 种食材，HSI 为 204 波段，RGB 与 HSI 均为 512×512，共享空间分布与像素标注。采集包括多次购买菜品及逐次倒入垃圾桶的过程。原论文针对四个成分检索子集分别随机按 9:1 划分训练/测试，没有规定通用的 RGB→NIR 划分。数据总大小超过 650 G。来源：[作者论文，III 与 V 节](https://doi.org/10.1109/ACCESS.2023.3243243)；本次实际读取[同一论文的公开全文镜像](https://www.researchgate.net/publication/368374434_HSIFoodIngr-64_A_Dataset_for_Hyperspectral_Food-Related_Studies_and_a_Benchmark_Method_on_Food_Ingredient_Retrieval)。

该论文未列出包号与类别的对应、独立物理餐盘 ID、RGB→NIR 推荐波段、逐图分位数归一化方案及本批数据白板/暗场校正记录。不能把这些未知事项表述为作者规定。[作者论文](https://doi.org/10.1109/ACCESS.2023.3243243)

## 配套 PNG 与 HSI 的关系

制造商资料给出 Specim IQ 400–1000 nm 范围、204 波段、512 像素空间采样和 ENVI 兼容输出；设备另有不同分辨率的 viewfinder 相机。[原厂规格](https://www.specim.com/iq/tech-specs/)、[完整规格 PDF](https://www.specim.com/wp-content/uploads/2024/11/Specim-IQ-Datasheet-Long-08.pdf)

原厂手册说明一次测量根目录可包含 viewfinder 和 spectral-camera 图像，Capture 保存 raw/dark/white，Results 保存应用结果；`.dat` 为反射率数据，`.hdr` 为配套头信息，`.png` 为 RGB 图像。[Specim 原厂手册第 14 页，公开镜像](https://www.kislab.kr/_files/ugd/795c68_72b2e5013599414e95e03c5909ede35a.pdf?index=true)

相机评估原始研究说明默认录制模式会显示从反射率数据生成的 RGB。因此配套 PNG 的存在并不证明它由独立普通 RGB 相机拍摄；上述来源也不能确定 HSIFoodIngr-64 发布 PNG 的具体颜色渲染函数。[原始相机研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC5855187/)

项目建议：用同一扫描的配套 PNG 三通道作为输入，用 HSI 的指定 NIR 波段作目标；称“数据集配套 RGB PNG”。先跨日期抽样检查尺寸、alpha、旋转、左右翻转及食物边缘。论文的空间对齐声明不能代替对本地 BIL 解析方向的检查。640/550/460 nm 三窄带组合是自定义合成伪 RGB，与 Nutrition5k 照片的响应不同；不应无说明地替换配套 PNG。

## 本地观测

以下来自主任务读盘，最终还应由生成器清单中的 source hash 绑定：8 包为 1、2、3、4、5、10、100、101，共 144 个 scan；采集日期计数为 2021-08-03:12、08-06:15、08-07:18、08-08:33、08-10:12、08-14:18、10-09:36。包号跨日期，不能当类别或独立餐盘 ID。

头文件为 SPECIM IQ、512×512×204、BIL、float32（ENVI type 4）、little endian（byte order 0）、header offset 0；默认显示波段字段为 70/53/19，没有 reflectance scale factor 字段。波长向量为 397.32–1003.58 nm；640/550/460/860 nm 最近零基索引为 83/52/22/156，对应 640.04/548.55/460.96/859.42 nm。它们是本地计算结果，不是论文推荐组合。默认显示波段字段与 numpy 索引的基数须由读取器明确处理。

主任务对一个 cube 的 NIR 测得约 14% 像素大于 1、最大约 1.49。它不证明绝对标定正确，也不支持直接把所有大于 1 的值裁掉或宣布损坏。原值和异常比例应保留。

## 当前方案审查

主任务拟按 `sha256('42:'+date)` 排序：2021-08-08 的 33 个 scan 为 test，08-07 的 18 个为 val，其余 93 个为 train；此划分是本项目选择，最终以冻结清单为准。日期是缺少物理餐盘 ID 时的关联组代理，不是已经证实的独立餐盘边界。只有 7 个日期，结果会受日期、菜品和垃圾场景分布影响。

建议落实以下协议：

- 同日期所有 scan、裁剪、增强保持同一 split；冻结日期、scan ID、输入规则与哈希。获得真实餐盘/序列标注后，若发现跨集合关联则建立新版本，不追改旧结果。
- 核验同名 PNG/HSI 严格一对一；做精确像素重复检查，以及跨日期抽样的可见光边缘配对检查。特别排除 BIL 轴顺序、转置、翻转错误。
- 目标固定为 859.42 nm。仅训练 scan 以 stride 4 网格采样计算全局 p0.5/p99.5，保存采样方式与参数，随后固定应用于三个集合。记录每个集合的低端/高端截断比例和非有限值数量。
- 共享缩放保留样本间的强弱关系；逐图 p2/p98 会将每张图改为自己的尺度。保留原始 band hash 和缩放参数，结果称“归一化 NIR 强度预测”，不把其数值当作经独立验证的绝对反射率。
- PSNR/SSIM 固定 data_range、缩放、尺寸和平均方式；与采用其他缩放的旧生成器、Capsicum 成绩不直接比较。测试集只用于最终模型评估。
- 生成器重建成绩不能单独证明热量估计提升；Nutrition5k 仍需固定餐盘划分下 RGB 与 RGB+预测 NIR 对照。

## 校正、包号、类别和许可的证据边界

制造商说明反射率转换需要白参考与暗参考，以修正环境和传感器因素。[原厂采集说明](https://www.specim.com/technology/how-to-record-data-with-hyperspectral-camera/) 但本批数据的参考板、几何条件及测量日志尚未验证。不要额外套用猜测的 1/10000 或灰板系数。

本次 web 工具不能成功读取 Harvard Dataverse API/数据页；主任务正在尝试直接读取官方元数据。因此数据版本、许可、准确总下载字节数、扫描标注文件和包号含义暂未核实。论文/插图的 CC BY-NC-ND 4.0 不自动等于数据许可；第三方代码的 MIT 也不等于数据许可。[作者指定发布点](https://doi.org/10.7910/DVN/E7WDNQ)

`HSIFoodIngr-64_data_N` 当前仅可视为分包名称。必须用官方标注按精确 scan ID 连接类别和食材，不能根据包号或图像观感生成标签。完整官方 file ID/字节/checksum 取得前，下载体量只能引用论文“超过 650 G”；当前 D 盘约 169 GiB 无法容纳该全量，扩容应优先获取小型标注与确有需要的分包。

本审计仅创建文档，没有修改原始数据、代码或权重。
