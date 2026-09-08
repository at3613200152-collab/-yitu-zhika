# 旧训练入口恢复说明

2026-09-04，旧 `src/training/train_multitask.py` 存在损失类导入失败、生成器版本与当前权重不匹配、默认回归维度不一致的问题，现替换为明确转入五目标 v2 实验训练器的兼容入口。

替换前尝试复制旧脚本，但当时 `archive/` 不存在，PowerShell 的 Copy-Item 非终止错误没有阻止后续补丁。**这是本轮操作失误：没有得到被替换文件的完整、逐字节备份，因此不能声称该入口可完整回退。** 此后操作使用终止错误检查，权重迁移在训练器内显式创建备份目录并校验文件指纹。

`train_multitask_upstream_reference.py` 来自本机另一份同项目源码 `C:/Users/user/Desktop/新建文件夹/food_calorie_estimation/training/train_multitask_yitu.py`。它与被替换入口并不完全相同（导入路径等不同），只作为历史参考，不能直接覆盖当前入口。原始文件的部分读取记录仍在本任务历史中；没有把推测重建的文件冒充原件。

Capsicum 训练权重是另一条独立流程，其迁移备份位于 `checkpoints/capsicum_unet4_seed42/before_amp_fix/`；是否完成备份与续训，见 `results/capsicum_unet4_seed42/amp_policy_migration.json` 和实时训练状态。本次没有删除任何数据集。
