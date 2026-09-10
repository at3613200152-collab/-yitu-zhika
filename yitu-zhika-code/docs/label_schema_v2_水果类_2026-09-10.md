# label_schema v2：新增 fruit 类（12 类）

日期：2026-09-10　决策人：项目负责人（本次会话确认）　影响面：标签规则 / 五目标 manifest / 分类头 / 接口与小程序的类别展示。

## 1. 决策
采用**方案 C**：升级 label_schema，在原有 11 类基础上新增 `fruit`（水果）→ **12 类**。
旧 11 类清单与旧冻结 manifest **保持不变**（作对照），新清单另存新路径。

原 11 类：`dairy, dessert, egg, grain, meat, mixed, other, sauce_condiment, seafood, soup_stew, vegetable`
新 12 类：`dairy, dessert, egg, **fruit**, grain, meat, mixed, other, sauce_condiment, seafood, soup_stew, vegetable`

## 2. 为什么必须加这一类（证据，来自真实数据）
对 `data/Nutrition5k/dish_ingredients.csv`（250 个去重食材、5006 餐盘、28376 行）审计发现：

1. **大量水果类食材此前被并进 `vegetable`**：apple(239 盘)、pineapple(249)、grapes(137)、
   orange(102)、strawberries(78)、honeydew melons(79)、pears(72)、banana(51)、blueberries(174)、
   raspberries(71)、figs(40)、lemon(58)、lime(255)、mandarin oranges(28)、fruit salad、apple cider、
   orange juice、raisins、dates、cranberries、blackberries 等。
   → 直接把「蔬菜」占比推高（修正前训练集 vegetable 占 **54%**），分类头偏置被放大。
2. **没有水果类就无处可归**：watermelon(110 盘)、kiwi(54)、grapefruit juice、cantaloupe 原本落 `other`。
3. **复合名一律按尾词归类不合理**：`tuna salad`(5)、`chicken salad`(6)、`pasta salad` 原按尾词 `salad`
   归为 `vegetable`，与主料（金枪鱼/鸡肉/意面）冲突，并进一步抬高 vegetable 占比。

## 3. 规则改动（`src/data/build_categories.py`）
| 改动 | 内容 |
|---|---|
| 新增类别 | `("fruit", [...])`：apple/applesauce/banana/berries/blackberries/blueberries/raspberries/strawberries/cranberries/cherry/date/fig/grape/grapefruit/kiwi/lemon/lime/mango/melon/cantaloupe/honeydew/watermelon/nectarine/orange/mandarin/peach/pear/pineapple/plum/apricot/pomegranate/raisin/dried fruit/fruit |
| 移出 vegetable | 上述水果词条（含 `lemon juice`）从 vegetable 列表删除，避免"同一个词出现在两类"的优先级歧义 |
| 口径（烹饪习惯，非植物学） | 番茄/鳄梨/橄榄/甜椒/黄瓜等仍为 `vegetable`；`cherry tomatoes`、`grape tomato` 因尾词 `tomato` 仍归 vegetable |
| 复合名按主料（OVERRIDES） | `tuna salad→seafood`、`chicken salad→meat`、`pasta salad→grain`、`egg salad→egg`、`potato salad→vegetable`、`fruit salad→fruit`、`greek salad→vegetable` |
| 防误收 | `wheat berry/wheat berries→grain`（否则会被 berry 收进 fruit） |

回归测试 `tests/test_build_categories.py` **9/9 通过**（含新增的 fruit 正例、番茄等反例、
wheat berry/chicken apple sausage 反例、salad 按主料用例）。

## 4. 修正清单（`results/meal_macros_corrected_v2/`）
`python src/data/meal_macros_corrected_manifest.py` 生成，**不覆盖** `meal_macros_v1` / `meal_macros_expanded_v1`：

- 基础 = `results/meal_macros_expanded_v1/manifest.json`，**划分/图像/五目标/mask 一律不动**，
  只重算 `category` + `category_idx`（保证与旧模型的差异只来自标签）；
- 逐字段溯源：`derived_from_sha256`、`label_rule_code_sha256`、`ingredients_sha256`、
  `label_schema_version=v2_fruit_12class`、逐 split 类别计数、`label_changes_vs_base`；
- 明细另存 `label_changes.json`（每道变更餐盘的 old/new/split）。

### 4.1 变更规模
| 指标 | 值 |
|---|---|
| 总行数 | 3390（train 2316 / val 567 / test 507，与旧清单一致） |
| 标签变化行数 | **649（19.1%）** |
| 主要迁移 | `vegetable→fruit` 454、`vegetable→mixed` 76、`other→fruit` 57、`egg→vegetable` 15、`vegetable→egg` 10、`other→grain` 6 等 |
| 缺失食材记录 | 0 |

### 4.2 训练集类别分布变化（`scripts/compare_label_schema_counts.py`，train n=2316 不变）
| 类别 | 旧（11 类） | 新（12 类） |
|---|---|---|
| vegetable | 1253（54.1%） | **940（40.6%）** |
| fruit | —（不存在） | **314（13.6%）** |
| meat | 257（11.1%） | 259（11.2%） |
| egg | 208（9.0%） | 199（8.6%） |
| mixed | 150（6.5%） | 195（8.4%） |
| other | 197（8.5%） | 151（6.5%） |
| seafood | 92（4.0%） | 94（4.1%） |
| dairy | 87（3.8%） | 87（3.8%） |
| grain | 62（2.7%） | 67（2.9%） |
| sauce_condiment | 6（0.3%） | 6（0.3%） |
| dessert | 4（0.2%） | 4（0.2%） |
| soup_stew | 0 | 0 |
| 合计 | 2316 | 2316 |

val 567 / test 507 均不变；test 上有真值支持的类别数为 10（12 类中 soup_stew、dessert 无测试样本 →
**这两类不得宣称已完成验收**）。

> 注：`vegetable→mixed` 的 76 道是因为"部分蔬果质量被划到 fruit 后，最高类别占比跌破 40% 阈值"，
> 属阈值规则的预期行为，不是数据缺失。

### 4.3 逐类别支持数（清单 §4 要求公开；train/val/test）
| 类别 | train（旧→新） | val（旧→新） | test（旧→新） |
|---|---|---|---|
| dairy | 87 → 87 | 13 → 13 | 19 → 19 |
| dessert | 4 → 4 | **0 → 0** | 3 → 3 |
| egg | 208 → 199 | 32 → 36 | 35 → 35 |
| fruit | — → **314** | — → **101** | — → **98** |
| grain | 62 → 67 | 8 → 12 | 14 → 14 |
| meat | 257 → 259 | 91 → 90 | 71 → 72 |
| mixed | 150 → 195 | 22 → 27 | 19 → 41 |
| other | 197 → 151 | 58 → 42 | 55 → 47 |
| sauce_condiment | 6 → 6 | 1 → 1 | **0 → 0** |
| seafood | 92 → 94 | 8 → 8 | 15 → 15 |
| soup_stew | 0 → 0 | **0 → 0** | **0 → 0** |
| vegetable | 1253 → 940 | 334 → 237 | 276 → 163 |

**明确不能宣称已验收的类别**：
- `soup_stew`：train/val/test 全为 0，模型从未见过该类，**任何情况下都不得声称支持汤炖菜**；
- `dessert`：val 为 0（仅 train 4、test 3）→ 无法用验证集选择该类，测试上的 3 个样本只作观察，不作验收依据；
- `sauce_condiment`：test 为 0（train 6、val 1）→ 无测试证据，不得宣称该类验收通过。

## 5. 对可比性的影响（重要）
- **卡路里/重量/蛋白/碳水/脂肪**五目标由 Nutrition5k 真值给出，**与类别标签无关**，
  因此新旧模型的回归指标可直接比较。
- **分类指标不可跨 schema 直接比较**：类别集合从 11 变 12，"全类别 Macro-F1"的分母不同，
  vegetable 的定义也变了。跨版本只能比较*相同定义*的子集（如只比两者都存在的类别，或看支持度分布），
  报告中必须写明这一点。
- 旧权重（`v1_expanded`）仍按自己的 manifest（11 类）解释类别；接口已改为
  **按 checkpoint 自带的训练 manifest** 解析 `category_idx→名称`（`resolve_macros_manifest`，
  校验 manifest SHA 与 `num_classes`），避免 12 类 logits 用 11 类映射解释导致错名。

## 6. 待办
1. 用修正清单重训五目标模型（多随机种子），与 `v1_expanded` 做同种子对照：
   热量/宏量回归指标直接对比；分类指标按 §5 的口径谨慎表述。
2. 评估通过后再决定是否把**在线 `/predict` 的主模型**切换到 12 类权重；切换需同步
   小程序类别展示（`result.js` / `nutritionist.js` 的中文映射增加"水果"）。
3. 类别集合升级后，历史记录里的旧类别名与旧模型预测需要保持可读（不重写历史）。

## 7. 边界
- 本 schema 仍是**关键词+质量阈值推导**，**不是官方食物类别**；不得对外宣称与官方类别对齐。
- 豆类/坚果/种子（beans、edamame、almonds…）、番茄酱类等仍落 `other`，未加专用类别。
- 本次未做官方 Nutrition5k 类别映射核对，也未做人标注一致性评估。
