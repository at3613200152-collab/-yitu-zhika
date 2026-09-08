"""
Nutrition5k 清理后 50 类别映射表 (与 data/Nutrition5k_real/labels_clean.csv 对应)
============================================================
基于清理后的真实数据: 3062 样本, 50 类, 仅保留样本数>=20 的类.
"""

# index -> (name_en, name_cn, category, calories_per_100g)
NUTRITION5K_CLASSES = {
     0: ('almonds', '杏仁', '坚果', 579),
     1: ('apple', '苹果', '水果', 52),
     2: ('arugula', '芝麻菜', '蔬菜', 25),
     3: ('bacon', '培根', '肉类', 541),
     4: ('beef', '牛肉', '肉类', 250),
     5: ('berries', '浆果', '水果', 57),
     6: ('bok choy', '小白菜', '蔬菜', 13),
     7: ('broccoli', '西兰花', '蔬菜', 34),
     8: ('brussels sprouts', '孢子甘蓝', '蔬菜', 43),
     9: ('caesar salad', '凯撒沙拉', '沙拉', 190),
    10: ('cantaloupe', '哈密瓜', '水果', 34),
    11: ('carrot', '胡萝卜', '蔬菜', 41),
    12: ('cauliflower', '花椰菜', '蔬菜', 25),
    13: ('cheese pizza', '芝士披萨', '快餐', 266),
    14: ('cherry tomatoes', '圣女果', '蔬菜', 18),
    15: ('chicken', '鸡肉', '肉类', 239),
    16: ('chicken breast', '鸡胸肉', '肉类', 165),
    17: ('chicken thighs', '鸡腿肉', '肉类', 209),
    18: ('corn on the cob', '玉米', '主食', 86),
    19: ('egg whites', '蛋白', '蛋类', 52),
    20: ('eggplant', '茄子', '蔬菜', 25),
    21: ('eggs', '鸡蛋', '蛋类', 155),
    22: ('fish', '鱼', '海鲜', 165),
    23: ('grapes', '葡萄', '水果', 67),
    24: ('green beans', '四季豆', '蔬菜', 31),
    25: ('grilled chicken', '烤鸡', '肉类', 215),
    26: ('honeydew melons', '哈密瓜（白兰瓜）', '水果', 36),
    27: ('mixed greens', '混合蔬菜', '蔬菜', 24),
    28: ('olives', '橄榄', '蔬菜', 115),
    29: ('other', 'other', '其他', 0),
    30: ('pears', '梨', '水果', 57),
    31: ('pepperoni pizza', '意式辣肠披萨', '快餐', 296),
    32: ('pineapple', '菠萝', '水果', 50),
    33: ('pizza', '披萨', '快餐', 266),
    34: ('pork', '猪肉', '肉类', 242),
    35: ('potatoes', '土豆', '主食', 77),
    36: ('quinoa', '藜麦', '主食', 120),
    37: ('roasted potatoes', '烤土豆', '主食', 149),
    38: ('salmon', '三文鱼', '海鲜', 208),
    39: ('sausage', '香肠', '肉类', 301),
    40: ('scrambled eggs', '炒蛋', '蛋类', 149),
    41: ('spinach (raw)', '菠菜（生）', '蔬菜', 23),
    42: ('squash', '南瓜', '蔬菜', 26),
    43: ('steak', '牛排', '肉类', 271),
    44: ('strawberries', '草莓', '水果', 32),
    45: ('sweet potato', '红薯', '主食', 86),
    46: ('tofu', '豆腐', '豆制品', 76),
    47: ('watermelon', '西瓜', '水果', 30),
    48: ('white rice', '白米饭', '主食', 130),
    49: ('yam', '山药', '主食', 118),
}

NUM_CLASSES = 50

# 反向索引: 英文名 -> 类别 id
NAME_TO_IDX = {v[0]: k for k, v in NUTRITION5K_CLASSES.items()}

def get_class_info(idx: int):
    """根据类别索引返回 (name_en, name_cn, category, calories_per_100g)"""
    if idx not in NUTRITION5K_CLASSES:
        return ("unknown", f"未知({idx})", "其他", 0)
    return NUTRITION5K_CLASSES[idx]

def get_class_name_cn(idx: int) -> str:
    return get_class_info(idx)[1]

def get_class_name_en(idx: int) -> str:
    return get_class_info(idx)[0]

def get_calories_per_100g(idx: int) -> float:
    return float(get_class_info(idx)[3])

def get_category(idx: int) -> str:
    return get_class_info(idx)[2]