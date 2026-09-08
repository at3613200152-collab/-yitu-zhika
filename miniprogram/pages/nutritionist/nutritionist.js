// pages/nutritionist/nutritionist.js - 营养师页面
// 支持：默认营养库生成 / 预设菜单(如饺子餐、商家推广) / 自定义食物池
const app = getApp()

// 11 类（与后端 model/recipe 保持一致）
const CATEGORY_IDS = [
  'dairy','dessert','egg','grain','meat','mixed','other',
  'sauce_condiment','seafood','soup_stew','vegetable'
]
const CATEGORY_ZH = {
  dairy:'乳制品', dessert:'甜点', egg:'蛋类', grain:'谷物主食', meat:'肉类',
  mixed:'混合餐食', other:'其他', sauce_condiment:'酱料调味品', seafood:'水产',
  soup_stew:'汤炖菜', vegetable:'蔬菜'
}

// 预设菜单（与后端 recipe/custom_menu.py PRESET_MENUS 对齐）
const PRESETS = [
  { key: '', zh: '默认（营养库）' },
  { key: 'dumpling', zh: '饺子餐（预设）' },
  { key: 'merchant_demo', zh: '商家推广（预设）' }
]

Page({
  data: {
    // 表单
    height: '170',
    weight: '65',
    age: '30',
    gender: 'male',
    activityLevels: ['久坐', '轻度', '中度', '活跃', '极高'],
    activityValues: ['sedentary', 'light', 'moderate', 'active', 'very_active'],
    activityIndex: 2,
    goal: 'maintain',
    allergies: [],
    // 预设 / 自定义食物
    presets: PRESETS.map(p => p.zh),
    presetIndex: 0,
    customFoods: [],
    foodFormName: '',
    foodCategoryIds: CATEGORY_IDS.map(id => CATEGORY_ZH[id]),
    foodCategoryIndex: 3,   // grain
    foodKcal: '',
    foodGrams: '',
    // 结果
    plan: null,
    loading: false
  },

  onInput(e) {
    const key = e.currentTarget.dataset.key
    this.setData({ [key]: e.detail.value })
  },

  onGenderSelect(e) {
    this.setData({ gender: e.currentTarget.dataset.v })
  },

  onActivityChange(e) {
    this.setData({ activityIndex: parseInt(e.detail.value) })
  },

  onGoalSelect(e) {
    this.setData({ goal: e.currentTarget.dataset.v })
  },

  toggleAllergy(e) {
    const v = e.currentTarget.dataset.v
    const list = this.data.allergies
    const idx = list.indexOf(v)
    if (idx >= 0) { list.splice(idx, 1) } else { list.push(v) }
    this.setData({ allergies: list })
  },

  // ---- 预设菜单 ----
  onPresetChange(e) {
    this.setData({ presetIndex: parseInt(e.detail.value) })
  },

  // ---- 自定义食物（用户补全数据）----
  onFoodField(e) {
    const key = e.currentTarget.dataset.key
    this.setData({ [key]: e.detail.value })
  },
  onFoodCategory(e) {
    this.setData({ foodCategoryIndex: parseInt(e.detail.value) })
  },
  addCustomFood() {
    const name = (this.data.foodFormName || '').trim()
    const kcal = parseFloat(this.data.foodKcal)
    const grams = parseFloat(this.data.foodGrams)
    if (!name) { wx.showToast({ title: '请输入食物名', icon: 'none' }); return }
    if (!(kcal > 0)) { wx.showToast({ title: '请输入每100g热量', icon: 'none' }); return }
    const catId = CATEGORY_IDS[this.data.foodCategoryIndex]
    const food = {
      name: name,
      category: catId,
      kcal_per_100g: kcal,
      protein_per_100g: 0,
      carb_per_100g: 0,
      fat_per_100g: 0,
      default_grams: grams > 0 ? grams : 100
    }
    const foods = this.data.customFoods.concat([food])
    this.setData({
      customFoods: foods,
      foodFormName: '', foodKcal: '', foodGrams: ''
    })
  },
  removeCustomFood(e) {
    const idx = e.currentTarget.dataset.idx
    const foods = this.data.customFoods.slice()
    foods.splice(idx, 1)
    this.setData({ customFoods: foods })
  },

  async generatePlan() {
    const d = this.data
    if (!d.height || !d.weight || !d.age) {
      wx.showToast({ title: '请填写基本信息', icon: 'none' })
      return
    }
    // 默认(营养库) 且无自定义食物 → 走原 /weekly-plan；否则走 /plan-from-menu
    const usePool = d.presetIndex > 0 || d.customFoods.length > 0

    this.setData({ loading: true })

    const base = {
      height_cm: parseFloat(d.height),
      weight_kg: parseFloat(d.weight),
      age: parseInt(d.age),
      gender: d.gender,
      activity_level: d.activityValues[d.activityIndex],
      goal: d.goal,
      allergies: d.allergies
    }

    try {
      let res
      if (usePool) {
        const payload = Object.assign({}, base, {
          preset: PRESETS[d.presetIndex].key || undefined,
          foods: d.customFoods
        })
        res = await this.callPlanFromMenu(payload)
      } else {
        res = await this.callWeeklyPlan(base)
      }
      if (res.statusCode === 200 && res.data.status === 'ok') {
        this.setData({ plan: res.data })
        wx.showToast({ title: '生成成功', icon: 'success' })
      } else if (res.data.status === 'refused') {
        wx.showModal({
          title: '无法生成',
          content: res.data.disclaimer || (res.data.message || '特殊人群需营养师审核'),
          showCancel: false
        })
      } else {
        wx.showModal({
          title: '生成失败',
          content: res.data.message || '未知错误',
          showCancel: false
        })
      }
    } catch (err) {
      wx.showModal({
        title: '网络错误',
        content: err.errMsg || '请检查网络后重试',
        showCancel: false
      })
    } finally {
      this.setData({ loading: false })
    }
  },

  callWeeklyPlan(payload) {
    return new Promise((resolve, reject) => {
      wx.request({
        url: app.globalData.apiBase + '/weekly-plan',
        method: 'POST',
        header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' },
        data: payload,
        success: resolve,
        fail: reject
      })
    })
  },

  callPlanFromMenu(payload) {
    return new Promise((resolve, reject) => {
      wx.request({
        url: app.globalData.apiBase + '/plan-from-menu',
        method: 'POST',
        header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' },
        data: payload,
        success: resolve,
        fail: reject
      })
    })
  },

  resetPlan() {
    this.setData({ plan: null })
  }
})
