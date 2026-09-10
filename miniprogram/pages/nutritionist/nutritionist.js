// pages/nutritionist/nutritionist.js - 饮食计划（可调整的建议，非强制菜单）
const app = getApp()

// 与 result.js 一致：label_schema v2 为 12 类（含水果）
const CATEGORY_IDS = [
  'dairy','dessert','egg','fruit','grain','meat','mixed','other',
  'sauce_condiment','seafood','soup_stew','vegetable'
]
const CATEGORY_ZH = {
  dairy:'乳制品', dessert:'甜点', egg:'蛋类', fruit:'水果', grain:'谷物主食', meat:'肉类',
  mixed:'混合餐食', other:'其他', sauce_condiment:'酱料调味品', seafood:'水产',
  soup_stew:'汤炖菜', vegetable:'蔬菜'
}
const MEAL_ZH = { breakfast: '早餐', lunch: '午餐', dinner: '晚餐' }

function recalc(food, grams) {
  const p = food.per_100g
  if (!p) return null
  const f = grams / 100
  return {
    calories: Math.round(p.kcal * f * 10) / 10,
    protein: Math.round((p.protein || 0) * f * 10) / 10,
    carb: Math.round((p.carb || 0) * f * 10) / 10,
    fat: Math.round((p.fat || 0) * f * 10) / 10
  }
}

Page({
  data: {
    height: '170', weight: '65', age: '30', gender: 'male',
    activityLevels: ['久坐', '轻度', '中度', '活跃', '极高'],
    activityValues: ['sedentary', 'light', 'moderate', 'active', 'very_active'],
    activityIndex: 2, goal: 'maintain', allergies: [],
    allergySelected: {},
    presets: undefined,
    menuOptions: ['默认（营养库）'],
    merchantMenus: [],
    menuId: '',
    presetIndex: 0,
    customFoods: [],
    foodFormName: '',
    foodCategoryIds: CATEGORY_IDS.map(id => CATEGORY_ZH[id]),
    foodCategoryIndex: 3,
    foodKcal: '', foodGrams: '',
    mealZh: MEAL_ZH,
    plan: null,
    loading: false
  },

  onInput(e) { this.setData({ [e.currentTarget.dataset.key]: e.detail.value }) },
  onGenderSelect(e) { this.setData({ gender: e.currentTarget.dataset.v }) },
  onActivityChange(e) { this.setData({ activityIndex: parseInt(e.detail.value) }) },
  onGoalSelect(e) { this.setData({ goal: e.currentTarget.dataset.v }) },
  toggleAllergy(e) {
    const v = e.currentTarget.dataset.v
    const list = this.data.allergies.slice()
    const i = list.indexOf(v)
    if (i >= 0) list.splice(i, 1); else list.push(v)
    const allergySelected = {}
    list.forEach(key => { allergySelected[key] = true })
    this.setData({ allergies: list, allergySelected })
  },

  onLoad() {
    this.loadMenus()
  },

  // 拉取已审核的商家菜单，动态生成"饮食范围"选项
  loadMenus() {
    wx.request({
      url: app.globalData.apiBase + '/public/menus',
      method: 'GET',
      success: (res) => {
        try {
          const menus = (res.data && res.data.menus) || []
          const opt = ['默认（营养库）'].concat(menus.map(m => `${m.merchant_name}·${m.menu_name}`))
          this.setData({ merchantMenus: menus, menuOptions: opt })
        } catch (e) {}
      },
      fail: () => {}
    })
  },

  onPresetChange(e) {
    const i = parseInt(e.detail.value)
    const menuId = (i > 0 && this.data.merchantMenus[i - 1]) ? this.data.merchantMenus[i - 1].menu_id : ''
    this.setData({ presetIndex: i, menuId: menuId })
  },
  onFoodField(e) { this.setData({ [e.currentTarget.dataset.key]: e.detail.value }) },
  onFoodCategory(e) { this.setData({ foodCategoryIndex: parseInt(e.detail.value) }) },
  addCustomFood() {
    const name = (this.data.foodFormName || '').trim()
    const kcal = parseFloat(this.data.foodKcal)
    const grams = parseFloat(this.data.foodGrams)
    if (!name) { wx.showToast({ title: '请输入食物名', icon: 'none' }); return }
    if (!(kcal > 0)) { wx.showToast({ title: '请输入每100g热量', icon: 'none' }); return }
    const food = {
      name, category: CATEGORY_IDS[this.data.foodCategoryIndex],
      kcal_per_100g: kcal, protein_per_100g: 0, carb_per_100g: 0, fat_per_100g: 0,
      default_grams: grams > 0 ? grams : 100
    }
    this.setData({ customFoods: this.data.customFoods.concat([food]), foodFormName: '', foodKcal: '', foodGrams: '' })
  },
  removeCustomFood(e) {
    const foods = this.data.customFoods.slice()
    foods.splice(e.currentTarget.dataset.idx, 1)
    this.setData({ customFoods: foods })
  },

  async generatePlan() {
    const d = this.data
    if (!d.height || !d.weight || !d.age) { wx.showToast({ title: '请填写基本信息', icon: 'none' }); return }
    const usePool = !!d.menuId || d.customFoods.length > 0
    const base = {
      height_cm: parseFloat(d.height), weight_kg: parseFloat(d.weight), age: parseInt(d.age),
      gender: d.gender, activity_level: d.activityValues[d.activityIndex], goal: d.goal, allergies: d.allergies
    }
    this.setData({ loading: true })
    try {
      let res
      if (usePool) {
        res = await this.callPlanFromMenu(Object.assign({}, base, { menu_id: d.menuId || undefined, foods: d.customFoods }))
      } else {
        res = await this.callWeeklyPlan(base)
      }
      if (res.statusCode === 200 && res.data.status === 'ok') {
        this.setData({ plan: res.data, loading: false })
        wx.showToast({ title: '已生成建议', icon: 'success' })
      } else if (res.data.status === 'refused') {
        wx.showModal({ title: '无法生成', content: res.data.disclaimer || (res.data.message || '特殊人群需营养师审核'), showCancel: false })
        this.setData({ loading: false })
      } else {
        wx.showModal({ title: '生成失败', content: res.data.message || '未知错误', showCancel: false })
        this.setData({ loading: false })
      }
    } catch (err) {
      wx.showModal({ title: '网络错误', content: err.errMsg || '请检查网络后重试', showCancel: false })
      this.setData({ loading: false })
    }
  },

  callWeeklyPlan(payload) {
    return new Promise((resolve, reject) => {
      wx.request({ url: app.globalData.apiBase + '/weekly-plan', method: 'POST',
        header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' }, data: payload, success: resolve, fail: reject })
    })
  },
  callPlanFromMenu(payload) {
    return new Promise((resolve, reject) => {
      wx.request({ url: app.globalData.apiBase + '/plan-from-menu', method: 'POST',
        header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' }, data: payload, success: resolve, fail: reject })
    })
  },

  // ---- 计划调整 ----
  swapFood(e) {
    const { day, meal, food } = e.currentTarget.dataset
    const plan = this.data.plan
    const f = plan.daily_recipes[day].meals[meal].foods[food]
    if (!f.substitutions || !f.substitutions.length) {
      wx.showToast({ title: '这道暂无可替换的同类选项', icon: 'none' }); return
    }
    const sub = f.substitutions[0]
    const grams = sub.grams_swap || f.grams
    const nutrition = f.per_100g
      ? recalc({ per_100g: { kcal: sub.kcal_per_100g_rag || f.per_100g.kcal, protein: f.per_100g.protein, carb: f.per_100g.carb, fat: f.per_100g.fat } }, grams)
      : null
    // 替换后仅 kcal 有来源，宏量标记为未知
    const nutritionOut = nutrition ? { ...nutrition, protein: null, carb: null, fat: null } : null
    f.name = sub.name
    f.grams = grams
    f.nutrition = nutritionOut
    f.swapped = true
    f.macro_unknown = true
    this.recomputeDay(plan, day, meal)
    this.setData({ plan })
    wx.showToast({ title: '已换一道（按同类替换）', icon: 'success' })
  },

  adjustPortion(e) {
    const { day, meal, food } = e.currentTarget.dataset
    const plan = this.data.plan
    const f = plan.daily_recipes[day].meals[meal].foods[food]
    if (!f.per_100g) { wx.showToast({ title: '该食物暂无每百克营养，无法重算份量', icon: 'none' }); return }
    wx.showModal({
      title: '调整份量',
      editable: true,
      placeholderText: `当前 ${f.grams} 克`,
      success: (res) => {
        const grams = parseFloat(res.content)
        if (!(grams > 0) || isNaN(grams)) { wx.showToast({ title: '请输入有效克数', icon: 'none' }); return }
        const nutrition = recalc(f, grams)
        f.grams = grams
        f.nutrition = nutrition || f.nutrition
        f.portion_edited = true
        this.recomputeDay(plan, day, meal)
        this.setData({ plan })
        wx.showToast({ title: '已按新份量估算', icon: 'success' })
      }
    })
  },

  dislikeFood(e) {
    const { day, meal, food } = e.currentTarget.dataset
    const plan = this.data.plan
    const mealRec = plan.daily_recipes[day].meals[meal]
    const [removed] = mealRec.foods.splice(food, 1)
    mealRec.removedNote = `已移除「${removed ? removed.name : '这道'}」`
    this.recomputeDay(plan, day, meal)
    this.setData({ plan })
    wx.showToast({ title: '已从建议中移除', icon: 'none' })
  },

  recomputeDay(plan, day, meal) {
    const days = plan.daily_recipes
    if (!days[day] || !days[day].meals[meal]) return
    const mealRec = days[day].meals[meal]
    const sum = mealRec.foods.reduce((s, f) => s + (f.nutrition && f.nutrition.calories ? f.nutrition.calories : 0), 0)
    mealRec.nutrition.calories = Math.round(sum * 10) / 10
    const daySum = days[day].meals.reduce((s, m) => s + (m.nutrition.calories || 0), 0)
    days[day].day_total_kcal = Math.round(daySum * 10) / 10
  },

  resetPlan() { this.setData({ plan: null }) }
})
