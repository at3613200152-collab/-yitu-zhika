// pages/result/result.js - 三档点击 + 折叠手动改 + 跳过 + quality 分级
const app = getApp()

// 11 类与后端 model manifest 对齐（yitu-zhika-code 主模型 category_to_idx）
// 后端返回中文 category_name + 数字 category_idx；这里用 id 列表回传，避免中英不一致
const CATEGORY_IDS = [
  'dairy', 'dessert', 'egg', 'grain', 'meat', 'mixed', 'other',
  'sauce_condiment', 'seafood', 'soup_stew', 'vegetable'
]
const CATEGORY_ZH = {
  dairy: '乳制品', dessert: '甜点', egg: '蛋类', grain: '谷物主食',
  meat: '肉类', mixed: '混合餐食', other: '其他',
  sauce_condiment: '酱料调味品', seafood: '水产', soup_stew: '汤炖菜',
  vegetable: '蔬菜'
}

// 中文名 -> id（兼容旧记录只存了中文名的场景）
function zhToId(zhName) {
  if (!zhName) return null
  if (CATEGORY_IDS.indexOf(zhName) >= 0) return zhName
  return CATEGORY_IDS.find(id => CATEGORY_ZH[id] === zhName) || null
}

Page({
  data: {
    record: null,
    editing: false,
    // 三档点击状态
    userGrade: '',  // 'over' | 'ok' | 'under' | ''
    // 折叠手动改
    showManualForm: false,
    // 手动修正表单：显示中文，回传 id
    categories: CATEGORY_IDS.map(id => CATEGORY_ZH[id]),  // ['乳制品','甜点',...]
    categoryIndex: -1,
    editCalories: '',
    editWeight: '',
    editCategory: ''
  },

  onLoad(options) {
    const id = options.id
    const record = app.globalData.history.find(r => r.id === id)
    if (!record) {
      wx.showModal({
        title: '错误',
        content: '找不到该记录',
        showCancel: false,
        success: () => wx.navigateBack()
      })
      return
    }
    // 用 category_idx（后端返回）预选类别；旧记录退回中文名映射
    let catId = null
    const idx = record.result && record.result.category_idx
    if (typeof idx === 'number' && idx >= 0 && idx < CATEGORY_IDS.length) {
      catId = CATEGORY_IDS[idx]
    } else if (record.result && record.result.category_id) {
      catId = record.result.category_id
    } else if (record.result && record.result.category_name) {
      catId = zhToId(record.result.category_name)
    }
    const categoryIndex = catId ? CATEGORY_IDS.indexOf(catId) : -1

    this.setData({
      record: record,
      editCalories: String(Math.round(record.result.calories || 0)),
      editWeight: String(Math.round(record.result.weight || 0)),
      editCategory: categoryIndex >= 0 ? CATEGORY_ZH[CATEGORY_IDS[categoryIndex]] : (record.result.category_name || ''),
      categoryIndex: categoryIndex
    })
  },

  // 模型类别 id（用于回传 model_category，与 corrected_category 同规范）
  modelCategoryId(record) {
    const r = record.result || {}
    if (typeof r.category_idx === 'number' && r.category_idx >= 0 && r.category_idx < CATEGORY_IDS.length) {
      return CATEGORY_IDS[r.category_idx]
    }
    return zhToId(r.category_name)
  },

  // 三档点击：选择"偏多/差不多/偏少"
  async onGradeSelect(e) {
    const grade = e.currentTarget.dataset.grade
    this.setData({ userGrade: grade })

    // grade=ok → confirm_only 质量
    // grade=over/under → directional（未手动输入具体值，仅方向性反馈）
    const quality = grade === 'ok' ? 'confirm_only' : 'directional'
    await this.submitGradeFeedback(grade, quality)
  },

  // 提交三档反馈（轻量，不带具体修正值）
  async submitGradeFeedback(grade, quality) {
    const record = this.data.record
    const payload = {
      dish_uuid: record.dish_uuid || record.id,
      image_hash: record.image_hash || null,
      prediction_id: record.id,
      user_grade: grade,  // 'over' | 'ok' | 'under'
      quality: quality,   // 'confirm_only' | 'directional'
      corrected_calories: null,
      corrected_weight: null,
      corrected_category: null,
      model_version: record.result.model_version || null,
      model_calories: record.result.calories || null,
      model_weight: record.result.weight || null,
      model_category: this.modelCategoryId(record)
    }

    try {
      const res = await this.callFeedbackApi(payload)
      if (res.statusCode === 200) {
        if (grade === 'ok') {
          wx.showToast({ title: '已确认', icon: 'success' })
        } else {
          wx.showToast({ title: '反馈已记录', icon: 'success' })
        }
        // 结构化落盘，供"我的"页统计（contribution.js 读取 feedbackQuality）
        record.grade = grade
        record.feedbackQuality = quality
        record.feedbackSubmitted = true
        app.saveHistory(record)
        setTimeout(() => wx.navigateBack(), 800)
      } else {
        wx.showToast({ title: '反馈提交失败，不影响识别', icon: 'none' })
      }
    } catch (err) {
      // 反馈失败不影响主流程，仅提示
      wx.showToast({ title: '反馈提交失败，不影响识别', icon: 'none' })
    }
  },

  // 折叠"我手动改…"
  toggleManualMode() {
    this.setData({ showManualForm: !this.data.showManualForm })
  },

  onCaloriesInput(e) {
    this.setData({ editCalories: e.detail.value })
  },

  onWeightInput(e) {
    this.setData({ editWeight: e.detail.value })
  },

  onCategoryChange(e) {
    const idx = parseInt(e.detail.value)
    if (idx >= 0 && idx < CATEGORY_IDS.length) {
      this.setData({
        categoryIndex: idx,
        editCategory: CATEGORY_ZH[CATEGORY_IDS[idx]]
      })
    }
  },

  // 手动修正提交（manual_typed 质量）
  async submitManualFeedback() {
    const record = this.data.record
    const catId = this.data.categoryIndex >= 0 ? CATEGORY_IDS[this.data.categoryIndex] : null
    const payload = {
      dish_uuid: record.dish_uuid || record.id,
      image_hash: record.image_hash || null,
      prediction_id: record.id,
      user_grade: this.data.userGrade || 'manual',
      quality: 'manual_typed',
      corrected_calories: parseFloat(this.data.editCalories) || null,
      corrected_weight: parseFloat(this.data.editWeight) || null,
      corrected_category: catId,
      model_version: record.result.model_version || null,
      model_calories: record.result.calories || null,
      model_weight: record.result.weight || null,
      model_category: this.modelCategoryId(record)
    }

    try {
      const res = await this.callFeedbackApi(payload)
      if (res.statusCode === 200) {
        wx.showToast({ title: '已提交', icon: 'success' })
        record.result.calories = payload.corrected_calories
        record.result.weight = payload.corrected_weight
        // 存 id 用于回传规范；展示名保持中文
        record.result.category_id = payload.corrected_category
        if (catId && CATEGORY_ZH[catId]) {
          record.result.category_name = CATEGORY_ZH[catId]
        }
        record.corrected = true
        record.feedbackQuality = 'manual_typed'
        record.corrected_calories = payload.corrected_calories
        record.corrected_weight = payload.corrected_weight
        record.corrected_category = payload.corrected_category
        record.feedbackSubmitted = true
        app.saveHistory(record)
        this.setData({ showManualForm: false, editing: false })
        setTimeout(() => wx.navigateBack(), 800)
      } else {
        wx.showModal({
          title: '提交失败',
          content: '服务器返回异常，请稍后重试',
          showCancel: false
        })
      }
    } catch (err) {
      wx.showModal({
        title: '提交失败',
        content: err.message || err.errMsg || '网络错误',
        showCancel: false
      })
    }
  },

  // 跳过这顿（skipped 质量，零摩擦退出）
  async skipFeedback() {
    const record = this.data.record
    const payload = {
      dish_uuid: record.dish_uuid || record.id,
      image_hash: record.image_hash || null,
      prediction_id: record.id,
      user_grade: 'skip',
      quality: 'skipped',
      corrected_calories: null,
      corrected_weight: null,
      corrected_category: null,
      model_version: record.result.model_version || null,
      model_calories: record.result.calories || null,
      model_weight: record.result.weight || null,
      model_category: this.modelCategoryId(record)
    }

    try {
      await this.callFeedbackApi(payload)
      record.grade = 'skip'
      record.feedbackQuality = 'skipped'
      record.feedbackSubmitted = true
      app.saveHistory(record)
    } catch (err) {
      // 跳过失败也不阻塞
    }
    wx.navigateBack()
  },

  // 调用后端 /feedback 接口
  callFeedbackApi(payload) {
    return new Promise((resolve, reject) => {
      wx.request({
        url: app.globalData.apiBase + '/feedback',
        method: 'POST',
        header: {
          'X-API-Key': app.globalData.apiKey,
          'Content-Type': 'application/json'
        },
        data: payload,
        success: resolve,
        fail: reject
      })
    })
  },

  // 保存记录（无网络时本地保存）
  saveLocal() {
    const record = this.data.record
    const catId = this.data.categoryIndex >= 0 ? CATEGORY_IDS[this.data.categoryIndex] : record.result.category_name
    record.result.calories = parseFloat(this.data.editCalories) || record.result.calories
    record.result.weight = parseFloat(this.data.editWeight) || record.result.weight
    record.result.category_name = CATEGORY_ZH[catId] || record.result.category_name
    record.savedLocal = true
    app.saveHistory(record)
    wx.showToast({ title: '已保存到本地', icon: 'success' })
    this.setData({ showManualForm: false, record: record })
  }
})
