// pages/result/result.js - 识别与确认
const app = getApp()
const record = require('../../utils/record.js')

// 11 类与后端 model manifest 对齐
const CATEGORY_IDS = [
  'dairy','dessert','egg','grain','meat','mixed','other',
  'sauce_condiment','seafood','soup_stew','vegetable'
]
const CATEGORY_ZH = {
  dairy:'乳制品', dessert:'甜点', egg:'蛋类', grain:'谷物主食', meat:'肉类',
  mixed:'混合餐食', other:'其他', sauce_condiment:'酱料调味品', seafood:'水产',
  soup_stew:'汤炖菜', vegetable:'蔬菜'
}

function zhToId(zhName) {
  if (!zhName) return null
  if (CATEGORY_IDS.indexOf(zhName) >= 0) return zhName
  return CATEGORY_IDS.find(id => CATEGORY_ZH[id] === zhName) || null
}

Page({
  data: {
    record: null,
    categories: CATEGORY_IDS.map(id => CATEGORY_ZH[id]),
    categoryIndex: -1,
    editCalories: '',
    editWeight: '',
    editCategory: '',
    userGrade: '',
    showManual: false,
    confirmed: false,
    celebrating: false,
    loading: false
  },

  onLoad(options) {
    const id = options.id
    const record = app.globalData.history.find(r => r.id === id)
    if (!record) {
      wx.showModal({ title: '提示', content: '没有找到这条记录', showCancel: false, success: () => wx.navigateBack() })
      return
    }
    let catId = null
    const idx = record.result && record.result.category_idx
    if (typeof idx === 'number' && idx >= 0 && idx < CATEGORY_IDS.length) catId = CATEGORY_IDS[idx]
    else if (record.result && record.result.category_id) catId = record.result.category_id
    else if (record.result && record.result.category_name) catId = zhToId(record.result.category_name)
    const categoryIndex = catId ? CATEGORY_IDS.indexOf(catId) : -1

    this.setData({
      record: record,
      editCalories: String(Math.round(record.result.calories || 0)),
      editWeight: String(Math.round(record.result.weight || 0)),
      editCategory: categoryIndex >= 0 ? CATEGORY_ZH[CATEGORY_IDS[categoryIndex]] : (record.result.category_name || ''),
      categoryIndex: categoryIndex,
      confirmed: !!record.feedbackSubmitted
    })
  },

  modelCategoryId(r) {
    const res = r.result || {}
    if (typeof res.category_idx === 'number' && res.category_idx >= 0 && res.category_idx < CATEGORY_IDS.length) {
      return CATEGORY_IDS[res.category_idx]
    }
    return zhToId(res.category_name)
  },

  // 确认记录（主操作）
  confirmRecord() {
    this.submitFeedback('ok', 'confirm_only', {})
  },

  onGradeSelect(e) {
    const grade = e.currentTarget.dataset.grade
    const quality = grade === 'ok' ? 'confirm_only' : 'directional'
    this.submitFeedback(grade, quality, {})
  },

  toggleManual() {
    this.setData({ showManual: !this.data.showManual })
  },

  onCaloriesInput(e) { this.setData({ editCalories: e.detail.value }) },
  onWeightInput(e) { this.setData({ editWeight: e.detail.value }) },
  onCategoryChange(e) {
    const idx = parseInt(e.detail.value)
    if (idx >= 0 && idx < CATEGORY_IDS.length) {
      this.setData({ categoryIndex: idx, editCategory: CATEGORY_ZH[CATEGORY_IDS[idx]] })
    }
  },

  // 用户手动填写的修正
  submitManual() {
    const record = this.data.record
    const catId = this.data.categoryIndex >= 0 ? CATEGORY_IDS[this.data.categoryIndex] : null
    this.submitFeedback(this.data.userGrade || 'manual', 'manual_typed', {
      corrected_calories: parseFloat(this.data.editCalories) || null,
      corrected_weight: parseFloat(this.data.editWeight) || null,
      corrected_category: catId
    })
  },

  submitFeedback(grade, quality, overrides) {
    const r = this.data.record
    const payload = {
      dish_uuid: r.dish_uuid || r.id,
      image_hash: r.image_hash || null,
      prediction_id: r.id,
      user_grade: grade,
      quality: quality,
      corrected_calories: overrides.corrected_calories != null ? overrides.corrected_calories : null,
      corrected_weight: overrides.corrected_weight != null ? overrides.corrected_weight : null,
      corrected_category: overrides.corrected_category != null ? overrides.corrected_category : null,
      model_version: r.result.model_version || null,
      model_calories: r.result.calories || null,
      model_weight: r.result.weight || null,
      model_category: this.modelCategoryId(r)
    }

    this.setData({ loading: true })
    wx.request({
      url: app.globalData.apiBase + '/feedback',
      method: 'POST',
      header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' },
      data: payload,
      success: (res) => {
        if (res.statusCode === 200) {
          // 更新本地记录
          r.grade = grade
          r.feedbackQuality = quality
          r.feedbackSubmitted = true
          if (quality === 'manual_typed') {
            r.corrected = true
            r.corrected_calories = payload.corrected_calories
            r.corrected_weight = payload.corrected_weight
            r.corrected_category = payload.corrected_category
            r.result.category_id = payload.corrected_category
            if (payload.corrected_calories != null) r.result.calories = payload.corrected_calories
            if (payload.corrected_weight != null) r.result.weight = payload.corrected_weight
            const catId = payload.corrected_category
            if (catId && CATEGORY_ZH[catId]) r.result.category_name = CATEGORY_ZH[catId]
          }
          app.saveHistory(r)
          this.setData({ confirmed: true, userGrade: grade, showManual: false, celebrating: true })
          wx.showToast({ title: '已记录这一餐', icon: 'success' })
          setTimeout(() => wx.navigateBack(), 900)
        } else {
          this.saveLocalOnly(r)
          wx.showToast({ title: '已保存到本地', icon: 'none' })
        }
      },
      fail: (err) => {
        this.saveLocalOnly(r)
        wx.showModal({ title: '网络不可用', content: '反馈未上传，已保存到本地。', showCancel: false })
        setTimeout(() => wx.navigateBack(), 600)
      },
      complete: () => this.setData({ loading: false })
    })
  },

  // 网络失败时的保底：仅本地保存
  saveLocalOnly(r) {
    r.feedbackSubmitted = true
    r.savedLocal = true
    app.saveHistory(r)
  },

  // 重新选择照片（从头再拍/选一张）
  repick() {
    record.recordMeal(this, 'album')
  }
})
