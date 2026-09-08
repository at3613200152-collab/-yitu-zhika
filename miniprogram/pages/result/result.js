// pages/result/result.js - 三档点击 + 折叠手动改 + 跳过 + quality 分级
const app = getApp()

Page({
  data: {
    record: null,
    editing: false,
    // 三档点击状态
    userGrade: '',  // 'over' | 'ok' | 'under' | ''
    // 折叠手动改
    showManualForm: false,
    // 手动修正表单
    editCalories: '',
    editWeight: '',
    editCategory: '',
    categories: ['mixed', 'vegetable', 'other', 'egg', 'meat', 'grain', 'dairy', 'seafood', 'sauce_condiment', 'dessert'],
    categoryIndex: -1
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
    this.setData({
      record: record,
      editCalories: String(Math.round(record.result.calories || 0)),
      editWeight: String(Math.round(record.result.weight || 0)),
      editCategory: record.result.category_name || '',
      categoryIndex: this.data.categories.indexOf(record.result.category_name || '')
    })
  },

  // 三档点击：选择"偏多/差不多/偏少"
  async onGradeSelect(e) {
    const grade = e.currentTarget.dataset.grade
    this.setData({ userGrade: grade })

    // grade=ok → confirm_only 质量
    // grade=over/under → fine_adjust（但未手动输入具体值，仅方向性反馈）
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
      model_category: record.result.category_name || null
    }

    try {
      const res = await this.callFeedbackApi(payload)
      if (res.statusCode === 200) {
        if (grade === 'ok') {
          wx.showToast({ title: '已确认', icon: 'success' })
        } else {
          wx.showToast({ title: '反馈已记录', icon: 'success' })
        }
        record.grade = grade
        record.feedbackSubmitted = true
        app.saveHistory(record)
        setTimeout(() => wx.navigateBack(), 800)
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
    this.setData({
      categoryIndex: idx,
      editCategory: this.data.categories[idx]
    })
  },

  // 手动修正提交（manual_typed 质量）
  async submitManualFeedback() {
    const record = this.data.record
    const payload = {
      dish_uuid: record.dish_uuid || record.id,
      image_hash: record.image_hash || null,
      prediction_id: record.id,
      user_grade: this.data.userGrade || 'manual',
      quality: 'manual_typed',
      corrected_calories: parseFloat(this.data.editCalories) || null,
      corrected_weight: parseFloat(this.data.editWeight) || null,
      corrected_category: this.data.editCategory || null,
      model_version: record.result.model_version || null,
      model_calories: record.result.calories || null,
      model_weight: record.result.weight || null,
      model_category: record.result.category_name || null
    }

    try {
      const res = await this.callFeedbackApi(payload)
      if (res.statusCode === 200) {
        wx.showToast({ title: '已提交', icon: 'success' })
        record.result.calories = payload.corrected_calories
        record.result.weight = payload.corrected_weight
        record.result.category_name = payload.corrected_category
        record.corrected = true
        record.feedbackSubmitted = true
        app.saveHistory(record)
        this.setData({ showManualForm: false, editing: false })
        setTimeout(() => wx.navigateBack(), 800)
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
      model_category: record.result.category_name || null
    }

    try {
      await this.callFeedbackApi(payload)
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
    record.result.calories = parseFloat(this.data.editCalories) || record.result.calories
    record.result.weight = parseFloat(this.data.editWeight) || record.result.weight
    record.result.category_name = this.data.editCategory
    record.savedLocal = true
    app.saveHistory(record)
    wx.showToast({ title: '已保存到本地', icon: 'success' })
    this.setData({ showManualForm: false, record: record })
  }
})
