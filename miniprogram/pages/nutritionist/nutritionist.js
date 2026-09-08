// pages/nutritionist/nutritionist.js - 营养师页面
const app = getApp()

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
    if (idx >= 0) {
      list.splice(idx, 1)
    } else {
      list.push(v)
    }
    this.setData({ allergies: list })
  },

  async generatePlan() {
    const d = this.data
    if (!d.height || !d.weight || !d.age) {
      wx.showToast({ title: '请填写基本信息', icon: 'none' })
      return
    }

    this.setData({ loading: true })

    const payload = {
      height_cm: parseFloat(d.height),
      weight_kg: parseFloat(d.weight),
      age: parseInt(d.age),
      gender: d.gender,
      activity_level: d.activityValues[d.activityIndex],
      goal: d.goal,
      allergies: d.allergies
    }

    try {
      const res = await this.callWeeklyPlan(payload)
      if (res.statusCode === 200 && res.data.status === 'ok') {
        this.setData({ plan: res.data })
        wx.showToast({ title: '生成成功', icon: 'success' })
      } else if (res.data.status === 'refused') {
        wx.showModal({
          title: '无法生成',
          content: res.data.disclaimer || '特殊人群需营养师审核',
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

  resetPlan() {
    this.setData({ plan: null })
  }
})
