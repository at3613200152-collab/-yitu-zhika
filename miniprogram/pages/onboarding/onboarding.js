// pages/onboarding/onboarding.js - 首次进入问卷（设计文档第 7 节）
const app = getApp()

Page({
  data: {
    goal: '',  // 'maintain' | 'reduce' | 'gain'
    allergens: [],  // ['egg', 'milk', 'seafood', 'nut', 'gluten', 'spicy']
    allergenSelected: {},
    advice: ''  // 'yes' | 'no'
  },

  onLoad() {
    // 检查是否已完成问卷
    const profile = wx.getStorageSync('user_profile')
    if (profile && profile.completed) {
      // 已完成，直接跳回拍照页
      wx.switchTab({ url: '/pages/today/today' })
    }
  },

  selectGoal(e) {
    this.setData({ goal: e.currentTarget.dataset.v })
  },

  toggleAllergen(e) {
    const v = e.currentTarget.dataset.v
    const list = this.data.allergens.slice()
    const idx = list.indexOf(v)
    if (idx >= 0) {
      list.splice(idx, 1)
    } else {
      list.push(v)
    }
    const allergenSelected = {}
    list.forEach(key => { allergenSelected[key] = true })
    this.setData({ allergens: list, allergenSelected })
  },

  selectAdvice(e) {
    this.setData({ advice: e.currentTarget.dataset.v })
  },

  finishOnboarding() {
    if (!this.data.goal) {
      wx.showToast({ title: '请选择饮食目标', icon: 'none' })
      return
    }
    const profile = {
      completed: true,
      completed_at: new Date().toISOString(),
      goal: this.data.goal,
      allergens: this.data.allergens,
      advice: this.data.advice || 'no'
    }
    wx.setStorageSync('user_profile', profile)
    app.globalData.userProfile = profile
    wx.showToast({ title: '已保存', icon: 'success' })
    setTimeout(() => {
      wx.switchTab({ url: '/pages/today/today' })
    }, 800)
  },

  skipOnboarding() {
    const profile = {
      completed: true,
      completed_at: new Date().toISOString(),
      goal: 'maintain',
      allergens: [],
      advice: 'no',
      skipped: true
    }
    wx.setStorageSync('user_profile', profile)
    app.globalData.userProfile = profile
    wx.switchTab({ url: '/pages/today/today' })
  }
})
