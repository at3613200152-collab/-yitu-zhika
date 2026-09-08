// app.js - 一图知卡小程序入口
App({
  globalData: {
    apiBase: 'http://localhost:8000',  // 模拟器测试用 localhost，真机需 HTTPS 域名
    apiKey: 'dev-key-change-in-prod',  // 开发 key，生产必须改
    userInfo: null,
    userProfile: null,  // onboarding 问卷结果
    history: []
  },

  onLaunch() {
    // 从本地存储恢复历史记录
    try {
      const history = wx.getStorageSync('predict_history') || []
      this.globalData.history = history
    } catch (e) {
      console.error('Failed to load history:', e)
    }

    // 首次进入检查：是否完成 onboarding
    try {
      const profile = wx.getStorageSync('user_profile')
      if (!profile || !profile.completed) {
        // 未完成问卷，跳到 onboarding 页
        wx.redirectTo({ url: '/pages/onboarding/onboarding' })
      } else {
        this.globalData.userProfile = profile
      }
    } catch (e) {
      console.error('Failed to load profile:', e)
    }
  },

  // 保存历史记录
  saveHistory(record) {
    this.globalData.history.unshift(record)
    if (this.globalData.history.length > 50) {
      this.globalData.history = this.globalData.history.slice(0, 50)
    }
    try {
      wx.setStorageSync('predict_history', this.globalData.history)
    } catch (e) {
      console.error('Failed to save history:', e)
    }
  },

  clearHistory() {
    this.globalData.history = []
    wx.removeStorageSync('predict_history')
  }
})
