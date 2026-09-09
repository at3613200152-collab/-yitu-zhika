// app.js - 一图知卡小程序入口
App({
  globalData: {
    apiBase: 'http://10.21.202.139:8000',  // 模拟器测试用 localhost，真机需 HTTPS 域名
    apiKey: 'dev-key-change-in-prod',  // 开发 key，生产必须改
    userInfo: null,
    userProfile: null,  // onboarding 问卷结果
    history: []
  },

  onLaunch() {
    // 微信登录：拿 wx.login 的 code 换去标识化 participant_id（后端 app/auth.py）
    this.wxLogin()

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

  // 微信登录：失败/未配置时静默降级，记录流程仍可用本地编号
  wxLogin() {
    wx.login({
      success: (res) => {
        if (!res.code) return
        wx.request({
          url: this.globalData.apiBase + '/auth/wx-login',
          method: 'POST',
          header: { 'X-API-Key': this.globalData.apiKey, 'Content-Type': 'application/json' },
          data: { code: res.code },
          success: (r) => {
            try {
              if (r.data && r.data.participant_id) {
                wx.setStorageSync('participant_id', r.data.participant_id)
              }
            } catch (e) {}
          },
          fail: () => {}
        })
      },
      fail: () => {}
    })
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
