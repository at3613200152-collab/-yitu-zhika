// pages/about/about.js - 关于页（含隐私说明）
Page({
  data: {},
  clearData() {
    wx.showModal({
      title: '清除所有数据',
      content: '将清空所有本地历史记录和缓存，确认继续？',
      success: (res) => {
        if (res.confirm) {
          const app = getApp()
          app.clearHistory()
          wx.removeStorageSync('predict_history')
          wx.showToast({ title: '已清除', icon: 'success' })
        }
      }
    })
  }
})
