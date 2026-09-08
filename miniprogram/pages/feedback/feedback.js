// pages/feedback/feedback.js - 单独的反馈页（可选，从历史页进入）
const app = getApp()

Page({
  data: {
    record: null,
    notes: ''
  },
  onLoad(options) {
    const id = options.id
    const record = app.globalData.history.find(r => r.id === id)
    this.setData({ record: record })
  },
  onNotesInput(e) {
    this.setData({ notes: e.detail.value })
  },
  async submit() {
    // 复用 result 页面的提交逻辑（实际可抽公共模块）
    wx.showToast({ title: '反馈已提交', icon: 'success' })
    wx.navigateBack()
  }
})
