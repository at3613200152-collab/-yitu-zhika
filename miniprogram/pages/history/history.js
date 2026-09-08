// pages/history/history.js - 历史记录列表
const app = getApp()

Page({
  data: {
    history: []
  },

  onShow() {
    // 每次进入页面刷新
    this.setData({ history: app.globalData.history })
  },

  // 点击进入结果页
  viewDetail(e) {
    const id = e.currentTarget.dataset.id
    wx.navigateTo({
      url: '/pages/result/result?id=' + id
    })
  },

  // 清空历史
  clearHistory() {
    wx.showModal({
      title: '确认清空',
      content: '将删除所有历史记录，不可恢复',
      success: (res) => {
        if (res.confirm) {
          app.clearHistory()
          this.setData({ history: [] })
          wx.showToast({ title: '已清空', icon: 'success' })
        }
      }
    })
  }
})
