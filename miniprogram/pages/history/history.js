// pages/history/history.js - 历史记录列表（含打卡摘要）
const app = getApp()

function keyOf(date) {
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const dd = String(date.getDate()).padStart(2, '0')
  return `${date.getFullYear()}-${m}-${dd}`
}

Page({
  data: {
    history: [],
    total: 0,
    streak: 0,
    daysInWeek: 0,
    weekTotal: 7,
    weekDots: [1, 2, 3, 4, 5, 6, 7]
  },

  onShow() {
    this.setData({ history: app.globalData.history })
    this.computeSummary()
  },

  computeSummary() {
    const history = app.globalData.history || []
    const dateSet = new Set()
    history.forEach(r => {
      if (r.timestamp) dateSet.add(String(r.timestamp).slice(0, 10))
    })
    this.setData({
      total: history.length,
      streak: this.calcStreak(dateSet),
      daysInWeek: this.daysInWeek(dateSet)
    })
  },

  calcStreak(dateSet) {
    let streak = 0
    const cur = new Date()
    if (!dateSet.has(keyOf(cur))) {
      cur.setDate(cur.getDate() - 1)
      if (!dateSet.has(keyOf(cur))) return 0
    }
    while (dateSet.has(keyOf(cur))) { streak++; cur.setDate(cur.getDate() - 1) }
    return streak
  },

  daysInWeek(dateSet) {
    const now = new Date()
    const day = now.getDay() === 0 ? 7 : now.getDay()
    let count = 0
    for (let i = 1; i <= day; i++) {
      const d = new Date(now)
      d.setDate(now.getDate() - (day - i))
      if (dateSet.has(keyOf(d))) count++
    }
    return count
  },

  viewDetail(e) {
    const id = e.currentTarget.dataset.id
    wx.navigateTo({ url: '/pages/result/result?id=' + id })
  },

  clearHistory() {
    wx.showModal({
      title: '确认清空',
      content: '将删除所有历史记录，不可恢复',
      success: (res) => {
        if (res.confirm) {
          app.clearHistory()
          this.setData({ history: [], total: 0, streak: 0, daysInWeek: 0 })
          wx.showToast({ title: '已清空', icon: 'success' })
        }
      }
    })
  }
})
