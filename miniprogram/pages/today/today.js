// pages/today/today.js - 今天吃了什么？
const app = getApp()
const record = require('../../utils/record.js')

function todayKey() {
  const d = new Date()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  return `${d.getFullYear()}-${m}-${dd}`
}

Page({
  data: {
    loading: false,
    dateLabel: '',
    todayMeals: [],
    mealCount: 0,
    sumCalories: 0,
    hasAny: false
  },

  onShow() {
    this.refresh()
  },

  refresh() {
    const history = app.globalData.history || []
    const key = todayKey()
    const dateLabel = this.formatToday()
    const todayMeals = history.filter(r => r.timestamp && String(r.timestamp).slice(0, 10) === key)
    const sumCalories = todayMeals.reduce((s, r) => s + (r.result && r.result.calories ? r.result.calories : 0), 0)
    this.setData({
      dateLabel: dateLabel,
      todayMeals: todayMeals,
      mealCount: todayMeals.length,
      sumCalories: Math.round(sumCalories),
      hasAny: todayMeals.length > 0
    })
  },

  formatToday() {
    const d = new Date()
    const week = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'][d.getDay()]
    return `${d.getMonth() + 1}月${d.getDate()}日 ${week}`
  },

  takePhoto() {
    record.recordMeal(this, 'camera')
  },

  chooseAlbum() {
    record.recordMeal(this, 'album')
  },

  goResult(e) {
    const id = e.currentTarget.dataset.id
    wx.navigateTo({ url: '/pages/result/result?id=' + id })
  },

  goHistory() {
    wx.navigateTo({ url: '/pages/history/history' })
  }
})
