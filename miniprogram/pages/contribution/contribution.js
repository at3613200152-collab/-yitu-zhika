// pages/contribution/contribution.js - 我的贡献 / 成就页
const app = getApp()

function dayKey(ts) {
  if (!ts) return ''
  return String(ts).slice(0, 10)
}

Page({
  data: {
    stats: {
      total: 0,
      confirmed: 0,
      skipped: 0,
      manual: 0,
      skipPct: 0,
      biasPct: '-',
      improved: ''
    },
    recentItems: [],
    clearing: false,
    // 成就感/打卡
    streak: 0,          // 连续记录天数
    totalDays: 0,       // 累计记录天数（去重日期）
    daysInWeek: 0,      // 本周记录天数
    weekTotal: 7,
    weekDots: [1, 2, 3, 4, 5, 6, 7],
    milestones: [],
    achievementCount: 0,
    profile: null
  },

  onLoad() {
    this.computeStats()
    this.loadProfile()
  },

  onShow() {
    this.computeStats()
    this.loadProfile()
  },

  loadProfile() {
    const profile = wx.getStorageSync('user_profile') || app.globalData.userProfile || null
    this.setData({ profile: profile })
  },

  goAbout() {
    wx.navigateTo({ url: '/pages/about/about' })
  },

  // 尚未实现：仅提示，不做假按钮
  exportData() {
    wx.showToast({ title: '数据导出暂未开放', icon: 'none' })
  },
  setReminder() {
    wx.showToast({ title: '提醒功能即将上线', icon: 'none' })
  },

  computeStats() {
    const history = app.globalData.history || []

    let confirmed = 0, directional = 0, manual = 0, skipped = 0
    let total = history.length
    let biasSum = 0, biasCount = 0
    const recentItems = []
    const dateSet = new Set()

    history.forEach(r => {
      const quality = r.feedbackQuality ||
        (r.corrected ? 'manual_typed'
          : r.grade === 'skip' ? 'skipped'
          : r.grade === 'ok' ? 'confirm_only'
          : r.grade ? 'directional' : '')

      if (quality === 'confirm_only') confirmed++
      else if (quality === 'manual_typed') {
        manual++
        if (r.result && r.result.calories && r.corrected_calories) {
          const m = r.result.calories, c = r.corrected_calories
          if (m > 0) { biasSum += Math.abs(c - m) / m; biasCount++ }
        }
      } else if (quality === 'skipped') skipped++
      else if (quality === 'directional') directional++

      const dk = dayKey(r.timestamp)
      if (dk) dateSet.add(dk)

      if (recentItems.length < 10) {
        const qualityLabel = {
          'confirm_only': '差不多', 'directional': r.grade === 'over' ? '偏多' : '偏少',
          'manual_typed': '手动改', 'skipped': '跳过'
        }[quality] || '未反馈'
        recentItems.push({
          id: r.id,
          label: this.friendlyDate(dk),
          quality: quality,
          qualityLabel: qualityLabel
        })
      }
    })

    const biasPct = biasCount > 0 ? Math.round(biasSum / biasCount * 100) : '-'
    const skipPct = total > 0 ? Math.round(skipped / total * 100) : 0
    const totalDays = dateSet.size

    // 连续打卡（从今天往前数连续有记录的日期；今天没记录则从昨天数起）
    const streak = this.computeStreak(dateSet)
    const daysInWeek = this.daysInCurrentWeek(dateSet)

    // 里程碑（成就）
    const milestones = this.buildMilestones({ total, confirmed, manual, streak, totalDays })

    this.setData({
      stats: { total, confirmed, skipped, manual, skipPct, biasPct, improved: '' },
      recentItems,
      streak,
      totalDays,
      daysInWeek,
      milestones,
      achievementCount: milestones.filter(m => m.achieved).length
    })
  },

  computeStreak(dateSet) {
    let streak = 0
    let cur = new Date()
    // 今天有记录则从今天算；否则允许从昨天算（今天还没记录不打断）
    if (!dateSet.has(this.keyOf(cur))) {
      cur.setDate(cur.getDate() - 1)
      if (!dateSet.has(this.keyOf(cur))) return 0
    }
    while (dateSet.has(this.keyOf(cur))) {
      streak++
      cur.setDate(cur.getDate() - 1)
    }
    return streak
  },

  daysInCurrentWeek(dateSet) {
    const now = new Date()
    const day = now.getDay() === 0 ? 7 : now.getDay()  // 周一=1 .. 周日=7
    let count = 0
    for (let i = 1; i <= day; i++) {
      const d = new Date(now)
      d.setDate(now.getDate() - (day - i))
      if (dateSet.has(this.keyOf(d))) count++
    }
    return count
  },

  buildMilestones({ total, confirmed, manual, streak, totalDays }) {
    return [
      { emoji: '🌱', label: '第一次记录', achieved: total >= 1 },
      { emoji: '🔥', label: '连续记录 3 天', achieved: streak >= 3 },
      { emoji: '🏆', label: '连续记录 7 天', achieved: streak >= 7 },
      { emoji: '🍽️', label: '累计记录 10 餐', achieved: total >= 10 },
      { emoji: '✅', label: '确认 10 次', achieved: confirmed >= 10 },
      { emoji: '✍️', label: '手动修正 1 次', achieved: manual >= 1 }
    ]
  },

  friendlyDate(dk) {
    if (!dk) return ''
    const parts = dk.split('-')
    if (parts.length !== 3) return dk
    const d = new Date(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]))
    const week = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'][d.getDay()]
    return `${Number(parts[1])}月${Number(parts[2])}日 ${week}`
  },

  keyOf(date) {
    const m = String(date.getMonth() + 1).padStart(2, '0')
    const dd = String(date.getDate()).padStart(2, '0')
    return `${date.getFullYear()}-${m}-${dd}`
  },

  goCamera() {
    wx.switchTab({ url: '/pages/today/today' })
  },

  clearAllData() {
    wx.showModal({
      title: '确认清除',
      content: '将删除所有本地历史记录和问卷数据，无法恢复',
      success: (res) => {
        if (res.confirm) {
          this.setData({ clearing: true })
          try {
            wx.removeStorageSync('predict_history')
            wx.removeStorageSync('user_profile')
            app.globalData.history = []
            app.globalData.userProfile = null
            wx.showToast({ title: '已清除', icon: 'success' })
            this.computeStats()
          } catch (e) {
            wx.showToast({ title: '清除失败', icon: 'none' })
          }
          this.setData({ clearing: false })
        }
      }
    })
  }
})
