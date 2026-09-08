// pages/contribution/contribution.js - 我的贡献页面
const app = getApp()

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
    clearing: false
  },

  onLoad() {
    this.computeStats()
  },

  onShow() {
    this.computeStats()
  },

  computeStats() {
    const history = app.globalData.history || []

    // 统计各 quality 分组
    let confirmed = 0  // confirm_only
    let directional = 0  // 偏多/偏少
    let manual = 0  // manual_typed
    let skipped = 0  // skipped
    let total = history.length

    // 计算偏差（仅 manual_typed 有真值）
    let biasSum = 0
    let biasCount = 0
    const recentItems = []

    history.forEach(r => {
      const quality = r.feedbackQuality || (r.corrected ? 'manual_typed' : (r.grade === 'skip' ? 'skipped' : (r.grade ? 'directional' : '')))
      
      if (quality === 'confirm_only') confirmed++
      else if (quality === 'manual_typed') {
        manual++
        // 偏差：|corrected - model| / model
        if (r.result && r.result.calories && r.corrected_calories) {
          const m = r.result.calories
          const c = r.corrected_calories
          if (m > 0) {
            biasSum += Math.abs(c - m) / m
            biasCount++
          }
        }
      }
      else if (quality === 'skipped') skipped++
      else if (quality === 'directional') directional++

      // 最近 10 条
      if (recentItems.length < 10) {
        const qualityLabel = {
          'confirm_only': '差不多',
          'directional': r.grade === 'over' ? '偏多' : '偏少',
          'manual_typed': '手动改',
          'skipped': '跳过'
        }[quality] || '未反馈'
        recentItems.push({
          id: r.id,
          label: new Date(r.timestamp).toLocaleDateString('zh-CN'),
          quality: quality,
          qualityLabel: qualityLabel
        })
      }
    })

    const biasPct = biasCount > 0 ? Math.round(biasSum / biasCount * 100) : '-'
    const skipPct = total > 0 ? Math.round(skipped / total * 100) : 0

    this.setData({
      stats: {
        total,
        confirmed,
        skipped,
        manual,
        skipPct,
        biasPct,
        improved: ''  // 需要后端批次对比才能算
      },
      recentItems
    })
  },

  goCamera() {
    wx.switchTab({ url: '/pages/camera/camera' })
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
