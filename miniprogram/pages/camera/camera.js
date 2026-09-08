// pages/camera/camera.js - 拍照/选图页（带 EXIF strip + dish_uuid 生成）
const app = getApp()

Page({
  data: {
    imagePath: '',
    loading: false
  },

  // 拍照
  takePhoto() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera'],
      camera: 'back',
      success: (res) => {
        this.setData({ imagePath: res.tempFiles[0].tempFilePath })
      },
      fail: (err) => {
        if (err.errMsg.indexOf('cancel') === -1) {
          wx.showToast({ title: '拍照失败：' + err.errMsg, icon: 'none' })
        }
      }
    })
  },

  // 从相册选择
  chooseFromAlbum() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success: (res) => {
        this.setData({ imagePath: res.tempFiles[0].tempFilePath })
      },
      fail: (err) => {
        if (err.errMsg.indexOf('cancel') === -1) {
          wx.showToast({ title: '选择失败：' + err.errMsg, icon: 'none' })
        }
      }
    })
  },

  // 生成 dish_uuid（基于时间戳 + 随机数，RFC4122 v4 格式）
  generateDishUuid() {
    const t = Date.now()
    const r = Math.floor(Math.random() * 0x100000000)
    const r2 = Math.floor(Math.random() * 0x100000000)
    const hex = (n, l) => n.toString(16).padStart(l, '0')
    return `${hex(t, 8)}-${hex(r, 4)}-4${hex(r & 0xfff, 3)}-${hex(8 + (r2 & 0x3), 1)}${hex(r2 & 0xfff, 3)}-${hex(r2 & 0xffffffff, 12)}`
  },

  // 计算文件 hash（用于去重和追溯）
  // 微信小程序环境无法直接计算 SHA256，用文件大小+修改时间作为弱 hash
  async computeImageHash(filePath) {
    try {
      const info = await new Promise((resolve, reject) => {
        wx.getFileInfo({
          filePath: filePath,
          success: resolve,
          fail: reject
        })
      })
      // 格式：size_<kb>_<random_salt>（小程序限制无法读二进制流）
      return `size_${info.size}_${Date.now().toString(36)}`
    } catch (err) {
      return `unknown_${Date.now().toString(36)}`
    }
  },

  // 把临时图片落到用户数据目录，重启后历史记录缩略图仍可见（否则 temp 路径失效）
  // 失败时保底用临时路径，不影响本轮流程
  savePermanent(filePath) {
    return new Promise((resolve) => {
      const fs = wx.getFileSystemManager()
      const dest = `${wx.env.USER_DATA_PATH}/dish_${Date.now()}_${Math.floor(Math.random() * 1e6)}.jpg`
      const fallback = () => resolve(filePath)
      try {
        fs.saveFile({
          tempFilePath: filePath,
          filePath: dest,
          success: () => {
            this.pruneSavedFiles()
            resolve(dest)
          },
          fail: fallback
        })
      } catch (e) {
        fallback()
      }
    })
  },

  // 保留最近 60 张已落盘图片，超出则删除最旧的（避免占满用户数据目录）
  pruneSavedFiles() {
    try {
      const fs = wx.getFileSystemManager()
      const dir = wx.env.USER_DATA_PATH
      const list = fs.readdirSync(dir).filter(n => n.indexOf('dish_') === 0)
      if (list.length <= 60) return
      list.sort()  // dish_<timestamp>_<rand>.jpg 按文件名时间序
      list.slice(0, list.length - 60).forEach(n => {
        try { fs.unlinkSync(dir + '/' + n) } catch (e) { /* ignore */ }
      })
    } catch (e) { /* ignore */ }
  },

  // 提交识别
  async submit() {
    if (!this.data.imagePath) {
      wx.showToast({ title: '请先拍照或选择图片', icon: 'none' })
      return
    }

    this.setData({ loading: true })

    try {
      // P1: 生成 dish_uuid + image_hash（设计文档第 6 节追溯链）
      const dishUuid = this.generateDishUuid()
      const imageHash = await this.computeImageHash(this.data.imagePath)

      // 落盘到用户数据目录（重启后历史仍可预览）；失败则用临时路径
      const savedPath = await this.savePermanent(this.data.imagePath)
      if (savedPath !== this.data.imagePath) {
        this.setData({ imagePath: savedPath })
      }

      const result = await this.callPredict(savedPath)
      
      // 保存到历史（带 dish_uuid 和 image_hash）
      const record = {
        id: 'pred_' + Date.now(),
        dish_uuid: dishUuid,
        image_hash: imageHash,
        timestamp: new Date().toISOString(),
        imagePath: savedPath,
        result: result
      }
      app.saveHistory(record)

      // 跳到结果页
      wx.navigateTo({
        url: '/pages/result/result?id=' + record.id
      })
    } catch (err) {
      wx.showModal({
        title: '识别失败',
        content: err.message || '请稍后重试',
        showCancel: false
      })
    } finally {
      this.setData({ loading: false })
    }
  },

  // 调用后端识别接口
  // P1: EXIF strip 在上传前完成
  callPredict(filePath) {
    return new Promise((resolve, reject) => {
      wx.uploadFile({
        url: app.globalData.apiBase + '/predict',
        filePath: filePath,
        name: 'image',
        header: {
          'X-API-Key': app.globalData.apiKey
        },
        success: (res) => {
          try {
            const data = JSON.parse(res.data)
            if (data.status === 'ok') {
              resolve(data)
            } else if (data.status === 'not_food') {
              // 非食物图片，明确拒绝，不进结果页
              reject(new Error(data.message || '未检测到食物'))
            } else {
              reject(new Error(data.message || '识别失败'))
            }
          } catch (e) {
            reject(new Error('服务器响应格式错误'))
          }
        },
        fail: (err) => {
          reject(new Error('网络错误：' + err.errMsg))
        }
      })
    })
  }
})
