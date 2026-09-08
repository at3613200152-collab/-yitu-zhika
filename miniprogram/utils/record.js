// utils/record.js - 记录一餐的公共流程（今天页 / 识别确认页共用）
// 拍照或选图 → 生成 dish_uuid/image_hash → 落盘 → 上传 → 存历史 → 跳「识别与确认」

function generateDishUuid() {
  const t = Date.now()
  const r = Math.floor(Math.random() * 0x100000000)
  const r2 = Math.floor(Math.random() * 0x100000000)
  const hex = (n, l) => n.toString(16).padStart(l, '0')
  return `${hex(t, 8)}-${hex(r, 4)}-4${hex(r & 0xfff, 3)}-${hex(8 + (r2 & 0x3), 1)}${hex(r2 & 0xfff, 3)}-${hex(r2 & 0xffffffff, 12)}`
}

function computeImageHash(filePath) {
  return new Promise((resolve) => {
    wx.getFileInfo({
      filePath: filePath,
      success: (info) => resolve(`size_${info.size}_${Date.now().toString(36)}`),
      fail: () => resolve(`unknown_${Date.now().toString(36)}`)
    })
  })
}

// 落盘到用户数据目录，避免重启后缩略图失效；保留最近 60 张
function savePermanent(filePath) {
  return new Promise((resolve) => {
    const fs = wx.getFileSystemManager()
    const dest = `${wx.env.USER_DATA_PATH}/dish_${Date.now()}_${Math.floor(Math.random() * 1e6)}.jpg`
    const fallback = () => resolve(filePath)
    try {
      fs.saveFile({ tempFilePath: filePath, filePath: dest, success: () => { prune(); resolve(dest) }, fail: fallback })
    } catch (e) { fallback() }
  })
}

function prune() {
  try {
    const fs = wx.getFileSystemManager()
    const dir = wx.env.USER_DATA_PATH
    const list = fs.readdirSync(dir).filter(n => n.indexOf('dish_') === 0)
    if (list.length <= 60) return
    list.sort()
    list.slice(0, list.length - 60).forEach(n => { try { fs.unlinkSync(dir + '/' + n) } catch (e) {} })
  } catch (e) {}
}

function chooseImage(sourceType) {
  return new Promise((resolve, reject) => {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: [sourceType],
      camera: 'back',
      success: (res) => resolve(res.tempFiles[0].tempFilePath),
      fail: (err) => {
        if ((err.errMsg || '').indexOf('cancel') >= 0) resolve(null)
        else reject(new Error(err.errMsg || '选择失败'))
      }
    })
  })
}

function upload(filePath) {
  const app = getApp()
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: app.globalData.apiBase + '/predict',
      filePath: filePath,
      name: 'image',
      header: { 'X-API-Key': app.globalData.apiKey },
      success: (res) => {
        try {
          const data = JSON.parse(res.data)
          if (data.status === 'ok') resolve(data)
          else reject(new Error(data.message || '未识别到食物'))
        } catch (e) {
          reject(new Error('服务器响应格式错误'))
        }
      },
      fail: (err) => reject(new Error('网络错误：' + err.errMsg))
    })
  })
}

// 供页面调用：page.setData({loading}) 由页面自身控制
async function recordMeal(page, sourceType) {
  const app = getApp()
  page.setData({ loading: true })
  try {
    const path = await chooseImage(sourceType)
    if (!path) { page.setData({ loading: false }); return }
    const dishUuid = generateDishUuid()
    const imageHash = await computeImageHash(path)
    const savedPath = await savePermanent(path)
    const result = await upload(savedPath)
    const record = {
      id: 'pred_' + Date.now(),
      dish_uuid: dishUuid,
      image_hash: imageHash,
      timestamp: new Date().toISOString(),
      imagePath: savedPath,
      result: result
    }
    app.saveHistory(record)
    wx.navigateTo({ url: '/pages/result/result?id=' + record.id })
  } catch (e) {
    wx.showModal({ title: '没能识别', content: e.message || '请重试或手动记录', showCancel: false })
  } finally {
    page.setData({ loading: false })
  }
}

module.exports = { recordMeal, generateDishUuid, computeImageHash, savePermanent, upload }
