// pages/result/result.js - 识别与确认（含内测授权 + 自愿标注入口）
const app = getApp()
const record = require('../../utils/record.js')

const CATEGORY_IDS = [
  'dairy','dessert','egg','grain','meat','mixed','other',
  'sauce_condiment','seafood','soup_stew','vegetable'
]
const CATEGORY_ZH = {
  dairy:'乳制品', dessert:'甜点', egg:'蛋类', grain:'谷物主食', meat:'肉类',
  mixed:'混合餐食', other:'其他', sauce_condiment:'酱料调味品', seafood:'水产',
  soup_stew:'汤炖菜', vegetable:'蔬菜'
}
const CONSENT_VERSION = '2026-09-08-v1'

function zhToId(zhName) {
  if (!zhName) return null
  if (CATEGORY_IDS.indexOf(zhName) >= 0) return zhName
  return CATEGORY_IDS.find(id => CATEGORY_ZH[id] === zhName) || null
}

function uuid() {
  const t = Date.now(), r = Math.floor(Math.random() * 0x100000000), r2 = Math.floor(Math.random() * 0x100000000)
  const hex = (n, l) => n.toString(16).padStart(l, '0')
  return `${hex(t, 8)}-${hex(r, 4)}-4${hex(r & 0xfff, 3)}-${hex(8 + (r2 & 0x3), 1)}${hex(r2 & 0xfff, 3)}-${hex(r2 & 0xffffffff, 12)}`
}

// 稳定去标识化参与者编号（本地生成，与身份分离）
function participantId() {
  let pid = wx.getStorageSync('participant_id')
  if (!pid) { pid = 'u_' + uuid().slice(0, 8) + '_' + Date.now().toString(36); wx.setStorageSync('participant_id', pid) }
  return pid
}

Page({
  data: {
    record: null,
    categories: CATEGORY_IDS.map(id => CATEGORY_ZH[id]),
    categoryIndex: -1,
    editCalories: '', editWeight: '', editCategory: '',
    userGrade: '', showManual: false,
    confirmed: false, celebrating: false, loading: false,
    // 内测：训练授权（默认不勾选）+ 自愿标注
    consent: false, consentVersion: CONSENT_VERSION,
    showAnnotate: false,
    annName: '', annIngredients: '', annWeight: '', annCooking: '', annImage: ''
  },

  onLoad(options) {
    const id = options.id
    const record = app.globalData.history.find(r => r.id === id)
    if (!record) {
      wx.showModal({ title: '提示', content: '没有找到这条记录', showCancel: false, success: () => wx.navigateBack() })
      return
    }
    if (!record.capture_session_id) record.capture_session_id = uuid()
    let catId = null
    const idx = record.result && record.result.category_idx
    if (typeof idx === 'number' && idx >= 0 && idx < CATEGORY_IDS.length) catId = CATEGORY_IDS[idx]
    else if (record.result && record.result.category_name) catId = zhToId(record.result.category_name)
    const categoryIndex = catId ? CATEGORY_IDS.indexOf(catId) : -1
    this.setData({
      record,
      editCalories: String(Math.round(record.result.calories || 0)),
      editWeight: String(Math.round(record.result.weight || 0)),
      editCategory: categoryIndex >= 0 ? CATEGORY_ZH[CATEGORY_IDS[categoryIndex]] : (record.result.category_name || ''),
      categoryIndex,
      confirmed: !!record.feedbackSubmitted,
      consent: !!wx.getStorageSync('training_consent'),   // 尊重用户在"我的"里的全局授权设置
      // 负值/异常字段（不作为正常营养值展示，显示"待确认/异常"）
      abnormalProtein: (record.result.abnormal_fields || []).includes('protein_g'),
      abnormalCarb: (record.result.abnormal_fields || []).includes('carbohydrate_g'),
      abnormalFat: (record.result.abnormal_fields || []).includes('fat_g'),
      abnormalKcal: (record.result.abnormal_fields || []).includes('calories'),
      valuesNote: record.result.values_note || ''
    })
  },

  toggleConsent() { this.setData({ consent: !this.data.consent }) },

  modelCategoryId(r) {
    const res = r.result || {}
    if (typeof res.category_idx === 'number' && res.category_idx >= 0 && res.category_idx < CATEGORY_IDS.length) {
      return CATEGORY_IDS[res.category_idx]
    }
    return zhToId(res.category_name)
  },

  // 确认记录（主操作）→ /record（含授权 + 溯源）
  confirmRecord() {
    const r = this.data.record
    const payload = {
      dish_uuid: r.dish_uuid || r.id,
      image_hash: r.image_hash || null,
      prediction_id: r.id,
      participant_id: participantId(),
      capture_session_id: r.capture_session_id,
      model_version: r.result.model_version || null,
      model_category: this.modelCategoryId(r),
      model_calories: r.result.calories || null,
      model_weight: r.result.weight || null,
      model_category_probs: r.result.category_probs || null,
      model_category_prob: r.result.category_prob || null,
      quality: 'confirm_only',
      training_consent: !!this.data.consent,
      consent_version: this.data.consentVersion
    }
    this.recordPost('/record', payload)
  },

  onGradeSelect(e) {
    const r = this.data.record
    const grade = e.currentTarget.dataset.grade
    // 快速反馈走 /feedback（only 方向性），不进训练池
    const payload = {
      dish_uuid: r ? (r.dish_uuid || r.id) : null,
      user_grade: grade,
      quality: grade === 'ok' ? 'confirm_only' : 'directional',
      prediction_id: r ? r.id : null
    }
    wx.request({
      url: app.globalData.apiBase + '/feedback',
      method: 'POST',
      header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' },
      data: payload,
      success: () => { this.setData({ userGrade: grade }); wx.showToast({ title: '已记录', icon: 'success' }) },
      fail: () => { this.saveLocalOnly(r); wx.showToast({ title: '已保存到本地', icon: 'none' }) }
    })
  },

  toggleManual() { this.setData({ showManual: !this.data.showManual }) },
  onCaloriesInput(e) { this.setData({ editCalories: e.detail.value }) },
  onWeightInput(e) { this.setData({ editWeight: e.detail.value }) },
  onCategoryChange(e) {
    const idx = parseInt(e.detail.value)
    if (idx >= 0 && idx < CATEGORY_IDS.length) this.setData({ categoryIndex: idx, editCategory: CATEGORY_ZH[CATEGORY_IDS[idx]] })
  },
  submitManual() {
    const r = this.data.record
    const catId = this.data.categoryIndex >= 0 ? CATEGORY_IDS[this.data.categoryIndex] : null
    this.recordPost('/record', {
      dish_uuid: r.dish_uuid || r.id,
      image_hash: r.image_hash || null,
      prediction_id: r.id,
      participant_id: participantId(),
      capture_session_id: r.capture_session_id,
      model_version: r.result.model_version || null,
      model_category: this.modelCategoryId(r),
      model_calories: r.result.calories || null,
      model_weight: r.result.weight || null,
      quality: 'manual_typed',
      corrected_calories: parseFloat(this.data.editCalories) || null,
      corrected_weight: parseFloat(this.data.editWeight) || null,
      corrected_category: catId,
      mass_basis: 'as_served',
      training_consent: !!this.data.consent,
      consent_version: this.data.consentVersion
    })
  },

  recordPost(url, payload) {
    const r = this.data.record
    this.setData({ loading: true })
    wx.request({
      url: app.globalData.apiBase + url,
      method: 'POST',
      header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' },
      data: payload,
      success: (res) => {
        if (res.statusCode === 200) {
          r.grade = payload.user_grade || 'ok'
          r.feedbackQuality = payload.quality
          r.feedbackSubmitted = true
          if (payload.corrected_calories != null) r.result.calories = payload.corrected_calories
          if (payload.corrected_weight != null) r.result.weight = payload.corrected_weight
          if (payload.corrected_category && CATEGORY_ZH[payload.corrected_category]) { r.result.category_name = CATEGORY_ZH[payload.corrected_category]; r.result.category_id = payload.corrected_category }
          app.saveHistory(r)
          this.setData({ confirmed: true, showManual: false, celebrating: true })
          wx.showToast({ title: url === '/record' ? '已记录这一餐' : '已保存', icon: 'success' })
          setTimeout(() => wx.navigateBack(), 900)
        } else {
          this.saveLocalOnly(r)
          wx.showToast({ title: '已保存到本地', icon: 'none' })
        }
      },
      fail: () => {
        this.saveLocalOnly(r)
        wx.showModal({ title: '网络不可用', content: '未上传成功，已保存到本地。', showCancel: false })
        setTimeout(() => wx.navigateBack(), 600)
      },
      complete: () => this.setData({ loading: false })
    })
  },

  saveLocalOnly(r) {
    r.feedbackSubmitted = true
    r.savedLocal = true
    app.saveHistory(r)
  },

  repick() { record.recordMeal(this, 'album') },

  // ---- 自愿标注（独立、可跳过）----
  toggleAnnotate() { this.setData({ showAnnotate: !this.data.showAnnotate }) },
  onAnnField(e) { this.setData({ [e.currentTarget.dataset.key]: e.detail.value }) },
  chooseAnnImage() {
    wx.chooseMedia({ count: 1, mediaType: ['image'], sourceType: ['album'], success: (res) => this.setData({ annImage: res.tempFiles[0].tempFilePath }) })
  },
  submitAnnotate() {
    const r = this.data.record
    const d = this.data
    const fields = {
      record_id: r.id,
      capture_session_id: r.capture_session_id,
      dish_uuid: r.dish_uuid || r.id,
      label_name: d.annName,
      ingredients: d.annIngredients ? d.annIngredients.split(/[,，、]/).map(s => s.trim()).filter(Boolean) : [],
      measured_weight: parseFloat(d.annWeight) || null,
      weight_unit: 'g',
      mass_basis: 'as_served',
      cooking_method: d.annCooking || null,
      participant_id: participantId(),
      training_consent: d.consent,
      consent_version: d.consentVersion
    }
    if (!fields.label_name && !fields.measured_weight && !d.annImage) {
      wx.showToast({ title: '可都留空，直接跳过', icon: 'none' }); return
    }
    const upload = d.annImage
    if (upload) {
      const formData = {}
      Object.keys(fields).forEach(k => {
        const v = fields[k]
        formData[k] = Array.isArray(v) ? v.join(',') : (v === null || v === undefined) ? '' : String(v)
      })
      wx.uploadFile({
        url: app.globalData.apiBase + '/annotate',
        filePath: upload, name: 'image',
        header: { 'X-API-Key': app.globalData.apiKey },
        formData: formData,
        success: (res) => this.annDone(res),
        fail: () => wx.showToast({ title: '上传失败，可稍后再试', icon: 'none' })
      })
    } else {
      wx.request({
        url: app.globalData.apiBase + '/annotate',
        method: 'POST',
        header: { 'X-API-Key': app.globalData.apiKey, 'Content-Type': 'application/json' },
        data: fields,
        success: (res) => this.annDone(res),
        fail: () => wx.showToast({ title: '上传失败，可稍后再试', icon: 'none' })
      })
    }
    this.setData({ showAnnotate: false, annName: '', annIngredients: '', annWeight: '', annCooking: '', annImage: '' })
  },
  annDone(res) {
    let ok = false
    try { ok = res.statusCode === 200 } catch (e) {}
    wx.showToast({ title: ok ? '谢谢你的帮助' : '已跳过', icon: ok ? 'success' : 'none' })
  }
})
