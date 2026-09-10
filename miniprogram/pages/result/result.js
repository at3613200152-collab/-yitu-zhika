// pages/result/result.js - 识别与确认（含内测授权 + 自愿标注入口）
const app = getApp()
const record = require('../../utils/record.js')

// label_schema v2（12 类，含水果）：与服务端 manifest 顺序一致
// 顺序 = dairy, dessert, egg, fruit, grain, meat, mixed, other, sauce_condiment, seafood, soup_stew, vegetable
const CATEGORY_IDS = [
  'dairy','dessert','egg','fruit','grain','meat','mixed','other',
  'sauce_condiment','seafood','soup_stew','vegetable'
]
// 旧 11 类顺序：历史记录（label_schema=v1_11class）的 category_idx 按此解释，不能混用
const CATEGORY_IDS_LEGACY = [
  'dairy','dessert','egg','grain','meat','mixed','other',
  'sauce_condiment','seafood','soup_stew','vegetable'
]
const CATEGORY_ZH = {
  dairy:'乳制品', dessert:'甜点', egg:'蛋类', fruit:'水果', grain:'谷物主食', meat:'肉类',
  mixed:'混合餐食', other:'其他', sauce_condiment:'酱料调味品', seafood:'水产',
  soup_stew:'汤炖菜', vegetable:'蔬菜'
}
const CONSENT_VERSION = '2026-09-08-v1'

function zhToId(zhName) {
  if (!zhName) return null
  if (CATEGORY_IDS.indexOf(zhName) >= 0) return zhName
  return CATEGORY_IDS.find(id => CATEGORY_ZH[id] === zhName) || null
}

// 兼容两版标签体系：优先按中文名（与 schema 无关），再用 category_idx 按对应版本顺序解释。
// 若直接按新数组解释旧记录的 idx，会把"谷物"错读成"水果"。
function categoryIdFromResult(res) {
  if (!res) return null
  const byName = zhToId(res.category_name)
  if (byName) return byName
  const idx = res.category_idx
  if (typeof idx === 'number' && idx >= 0) {
    const ids = res.category_label_schema === 'v2_fruit_12class' ? CATEGORY_IDS : CATEGORY_IDS_LEGACY
    if (idx < ids.length) return ids[idx]
  }
  return null
}

function displayNumber(value) {
  if (value === null || value === undefined || value === '') return null
  const n = Number(value)
  return Number.isFinite(n) ? Math.round(n * 10) / 10 : null
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
    displayNutrition: {},
    editCalories: '', editWeight: '',
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
    this.setData({
      record,
      displayNutrition: {
        calories: displayNumber(record.result.calories),
        weight: displayNumber(record.result.weight),
        protein: displayNumber(record.result.protein_g),
        carbohydrate: displayNumber(record.result.carbohydrate_g),
        fat: displayNumber(record.result.fat_g)
      },
      editCalories: String(Math.round(record.result.calories || 0)),
      editWeight: String(Math.round(record.result.weight || 0)),
      // 决策 C：类别来自独立类别模型（12 类，含水果），热量/宏量来自五目标模型
      categorySourceNote: record.result.category_label_schema === 'v2_fruit_12class'
        ? '类别建议来自独立的 12 类模型；热量与宏量来自五目标模型。'
        : '',
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
    return categoryIdFromResult(r.result || {})
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
      // 决策 C 溯源：类别来自独立类别模型，需随记录落库以区分 11/12 类口径
      category_model: r.result.category_model || null,
      category_label_schema: r.result.category_label_schema || null,
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
  submitManual() {
    const r = this.data.record
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
      category_model: r.result.category_model || null,
      category_label_schema: r.result.category_label_schema || null,
      quality: 'manual_typed',
      corrected_calories: parseFloat(this.data.editCalories) || null,
      corrected_weight: parseFloat(this.data.editWeight) || null,
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
          this.applyCorrections(r, payload)
          app.saveHistory(r)
          this.setData({ confirmed: true, showManual: false, celebrating: true })
          wx.showToast({ title: url === '/record' ? '已记录这一餐' : '已保存', icon: 'success' })
          setTimeout(() => wx.navigateBack(), 900)
        } else {
          this.saveLocalOnly(r, payload)
          wx.showToast({ title: '已保存到本地', icon: 'none' })
        }
      },
      fail: () => {
        this.saveLocalOnly(r, payload)
        wx.showModal({ title: '网络不可用', content: '未上传成功，已保存到本地。', showCancel: false })
        setTimeout(() => wx.navigateBack(), 600)
      },
      complete: () => this.setData({ loading: false })
    })
  },

  applyCorrections(r, payload) {
    if (!r || !payload) return
    if (payload.corrected_calories != null) r.result.calories = payload.corrected_calories
    if (payload.corrected_weight != null) r.result.weight = payload.corrected_weight
    if (Array.isArray(payload.corrected_categories)) {
      r.result.confirmed_category_ids = payload.corrected_categories.slice()
      r.result.confirmed_category_names = payload.corrected_categories.map(id => CATEGORY_ZH[id]).filter(Boolean)
    }
  },

  saveLocalOnly(r, payload) {
    this.applyCorrections(r, payload)
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
