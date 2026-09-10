const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const root = path.resolve(__dirname, '..')
const read = file => fs.readFileSync(path.join(root, file), 'utf8')
function page(name) {
  let definition
  const app = { globalData: { history: [], userProfile: null } }
  app.clearHistory = () => { app.globalData.history = [] }
  app.saveHistory = () => {}
  const wx = { getStorageSync: () => null, setStorageSync: () => {}, switchTab: () => {}, request: () => {} }
  vm.runInNewContext(read(`pages/${name}/${name}.js`), {
    Page: p => { definition = p }, getApp: () => app, wx, setTimeout: f => f(),
    require: () => ({ recordMeal: () => {} })
  })
  definition.data = JSON.parse(JSON.stringify(definition.data))
  definition.setData = values => Object.assign(definition.data, values)
  return { definition, app, wx }
}
test('milestone template declares the alias used by all six cards', () => {
  const markup = read('pages/contribution/contribution.wxml')
  const loop = markup.match(/<view[^>]*wx:for="{{milestones}}"[^>]*>/)[0]
  const alias = (loop.match(/wx:for-item="([^"]+)"/) || [null, 'item'])[1]
  assert.equal(alias, 'm', 'm.label and m.achieved must bind to the loop item')
  const { definition: p, app } = page('contribution')
  p.computeStats()
  assert.equal(p.data.milestones.length, 6)
  assert.equal(p.data.achievementCount, 0)
  app.globalData.history = [{ timestamp: '2026-09-09 12:00', grade: 'ok' }]
  p.computeStats()
  assert.equal(p.data.achievementCount, 1)
  assert.ok(p.data.milestones.every(m => m.label))
})
test('templates do not call unsupported array methods for chip selection', () => {
  for (const name of ['onboarding', 'nutritionist']) {
    assert.doesNotMatch(read(`pages/${name}/${name}.wxml`), /\.(includes|indexOf)\(/)
  }
})
test('allergen selection exposes selected state and can be toggled off', () => {
  for (const [name, handler, field, value] of [
    ['onboarding', 'toggleAllergen', 'allergenSelected', 'egg'],
    ['nutritionist', 'toggleAllergy', 'allergySelected', '鸡蛋']
  ]) {
    const { definition: p } = page(name)
    const event = { currentTarget: { dataset: { v: value } } }
    p[handler](event)
    assert.equal(p.data[field][value], true)
    p[handler](event)
    assert.equal(Boolean(p.data[field][value]), false)
  }
})
test('skip questionnaire navigates to an actual tab', () => {
  const { definition: p, wx } = page('onboarding')
  wx.setStorageSync = () => {}
  let target
  wx.switchTab = ({ url }) => { target = url.slice(1) }
  p.skipOnboarding()
  assert.ok(JSON.parse(read('app.json')).tabBar.list.some(t => t.pagePath === target))
})

test('result page rounds nutrition values and submits user-confirmed multi labels', () => {
  const { definition: p, app } = page('result')
  app.globalData.history = [{
    id: 'meal-1',
    result: {
      calories: 466.94,
      weight: 352.44,
      protein_g: 47.483219146728516,
      carbohydrate_g: 21.823104858398438,
      fat_g: 19.157428741455078,
      category_name: '蔬菜',
      category_idx: 11,
      category_label_schema: 'v2_fruit_12class',
      category_probs: [
        { id: 'vegetable', name: '蔬菜', pct: 88.6 },
        { id: 'meat', name: '肉类', pct: 5.9 }
      ]
    }
  }]

  p.onLoad({ id: 'meal-1' })
  assert.equal(p.data.displayNutrition.calories, 466.9)
  assert.equal(p.data.displayNutrition.weight, 352.4)
  assert.equal(p.data.displayNutrition.protein, 47.5)
  assert.equal(p.data.displayNutrition.carbohydrate, 21.8)
  assert.equal(p.data.displayNutrition.fat, 19.2)
  assert.deepEqual(Array.from(p.data.selectedCategoryIds || []), [])
  assert.equal(typeof p.toggleCategory, 'undefined')

  let submitted
  p.recordPost = (url, payload) => { submitted = { url, payload } }
  p.confirmRecord()
  assert.equal(submitted.url, '/record')
  assert.equal(submitted.payload.corrected_categories, undefined)
  assert.equal(submitted.payload.quality, 'confirm_only')
})

test('result template no longer renders the multi-label confirmation grid', () => {
  const markup = read('pages/result/result.wxml')
  assert.doesNotMatch(markup, /餐盘包含哪些类别/)
  assert.doesNotMatch(markup, /可多选/)
  // 类别来源说明保留（回归与类别解耦的披露）
  assert.match(markup, /categorySourceNote/)
})

test('result page keeps manual values and multi labels when upload falls back to local', () => {
  const { definition: p } = page('result')
  const meal = { result: { calories: 400, weight: 300 } }
  p.saveLocalOnly(meal, {
    corrected_calories: 466.9,
    corrected_weight: 352.4,
    corrected_categories: ['vegetable', 'meat']
  })
  assert.equal(meal.result.calories, 466.9)
  assert.equal(meal.result.weight, 352.4)
  assert.deepEqual(Array.from(meal.result.confirmed_category_ids), ['vegetable', 'meat'])
  assert.equal(meal.savedLocal, true)
})

test('today and history cards round model numbers to one decimal', () => {
  const now = new Date()
  const m = String(now.getMonth() + 1).padStart(2, '0')
  const d = String(now.getDate()).padStart(2, '0')
  const meal = {
    id: 'meal-2', timestamp: `${now.getFullYear()}-${m}-${d} 12:00`,
    result: { calories: 466.94, weight: 352.44 }
  }
  for (const name of ['today', 'history']) {
    const { definition: p, app } = page(name)
    app.globalData.history = [meal]
    p.onShow()
    assert.equal(p.data[name === 'today' ? 'todayMeals' : 'history'][0].displayCalories, 466.9)
    assert.equal(p.data[name === 'today' ? 'todayMeals' : 'history'][0].displayWeight, 352.4)
  }
})

test('both privacy entry points clear all local identity and consent keys', () => {
  for (const [name, method] of [['about', 'clearData'], ['contribution', 'clearAllData']]) {
    const { definition: p, wx } = page(name)
    const removed = []
    wx.showModal = ({ success }) => success({ confirm: true })
    wx.removeStorageSync = key => removed.push(key)
    wx.showToast = () => {}
    p[method]()
    for (const key of ['predict_history', 'user_profile', 'training_consent', 'participant_id']) {
      assert.ok(removed.includes(key), `${name} should clear ${key}`)
    }
  }
})

test('main pages avoid duplicate native titles and keep a consistent visual system', () => {
  for (const [pageName, duplicateTitle] of [
    ['nutritionist', '饮食计划'],
    ['contribution', '我的'],
    ['history', '历史记录'],
    ['camera', '拍照识别']
  ]) {
    const markup = read(`pages/${pageName}/${pageName}.wxml`)
    assert.doesNotMatch(markup, new RegExp(`class="(?:page-)?title">${duplicateTitle}<`))
  }
  assert.match(read('app.wxss'), /safe-area-inset-bottom/)
  assert.doesNotMatch(read('pages/history/history.wxss'), /#07c160|#f6f7fb/)
  assert.doesNotMatch(read('pages/history/history.wxml'), /#07c160/)
  assert.doesNotMatch(read('pages/today/today.wxml'), /🥣/)
  assert.doesNotMatch(read('pages/result/result.wxml'), /🎉/)
  assert.match(read('pages/nutritionist/nutritionist.wxss'), /\.food-add\s*\{[^}]*flex-direction:\s*column/s)
  for (const name of ['about', 'camera', 'onboarding', 'result', 'today', 'history', 'nutritionist', 'contribution']) {
    assert.equal(JSON.parse(read(`pages/${name}/${name}.json`)).navigationBarBackgroundColor, '#faf7f4')
  }
})
