const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const root = path.resolve(__dirname, '..')
const read = file => fs.readFileSync(path.join(root, file), 'utf8')
function page(name) {
  let definition
  const app = { globalData: { history: [] } }
  const wx = { getStorageSync: () => null, switchTab: () => {}, request: () => {} }
  vm.runInNewContext(read(`pages/${name}/${name}.js`), {
    Page: p => { definition = p }, getApp: () => app, wx, setTimeout: f => f()
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
