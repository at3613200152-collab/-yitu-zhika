// Browser approximation of WXML/WXSS, NOT a substitute for WeChat device QA.
// Requires playwright and xml-js (set NODE_PATH to your installed packages).
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const { chromium } = require('playwright')
const { xml2js } = require('xml-js')
const root = path.resolve(__dirname, '..')
const read = p => fs.readFileSync(path.join(root, p), 'utf8')
const escape = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;')
function render(name) {
  let p
  vm.runInNewContext(read(`pages/${name}/${name}.js`), {
    Page: v => { p = v }, getApp: () => ({ globalData: { history: [] } }),
    wx: { getStorageSync: () => null }, require: () => ({}), setTimeout: () => {}
  })
  p.setData = values => Object.assign(p.data, values)
  if (name === 'contribution') p.computeStats()
  const expressions = []
  const source = read(`pages/${name}/${name}.wxml`).replace(/{{([\s\S]*?)}}/g, (_, e) => {
    expressions.push(e); return `WXEXPR${expressions.length - 1}END`
  }).replace(/wx:else(?=[\s>])/g, 'wx:else=""')
  const evaluate = (s, data) => {
    const index = /^WXEXPR(\d+)END$/.exec(s)
    if (!index) return s
    try { return vm.runInNewContext(expressions[+index[1]], data) } catch { return undefined }
  }
  const substitute = (s, data) => String(s || '').replace(/WXEXPR\d+END/g, token => escape(evaluate(token, data)))
  function nodes(items, data) {
    let matched = false
    return (items || []).map(n => {
      if (n.type === 'text') return substitute(n.text, data)
      if (n.type !== 'element') return ''
      const a = n.attributes || {}
      if (a['wx:if']) { matched = !!evaluate(a['wx:if'], data); if (!matched) return '' }
      else if ('wx:else' in a && matched) return ''
      if (a['wx:for']) {
        return (evaluate(a['wx:for'], data) || []).map((item, index) => {
          const attributes = { ...a }; delete attributes['wx:for']
          return nodes([{ ...n, attributes }], { ...data, [a['wx:for-item'] || 'item']: item, [a['wx:for-index'] || 'index']: index })
        }).join('')
      }
      if (n.name === 'block') return nodes(n.elements, data)
      const tag = ({ view: 'div', text: 'span', picker: 'div', image: 'img' })[n.name] || n.name
      let attributes = ''
      for (const [key, value] of Object.entries(a)) {
        if (key.startsWith('wx:') || key.startsWith('bind') || key === 'type') continue
        if (key === 'disabled' && !evaluate(value, data)) continue
        if (key === 'src' && value.startsWith('/assets/')) {
          const mime = value.endsWith('.svg') ? 'image/svg+xml' : 'image/png'
          attributes += ` src="data:${mime};base64,${fs.readFileSync(path.join(root, value)).toString('base64')}"`
        } else attributes += ` ${key}="${substitute(value, data)}"`
      }
      return `<${tag}${attributes}>${nodes(n.elements, data)}</${tag}>`
    }).join('')
  }
  return nodes(xml2js(source).elements, p.data)
}
async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true })
  let failures = 0
  try {
    const widths = [320, 375, 430]
    for (const width of widths) for (const name of ['onboarding', 'nutritionist', 'contribution', 'today']) {
      const page = await browser.newPage({ viewport: { width, height: 900 } })
      const css = (read('app.wxss') + read(`pages/${name}/${name}.wxss`))
        .replace(/\bpage\s*{/g, 'body {').replace(/(-?[\d.]+)rpx/g, (_, n) => `${+n * width / 750}px`)
      await page.setContent(`<meta charset="utf-8"><style>
        body{margin:0} button{display:block;width:184px;margin:0 auto;padding:8px 24px;box-sizing:border-box;border:0;line-height:1.4;font:inherit}
        input{box-sizing:content-box;min-width:auto;width:140px;border:0} img{object-fit:contain}
        ${css}</style>${render(name)}`)
      const report = await page.evaluate(() => {
        const overflow = [...document.querySelectorAll('div,button,input')].filter(e => {
          const r = e.getBoundingClientRect(); return r.width && (r.right > innerWidth + 1 || r.left < -1)
        }).map(e => e.className).slice(0, 8)
        const offCenter = [...document.querySelectorAll('.opt')].filter(e => {
          const range = document.createRange(); range.selectNodeContents(e)
          const t = range.getBoundingClientRect(), b = e.getBoundingClientRect()
          return Math.abs((t.top + t.bottom - b.top - b.bottom) / 2) > 3
        }).map(e => e.textContent)
        const blankMilestones = [...document.querySelectorAll('.ach-label')].filter(e => !e.textContent.trim()).length
        return { overflow, offCenter, blankMilestones }
      })
      console.log(JSON.stringify({ name, width, ...report }))
      if (report.overflow.length || report.offCenter.length || report.blankMilestones) failures++
      if (process.env.UI_SCREENSHOT_DIR && width === 375) {
        fs.mkdirSync(process.env.UI_SCREENSHOT_DIR, { recursive: true })
        await page.screenshot({ path: path.join(process.env.UI_SCREENSHOT_DIR, `${name}.png`), fullPage: true })
      }
      await page.close()
    }
  } finally { await browser.close() }
  if (failures) process.exitCode = 1
}
main().catch(e => { console.error(e); process.exitCode = 1 })
