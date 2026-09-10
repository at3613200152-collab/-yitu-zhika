// Original vector geometry, rasterized for the native WeChat tab bar.
// NODE_PATH may point to a local installation of sharp; no network is used.
const sharp = require('sharp')
const path = require('node:path')
const shapes = {
  camera: '<path d="M8 7l2-3h4l2 3h3a2 2 0 0 1 2 2v10H3V9a2 2 0 0 1 2-2z"/><circle cx="12" cy="13" r="3.5"/>',
  calendar: '<rect x="4" y="5" width="16" height="16" rx="3"/><path d="M8 3v5m8-5v5M4 10h16M8 14h2m4 0h2m-8 4h2"/>',
  person: '<circle cx="12" cy="7" r="4"/><path d="M4 21v-2a8 7 0 0 1 16 0v2"/>'
}
async function main() {
  for (const [name, shape] of Object.entries(shapes)) {
    const variants = { '': '#78827d', '-active': '#1f7a4d' }
    if (name === 'camera') variants['-white'] = '#ffffff'
    for (const [suffix, color] of Object.entries(variants)) {
      const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 24 24"><g fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + shape + '</g></svg>'
      await sharp(Buffer.from(svg)).png().toFile(path.join(__dirname, '../assets/tabbar', 'ui-' + name + suffix + '.png'))
    }
  }
}
main().catch(e => { console.error(e); process.exitCode = 1 })
