// Usage: node screenshot.js in1.html out1.png [in2.html out2.png ...]
// Photographs the terminal frame of each page at deviceScaleFactor 2.
const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const pairs = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 }, deviceScaleFactor: 2 });
  for (let i = 0; i < pairs.length; i += 2) {
    await page.goto('file://' + path.resolve(pairs[i]));
    await page.evaluate(() => document.fonts.ready);
    await page.locator('body').screenshot({ path: pairs[i + 1] });
  }
  await browser.close();
})();
