const {chromium}=require('/Users/filippofocaccia/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path=require('path');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--enable-unsafe-swiftshader']});
  const page=await browser.newPage({viewport:{width:736,height:860}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('file:///tmp/salt-preview.html');
  const f=page.frameLocator('iframe');
  await f.locator('#salt-scene canvas').waitFor({timeout:45000});
  await f.locator('#salt-detail').filter({hasText:'model_0002'}).waitFor();
  await page.screenshot({path:path.join(__dirname,'one-body.png'),fullPage:true});
  const sample=await page.frames()[1].evaluate(()=>{
    const pixels=id=>{
      const canvas=document.querySelector(id);
      const data=canvas.getContext('2d').getImageData(0,0,48,48).data;
      const colors=new Set();for(let p=0;p<data.length;p+=4)colors.add([...data.slice(p,p+3)].join(','));
      return colors.size;
    };
    return {inputColors:pixels('#salt-input'),maskColors:pixels('#salt-mask'),overlayColors:pixels('#salt-overlay')};
  });
  await f.locator('#salt-case').selectOption('1');
  await f.locator('#salt-bands').check();
  await f.locator('#salt-rotate').click();
  await page.screenshot({path:path.join(__dirname,'three-bodies-bands.png'),fullPage:true});
  const firstSlice=await f.locator('#salt-slice-value').innerText();
  await f.locator('#salt-slice').fill('25');
  await f.locator('#salt-slice').dispatchEvent('input');
  const updatedSlice=await f.locator('#salt-slice-value').innerText();
  await page.setViewportSize({width:320,height:900});
  await page.screenshot({path:path.join(__dirname,'narrow.png'),fullPage:true});
  const widths=await page.frames()[1].evaluate(()=>({root:document.querySelector('#salt-ground-truth').getBoundingClientRect().width,scroll:document.documentElement.scrollWidth,viewport:innerWidth}));
  console.log(JSON.stringify({errors,sample,firstSlice,updatedSlice,widths,detail:await f.locator('#salt-detail').innerText()}));
  await browser.close();
  if(errors.length||sample.maskColors!==2||sample.inputColors<10||firstSlice===updatedSlice||widths.scroll>widths.viewport)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
