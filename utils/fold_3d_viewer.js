
(()=>{
 const DATA=JSON.parse(document.getElementById('viewer-data').textContent);
 let D,gt,pred,mri,nx,ny,nz;
 const root=document.getElementById('hippo-viewer'),q=s=>root.querySelector(s);
 const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
 function decodeMask(runs,size){const a=new Uint8Array(size);let offset=0;for(let i=0;i<runs.length;i+=2){a.fill(runs[i],offset,offset+runs[i+1]);offset+=runs[i+1];}if(offset!==size)throw new Error('Invalid mask length');return a;}
 const index=(x,y,z)=>(x*ny+y)*nz+z,inside=(x,y,z)=>x>=0&&y>=0&&z>=0&&x<nx&&y<ny&&z<nz;
 const canvas=q('#hippo-3d'),slice=q('#hippo-slice'),sc=slice.getContext('2d');
 const gl=canvas.getContext('webgl',{alpha:true,antialias:true,premultipliedAlpha:false});
 const slider=q('#hippo-range'),mode=q('#hippo-mode'),view=q('#hippo-view');
 const imageCanvas=document.createElement('canvas'),ic=imageCanvas.getContext('2d');
 const texCanvas=document.createElement('canvas'),tc=texCanvas.getContext('2d');
 const rgbCanvas=document.createElement('canvas'),rgbContext=rgbCanvas.getContext('2d');rgbCanvas.width=rgbCanvas.height=1;
 let yaw=.7,tilt=.2,y=0,palette=[],pending=false;
 function resolve(token){const el=q('#hippo-color');el.style.color='var('+token+')';const c=getComputedStyle(el).color;rgbContext.clearRect(0,0,1,1);rgbContext.fillStyle=c;rgbContext.fillRect(0,0,1,1);return Array.from(rgbContext.getImageData(0,0,1,1).data).slice(0,3).map(v=>v/255);}
 function getPalette(){palette=[resolve('--foreground'),resolve('--viz-series-1'),resolve('--viz-series-2'),resolve('--muted-foreground'),resolve('--viz-series-3'),resolve('--viz-series-4'),resolve('--viz-series-5')];}
 function category(i,m){const g=gt[i],p=pred?pred[i]:0;if(m==='fg')return g?3:0;if(m==='gt')return g;if(m==='pred')return p;if(!g&&p)return 4;if(g&&!p)return 5;if(g&&p&&g!==p)return 6;return m==='pred_errors'?p:g?3:0;}
 const xyz=(x,y,z)=>[(D.center[1]-y)*D.spacing[1],(x-D.center[0])*D.spacing[0],(z-D.center[2])*D.spacing[2]];
 const acquired=(x,y,z)=>[x,y,z].every((v,i)=>v>=D.imageBounds[0][i]&&v<D.imageBounds[1][i]);
 const directions=[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]];
 const cornerSets=[[[0,0,0],[0,1,0],[0,1,1],[0,0,1]],[[1,0,0],[1,0,1],[1,1,1],[1,1,0]],[[0,0,0],[0,0,1],[1,0,1],[1,0,0]],[[0,1,0],[1,1,0],[1,1,1],[0,1,1]],[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],[[0,0,1],[0,1,1],[1,1,1],[1,0,1]]];
 const meshCache=new Map();
 function mesh(m){if(meshCache.has(m))return meshCache.get(m);let a=[];for(let x=0;x<nx;x++)for(let yy=0;yy<ny;yy++)for(let z=0;z<nz;z++){let cat=category(index(x,yy,z),m);if(!cat)continue;directions.forEach((d,k)=>{let [xx,yyy,zz]=[x+d[0],yy+d[1],z+d[2]];if(inside(xx,yyy,zz)&&category(index(xx,yyy,zz),m))return;let corners=cornerSets[k].map(c=>xyz(x+c[0]-.5,yy+c[1]-.5,z+c[2]-.5)),n=[-d[1],d[0],d[2]];for(const j of [0,1,2,0,2,3])a.push(...corners[j],...n,cat,0,0);});}const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(a),gl.STATIC_DRAW);const result={buffer,count:a.length/9};meshCache.set(m,result);return result;}
 let program,locations,texture,dynamic;
 if(gl){
  const vs='attribute vec3 aPosition;attribute vec3 aNormal;attribute float aCategory;attribute vec2 aUV;uniform vec4 uRotation;uniform vec2 uScale;varying float vCategory;varying float vLight;varying vec2 vUV;void main(){float x=aPosition.x*uRotation.x-aPosition.y*uRotation.y;float d=aPosition.x*uRotation.y+aPosition.y*uRotation.x;float y=aPosition.z*uRotation.z-d*uRotation.w;float depth=d*uRotation.z+aPosition.z*uRotation.w;gl_Position=vec4(x*uScale.x,y*uScale.y,-depth/100.0,1.0);vCategory=aCategory;vLight=.6+.4*abs(dot(aNormal,normalize(vec3(.3,.6,.8))));vUV=aUV;}';
  const fs='precision mediump float;varying float vCategory;varying float vLight;varying vec2 vUV;uniform vec3 uColors[7];uniform int uMode;uniform vec3 uFixed;uniform sampler2D uTexture;void main(){if(uMode==1){vec4 c=texture2D(uTexture,vUV);if(c.a<.01)discard;gl_FragColor=c;return;}if(uMode==2){gl_FragColor=vec4(uFixed,1.0);return;}vec3 c=uColors[0];if(vCategory<1.5)c=uColors[1];else if(vCategory<2.5)c=uColors[2];else if(vCategory<3.5)c=uColors[3];else if(vCategory<4.5)c=uColors[4];else if(vCategory<5.5)c=uColors[5];else c=uColors[6];gl_FragColor=vec4(c*vLight,1.0);}';
  function shader(type,code){let s=gl.createShader(type);gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
  program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));gl.useProgram(program);
  locations={};for(const name of ['aPosition','aNormal','aCategory','aUV'])locations[name]=gl.getAttribLocation(program,name);for(const name of ['uRotation','uScale','uColors[0]','uMode','uFixed','uTexture'])locations[name]=gl.getUniformLocation(program,name);
  texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);dynamic=gl.createBuffer();gl.enable(gl.DEPTH_TEST);
 }else{q('#hippo-failure').hidden=false;q('#hippo-failure').textContent='3D needs WebGL in this viewer. The actual MRI slice and mask controls remain available.';}
 function bind(buffer){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(const [name,size,offset] of [['aPosition',3,0],['aNormal',3,12],['aCategory',1,24],['aUV',2,28]]){let loc=locations[name];gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,36,offset);}}
 function geometry(verts,type,renderMode){gl.bindBuffer(gl.ARRAY_BUFFER,dynamic);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.DYNAMIC_DRAW);bind(dynamic);gl.uniform1i(locations.uMode,renderMode);gl.drawArrays(type,0,verts.length/9);}
 function vertex(p,u=0,v=0){return [...p,0,1,0,0,u,v];}
 function outline(yy,dashed=false){let corners=[xyz(-.5,yy,-.5),xyz(nx-.5,yy,-.5),xyz(nx-.5,yy,nz-.5),xyz(-.5,yy,nz-.5)],a=[];for(let i=0;i<4;i++){let p=corners[i],q=corners[(i+1)%4];if(!dashed)a.push(...vertex(p),...vertex(q));else for(let t=0;t<1;t+=.12){let end=Math.min(1,t+.07);a.push(...vertex(p.map((v,j)=>v+(q[j]-v)*t)),...vertex(p.map((v,j)=>v+(q[j]-v)*end)));}}return a;}
 function renderSlice(){const overlay=q('#hippo-overlay').checked,m=mode.value,im=ic.createImageData(nx,nz),tex=tc.createImageData(nx,nz),mriIn3d=q('#hippo-mri3d').checked;
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
   const i=index(x,y,z),k=((nz-1-z)*nx+x)*4,cat=category(i,m);
   const g=acquired(x,y,z)?mri[i]:((Math.floor(x/2)+Math.floor(z/2))%2?45:75);
   let rgb=[g,g,g],alpha=cat>=4?.85:m==='errors'?0:.43;
   if(cat&&overlay)rgb=rgb.map((v,j)=>v*(1-alpha)+palette[cat][j]*255*alpha);
   im.data.set([...rgb,255],k);
   const tcolor=cat?palette[cat].map(v=>v*255):[g,g,g];
   if(mriIn3d)tex.data.set([...rgb,220],k);else tex.data.set([...tcolor,cat?255:0],k);
  }
  ic.putImageData(im,0,0);tc.putImageData(tex,0,0);const r=slice.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);slice.width=Math.round(r.width*d);slice.height=Math.round(r.height*d);sc.setTransform(d,0,0,d,0,0);sc.clearRect(0,0,r.width,r.height);
  const scale=Math.min((r.width-34)/(nx*D.spacing[0]),(r.height-34)/(nz*D.spacing[2])),xs=scale*D.spacing[0],zs=scale*D.spacing[2],ww=nx*xs,hh=nz*zs,left=(r.width-ww)/2,top=(r.height-hh)/2;
  sc.imageSmoothingEnabled=false;sc.drawImage(imageCanvas,left,top,ww,hh);
  // Contour around the selected foreground, computed on the exact voxel grid.
  sc.lineWidth=1.25;sc.strokeStyle='rgb('+palette[0].map(v=>Math.round(v*255)).join(',')+')';
  const reference=q('#hippo-reference').checked;
  if(overlay||reference){const isForeground=(x,z)=>inside(x,y,z)&&(reference?gt[index(x,y,z)]>0:category(index(x,y,z),m)>0);sc.beginPath();for(let x=0;x<nx;x++)for(let z=0;z<nz;z++){if(!isForeground(x,z))continue;let xx=left+x*xs,zz=top+(nz-1-z)*zs;for(const [dx,dz,ax,az,bx,bz] of [[-1,0,0,0,0,1],[1,0,1,0,1,1],[0,1,0,0,1,0],[0,-1,0,1,1,1]])if(!isForeground(x+dx,z+dz)){sc.moveTo(xx+ax*xs,zz+az*zs);sc.lineTo(xx+bx*xs,zz+bz*zs);}}sc.stroke();}
  sc.fillStyle='rgb('+palette[0].map(v=>Math.round(v*255)).join(',')+')';sc.font='12px '+getComputedStyle(root).fontFamily;sc.textAlign='center';sc.fillText('S',r.width/2,12);sc.fillText('I',r.width/2,r.height-1);sc.fillText('L',9,r.height/2);sc.fillText('R',r.width-9,r.height/2);
 }
 function render3D(){if(!gl)return;const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);const radius=Math.hypot(...D.shape.map((n,i)=>Math.max(D.center[i]+.5,n-.5-D.center[i])*D.spacing[i])),diameter=radius*2.15,scale=Math.min(r.width/diameter,r.height/diameter);gl.uniform2f(locations.uScale,2*scale/r.width,2*scale/r.height);gl.uniform4f(locations.uRotation,Math.cos(yaw),Math.sin(yaw),Math.cos(tilt),Math.sin(tilt));gl.uniform3fv(locations['uColors[0]'],new Float32Array(palette.flat()));
  const m=mesh(mode.value);bind(m.buffer);gl.uniform1i(locations.uMode,0);gl.disable(gl.BLEND);gl.depthMask(true);gl.drawArrays(gl.TRIANGLES,0,m.count);
  let p=[xyz(-.5,y,-.5),xyz(nx-.5,y,-.5),xyz(nx-.5,y,nz-.5),xyz(-.5,y,nz-.5)],uv=[[0,1],[1,1],[1,0],[0,0]],a=[];for(const j of [0,1,2,0,2,3])a.push(...vertex(p[j],...uv[j]));gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,texCanvas);gl.uniform1i(locations.uTexture,0);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);geometry(a,gl.TRIANGLES,1);gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);gl.uniform3fv(locations.uFixed,palette[0]);if(D.cut!==null)geometry(outline(D.cut,true),gl.LINES,2);gl.uniform3fv(locations.uFixed,palette[4]);geometry(outline(y),gl.LINES,2);gl.enable(gl.DEPTH_TEST);gl.depthMask(true);
  drawAxes(r.width,r.height,d);
 }
 function drawAxes(width,height,dpr){
  const a=q('#hippo-axes'),c=a.getContext('2d');a.width=Math.round(width*dpr);a.height=Math.round(height*dpr);c.setTransform(dpr,0,0,dpr,0,0);
  c.strokeStyle=c.fillStyle='rgb('+palette[0].map(v=>Math.round(v*255)).join(',')+')';c.font='12px '+getComputedStyle(root).fontFamily;c.textAlign='center';
  for(const [name,p] of [['A',[-1,0,0]],['R',[0,1,0]],['S',[0,0,1]]]){
   const x=p[0]*Math.cos(yaw)-p[1]*Math.sin(yaw),depth=p[0]*Math.sin(yaw)+p[1]*Math.cos(yaw),z=p[2]*Math.cos(tilt)-depth*Math.sin(tilt);
   c.beginPath();c.moveTo(48,height-48);c.lineTo(48+x*26,height-48-z*26);c.stroke();c.fillText(name,48+x*38,height-44-z*38);
  }
 }
 const format=n=>Number(n).toLocaleString();
 function updateLabels(){
  const s=D.slices[y],nativeY=y+D.origin[1],m=mode.value;
  q('#hippo-position').textContent='native y = '+nativeY+' · '+(ny-y)+'/'+ny;
  q('#hippo-count').textContent='Annotation: '+s.anterior+' anterior · '+s.posterior+' posterior · '+s.components+' component'+(s.components===1?'':'s');
  let detail=pred?'This slice: '+s.fp+' extra foreground · '+s.fn+' missed foreground · '+s.swaps+' A/P swaps':'Annotation foreground: '+format(D.stats.foreground)+' voxels';
  if(y<D.imageBounds[0][1]||y>=D.imageBounds[1][1])detail+=' · outside acquired MRI';
  q('#hippo-detail').textContent=detail;
  const errorLegend=[[4,'Extra foreground'],[5,'Missed foreground'],[6,'A/P swap']];
  const legend=m==='errors'?[[3,'Correct foreground'],...errorLegend]:m==='pred_errors'?[[1,'Correct anterior'],[2,'Correct posterior'],...errorLegend]:m==='fg'?[[3,'Whole foreground']]:[[1,'Anterior'],[2,'Posterior']];
  const el=q('#hippo-legend');el.replaceChildren();
  for(const [i,label] of legend){let span=document.createElement('span'),swatch=document.createElement('i');swatch.className='hippo-swatch';swatch.style.background='rgb('+palette[i].map(v=>Math.round(v*255)).join(',')+')';span.append(swatch,document.createTextNode(label));el.append(span);}
  q('#hippo-slice-title').textContent='Coronal MRI + '+(m==='pred'?'prediction':m==='pred_errors'?'prediction / errors':m==='errors'?'errors':'annotation')+' · R on right';
  q('#hippo-prev').disabled=+slider.value===0;q('#hippo-next').disabled=+slider.value===ny-1;
 }
 function render(){pending=false;if(!D)return;y=ny-1-(+slider.value);getPalette();renderSlice();render3D();updateLabels();}
 function schedule(){if(!pending){pending=true;requestAnimationFrame(render);}}
 slider.addEventListener('input',schedule);mode.addEventListener('change',schedule);q('#hippo-mri3d').addEventListener('change',schedule);q('#hippo-overlay').addEventListener('change',schedule);q('#hippo-reference').addEventListener('change',schedule);
 q('#hippo-prev').addEventListener('click',()=>{slider.value=Math.max(0,+slider.value-1);schedule();});q('#hippo-next').addEventListener('click',()=>{slider.value=Math.min(ny-1,+slider.value+1);schedule();});q('#hippo-jump').addEventListener('click',()=>{slider.value=ny-1-D.start;schedule();});
 view.addEventListener('change',()=>{[yaw,tilt]=view.value==='sagittal'?[0,0]:view.value==='coronal'?[-Math.PI/2,0]:view.value==='axial'?[0,Math.PI/2]:[.7,.2];schedule();});
 let drag=null;canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.009;tilt=Math.max(-1.57,Math.min(1.57,tilt+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];view.value='oblique';schedule();});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
 function chooseCase(name){
  D=DATA.cases.find(c=>c.name===name);if(!D)throw new Error('Unknown patient '+name);
  if(gl)for(const m of meshCache.values())gl.deleteBuffer(m.buffer);meshCache.clear();
  [nx,ny,nz]=D.shape;gt=decodeMask(D.gt,nx*ny*nz);pred=D.pred?decodeMask(D.pred,nx*ny*nz):null;mri=decode(D.image);
  imageCanvas.width=texCanvas.width=nx;imageCanvas.height=texCanvas.height=nz;
  slider.max=ny-1;slider.value=ny-1-D.start;
  for(const option of mode.options)option.disabled=!pred&&!['gt','fg'].includes(option.value);
  if(!pred&&!['gt','fg'].includes(mode.value))mode.value='gt';
  q('#hippo-case-title').textContent=D.name;
  const stat=D.stats;
  q('#hippo-case-stats').textContent=pred?format(stat.total)+' errors · '+format(stat.swaps)+' A/P swaps · foreground Dice '+(stat.dice_fg===null?'n/a':stat.dice_fg.toFixed(3)):format(stat.foreground)+' annotation voxels';
  q('#hippo-frame-label').textContent='Solid frame: selected slice · '+(D.cut===null?'no single planar annotation cut':'dashed frame: annotation cut at native y = '+(D.cut+D.origin[1]));
  q('#hippo-jump').disabled=D.cut===null;schedule();
 }
 function fillPatients(selected){
  const sorting=q('#hippo-sort').value,patients=[...DATA.cases];
  patients.sort((a,b)=>sorting==='name'?a.name.localeCompare(b.name):(b.stats[sorting]||0)-(a.stats[sorting]||0)||a.name.localeCompare(b.name));
  const select=q('#hippo-patient');select.replaceChildren();
  for(const c of patients){const option=document.createElement('option');option.value=c.name;option.textContent=c.name+(c.pred?' · '+format(c.stats.swaps)+' swaps / '+format(c.stats.total)+' errors':'');select.append(option);}
  select.value=selected;
 }
 q('#hippo-run').textContent='Fold '+DATA.fold+' · '+DATA.cases.length+' validation patients · '+DATA.predictionName+' · saved predictions';
 q('#hippo-patient').addEventListener('change',e=>chooseCase(e.target.value));q('#hippo-sort').addEventListener('change',()=>fillPatients(D.name));
 if(!DATA.cases.some(c=>c.pred)){q('#hippo-sort').value='name';q('#hippo-sort').disabled=true;}
 fillPatients(DATA.initialCase);chooseCase(DATA.initialCase);
 new ResizeObserver(schedule).observe(root);new MutationObserver(schedule).observe(document.documentElement,{attributes:true,attributeFilter:['class','style','data-theme']});
 matchMedia('(prefers-color-scheme: dark)').addEventListener('change',schedule);
})();
