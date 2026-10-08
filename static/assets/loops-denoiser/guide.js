(function(){
  const root=document.querySelector('.ld-guide');
  if(!root)return;
  const english=root.lang==='en';
  const tr=(zh,en)=>english?en:zh;
  const $ = s => root.querySelector(s);
  const css = n => getComputedStyle(root).getPropertyValue(n).trim();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs={}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) if (attrs[k]!=null) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

  /* ============ HERO: two clocks ============ */
  const Ts = [10, 25, 50, 100];
  let K = 4, T = 25;
  const canvas = $('#xt'), ctx = canvas.getContext('2d');
  const R = 64;
  // procedural scene, values 0..1
  const scene = new Float32Array(R*R*3);
  (function build(){
    for (let y=0;y<R;y++) for (let x=0;x<R;x++){
      const i=(y*R+x)*3, v=y/R;
      let c=[0.12+0.85*v*v, 0.16+0.5*v*v, 0.38-0.05*v];
      const dx=x-42, dy=y-24, d=Math.sqrt(dx*dx+dy*dy);
      if (d<8) c=[1,0.84,0.5]; else if (d<13){ const g=(13-d)/5*0.35; c=[c[0]+g,c[1]+g*0.7,c[2]+g*0.2]; }
      const h1 = 38 + 5*Math.sin(x*0.13+1) + 3*Math.sin(x*0.31);
      const h2 = 47 + 4*Math.sin(x*0.09+3) + 2*Math.sin(x*0.4+1);
      if (y>h1) c=[0.42,0.30,0.55];
      if (y>h2) c=[0.13,0.19,0.27];
      if (y>56) { const w=0.5+0.5*Math.sin(x*0.6+y*1.3); c=[0.10+0.05*w,0.16+0.06*w,0.26+0.08*w]; }
      scene[i]=Math.min(1,c[0]); scene[i+1]=Math.min(1,c[1]); scene[i+2]=Math.min(1,c[2]);
    }
  })();
  const noise = new Float32Array(R*R*3);
  for (let i=0;i<noise.length;i++){ let u=0; for(let j=0;j<4;j++) u+=Math.random(); noise[i]=0.5+(u-2)*0.42; }
  const small = document.createElement('canvas'); small.width=R; small.height=R;
  const sctx = small.getContext('2d'); const img = sctx.createImageData(R,R);
  function drawImage(t){
    for (let p=0;p<R*R;p++){
      for (let c=0;c<3;c++){ const i=p*3+c; const v=t*scene[i]+(1-t)*noise[i]; img.data[p*4+c]=Math.max(0,Math.min(255,v*255)); }
      img.data[p*4+3]=255;
    }
    sctx.putImageData(img,0,0);
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(small,0,0,256,256);
  }

  const dial = $('#dial');
  function drawDial(step, loopPhase){
    dial.innerHTML='';
    const cx=120, cy=120, ro=104, ri=56;
    el('circle',{cx,cy,r:ro,fill:'none',stroke:css('--rule'),'stroke-width':1},dial);
    for (let i=0;i<T;i++){
      const a=-Math.PI/2 + (i/T)*Math.PI*2;
      const len = T>50?6:9;
      const done = i < step;
      el('line',{x1:cx+(ro-len)*Math.cos(a),y1:cy+(ro-len)*Math.sin(a),x2:cx+(ro+len*0.4)*Math.cos(a),y2:cy+(ro+len*0.4)*Math.sin(a),stroke: done?css('--ink'):css('--rule'),'stroke-width': T>50?1.2:2,'stroke-linecap':'round'},dial);
    }
    const ha = -Math.PI/2 + (Math.min(step,T)/T)*Math.PI*2;
    el('line',{x1:cx,y1:cy,x2:cx+(ro-16)*Math.cos(ha),y2:cy+(ro-16)*Math.sin(ha),stroke:css('--ink-3'),'stroke-width':1.5,'stroke-linecap':'round'},dial);
    el('circle',{cx,cy,r:ri,fill:'none',stroke:css('--accent'),'stroke-opacity':.25,'stroke-width':8},dial);
    // K segments
    for (let k=0;k<K;k++){
      const a0=-Math.PI/2 + k/K*Math.PI*2 + 0.06, a1=-Math.PI/2 + (k+1)/K*Math.PI*2 - 0.06;
      const done = loopPhase >= (k+1)/K || step>=T;
      const cur = !done && loopPhase >= k/K;
      if (!(done||cur)) continue;
      const end = cur ? -Math.PI/2 + loopPhase*Math.PI*2 : a1;
      if (end<=a0) continue;
      const large = (end-a0)>Math.PI?1:0;
      el('path',{d:`M ${cx+ri*Math.cos(a0)} ${cy+ri*Math.sin(a0)} A ${ri} ${ri} 0 ${large} 1 ${cx+ri*Math.cos(end)} ${cy+ri*Math.sin(end)}`,fill:'none',stroke:css('--accent'),'stroke-width':8,'stroke-linecap':'round'},dial);
    }
    const da = -Math.PI/2 + loopPhase*Math.PI*2;
    el('circle',{cx:cx+ri*Math.cos(da),cy:cy+ri*Math.sin(da),r:6,fill:css('--accent'),stroke:css('--panel'),'stroke-width':2},dial);
    const tx = el('text',{x:cx,y:cy-4,'text-anchor':'middle','font-family':'ui-monospace, SFMono-Regular, Menlo, monospace','font-size':15,fill:css('--ink')},dial);
    tx.textContent = tr(`步 ${Math.min(step+1,T)}/${T}`,`Step ${Math.min(step+1,T)}/${T}`);
    const tx2 = el('text',{x:cx,y:cy+16,'text-anchor':'middle','font-family':'ui-monospace, SFMono-Regular, Menlo, monospace','font-size':12,fill:css('--accent')},dial);
    tx2.textContent = step>=T ? `t = 1` : tr(`圈 ${Math.min(K,Math.floor(loopPhase*K)+1)}/${K}`,`Loop ${Math.min(K,Math.floor(loopPhase*K)+1)}/${K}`);
  }

  function updateReadout(){
    const per = 6 + 5*K + 6;
    $('#rPer').textContent = per;
    $('#rTot').textContent = (per*T).toLocaleString('en-US');
    $('#kOut').textContent = K; $('#tOut').textContent = T;
  }

  let playing = !reduce, start = performance.now(), raf=null;
  function period(){ return Math.min(0.4, Math.max(0.012, 4.5/(T*K))); }
  function frame(now){
    const per = period();
    const total = T*K*per, hold=1.4;
    let el_ = (now-start)/1000;
    if (el_ > total+hold){ start=now; el_=0; }
    const loopsDone = Math.min(T*K, el_/per);
    const step = Math.min(T, Math.floor(loopsDone/K));
    const phase = step>=T ? 1 : (loopsDone - step*K)/K;
    drawImage(step/T);
    drawDial(step, phase);
    if (playing) raf=requestAnimationFrame(frame);
  }
  function restart(){ start=performance.now(); updateReadout(); if (!playing){ drawImage(1); drawDial(T,1); } }
  $('#kSlider').addEventListener('input', e=>{ K=+e.target.value; restart(); });
  $('#tSlider').addEventListener('input', e=>{ T=Ts[+e.target.value]; restart(); });
  $('#playBtn').addEventListener('click', ()=>{
    playing=!playing; $('#playBtn').textContent = playing?tr('暂停','Pause'):tr('播放','Play');
    if (playing){ start=performance.now(); raf=requestAnimationFrame(frame); } else { cancelAnimationFrame(raf); }
  });
  updateReadout();
  if (playing) raf=requestAnimationFrame(frame); else { $('#playBtn').textContent=tr('播放','Play'); drawImage(1); drawDial(T,1); }

  /* ============ tooltip helper ============ */
  function makeTip(fig){
    let t=fig.querySelector('.tip'); if(!t){t=document.createElement('div'); t.className='tip'; fig.appendChild(t);}
    return {
      show(html, evt){
        t.innerHTML=html; t.classList.add('on');
        const fr=fig.getBoundingClientRect();
        let x=evt.clientX-fr.left+14, y=evt.clientY-fr.top-12;
        const w=t.offsetWidth; if (x+w>fr.width-8) x=evt.clientX-fr.left-w-14;
        t.style.left=Math.max(6,x)+'px'; t.style.top=Math.max(6,y-t.offsetHeight)+'px';
      },
      hide(){ t.classList.remove('on'); }
    };
  }

  /* ============ §3 explorer ============ */
  const ex = {
    naive:{
      color:'--dense', title:tr('朴素循环：只监督最后一圈','Naive looping: final-loop supervision'),
      text:tr('只有 u₄ 被拉向目标。中间圈没有任何约束，可以停在离解很远的地方，所以推理时少转一圈就没法用。多转一圈时，同一个变换会继续改写已经正确的状态。','Only u₄ is pulled toward the target. Intermediate loops are unconstrained and may remain far from a solution, so stopping one loop early can fail. An extra loop applies the same transformation again, potentially changing a state that was already correct.'),
      formula:'L = ‖u<sub>K</sub> − u★‖²',
      pts:[[90,300],[210,330],[330,300],[430,220],[540,96]], arrows:[[4,'t']], seg:false
    },
    elt:{
      color:'--elt', title:tr('ELT：浅层出口模仿深层出口','ELT: shallow exits imitate deeper exits'),
      text:tr('每步随机抽一个学生出口（图中 u₂）。它一边学真值 u★，一边学老师 u₄ 的输出（stop-grad）。λ 从 1 线性降到 0，训练后期学生主要模仿老师。所有出口共享同一组权重，学生路径是老师路径的前缀。','Each step samples a student exit (u₂ here). It learns from ground truth u★ and the stop-gradient teacher output u₄. As λ decreases linearly from 1 to 0, the student increasingly imitates the teacher. All exits share weights, and the student path is a prefix of the teacher path.'),
      formula:'L = L<sub>GT</sub>(u<sub>Lmax</sub>) + λ·L<sub>GT</sub>(u<sub>Lint</sub>)<br>  + (1−λ)·L<sub>dist</sub>(u<sub>Lint</sub>, sg(u<sub>Lmax</sub>))<br><span class="c">L<sub>int</sub> ~ U(L<sub>min</sub>, L<sub>max</sub>), λ: 1 → 0</span>',
      pts:[[90,300],[200,268],[300,222],[415,165],[512,112]], arrows:[[4,'t'],[2,'t','λ'],[2,4,'1−λ']], seg:false, hl:2, bend:-1.6
    },
    ldit:{
      color:'--ldit', title:tr('Looped-DiT：每圈都预测同一个目标','Looped-DiT: the same target at every loop'),
      text:tr('第 1 到 4 圈的读出都经过共享的 post-loop 块，对同一个干净图 x₀ 算 flow-matching 损失。早期圈被迫也给出可用的答案，所以早退的代价变小，到 8 圈仍不崩。最后一圈权重最大。','Readouts from loops 1–4 pass through the shared post-loop blocks and use a flow-matching loss against the same clean image x₀. Early loops must produce usable outputs, reducing the cost of early exits; performance remains stable up to 8 loops. The final loop has the largest weight.'),
      formula:'L = Σ<sub>n</sub> w<sub>n</sub>·‖x̂₀<sup>(n)</sup> − x₀‖² / c(t)²<br><span class="c">w = (1/3, 1/3, 1/3, 1)  final + mean</span>',
      pts:[[90,300],[300,160],[420,118],[490,100],[528,92]], arrows:[[1,'t','1/3'],[2,'t','1/3'],[3,'t','1/3'],[4,'t','1']], seg:false
    },
    lift:{
      color:'--lift', title:tr('LiFT：每圈预测路径上对应的点','LiFT: a different point on the path at each loop'),
      text:tr('从初始估计 b = sg(u₀) 到真值 u★ 连一条直线。训练时随机抽排序后的深度坐标 s，第 k 圈只需到达直线上的 ū(s_k)。各圈有不同分工，推理时把 [0, 1] 切得更细，就是多转几圈。','Draw a straight line from the initial estimate b = sg(u₀) to ground truth u★. Training samples and sorts depth coordinates s; loop k targets ū(s_k) on the line. Each loop has a different role. More inference loops use a finer partition of [0, 1].'),
      formula:tr('ū<sub>s</sub> = (1 − s)·sg(u₀) + s·u★<br>L = 1/K Σ<sub>k</sub> ‖u<sub>k</sub> − ū<sub>s<sub>k</sub></sub>‖²<br><span class="c">训练：s₁…s<sub>K−1</sub> ~ U(0,1) 排序；推理：s<sub>k</sub> = k/K</span>','ū<sub>s</sub> = (1 − s)·sg(u₀) + s·u★<br>L = 1/K Σ<sub>k</sub> ‖u<sub>k</sub> − ū<sub>s<sub>k</sub></sub>‖²<br><span class="c">Train: sort s₁…s<sub>K−1</sub> ~ U(0,1); infer: s<sub>k</sub> = k/K</span>'),
      pts:[[90,300],[218,262],[325,190],[440,138],[540,92]], arrows:[], seg:true, s:[0.27,0.52,0.8,1]
    }
  };
  const target=[555,78];
  function renderEx(key){
    const d=ex[key], svg=$('#exSvg'); svg.innerHTML='';
    const col=css(d.color), ink=css('--ink'), ink3=css('--ink-3');
    const defs=el('defs',{},svg);
    const mk=el('marker',{id:'exa',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'},defs);
    el('path',{d:'M0,0 L10,5 L0,10 z',fill:col},mk);
    const g=el('g',{class:'fade'},svg);
    // faint grid
    for (let x=40;x<640;x+=60) el('line',{x1:x,y1:10,x2:x,y2:370,stroke:css('--grid'),'stroke-width':1},g);
    for (let y=30;y<380;y+=60) el('line',{x1:10,y1:y,x2:630,y2:y,stroke:css('--grid'),'stroke-width':1},g);
    // target
    el('circle',{cx:target[0],cy:target[1],r:18,fill:'none',stroke:ink,'stroke-width':1.5,'stroke-dasharray':'3 3'},g);
    el('circle',{cx:target[0],cy:target[1],r:5,fill:ink},g);
    const tt=el('text',{x:target[0]-8,y:target[1]-26,'text-anchor':'end'},g); tt.textContent=tr('u★ 目标（真值速度 / x₀）','u★ target (true velocity / x₀)'); tt.setAttribute('style',`fill:${ink}`);
    const P=d.pts;
    // trajectory
    el('path',{d:'M '+P.map(p=>p.join(' ')).join(' L '),fill:'none',stroke:ink3,'stroke-width':1.2,'stroke-opacity':.6},g);
    if (d.seg){
      el('line',{x1:P[0][0],y1:P[0][1],x2:target[0],y2:target[1],stroke:col,'stroke-width':2,'stroke-dasharray':'6 5'},g);
      d.s.forEach((s,i)=>{
        const rx=P[0][0]+s*(target[0]-P[0][0]), ry=P[0][1]+s*(target[1]-P[0][1]);
        if (s<1){ el('rect',{x:rx-5,y:ry-5,width:10,height:10,fill:col,transform:`rotate(45 ${rx} ${ry})`},g);
          const st=el('text',{x:rx+10,y:ry+22},g); st.textContent=`ū(s=${s})`; }
        const p=P[i+1];
        if (Math.hypot(p[0]-rx,p[1]-ry)>14) el('line',{x1:p[0],y1:p[1],x2:rx,y2:ry,stroke:col,'stroke-width':2,'marker-end':'url(#exa)'},g);
      });
    }
    d.arrows.forEach(a=>{
      const p=P[a[0]]; const q = a[1]==='t'?target:P[a[1]];
      const dx=q[0]-p[0], dy=q[1]-p[1], L=Math.hypot(dx,dy), cut=a[1]==='t'?20:12;
      const ex2=p[0]+dx*(1-cut/L), ey2=p[1]+dy*(1-cut/L);
      const bend = a[1]==='t' && a[0]!==4 ? 1 : 0;
      if (bend){ const sg=d.bend||1; const mx=(p[0]+ex2)/2 - dy*0.12*sg, my=(p[1]+ey2)/2 + dx*0.12*sg;
        el('path',{d:`M ${p[0]} ${p[1]} Q ${mx} ${my} ${ex2} ${ey2}`,fill:'none',stroke:col,'stroke-width':2,'marker-end':'url(#exa)'},g);
        if (a[2]){ const lt=el('text',{x:mx+(sg<0?-14:4),y:my+(sg<0?-4:4)},g); lt.textContent=a[2]; lt.setAttribute('style',`fill:${ink}`); }
      } else {
        el('line',{x1:p[0],y1:p[1],x2:ex2,y2:ey2,stroke:col,'stroke-width':2,'marker-end':'url(#exa)','stroke-dasharray':a[1]===4?'5 4':null},g);
        if (a[2]){ const lt=el('text',{x:(p[0]+ex2)/2+8,y:(p[1]+ey2)/2+16},g); lt.textContent=a[2]; lt.setAttribute('style',`fill:${ink}`); }
      }
    });
    P.forEach((p,i)=>{
      const sup = i===0 ? false : (key==='naive' ? i===4 : true);
      const isB = i===0;
      el('circle',{cx:p[0],cy:p[1],r:isB?7:(d.hl===i?8:6),fill: isB?css('--panel'):(sup?col:css('--panel')),stroke: isB?ink:(sup?css('--paper'):ink3),'stroke-width':2},g);
      const lt=el('text',{x:p[0]+(i===0?-6:10),y:p[1]+(i===0?24:-10),'text-anchor':i===0?'start':'start'},g);
      lt.textContent = i===0 ? (key==='lift'?'u₀ = b':'u₀（prelude）') : `u${'₁₂₃₄'[i-1]}`;
      if (key==='elt' && i===2){ const s2=el('text',{x:p[0]+10,y:p[1]+20},g); s2.textContent=tr('学生 L_int','Student L_int'); s2.setAttribute('style',`fill:${col}`); }
      if (key==='elt' && i===4){ const s2=el('text',{x:p[0]-12,y:p[1]+24,'text-anchor':'end'},g); s2.textContent=tr('老师 L_max','Teacher L_max'); s2.setAttribute('style',`fill:${col}`); }
      if (key==='naive' && i>0 && i<4){ const q=el('text',{x:p[0]+10,y:p[1]+18},g); q.textContent=tr('无监督','Unsupervised'); q.setAttribute('style',`fill:${ink3}`); }
    });
    const side=$('#exSide');
    side.innerHTML=`<h3 class="fade">${d.title}</h3><p class="fade">${d.text}</p><div class="formula fade">${d.formula}</div>`;
    root.querySelectorAll('.tab').forEach(b=>b.setAttribute('aria-selected', b.dataset.s===key?'true':'false'));
    current=key;
  }
  let current='elt';
  root.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>renderEx(b.dataset.s)));
  renderEx('elt');

  /* ============ §4 ELT bars ============ */
  function eltChart(){
    const box=$('#eltChart'); box.innerHTML='';
    const data=[
      {n:tr('DiT · 16 层','DiT · 16 layers'),fid:3.87,p:'1.1B',k:'dense',d:16},
      {n:tr('DiT · 32 层','DiT · 32 layers'),fid:3.43,p:'2.1B',k:'dense',d:32},
      {n:'ELT 1N × 32L',fid:10.30,p:'69M',k:'elt',d:32},
      {n:'ELT 4N × 8L',fid:3.96,p:'271M',k:'elt',d:32},
      {n:'ELT 8N × 4L',fid:3.16,p:'539M',k:'elt',d:32},
      {n:'ELT 16N × 2L',fid:2.83,p:'1.1B',k:'elt',d:32},
    ];
    const W=760, rowH=34, top=26, left=130, right=110, H=top+data.length*rowH+30;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':tr('ELT 与 dense DiT 的 FID 对比','ELT versus dense DiT FID')},box);
    const x=v=>left+v/11*(W-left-right);
    [0,2,4,6,8,10].forEach(v=>{ el('line',{x1:x(v),y1:top-6,x2:x(v),y2:H-26,stroke:css('--grid')},svg); const t=el('text',{x:x(v),y:H-8,'text-anchor':'middle',class:'muted'},svg); t.textContent=v; });
    const xl=el('text',{x:W-right,y:H-8,'text-anchor':'start',class:'muted'},svg); xl.textContent='  FID ↓';
    const fig=box.closest('.fig'); const tip=makeTip(fig);
    data.forEach((d,i)=>{
      const y=top+i*rowH, bh=18;
      const lt=el('text',{x:left-12,y:y+bh/2+4,'text-anchor':'end',class:'lab'},svg); lt.textContent=d.n;
      const w=x(d.fid)-left;
      el('path',{d:`M ${left} ${y} h ${w-4} q 4 0 4 4 v ${bh-8} q 0 4 -4 4 h ${-(w-4)} z`,fill:d.k==='elt'?css('--elt'):css('--dense')},svg);
      const vt=el('text',{x:x(d.fid)+8,y:y+bh/2+4,class:'lab'},svg); vt.textContent=d.fid.toFixed(2);
      const pt=el('text',{x:x(d.fid)+52,y:y+bh/2+4,class:'muted'},svg); pt.textContent=d.p;
      const hit=el('rect',{x:0,y:y-6,width:W,height:rowH,fill:'transparent'},svg);
      hit.addEventListener('mousemove',e=>tip.show(tr(`${d.n}<br>FID ${d.fid} · ${d.p} 参数 · 深度 ${d.d}`,`${d.n}<br>FID ${d.fid} · ${d.p} parameters · depth ${d.d}`),e));
      hit.addEventListener('mouseleave',()=>tip.hide());
    });
    // reference line at dense 32
    el('line',{x1:x(3.43),y1:top-10,x2:x(3.43),y2:H-26,stroke:css('--ink-3'),'stroke-width':1},svg);
    const rl=el('text',{x:x(3.43)+4,y:top-12,class:'muted'},svg); rl.textContent=tr('dense 32 层 = 3.43','dense 32 layers = 3.43');
  }

  /* ============ §5 Looped-DiT scatter ============ */
  function ldChart(){
    const box=$('#ldChart'); box.innerHTML='';
    const data=[
      ['E-MMDiT',0.30,56.8,0],['SANA-0.6B',0.59,60.4,0],['DeCo-XXL/16',1.1,61.9,0],['URSA-0.6B',0.86,62.1,0],['TiM-T2I',0.87,62.2,0],['DreamLite',0.39,63.5,0],['CogView4',6.4,64.0,0],['MiniT2I-B/16',0.26,66.4,0],['MiniT2I-L/16',0.91,67.3,0],['UniLiP-3B',1.6,67.8,0],['InternVL-U',1.7,69.0,0],
      ['GoT-R1',6.9,63.2,1],['T2I-R1',6.9,63.9,1],['Uni-CoT',6.6,66.8,1],
      ['Looped-DiT B/16',0.26,71.5,2]
    ];
    const W=760,H=360,L=56,Rr=30,Tp=20,B=46;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':tr('平均分对参数量散点图','Average score versus parameter count')},box);
    const lx=v=>L+(Math.log10(v)-Math.log10(0.2))/(Math.log10(10)-Math.log10(0.2))*(W-L-Rr);
    const ly=v=>Tp+(73-v)/(73-55)*(H-Tp-B);
    [56,60,64,68,72].forEach(v=>{ el('line',{x1:L,y1:ly(v),x2:W-Rr,y2:ly(v),stroke:css('--grid')},svg); const t=el('text',{x:L-8,y:ly(v)+4,'text-anchor':'end',class:'muted'},svg); t.textContent=v; });
    [0.2,0.5,1,2,5,10].forEach(v=>{ const t=el('text',{x:lx(v),y:H-B+20,'text-anchor':'middle',class:'muted'},svg); t.textContent=v+'B'; el('line',{x1:lx(v),y1:H-B,x2:lx(v),y2:H-B+5,stroke:css('--ink-3')},svg); });
    el('line',{x1:L,y1:H-B,x2:W-Rr,y2:H-B,stroke:css('--ink-3')},svg);
    const xt=el('text',{x:W-Rr,y:H-8,'text-anchor':'end',class:'muted'},svg); xt.textContent=tr('参数量（对数轴）','Parameters (log scale)');
    const yt=el('text',{x:L,y:12,class:'muted'},svg); yt.textContent=tr('六基准平均分 ↑','Six-benchmark mean ↑');
    const fig=box.closest('.fig'); const tip=makeTip(fig);
    // arrow from MiniT2I-B/16 to Looped
    el('line',{x1:lx(0.26),y1:ly(66.4)-9,x2:lx(0.26),y2:ly(71.5)+11,stroke:css('--ldit'),'stroke-width':1.5,'stroke-dasharray':'3 3'},svg);
    const lab=el('text',{x:lx(0.26)+10,y:ly(69)+4,class:'lab'},svg); lab.textContent=tr('+5.1（只加循环）','+5.1 (loops only)');
    const labels={'Looped-DiT B/16':[12,-10,'start'],'InternVL-U':[10,-8,'start'],'MiniT2I-B/16':[10,14,'start'],'CogView4':[0,-12,'middle'],'Uni-CoT':[-10,-8,'end']};
    data.forEach(d=>{
      const cx=lx(d[1]), cy=ly(d[2]);
      const c = d[3]===2?css('--ldit'):css('--dense');
      if (d[3]===1) el('circle',{cx,cy,r:5,fill:css('--panel'),stroke:c,'stroke-width':2},svg);
      else el('circle',{cx,cy,r:d[3]===2?7:5,fill:c,stroke:css('--panel'),'stroke-width':2},svg);
      if (labels[d[0]]){ const o=labels[d[0]]; const t=el('text',{x:cx+o[0],y:cy+o[1],'text-anchor':o[2],class:d[3]===2?'lab':''},svg); t.textContent=d[0]; if(d[3]===2) t.setAttribute('font-weight','600'); }
      const hit=el('circle',{cx,cy,r:13,fill:'transparent'},svg);
      hit.addEventListener('mousemove',e=>tip.show(tr(`${d[0]}${d[3]===1?'（CoT）':''}<br>${d[1]}B 参数 · 平均分 ${d[2]}`,`${d[0]}${d[3]===1?' (CoT)':''}<br>${d[1]}B parameters · mean ${d[2]}`),e));
      hit.addEventListener('mouseleave',()=>tip.hide());
    });
  }

  /* ============ §6 LiFT line ============ */
  function liftLine(){
    const box=$('#liftLine'); box.innerHTML='';
    const Ks=[1,2,3,4,5,8,16,32];
    const series=[
      {n:'L/2 R10',c:'--lift',tr:2,v:[31.83,20.30,14.64,12.25,11.70,10.95,10.99,11.57]},
      {n:'XL/2 R12',c:'--accent',tr:2,v:[27.71,17.07,12.74,10.69,10.25,9.55,9.31,9.71]},
    ];
    const W=760,H=340,L=50,Rr=150,Tp=18,B=42;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':tr('LiFT FID 随推理圈数变化','LiFT FID versus inference loops')},box);
    const lx=k=>L+Math.log2(k)/5*(W-L-Rr);
    const ly=v=>Tp+(34-v)/(34-6)*(H-Tp-B);
    [10,15,20,25,30].forEach(v=>{ el('line',{x1:L,y1:ly(v),x2:W-Rr,y2:ly(v),stroke:css('--grid')},svg); const t=el('text',{x:L-8,y:ly(v)+4,'text-anchor':'end',class:'muted'},svg); t.textContent=v; });
    Ks.forEach(k=>{ const t=el('text',{x:lx(k),y:H-B+20,'text-anchor':'middle',class:'muted'},svg); t.textContent=k; });
    el('line',{x1:L,y1:H-B,x2:W-Rr,y2:H-B,stroke:css('--ink-3')},svg);
    const xt=el('text',{x:W-Rr,y:H-6,'text-anchor':'end',class:'muted'},svg); xt.textContent=tr('推理圈数 K_inf','Inference loops K_inf');
    const yt=el('text',{x:L,y:12,class:'muted'},svg); yt.textContent='FID ↓';
    // dense refs
    [['dense L/2 · 16.94',16.94],['dense XL/2 · 15.54',15.54]].forEach((r,i)=>{
      el('line',{x1:L,y1:ly(r[1]),x2:W-Rr,y2:ly(r[1]),stroke:css('--dense'),'stroke-width':1.5},svg);
      const t=el('text',{x:W-Rr+8,y:ly(r[1])+(i===0?-2:10),class:'muted'},svg); t.textContent=r[0];
    });
    const fig=box.closest('.fig'); const tip=makeTip(fig);
    series.forEach(s=>{
      const c=css(s.c);
      el('path',{d:'M '+s.v.map((v,i)=>`${lx(Ks[i])} ${ly(v)}`).join(' L '),fill:'none',stroke:c,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
      s.v.forEach((v,i)=>{
        const cx=lx(Ks[i]), cy=ly(v), isTr=Ks[i]===s.tr;
        if (isTr) el('circle',{cx,cy,r:8,fill:css('--panel'),stroke:c,'stroke-width':2},svg);
        el('circle',{cx,cy,r:isTr?3:4,fill:c,stroke:isTr?'none':css('--panel'),'stroke-width':2},svg);
        const hit=el('circle',{cx,cy,r:12,fill:'transparent'},svg);
        hit.addEventListener('mousemove',e=>tip.show(tr(`${s.n} · K_inf = ${Ks[i]}${isTr?'（训练深度）':''}<br>FID ${v.toFixed(2)}`,`${s.n} · K_inf = ${Ks[i]}${isTr?' (training depth)':''}<br>FID ${v.toFixed(2)}`),e));
        hit.addEventListener('mouseleave',()=>tip.hide());
      });
      const last=s.v[s.v.length-1];
      const t=el('text',{x:lx(32)+10,y:ly(last)+(s.n==='XL/2 R12'?12:-4),class:'lab'},svg); t.textContent=s.n;
    });
    const a=el('text',{x:lx(2)+12,y:ly(20.3)-8,class:'muted'},svg); a.textContent=tr('训练深度','Training depth');
  }


  /* ============ §7 Diffusion-as-curriculum bars ============ */
  function dcChart(){
    const box=$('#dcChart'); box.innerHTML='';
    const groups=[
      {h:tr('训练时的腐蚀方式','Training corruption'),rows:[
        {n:tr('退火噪声（默认）','Annealed noise (default)'),v:99.54,e:0.02,on:1},
        {n:tr('固定最大噪声','Fixed maximum noise'),v:82.73,e:6.60},
        {n:'free-running',v:80.50,e:7.31},
        {n:tr('每步随机噪声水平','Random noise per step'),v:20.59,e:3.40},
        {n:tr('不加噪声','No noise'),v:0.00,e:0}]},
      {h:tr('推理时的噪声（同一模型）','Inference noise (same model)'),rows:[
        {n:tr('保持最大噪声 t=0','Maximum noise t=0'),v:99.90,on:1},
        {n:tr('退火噪声','Annealed noise'),v:99.56},
        {n:tr('不加噪声 t=1','No noise t=1'),v:49.39}]}
    ];
    const W=760,left=150,right=70,rowH=30,headH=30,top=8;
    let n=0; groups.forEach(g=>n+=g.rows.length);
    const H=top+groups.length*headH+n*rowH+34;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':tr('训练与推理噪声设置对数独解出率的影响','Sudoku solution rate by training and inference noise')},box);
    const x=v=>left+v/100*(W-left-right);
    [0,25,50,75,100].forEach(v=>{ el('line',{x1:x(v),y1:top,x2:x(v),y2:H-26,stroke:css('--grid')},svg); const t=el('text',{x:x(v),y:H-8,'text-anchor':'middle',class:'muted'},svg); t.textContent=v+'%'; });
    const fig=box.closest('.fig'); const tip=makeTip(fig);
    let y=top;
    groups.forEach(g=>{
      const ht=el('text',{x:0,y:y+20,class:'lab'},svg); ht.textContent=g.h; ht.setAttribute('font-weight','600');
      y+=headH;
      g.rows.forEach(d=>{
        const bh=16;
        const lt=el('text',{x:left-12,y:y+bh/2+4,'text-anchor':'end',class:'lab'},svg); lt.textContent=d.n;
        const w=Math.max(0,x(d.v)-left);
        if (w>4) el('path',{d:`M ${left} ${y} h ${w-4} q 4 0 4 4 v ${bh-8} q 0 4 -4 4 h ${-(w-4)} z`,fill:d.on?css('--dc'):css('--dense')},svg);
        else el('line',{x1:left,y1:y,x2:left,y2:y+bh,stroke:css('--dense'),'stroke-width':2},svg);
        if (d.e){ el('line',{x1:x(Math.max(0,d.v-d.e)),y1:y+bh/2,x2:x(Math.min(100,d.v+d.e)),y2:y+bh/2,stroke:css('--ink'),'stroke-width':1.2},svg);
          [d.v-d.e,d.v+d.e].forEach(v=>{ if(v>=0&&v<=100) el('line',{x1:x(v),y1:y+bh/2-4,x2:x(v),y2:y+bh/2+4,stroke:css('--ink'),'stroke-width':1.2},svg); }); }
        const vt=el('text',{x:x(Math.min(100,d.v+(d.e||0)))+8,y:y+bh/2+4,class:'lab'},svg); vt.textContent=d.v.toFixed(2);
        const hit=el('rect',{x:0,y:y-6,width:W,height:rowH,fill:'transparent'},svg);
        hit.addEventListener('mousemove',ev=>tip.show(tr(`${g.h} · ${d.n}<br>解出率 ${d.v.toFixed(2)}%${d.e?` ± ${d.e}`:''}`,`${g.h} · ${d.n}<br>Solution rate ${d.v.toFixed(2)}%${d.e?` ± ${d.e}`:''}`),ev));
        hit.addEventListener('mouseleave',()=>tip.hide());
        y+=rowH;
      });
    });
  }

  /* ============ §6 heatmap ============ */
  const Tg=[10,25,50,100], Kg=[1,2,4,8,16,32];
  const FID=[[48.613,33.633,18.955,15.289,15.094,16.435],[34.819,22.570,13.165,11.390,11.402,12.142],[31.828,20.304,12.255,10.952,10.994,11.567],[30.586,19.422,12.043,10.961,10.999,11.484]];
  const TF=[[0.942,1.614,2.959,5.648,11.027,21.785],[2.354,4.035,7.397,14.120,27.568,54.463],[4.708,8.070,14.793,28.241,55.136,108.926],[9.415,16.139,29.587,56.482,110.271,217.851]];
  const extra=[{T:50,K:3,f:14.642,c:11.431},{T:50,K:5,f:11.702,c:18.155}];
  const dense=[[10,1.614,29.591],[25,4.035,19.047],[50,8.069,16.940],[100,16.139,16.173],[250,40.347,15.743]];
  const ramp=['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5','#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b'];
  function fidColor(f){ const lo=Math.log(10.9), hi=Math.log(48.7); const t=1-(Math.log(f)-lo)/(hi-lo); const i=Math.round(Math.max(0,Math.min(1,t))*(ramp.length-1)); return {c:ramp[i], dark:i>=6}; }
  let heatCells=[];
  function heat(){
    const box=$('#heat'); box.innerHTML='';
    const W=520,H=300,L=58,Tp=36,cw=(W-L-8)/6,ch=(H-Tp-20)/4;
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':tr('积分步数与推理圈数网格上的 FID','FID over integration steps and inference loops')},box);
    Kg.forEach((k,j)=>{ const t=el('text',{x:L+j*cw+cw/2,y:Tp-12,'text-anchor':'middle',fill:css('--ink-2')},svg); t.textContent='K='+k; });
    Tg.forEach((tv,i)=>{ const t=el('text',{x:L-10,y:Tp+i*ch+ch/2+4,'text-anchor':'end',fill:css('--ink-2')},svg); t.textContent='T='+tv; });
    const fig=$('#heatFig'); const tip=makeTip(fig);
    heatCells=[];
    Tg.forEach((tv,i)=>Kg.forEach((k,j)=>{
      const f=FID[i][j], c=TF[i][j], col=fidColor(f);
      const g=el('g',{},svg);
      const r=el('rect',{x:L+j*cw+1,y:Tp+i*ch+1,width:cw-2,height:ch-2,rx:6,fill:col.c},g);
      const t=el('text',{x:L+j*cw+cw/2,y:Tp+i*ch+ch/2+1,'text-anchor':'middle',style:'fill:'+(col.dark?'#ffffff':'#0b0b0b'),'font-weight':500},g); t.textContent=f.toFixed(1);
      const t2=el('text',{x:L+j*cw+cw/2,y:Tp+i*ch+ch/2+16,'text-anchor':'middle',style:'fill:'+(col.dark?'#e6efff':'#24324a'),'font-size':10},g); t2.textContent=c<10?c.toFixed(2)+'TF':c.toFixed(1)+'TF';
      if (tv===50 && k===2) el('rect',{x:L+j*cw+3,y:Tp+i*ch+3,width:cw-6,height:ch-6,rx:5,fill:'none',stroke:'#0b0b0b','stroke-width':2.5},g);
      const ring=el('rect',{x:L+j*cw-1,y:Tp+i*ch-1,width:cw+2,height:ch+2,rx:7,fill:'none',stroke:css('--ldit'),'stroke-width':3,opacity:0},svg);
      g.addEventListener('mousemove',e=>tip.show(tr(`T = ${tv} 步 · K = ${k} 圈<br>FID ${f.toFixed(2)} · ${c.toFixed(2)} TFLOPs/图`,`T = ${tv} steps · K = ${k} loops<br>FID ${f.toFixed(2)} · ${c.toFixed(2)} TFLOPs/image`),e));
      g.addEventListener('mouseleave',()=>tip.hide());
      heatCells.push({T:tv,K:k,f,c,g,ring});
    }));
  }
  function budgetVal(){ const v=+$('#budget').value/1000; return Math.exp(Math.log(0.9)+v*(Math.log(220)-Math.log(0.9))); }
  function setBudgetTo(tf){ const v=(Math.log(tf)-Math.log(0.9))/(Math.log(220)-Math.log(0.9)); $('#budget').value=Math.round(v*1000); }
  function updateBudget(){
    const b=budgetVal();
    $('#bVal').textContent=(b<10?b.toFixed(2):b.toFixed(1))+' TFLOPs';
    let best=null;
    heatCells.forEach(c=>{ const ok=c.c<=b+1e-9; c.g.setAttribute('opacity',ok?1:0.22); c.ring.setAttribute('opacity',0); if(ok&&(!best||c.f<best.f)) best=c; });
    extra.forEach(e=>{ if(e.c<=b && (!best||e.f<best.f)) best={T:e.T,K:e.K,f:e.f,c:e.c,extra:true}; });
    if (best && best.ring) best.ring.setAttribute('opacity',1);
    let bd=null; dense.forEach(d=>{ if(d[1]<=b+1e-9 && (!bd||d[2]<bd[2])) bd=d; });
    $('#bLift').textContent = best ? `T=${best.T}, K=${best.K} · FID ${best.f.toFixed(2)}` : tr('预算不够一次采样','Budget below one sample');
    $('#bDense').textContent = bd ? `T=${bd[0]} · FID ${bd[2].toFixed(2)}` : tr('预算不够 10 步','Budget below 10 steps');
    let v='';
    if (best && bd){ const d=bd[2]-best.f; v = d>0 ? tr(`在这个预算内，LiFT 比 dense L/2 低 <strong>${d.toFixed(2)}</strong> FID，参数少约 41%。`,`Within this budget, LiFT has <strong>${d.toFixed(2)}</strong> lower FID than dense L/2, with about 41% fewer parameters.`) : tr(`在这个预算内，dense L/2 更好（低 ${(-d).toFixed(2)} FID）。预算很小时，把计算花在步数上更划算。`,`Within this budget, dense L/2 is better by ${(-d).toFixed(2)} FID. At tight budgets, spending compute on sampling steps is more effective.`); }
    else if (best) v=tr('dense L/2 在这个预算内连 10 步都跑不完。','The budget is too small for even 10 steps of dense L/2.');
    else v=tr('增加预算可以查看可用的采样设置。','Increase the budget to see available sampling settings.');
    if (best && best.extra) v+=tr(' （最佳点 K = '+best.K+' 只在 50 步下测过，不在网格里。）',' (The best point, K = '+best.K+', was measured only at 50 steps and is outside the grid.)');
    $('#bVerdict').innerHTML=v;
  }
  $('#budget').addEventListener('input',updateBudget);

  function drawAll(){ eltChart(); ldChart(); liftLine(); dcChart(); heat(); updateBudget(); renderEx(current); if(!playing){ drawImage(1); drawDial(T,1);} }
  setBudgetTo(8.07);
  drawAll();
  const mq=window.matchMedia('(prefers-color-scheme: dark)');
  if (mq.addEventListener) mq.addEventListener('change',drawAll);
  new MutationObserver(drawAll).observe(document.body,{attributes:true,attributeFilter:['class']});
})();
