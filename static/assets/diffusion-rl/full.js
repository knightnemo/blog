(function(){
'use strict';
var cfg=JSON.parse(document.getElementById('diffusion-full-spec').textContent);
var NS='http://www.w3.org/2000/svg';
var ROWORDER=cfg.ROWORDER,ROWLABEL=cfg.ROWLABEL,ATTR_LABEL=cfg.ATTR_LABEL,M=cfg.M,G=cfg.G;
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, anchor) {
    var t = el('text', { x: x, y: y, 'class': cls || 'g-t', 'text-anchor': anchor || 'middle' }, parent);
    t.textContent = s; return t;
  }
  function pick(v, c) { return Array.isArray(v) ? v[c] : v; }
  function defState(r) { return r === 'noise' || r === 'out' || r === 'env' ? 'data' : (r === 'res' ? 'act' : 'frozen'); }

  function renderGrid(key, host) {
    var S = M[key];
    host.innerHTML = '';
    var rows = ROWORDER.filter(function (r) { return r !== 'res' || S.res; });
    var cols = S.cols;
    var LW = 66, CW = 82, cw = 62, ch = 22, RH = 28, TOP = 26, SIDE = cfg.lang === "en" ? 235 : 150;
    var W = LW + cols.length * CW + SIDE;
    var H = TOP + rows.length * RH + (S.gae ? 24 : 8);
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': S.name + (cfg.lang === 'en' ? ' computation grid' : ' 的 t×k 计算网格') });
    var yOf = function (r) { return TOP + rows.indexOf(r) * RH; };
    var cxOf = function (c) { return LW + c * CW + CW / 2; };
    var sideX = LW + cols.length * CW + 4;
    var defs = el('defs', {}, svg);
    var mk = el('marker', { id: 'gm-' + key + '-' + host.id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', style: 'fill:var(--pathwise)' }, mk);
    var mk2 = el('marker', { id: 'gg-' + key + '-' + host.id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', style: 'fill:var(--accent)' }, mk2);
    var mk3 = el('marker', { id: 'ge-' + key + '-' + host.id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', style: 'fill:var(--muted)' }, mk3);

    // row labels and column headers
    rows.forEach(function (r) {
      var lab = r === 'env' && S.envLabel ? S.envLabel : ROWLABEL[r];
      txt(svg, 6, yOf(r) + 14 + 3, lab, 'g-label', 'start');
    });
    cols.forEach(function (c, i) { txt(svg, cxOf(i), 16, c, 'g-colhead'); });

    // groups (drawn first, behind)
    if (S.group) {
      var gc = S.group.cols || cols.map(function (_, i) { return i; });
      gc.forEach(function (c, idx) {
        var x = cxOf(c) - cw / 2 - 5, y = yOf(S.group.from) - 1, h = yOf(S.group.to) + ch + 7 - y, w = cw + 10;
        el('rect', { x: x + 4, y: y + 4, width: w, height: h, rx: 5, 'class': 'g-group' }, svg);
        el('rect', { x: x, y: y, width: w, height: h, rx: 5, 'class': 'g-group', style: 'fill:var(--surface)' }, svg);
        if (idx === 0) txt(svg, x + w + 2, 16, S.group.label, 'g-side', 'end');
      });
    }
    // connectors
    cols.forEach(function (_, c) {
      for (var i = 0; i + 1 < rows.length; i++) {
        el('line', { x1: cxOf(c), y1: yOf(rows[i]) + 3 + ch, x2: cxOf(c), y2: yOf(rows[i + 1]) + 3, 'class': 'g-conn' }, svg);
      }
    });
    // env transitions
    if (cols.length > 1) {
      for (var c = 0; c + 1 < cols.length; c++) {
        var ye = yOf('env') + 3 + ch / 2;
        el('line', { x1: cxOf(c) + cw / 2, y1: ye, x2: cxOf(c + 1) - cw / 2 - 1, y2: ye, 'class': 'g-envarrow', 'marker-end': 'url(#ge-' + key + '-' + host.id + ')' }, svg);
      }
    }
    // cells
    rows.forEach(function (r) {
      cols.forEach(function (_, c) {
        var st = pick(S.cells && S.cells[r] !== undefined ? S.cells[r] : defState(r), c) || defState(r);
        var x = cxOf(c) - cw / 2, y = yOf(r) + 3;
        var rect = el('rect', { x: x, y: y, width: cw, height: ch, rx: 4, 'class': 'g-cell-' + st }, svg);
        var op = S.op && S.op[r] !== undefined ? pick(S.op[r], c) : 1;
        if (st === 'act') rect.setAttribute('fill-opacity', op);
        var label = S.w && S.w[r] !== undefined ? pick(S.w[r], c) : '';
        if (label) {
          var cls = st === 'act' ? (op >= 0.7 ? 'g-t-on' : 'g-t') : (st === 'reuse' || st === 'frozen' ? 'g-t-muted' : 'g-t');
          txt(svg, cxOf(c), y + 15, label, cls);
        }
      });
    });
    // stars
    (S.stars || []).forEach(function (r) {
      cols.forEach(function (_, c) { txt(svg, cxOf(c) + cw / 2 - 1, yOf(r) + 9, '★', 'g-star'); });
    });
    // GAE arrows
    if (S.gae && cols.length > 1) {
      var yg = yOf('env') + 3 + ch + 12;
      for (var c2 = cols.length - 1; c2 > 0; c2--) {
        el('line', { x1: cxOf(c2) - 6, y1: yg, x2: cxOf(c2 - 1) + 8, y2: yg, 'class': 'g-gae', 'marker-end': 'url(#gg-' + key + '-' + host.id + ')' }, svg);
      }
      txt(svg, cxOf(cols.length - 1) + 10, yg + 4, 'GAE', 'g-side-acc', 'start');
    }
    // pathwise arrows
    (S.path || []).forEach(function (p) {
      var pc = p.cols || cols.map(function (_, i) { return i; });
      pc.forEach(function (c) {
        var x = cxOf(c) + cw / 2 + 6;
        el('line', { x1: x, y1: yOf(p.from) + 3, x2: x, y2: yOf(p.to) + 3 + ch / 2, 'class': 'g-path', 'marker-end': 'url(#gm-' + key + '-' + host.id + ')' }, svg);
      });
    });
    // brackets
    (S.brackets || []).forEach(function (b) {
      var y1 = yOf(b.from) + 4, y2 = yOf(b.to) + ch + 2;
      el('path', { d: 'M' + sideX + ',' + y1 + ' H' + (sideX + 5) + ' V' + y2 + ' H' + sideX, 'class': 'g-bracket' }, svg);
      txt(svg, sideX + 10, (y1 + y2) / 2 + 4, b.text, 'g-side-acc', 'start');
    });
    // critic pills
    (S.critic || []).forEach(function (cr) {
      var y = yOf(cr.row) + 3;
      var w = Math.min(SIDE - 14, 14 + cr.text.length * 7.2);
      el('rect', { x: sideX + 8, y: y + 1, width: w, height: ch - 2, rx: (ch - 2) / 2, 'class': 'g-pill' }, svg);
      txt(svg, sideX + 8 + w / 2, y + 15, cr.text, 'g-side-acc');
    });
    // side notes
    (S.side || []).forEach(function (s) { txt(svg, sideX + 8, yOf(s.row) + 18, s.text, 'g-side', 'start'); });
    // numbered notes
    (S.notes || []).forEach(function (n) {
      var x, y;
      if (n.side) { x = sideX + 4; y = yOf(n.side) + 3 + ch / 2; }
      else { x = cxOf(n.col) - cw / 2 + (n.dx !== undefined ? n.dx + cw / 2 : 0); y = yOf(n.row) + 3; }
      el('circle', { cx: x, cy: y, r: 7, 'class': 'g-note-c' }, svg);
      txt(svg, x, y + 3.5, String(n.n), 'g-note-t');
    });
    host.appendChild(svg);
    if (S.flow) {
      var f = document.createElement('div'); f.className = 'flow';
      S.flow.forEach(function (st) {
        var d = document.createElement('div'); d.className = 'st' + (st[2] ? ' upd' : '');
        var b = document.createElement('b'); b.textContent = st[0]; d.appendChild(b);
        d.appendChild(document.createTextNode(st[1])); f.appendChild(d);
      });
      host.appendChild(f);
    }
  }

  var counter = 0;
  document.querySelectorAll('.dr-full .gridfig[data-m]').forEach(function (h) {
    if (!h.id) h.id = 'gf' + (counter++);
    try { renderGrid(h.getAttribute('data-m'), h); } catch (e) { h.textContent = (cfg.lang === 'en' ? 'Diagram failed: ' : '图未能渲染：') + e.message; }
  });

  // ---------- genealogy ----------
  function nodeBox(id) { var n = G.nodes[id]; return { x: n[0], y: n[1], hw: 66, hh: n[2].length > 1 ? 22 : 17 }; }
  function clipTo(b, dx, dy) {
    var tx = dx === 0 ? Infinity : b.hw / Math.abs(dx), ty = dy === 0 ? Infinity : b.hh / Math.abs(dy);
    var t = Math.min(tx, ty); return [b.x + dx * t, b.y + dy * t];
  }
  function renderGenealogy(host) {
    var svg = el('svg', { viewBox: '0 0 ' + G.W + ' ' + G.H, role: 'img', 'aria-label': (cfg.lang === 'en' ? 'Method genealogy: lanes identify RL intervention points; edges describe mechanism changes' : '方法谱系图：泳道区分 RL 作用位置，边标注机制改动') });
    var defs = el('defs', {}, svg);
    var mk = el('marker', { id: 'gen-ah', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', style: 'fill:var(--muted)' }, mk);
    G.lanes.forEach(function (l) {
      el('rect', { x: 2, y: l[0], width: G.W - 4, height: l[1] - l[0], rx: 8, 'class': 'gen-lane' }, svg);
      txt(svg, 12, l[0] + 16, l[2], 'gen-lane-t', 'start');
    });
    var labels = document.createElementNS(NS, 'g');
    G.edges.forEach(function (e) {
      var a = nodeBox(e[0]), b = nodeBox(e[1]);
      var dx = b.x - a.x, dy = b.y - a.y;
      var p1 = clipTo(a, dx, dy), p2 = clipTo(b, -dx, -dy);
      var attrs = { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], 'class': 'gen-edge', 'marker-end': 'url(#gen-ah)' };
      if (e[4]) attrs['marker-start'] = 'url(#gen-ah)';
      el('line', attrs, svg);
      var t = e[3], lx = a.x + dx * t, ly = a.y + dy * t;
      var g = el('g', {}, labels);
      var bg = el('rect', { 'class': 'gen-ebg', rx: 3 }, g);
      var lines = e[2], n = lines.length;
      lines.forEach(function (s, i) { txt(g, lx, ly + 4 + (i - (n - 1) / 2) * 13, s, 'gen-elabel'); });
      g._bg = bg;
    });
    svg.appendChild(labels);
    Object.keys(G.nodes).forEach(function (id) {
      var b = nodeBox(id), n = G.nodes[id];
      var a = el('a', { href: '#' + G.href[id], 'class': 'gen-node' }, svg);
      el('rect', { x: b.x - b.hw, y: b.y - b.hh, width: b.hw * 2, height: b.hh * 2, rx: 7 }, a);
      n[2].forEach(function (s, i) { txt(a, b.x, b.y + 4.5 + (i - (n[2].length - 1) / 2) * 15, s, '', 'middle'); });
    });
    host.appendChild(svg);
    function sizeLabels() {
      Array.prototype.forEach.call(labels.childNodes, function (g) {
        try {
          var texts = g.querySelectorAll('text'); var x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
          texts.forEach(function (t) { var bb = t.getBBox(); x0 = Math.min(x0, bb.x); y0 = Math.min(y0, bb.y); x1 = Math.max(x1, bb.x + bb.width); y1 = Math.max(y1, bb.y + bb.height); });
          if (x1 > x0) { g._bg.setAttribute('x', x0 - 3); g._bg.setAttribute('y', y0 - 1); g._bg.setAttribute('width', x1 - x0 + 6); g._bg.setAttribute('height', y1 - y0 + 2); }
        } catch (err) {}
      });
    }
    sizeLabels();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(sizeLabels);
  }
  var gh = document.getElementById('genealogy-fig');
  if (gh) { try { renderGenealogy(gh); } catch (e) { gh.textContent = (cfg.lang === 'en' ? 'Genealogy failed: ' : '谱系图未能渲染：') + e.message; } }

  // ---------- comparator ----------
  var selA = document.getElementById('cmp-a'), selB = document.getElementById('cmp-b');
  if (selA && selB) {
    Object.keys(M).forEach(function (k) {
      if (k === 'example') return;
      [selA, selB].forEach(function (s) { var o = document.createElement('option'); o.value = k; o.textContent = M[k].name; s.appendChild(o); });
    });
    selA.value = 'ddpo'; selB.value = 'dppo';
    var update = function () {
      var a = selA.value, b = selB.value;
      document.getElementById('cmp-a-name').textContent = M[a].name;
      document.getElementById('cmp-b-name').textContent = M[b].name;
      document.getElementById('cmp-th-a').textContent = M[a].name;
      document.getElementById('cmp-th-b').textContent = M[b].name;
      renderGrid(a, document.getElementById('cmp-a-grid'));
      renderGrid(b, document.getElementById('cmp-b-grid'));
      var tb = document.querySelector('#cmp-table tbody'); tb.innerHTML = '';
      Object.keys(ATTR_LABEL).forEach(function (k) {
        var tr = document.createElement('tr');
        var va = M[a].attrs[k], vb = M[b].attrs[k];
        if (va !== vb) tr.className = 'differs';
        [ATTR_LABEL[k], va, vb].forEach(function (v) { var td = document.createElement('td'); td.textContent = v; tr.appendChild(td); });
        tb.appendChild(tr);
      });
    };
    selA.addEventListener('change', update); selB.addEventListener('change', update);
    update();
  }



  function finish() {
    var root=document.querySelector('.dr-full');
    if (typeof renderMathInElement === 'function') renderMathInElement(root, {
      delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],
      throwOnError:false, ignoredTags:['script','noscript','style','textarea','pre','code','option','svg']
    });
    var toc=root.querySelector('.toc details');
    if (toc) {
      var wide=window.matchMedia('(min-width:1200px)');
      var syncToc=function(){toc.open=wide.matches;};
      syncToc();wide.addEventListener('change',syncToc);
    }
    var headings=root.querySelectorAll('.research-content section[id],.research-content article[id]');
    if ('IntersectionObserver' in window) {
      var observer=new IntersectionObserver(function(entries){entries.forEach(function(entry){
        if(entry.isIntersecting){root.querySelectorAll('.toc a').forEach(function(a){a.classList.toggle('active',a.getAttribute('href')==='#'+entry.target.id);});}
      });},{rootMargin:'-5% 0px -75% 0px',threshold:0});
      headings.forEach(function(h){observer.observe(h);});
    }
    root.querySelectorAll('.fig-frame').forEach(function(frame){
      var svg=frame.querySelector(':scope > svg');if(!svg)return;
      var bar=document.createElement('div');bar.className='fig-tools';
      var scale=1, base=svg.viewBox.baseVal.width;
      [['−',-0.2],['100%',0],['+',0.2]].forEach(function(item){
        var b=document.createElement('button');b.type='button';b.textContent=item[0];
        b.setAttribute('aria-label',(cfg.lang==='zh'?'图示缩放 ':'Diagram zoom ')+item[0]);
        b.addEventListener('click',function(){scale=item[1]===0?1:Math.max(.6,Math.min(2.4,scale+item[1]));svg.style.width=Math.round(base*scale)+'px';svg.style.minWidth=Math.round(base*scale)+'px';});bar.appendChild(b);
      });frame.before(bar);
      frame.tabIndex=0;frame.setAttribute('aria-label',cfg.lang==='zh'?'可横向滚动的图示':'Horizontally scrollable diagram');
    });
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',finish); else finish();
})();
