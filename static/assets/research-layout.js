/* Derive navigation from the article, so translated headings stay in sync. */
(() => {
  const layout = document.querySelector('.research-layout');
  if (!layout) return;
  const directory = layout.querySelector('.research-directory');
  const nav = directory.querySelector('nav');
  const original = layout.querySelector('.dr-full .toc details > ol');
  if (original) {
    nav.append(original.cloneNode(true));
  } else {
    const list = document.createElement('ol');
    const loops = layout.querySelector('.ld-guide');
    const headings = loops
      ? loops.querySelectorAll('section[id] h2')
      : layout.querySelectorAll(layout.querySelector('.dex-language')
        ? '.post-content > :is(h2,h3,h4,h5)[id]'
        : '.post-content > :is(h2,h3)[id]');
    const names = {rudin:'rsl-rl',andrychowicz:'What Matters in On-Policy RL?','parallel-collection':'Parallel collection','staggered-resets':'Staggered resets',dexpbt:'DexPBT',sapg:'SAPG',epo:'EPO',omnireset:'OmniReset',sgs:'SGS',dextrah:'DextrAH',pql:'PQL',pqn:'PQN',fasttd3:'FastTD3',fastsac:'FastSAC',flashsac:'FlashSAC'};
    const parents = [{level: 1, list}];
    headings.forEach(heading => {
      const target = loops ? heading.closest('section[id]') : heading;
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = '#' + target.id;
      link.textContent = names[target.id] || heading.textContent.replace(/#\s*$/, '').trim();
      item.append(link);
      const level = loops ? 2 : Number(heading.tagName.slice(1));
      while (parents.length > 1 && parents[parents.length - 1].level >= level) parents.pop();
      parents[parents.length - 1].list.append(item);
      const children = document.createElement('ol');
      item.append(children);
      parents.push({level, list: children});
    });
    nav.append(list);
  }
  const links = [...nav.querySelectorAll('a[href^="#"]')];
  const entries = links.map(link => ({link, target:document.getElementById(decodeURIComponent(link.hash.slice(1)))})).filter(entry => entry.target);
  if (!entries.length) return;
  layout.querySelector('.research-sidebar').hidden = false;
  const wide = matchMedia('(min-width:1100px)');
  const sync = () => { directory.open = wide.matches; };
  wide.addEventListener('change', sync);
  sync();
  directory.addEventListener('toggle', () => {
    if (wide.matches && !directory.open) directory.open = true;
  });
  nav.addEventListener('click', event => {
    if (event.target.closest('a') && !wide.matches) directory.open = false;
  });
  let queued = false;
  const update = () => {
    let current = null;
    for (const entry of entries) {
      if (entry.target.getBoundingClientRect().top <= 140) current = entry;
    }
    entries.forEach(entry => {
      if (entry === current) entry.link.setAttribute('aria-current','location');
      else entry.link.removeAttribute('aria-current');
    });
    queued = false;
  };
  addEventListener('scroll', () => {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, {passive:true});
  update();
})();
