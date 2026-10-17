"""Regenerate the four English Robo VLM figures; Python standard library only.
Nunito is embedded so SVGs retain the blog's typography when opened separately.
"""
from pathlib import Path
from html import escape
import base64
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'static/assets/robo-centric-vlms'
FONT = base64.b64encode((OUT / 'nunito-latin.woff2').read_bytes()).decode()
INK, MUTED, LINE, BLUE, TINT = '#263742', '#637581', '#d9e1e4', '#216c8b', '#edf5f8'
SCENE, FILL = '#83949e', '#e7edef'

class SVG:
    def __init__(self, h, title, subtitle, width=900):
        self.s = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}" role="img" aria-label="{escape(title)}">
<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>
<defs><style>@font-face{{font-family:Nunito;src:url(data:font/woff2;base64,{FONT}) format('woff2');font-weight:200 1000;font-style:normal}}text{{font-family:Nunito,sans-serif}}</style>
<marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M1 1L6 3.5L1 6" fill="none" stroke="{BLUE}" stroke-width="1.3" stroke-linejoin="round"/></marker></defs>
<rect width="{width}" height="{h}" rx="8" fill="#fff"/>
<g fill="{INK}" stroke-linecap="round" stroke-linejoin="round">''']
        if width == 900:
            self.text(32, 46, title, 29, 750)
            self.text(32, 78, subtitle, 19, color=MUTED)
        else:
            end=self.lines(22,40,title,width=27,size=27,color=INK,leading=32,weight=750)
            self.lines(22,end+4,subtitle,width=41,size=18,leading=24)
    def text(self,x,y,t,size=20,weight=450,color=INK):
        self.s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(t)}</text>')
    def lines(self,x,y,t,width=30,size=20,color=MUTED,leading=27,weight=450):
        for paragraph in t.split('\n'):
            for line in textwrap.wrap(paragraph,width,break_long_words=False,break_on_hyphens=False) or ['']:
                self.text(x,y,line,size,weight,color);y+=leading
        return y
    def rect(self,x,y,w,h,fill='#fff',stroke=LINE,r=6):
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    def path(self,d,color=SCENE,width=2,arrow=False,dash=False):
        self.s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="4 5"' if dash else '')+'/>')
    def circle(self,x,y,r=4,fill=BLUE,stroke='none'):
        self.s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def legend(self,y,model=False):
        if model:
            self.rect(32,y-12,17,17,TINT,BLUE,3);self.text(59,y+2,'Robo VLM',17,color=MUTED)
            self.path(f'M190 {y-4}h27',BLUE,1.7,True);self.text(229,y+2,'Information flow',17,color=MUTED)
        else:
            self.rect(32,y-12,17,17,FILL,SCENE,3);self.text(59,y+2,'Scene / observation',17,color=MUTED)
            self.path(f'M270 {y-4}h24',BLUE,2);self.circle(282,y-4,4);self.text(306,y+2,'Prediction / target',17,color=MUTED)
        self.path(f'M32 {y+22}H868',LINE,1)
    def save(self,n):
        (OUT/f'i-{n}.svg').write_text(''.join(self.s)+'</g></svg>')

# Scene drawings share line weights, projection marks and output colors.
def sketch(a,n,x,y):
    a.s.append(f'<g transform="translate({x} {y})">')
    a.rect(0,0,152,140,'#f7f9fa','none',6)
    def cup(cx,cy):
        a.path(f'M{cx} {cy}v34q0 9 9 9h17q9 0 9-9v-34z',SCENE)
        a.path(f'M{cx+35} {cy+8}h5q15 12 0 23h-5',SCENE)
        a.path(f'M{cx+3} {cy+4}h29',LINE,1)
    def table(): a.path('M12 114H140',LINE,1.5)
    if n==1:
        table();cup(29,64)
        a.path('M100 91h34q-3 21-17 21t-17-21',SCENE)
        a.rect(20,55,62,65,'none',BLUE,2)
        a.text(21,43,'cup',17,650,BLUE)
    elif n==2:
        table()
        for xx,yy in [(31,90),(77,79),(122,92)]:
            a.path(f'M{xx} {yy-13}q-3-12 5-15',SCENE)
            a.circle(xx,yy,15,FILL,SCENE);a.circle(xx,yy,4)
        a.text(64,35,'3',24,700,BLUE)
    elif n==3:
        a.path('M12 129L71 50L142 129M18 101H126M37 76H105',LINE,1)
        a.rect(65,45,35,38,FILL,SCENE,2)
        a.rect(12,14,20,15,'#fff',SCENE,3)
        a.path('M32 19l8-4v14l-8-4',SCENE)
        a.path('M31 33L80 62',BLUE,2,True,True);a.circle(82,64,4)
        a.text(8,63,'2.4 m',17,650,BLUE)
    elif n==4:
        a.rect(13,49,57,35,FILL,SCENE,4);a.rect(9,65,65,24,FILL,SCENE,4)
        a.path('M17 90v12M66 90v12M102 79h36M107 79v25M132 79v25',SCENE)
        a.path('M72 115H133M72 110v10M133 110v10',BLUE)
        a.text(76,136,'1.5 m',17,650,BLUE)
    elif n==5:
        for i in range(3):
            xx=8+i*48;a.rect(xx,29,41,74,'#fff',LINE,3)
            a.path(f'M{xx+3} 82h35',LINE,1)
            a.rect(xx+19,68-i*6,11,20,FILL,SCENE,2)
            a.text(xx+8,123,f't{i+1}',16,color=MUTED)
        a.rect(119,52,20,29,'none',BLUE,2)
    elif n==6:
        a.path('M28 63l31-24 64 0-31 24zM28 63v48l64 0 31-24V39M92 63v48',BLUE,2)
        a.path('M59 39v48h64M59 87l-31 24',BLUE,1.5,dash=True)
        a.circle(76,76,4);a.path('M76 76h33M76 76V49M76 76l-16 15',BLUE,1.4,True)
    elif n==7:
        table();cup(41,63);a.circle(91,83,5)
        a.path('M90 31v28M78 59v15M102 59v15M78 59h24',BLUE,2.3)
        a.path('M106 37q12 12 9 25',BLUE,1.5);a.text(124,54,'θ',17,650,BLUE)
    elif n==8:
        table();a.path('M106 54v58M91 113h30M83 54l11-28h25l12 28z',SCENE)
        a.path('M21 84l41-7 15 23-43 7z',BLUE,1.6,dash=True)
        for xx,yy in [(34,96),(51,90),(64,99)]:a.circle(xx,yy,3.5)
    elif n==9:
        table();cup(14,71);a.rect(108,104,33,8,FILL,SCENE,3)
        a.path('M31 66Q60 4 96 32T127 91',BLUE,2,True)
        for xx,yy in [(31,66),(54,33),(80,24),(108,47)]:a.circle(xx,yy,3.5)
    elif n==10:
        for xx,yy in [(29,87),(75,74),(122,92)]:
            a.path(f'M{xx} {yy}v32m0-12l-8-7m8 13l8-7',SCENE,1.5)
            for dx,dy in [(-5,0),(5,0),(0,-5),(0,5)]:a.circle(xx+dx,yy+dy,4,FILL,SCENE)
        a.path('M24 57Q65 20 125 56',BLUE,2,True)
        for xx,yy in [(30,53),(69,38),(107,46)]:a.circle(xx,yy,3.5)
        a.path('M75 45v18',BLUE,1.3,dash=True)
        a.text(57,24,'1–5 cm',16,650,BLUE)
    elif n==11:
        a.rect(9,69,82,53,FILL,SCENE,5);a.rect(17,77,65,36,'#fff',LINE,4)
        a.rect(29,87,31,18,TINT,BLUE,2)
        a.path('M108 53h34M111 46v16M122 46v16M133 46v16',SCENE)
        a.path('M49 81Q73 29 122 37',BLUE,2,True)
        a.circle(123,37,4)
    else:
        for xx in [9,86]:
            a.rect(xx,17,57,58,'#fff',LINE,4);a.path(f'M{xx+6} 61h45',LINE,1)
        a.rect(20,45,17,15,FILL,SCENE,2);a.rect(113,29,17,31,TINT,BLUE,2)
        a.text(10,96,'before',15,color=MUTED);a.text(95,96,'after',15,color=MUTED)
        a.rect(11,111,130,7,FILL,'none',3);a.rect(11,111,78,7,BLUE,'none',3)
        a.text(111,136,'0.6',17,650,BLUE)
    a.s.append('</g>')
rows = [('01  DATA SOURCES', [('Public datasets', 'RefCOCO, RoboPoint\nVSI-590K, ShareRobot'), ('Geometry', 'RGB-D, 3D boxes\nExact simulation poses'), ('Robot & human video', 'Open-X, AgiBot, DROID\nEgo4D first-person video'), ('Model generation', 'Rewrite instructions\nGenerate QA and CoT')]), ('02  ANNOTATION & VERIFICATION', [('Automatic labels', 'Detect, segment, track\nSLAM and depth'), ('Teacher checks', 'Strong-VLM verification\nField-wise teacher votes'), ('Human review', 'Points, paths, keyframes\nReview uncertain cases'), ('Shared formats', 'Text + coordinates\nFrame indices / depth')]), ('03  TRAINING', [('General VLM', 'Qwen-VL, Hunyuan\nExisting pretrained weights'), ('Embodied training', 'Perception + embodied\n+ general supervision'), ('Reasoning SFT', 'Checked reasoning traces\nHY: ~100K cold-start data'), ('RL where used', 'GRPO: HY, RynnBrain,\nGroundingPI; task rewards')])]
tasks = [('Detection / referring', 'All four', 'Box the cup on the table.', 'Cup box:\n[129, 393, 357, 750]'), ('Pointing / counting', 'HY, RoboBrain, others', 'Point to all apples.', '(250, 625), (500, 589),\n(750, 634): 3 apples'), ('Depth estimation', 'HY', 'How far is the marked point?', 'About 2.4 m.\nTargets: LiDAR / RGB-D'), ('Spatial QA', 'All four', 'How far is the sofa from the table?', 'About 1.5 m.\nAlso direction, size, count'), ('Video localization', 'RynnBrain', 'Where is the remote control?', 'Frame 3, box:\n[757, 518, 914, 643]'), ('3D box grounding', 'RynnBrain 1.1, HY', 'Locate the carton (image + intrinsics).', 'Camera-frame 3D box:\ncenter, dimensions, rotation'), ('Affordance / grasp point', 'Across the models', 'Where should I grasp the cup?', 'Contact: (664, 589).\nRynnBrain 1.1 adds angle θ'), ('Placement region', 'RynnBrain, RoboBrain', 'Put the book left of the lamp.', 'Region points:\n(286, 625), (357, 643), …'), ('2D trajectory', 'Trajectory-capable models', 'Put the cup on the tray.', 'Up to 10 waypoints here:\n(214, 482), (357, 357), …'), ('3D trajectory', 'RoboBrain 2.5', 'Water left to right, spout 1–5 cm above flowers.', 'Sequence of (u, v, d);\nd is absolute depth in m'), ('Grounded planning', 'RynnBrain, HY, others', 'Tidy the sink (with observations).', 'Pick up sponge [box];\nplace on rack at (x, y)'), ('Progress / value', 'RoboBrain 2.5', 'Start, goal, and before/after frames.', 'Hop: +0.4 of remainder;\naccumulated progress: 0.6')]
lanes = [('01  HIGH-LEVEL PLANNER', 1, [('Observation', 'Camera image\n+ task instruction'), ('Robo VLM', 'Subtask + target box\n/ placement points'), ('Low-level executor', 'VLA policy\nor motion planner'), ('Robot action', 'Execute, observe again,\nrequest the next step')], 'Example: RynnBrain-Plan; grounded plans condition a downstream VLA.'), ('02  VLA BACKBONE', 0, [('Robo VLM weights', 'Initialize the\nvision-language backbone'), ('Action expert', 'Add a flow-matching\naction head / DiT'), ('Fine-tuning', 'Teleoperated or simulated\nrobot demonstrations'), ('Action output', 'Joint / end-effector\naction chunks')], 'Examples: RynnBrain-VLA (G1, Astribot, Tianji-Wuji), HY VLA (Xtrainer), GroundingPI + DiT.'), ('03  REWARD / PROGRESS MODEL', 1, [('Policy rollout', 'Simulation or real robot\ntrajectory video'), ('Robo VLM', 'Estimate task progress\nfrom observations'), ('Dense reward', 'Use progress increments\nas a learning signal'), ('Policy update', 'Run another round of RL,\nthen collect new rollouts')], 'Example: RoboBrain 2.5 hops; disagreeing anchor estimates trigger conservative updates.')]

# Recipe: two readable columns per stage instead of four compressed columns.
a=SVG(1320,'From data to a Robo-centric VLM', 'Images / video + instructions  →  text + spatial targets')
for ri,(name,cols) in enumerate(rows):
    y=112+ri*294
    number,label=name.split('  ')
    a.text(32,y+20,number,21,700,BLUE)
    a.lines(32,y+54,label.title().replace(' & ', ' &\n'),width=14,size=20,color=INK,weight=700,leading=26)
    if ri<2:a.path(f'M48 {y+125}v125',LINE,1.5)
    for j,(title,body) in enumerate(cols):
        x=204+(j%2)*340;yy=y+(j//2)*130
        a.rect(x,yy,324,112,'#f8fafb','none')
        a.text(x+18,yy+31,title,21,700)
        a.lines(x+18,yy+61,body,width=30,size=19,leading=25)
    if ri<2:a.path(f'M530 {y+251}v29',BLUE,1.5,True)
a.path('M530 951v31',BLUE,1.5,True)
a.rect(204,994,664,86,TINT,'#b8d1dd')
a.text(226,1028,'Robo-centric VLM',25,750,BLUE)
a.text(226,1058,'Grounded plans, points, boxes, paths, progress',20,color=BLUE)
a.path('M536 1081v26M168 1107H732M168 1107v21M450 1107v21M732 1107v21',BLUE,1.5)
for x,title,body in [(32,'High-level planner','Grounded subgoals → low-level policy'),(316,'VLA backbone','Initialize a VLA, then add an action head'),(600,'Reward / progress','Estimate completion → dense rewards')]:
    a.rect(x,1130,268,145)
    a.text(x+18,1163,title,21,700)
    a.lines(x+18,1197,body,width=24,size=20,leading=27)
a.save(0)

for group in range(2):
    a=SVG(1430,'Perception & space' if group==0 else 'Manipulation & time','Image / video + query  →  structured answer')
    a.legend(111)
    for i,(title,who,query,answer) in enumerate(tasks[group*6:group*6+6]):
        n=group*6+i+1;x=32+(i%2)*430;y=157+(i//2)*410
        a.rect(x,y,406,390)
        a.text(x+18,y+34,f'{n:02d}',18,700,BLUE)
        a.text(x+53,y+34,title,22,700)
        a.text(x+18,y+63,who,17,color=MUTED)
        sketch(a,n,x+18,y+89)
        a.text(x+187,y+108,'QUERY',14,750,MUTED)
        a.lines(x+187,y+136,query,width=20,size=20,color=INK,leading=27)
        a.rect(x+16,y+252,374,119,TINT,'none',5)
        a.text(x+30,y+276,'OUTPUT',14,750,BLUE)
        a.lines(x+30,y+305,answer,width=34,size=20,color=BLUE,leading=26)
    a.text(32,1410,'Schematic examples; numerical values are illustrative.',17,color=MUTED)
    a.save(group+1)

# Deployment: a consistent reading direction and one highlighted VLM stage.
a=SVG(1140,'Three ways to use a Robo VLM','The same model plays a different role in each pipeline.')
a.legend(111,True)
for i,(label,active,cols,note) in enumerate(lanes):
    y=158+i*318
    num,title=label.split('  ')
    a.text(32,y+22,num,18,700,BLUE);a.text(67,y+22,title.title().replace("Vla", "VLA"),24,750)
    for k,(title,body) in enumerate(cols):
        x=32+k*214;by=y+47
        a.rect(x,by,194,170,TINT if k==active else '#f8fafb','#a7c7d6' if k==active else 'none')
        a.lines(x+14,by+30,title.replace("Robo VLM weights", "Robo VLM\nweights"),width=17,size=21,color=BLUE if k==active else INK,leading=26,weight=750)
        a.lines(x+14,by+87,body.replace("\n", " "),width=18,size=19,leading=25)
        if k<3:a.path(f'M{x+198} {by+80}h12',BLUE,1.4,True)
    a.lines(32,y+251,note,width=86,size=18,leading=25)
    if i<2:a.path(f'M32 {y+295}H868',LINE,1)
a.save(3)

# Narrow screens get the same information in one column, without tiny type.
for group in range(2):
    a=SVG(2650,'Perception & space' if group==0 else 'Manipulation & time','Image / video + query → structured answer',450)
    a.rect(22,109,16,16,FILL,SCENE,3);a.text(49,122,'Scene / observation',17,color=MUTED)
    a.path('M23 148h22',BLUE,2);a.circle(34,148,4);a.text(56,154,'Prediction / target',17,color=MUTED)
    for i,(title,who,query,answer) in enumerate(tasks[group*6:group*6+6]):
        x=22;y=180+i*400;n=group*6+i+1
        a.rect(x,y,406,382)
        a.text(x+18,y+34,f'{n:02d}',18,700,BLUE);a.text(x+53,y+34,title,22,700)
        a.text(x+18,y+63,who,17,color=MUTED);sketch(a,n,x+18,y+89)
        a.text(x+187,y+108,'QUERY',14,750,MUTED)
        a.lines(x+187,y+136,query,width=20,size=20,color=INK,leading=27)
        a.rect(x+16,y+248,374,117,TINT,'none',5);a.text(x+30,y+272,'OUTPUT',14,750,BLUE)
        a.lines(x+30,y+301,answer,width=34,size=20,color=BLUE,leading=26)
    a.lines(22,2605,'Schematic examples; numerical values are illustrative.',width=43,size=17,leading=23)
    a.save(f'{group+1}-mobile')

a=SVG(2450,'From data to a Robo-centric VLM','Images / video + instructions → text + spatial targets',450)
for ri,(name,cols) in enumerate(rows):
    y=167+ri*554;num,label=name.split('  ')
    a.text(22,y,num,18,750,BLUE);a.text(55,y,label.title(),22,750)
    for j,(title,body) in enumerate(cols):
        yy=y+23+j*122;a.rect(22,yy,406,106,'#f8fafb','none')
        a.text(40,yy+30,title,22,700);a.lines(40,yy+60,body,width=38,size=20,leading=26)
    a.path(f'M225 {y+499}v25',BLUE,1.5,True)
a.rect(22,1826,406,133,TINT,'#b8d1dd');a.text(40,1861,'Robo-centric VLM',25,750,BLUE)
a.lines(40,1896,'Grounded plans, points, boxes, paths, progress',width=35,size=21,color=BLUE)
for j,(title,body) in enumerate([('High-level planner','Grounded subgoals → low-level policy'),('VLA backbone','Initialize a VLA, then add an action head'),('Reward / progress model','Estimate completion → dense rewards')]):
    y=1998+j*139;a.rect(22,y,406,121);a.text(40,y+35,title,23,700);a.lines(40,y+69,body,width=35,size=21,leading=27)
a.save('0-mobile')

a=SVG(2255,'Three ways to use a Robo VLM','The same model plays a different role in each pipeline.',450)
a.rect(22,140,17,17,TINT,BLUE,3);a.text(49,155,'Robo VLM',17,color=MUTED)
a.path('M177 148h25',BLUE,1.5,True);a.text(213,155,'Information flow',17,color=MUTED)
for i,(label,active,cols,note) in enumerate(lanes):
    y=201+i*676;num,title=label.split('  ')
    a.text(22,y,num,18,700,BLUE);a.text(57,y,title.title().replace("Vla", "VLA"),23,750)
    for k,(title,body) in enumerate(cols):
        by=y+23+k*122
        a.rect(22,by,406,104,TINT if k==active else '#f8fafb','#a7c7d6' if k==active else 'none')
        a.text(40,by+31,title,22,750,BLUE if k==active else INK)
        a.lines(40,by+61,body.replace('\n',' '),width=37,size=20,leading=26)
        if k<3:a.path(f'M225 {by+108}v9',BLUE,1.5,True)
    a.lines(22,y+535,note,width=42,size=18,leading=25)
    if i<2:a.path(f'M22 {y+640}H428',LINE,1)
a.save('3-mobile')
