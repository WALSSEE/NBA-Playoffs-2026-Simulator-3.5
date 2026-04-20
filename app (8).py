import streamlit as st
import pandas as pd
import numpy as np
import io, json, base64, zlib

st.set_page_config(page_title="NBA Playoffs 2026", page_icon="🏀",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
h1,h2,h3{font-family:'Bebas Neue',sans-serif;letter-spacing:2px;}
.stApp{background:#0a0a0f;color:#e8e8e8;}
[data-testid="stSidebar"]{background:#111118!important;border-right:1px solid #222230;}
.mc{background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #2a2a4a;
    border-radius:12px;padding:18px;text-align:center;margin-bottom:8px;}
.mv{font-family:'Bebas Neue',sans-serif;font-size:2.4rem;color:#f97316;line-height:1;margin:0;}
.ml{font-size:0.7rem;color:#8888aa;text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;}
hr{border-color:#222230!important;}
.stButton>button{background:linear-gradient(135deg,#f97316,#ea580c)!important;color:white!important;
 border:none!important;font-family:'Bebas Neue',sans-serif!important;font-size:1rem!important;
 letter-spacing:2px!important;border-radius:8px!important;padding:8px 20px!important;}
.stTabs [data-baseweb="tab-list"]{background:#111118;border-bottom:2px solid #f97316;}
.stTabs [data-baseweb="tab"]{font-family:'Bebas Neue',sans-serif;letter-spacing:1.5px;color:#8888aa!important;}
.stTabs [aria-selected="true"]{color:#f97316!important;background:#1a1a2e!important;}
.pibox{background:linear-gradient(135deg,#1a0a2e,#0a1a2e);border:1px solid #7c3aed;
       border-radius:10px;padding:12px 16px;margin:8px 0;}
.live-box{background:linear-gradient(135deg,#0a1a0a,#0f2a0f);border:1px solid #22c55e;
          border-radius:10px;padding:14px 18px;margin:8px 0;}
.series-score{font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#f97316;text-align:center;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATE ENCODING — pack all ratings into a short URL-safe string
# ══════════════════════════════════════════════════════════════════════════════

def encode_state(data: dict) -> str:
    """Compress and base64-encode state dict for URL storage."""
    raw = json.dumps(data, separators=(',', ':'))
    compressed = zlib.compress(raw.encode(), level=9)
    return base64.urlsafe_b64encode(compressed).decode()

def decode_state(s: str) -> dict:
    """Decode URL state string back to dict."""
    compressed = base64.urlsafe_b64decode(s.encode())
    raw = zlib.decompress(compressed).decode()
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════════
# MATH
# ══════════════════════════════════════════════════════════════════════════════

def win_prob(nr_a: float, nr_b: float, home_adv: float) -> float:
    return float(np.clip(1 / (1 + np.exp(-(nr_a + home_adv - nr_b) * 0.116)), 0.001, 0.999))


def series_probs(nr_home: float, nr_away: float, home_adv: float,
                 variance: float = 0.0, best_of: int = 7,
                 wins_home: int = 0, wins_away: int = 0) -> dict:
    """
    Exact series probs from CURRENT game state (wins_home, wins_away already played).
    Margin lines:
      +3.5 → sweep 4-0 only
      +2.5 → 4-0 or 4-1
      +1.5 → 4-0, 4-1 or 4-2
    """
    wn = (best_of + 1) // 2
    # Already done?
    if wins_home == wn: return {'p_home':1.0,'p_away':0.0,'outcomes':{(wn,wins_away):1.0},
        'p_home_p35':float(wn-wins_away>3.5),'p_home_p25':float(wn-wins_away>2.5),
        'p_home_p15':float(wn-wins_away>1.5),'p_away_p35':0.,'p_away_p25':0.,'p_away_p15':0.}
    if wins_away == wn: return {'p_home':0.0,'p_away':1.0,'outcomes':{(wins_home,wn):1.0},
        'p_home_p35':0.,'p_home_p25':0.,'p_home_p15':0.,
        'p_away_p35':float(wn-wins_home>3.5),'p_away_p25':float(wn-wins_home>2.5),
        'p_away_p15':float(wn-wins_home>1.5)}

    schedule = {1:True,2:True,3:False,4:False,5:True,6:False,7:True}  # True=home court
    ph = win_prob(nr_home, nr_away,  home_adv) * (1-variance*0.5) + 0.5*variance*0.5
    pa = win_prob(nr_home, nr_away, -home_adv) * (1-variance*0.5) + 0.5*variance*0.5

    games_played = wins_home + wins_away
    states = {(wins_home, wins_away): 1.0}
    outcomes = {}

    for g in range(games_played + 1, best_of + 1):
        pw = ph if schedule[g] else pa
        ns = {}
        for (wh, wa), prob in states.items():
            for dwh, dwa, p in [(1,0,pw),(0,1,1-pw)]:
                nwh, nwa = wh+dwh, wa+dwa
                if nwh==wn or nwa==wn:
                    outcomes[(nwh,nwa)] = outcomes.get((nwh,nwa),0) + prob*p
                else:
                    ns[(nwh,nwa)] = ns.get((nwh,nwa),0) + prob*p
        states = ns

    ph_win = sum(v for (wh,wa),v in outcomes.items() if wh==wn)
    pa_win = sum(v for (wh,wa),v in outcomes.items() if wa==wn)
    aw = {k:v for k,v in outcomes.items() if k[0]==wn}
    bw = {k:v for k,v in outcomes.items() if k[1]==wn}

    def mgn_h(d, t): return sum(v for (wh,wa),v in d.items() if (wh-wa) > t)
    def mgn_a(d, t): return sum(v for (wh,wa),v in d.items() if (wa-wh) > t)

    return {
        'p_home': ph_win, 'p_away': pa_win, 'outcomes': outcomes,
        'p_home_p35': mgn_h(aw,3.5), 'p_home_p25': mgn_h(aw,2.5), 'p_home_p15': mgn_h(aw,1.5),
        'p_away_p35': mgn_a(bw,3.5), 'p_away_p25': mgn_a(bw,2.5), 'p_away_p15': mgn_a(bw,1.5),
    }


def playin_exact(t7, t8, t9, t10, nr, home_adv):
    p78=win_prob(nr[t7],nr[t8],home_adv); p910=win_prob(nr[t9],nr[t10],home_adv)
    p89=win_prob(nr[t8],nr[t9],home_adv); p810=win_prob(nr[t8],nr[t10],home_adv)
    p79=win_prob(nr[t7],nr[t9],home_adv); p710=win_prob(nr[t7],nr[t10],home_adv)
    r={t:{'p7':0.,'p8':0.} for t in [t7,t8,t9,t10]}
    r[t7]['p7']+=p78; r[t8]['p7']+=(1-p78)
    r[t8]['p8']+=p78*p910*p89;      r[t9]['p8'] +=p78*p910*(1-p89)
    r[t8]['p8']+=p78*(1-p910)*p810; r[t10]['p8']+=p78*(1-p910)*(1-p810)
    r[t7]['p8']+=(1-p78)*p910*p79;      r[t9]['p8'] +=(1-p78)*p910*(1-p79)
    r[t7]['p8']+=(1-p78)*(1-p910)*p710; r[t10]['p8']+=(1-p78)*(1-p910)*(1-p710)
    for t in r: r[t]['p_qualify']=r[t]['p7']+r[t]['p8']
    return r


def sim_series_mc(home, away, nr, home_adv, variance, rng, best_of=7,
                  wh_start=0, wa_start=0):
    """Simulate series from current score (wh_start, wa_start)."""
    wn=(best_of+1)//2; wh=wh_start; wa=wa_start
    ph=win_prob(nr[home],nr[away], home_adv)*(1-variance*0.5)+0.5*variance*0.5
    pa=win_prob(nr[home],nr[away],-home_adv)*(1-variance*0.5)+0.5*variance*0.5
    schedule=[True,True,False,False,True,False,True]
    g = wh_start + wa_start  # games already played
    while wh<wn and wa<wn:
        if rng.random()<(ph if schedule[g] else pa): wh+=1
        else: wa+=1
        g+=1
    return home if wh==wn else away


def sim_playin_mc(t7,t8,t9,t10,nr,home_adv,rng):
    g1=t7 if rng.random()<win_prob(nr[t7],nr[t8],home_adv) else t8
    g1l=t8 if g1==t7 else t7
    g2=t9 if rng.random()<win_prob(nr[t9],nr[t10],home_adv) else t10
    g3=g1l if rng.random()<win_prob(nr[g1l],nr[g2],home_adv) else g2
    return g1,g3


def sim_full(east, west, home_adv, variance, n_sim,
             locked_e=None, locked_w=None, scores=None):
    """
    Path-aware MC. locked_X keys: playin7/8, r1_1v8/4v5/2v7/3v6, r2_A/B, cf
    scores: dict of {series_key: (wh, wa)} for in-progress series
    """
    rng=np.random.default_rng(42)
    nr={t['name']:t['nr'] for t in east+west}
    locked_e=locked_e or {}; locked_w=locked_w or {}
    scores=scores or {}

    cw_e={t['name']:0 for t in east}; cw_w={t['name']:0 for t in west}
    nba_w={t['name']:0 for t in east+west}
    rnd_e={t['name']:{1:0,2:0,3:0} for t in east}
    rnd_w={t['name']:{1:0,2:0,3:0} for t in west}
    pq_e={t['name']:{7:0,8:0} for t in east}
    pq_w={t['name']:{7:0,8:0} for t in west}
    e_seed={t['seed']:t['name'] for t in east}
    w_seed={t['seed']:t['name'] for t in west}

    def sim_conf(seed_map, cw, rnd, pq, locked, conf_prefix):
        pi7,pi8,pi9,pi10=seed_map[7],seed_map[8],seed_map[9],seed_map[10]
        if "playin7" in locked and "playin8" in locked:
            q7,q8=locked["playin7"],locked["playin8"]
        elif "playin7" in locked:
            q7=locked["playin7"]
            _,q8=sim_playin_mc(pi7,pi8,pi9,pi10,nr,home_adv,rng)
            if q8==q7: _,q8=sim_playin_mc(pi7,pi8,pi9,pi10,nr,home_adv,rng)
        elif "playin8" in locked:
            q8=locked["playin8"]
            q7,_=sim_playin_mc(pi7,pi8,pi9,pi10,nr,home_adv,rng)
            if q7==q8: q7,_=sim_playin_mc(pi7,pi8,pi9,pi10,nr,home_adv,rng)
        else:
            q7,q8=sim_playin_mc(pi7,pi8,pi9,pi10,nr,home_adv,rng)
        pq[q7][7]+=1; pq[q8][8]+=1

        bseed={seed_map[s]:s for s in range(1,7)}
        bseed[q7]=7; bseed[q8]=8

        def ps(a,b,lock_key,score_key=None):
            if lock_key in locked: return locked[lock_key]
            h=a if bseed.get(a,99)<bseed.get(b,99) else b
            aw=b if h==a else a
            sk=f"{conf_prefix}_{score_key}" if score_key else None
            wh_s,wa_s=(scores[sk] if sk and sk in scores else (0,0))
            return sim_series_mc(h,aw,nr,home_adv,variance,rng,7,wh_s,wa_s)

        s1,s2,s3,s4,s5,s6=(seed_map[i] for i in range(1,7))
        w1=ps(s1,q8,"r1_1v8","r1_1v8"); w2=ps(s4,s5,"r1_4v5","r1_4v5")
        w3=ps(s2,q7,"r1_2v7","r1_2v7"); w4=ps(s3,s6,"r1_3v6","r1_3v6")
        for w in [w1,w2,w3,w4]: rnd[w][1]+=1
        wA=ps(w1,w2,"r2_A","r2_A"); wB=ps(w3,w4,"r2_B","r2_B")
        rnd[wA][2]+=1; rnd[wB][2]+=1
        cf=ps(wA,wB,"cf","cf"); rnd[cf][3]+=1; cw[cf]+=1
        return cf

    for _ in range(n_sim):
        e_cf=sim_conf(e_seed,cw_e,rnd_e,pq_e,locked_e,"east")
        w_cf=sim_conf(w_seed,cw_w,rnd_w,pq_w,locked_w,"west")
        sk="finals"
        wh_f,wa_f=scores.get(sk,(0,0)) if scores else (0,0)
        champ=sim_series_mc(e_cf if nr[e_cf]>=nr[w_cf] else w_cf,
                            w_cf if nr[e_cf]>=nr[w_cf] else e_cf,
                            nr,home_adv,variance,rng,7,wh_f,wa_f)
        nba_w[champ]+=1

    return cw_e,cw_w,nba_w,rnd_e,rnd_w,pq_e,pq_w


def pct(v,n=1): return f"{v*100:.{n}f}%"


# ══════════════════════════════════════════════════════════════════════════════
# 2026 BRACKET
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS={
    "west":[
        ("Oklahoma City Thunder",1,0.0),("San Antonio Spurs",2,0.0),
        ("Denver Nuggets",3,0.0),("Los Angeles Lakers",4,0.0),
        ("Houston Rockets",5,0.0),("Minnesota Timberwolves",6,0.0),
        ("Phoenix Suns",7,0.0),("Portland Trail Blazers",8,0.0),
        ("LA Clippers",9,0.0),("Golden State Warriors",10,0.0),
    ],
    "east":[
        ("Detroit Pistons",1,0.0),("Boston Celtics",2,0.0),
        ("Atlanta Hawks",3,0.0),("Cleveland Cavaliers",4,0.0),
        ("New York Knicks",5,0.0),("Toronto Raptors",6,0.0),
        ("Philadelphia 76ers",7,0.0),("Orlando Magic",8,0.0),
        ("Charlotte Hornets",9,0.0),("Miami Heat",10,0.0),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# STATE INIT & URL PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def collect_state() -> dict:
    """Collect all user-editable state."""
    d={}
    for ck in ("west","east"):
        for i in range(10):
            for s in "nsr":
                k=f"{ck}{i}{s}"
                if k in st.session_state: d[k]=st.session_state[k]
    for k in ("locked_west","locked_east","series_scores"):
        if k in st.session_state and st.session_state[k]:
            d[k]=st.session_state[k]
    return d

def apply_state(d: dict):
    for k,v in d.items():
        st.session_state[k]=v

# Load from URL on first visit
if "_url_loaded" not in st.session_state:
    st.session_state["_url_loaded"] = True
    qp = st.query_params
    if "s" in qp:
        try:
            apply_state(decode_state(qp["s"]))
        except Exception:
            pass

# Defaults for missing keys
for _ck in ("west","east"):
    for _i,(_dn,_ds,_dnr) in enumerate(DEFAULTS[_ck]):
        st.session_state.setdefault(f"{_ck}{_i}n", _dn)
        st.session_state.setdefault(f"{_ck}{_i}s", _ds)
        st.session_state.setdefault(f"{_ck}{_i}r", _dnr)

st.session_state.setdefault("locked_west",{})
st.session_state.setdefault("locked_east",{})
st.session_state.setdefault("series_scores",{})  # {key: [wh, wa]}

# Tallenna automaattisesti URL:iin joka renderöinnillä
# (kutsutaan ENNEN widgettejä jotta edellisen kierroksen arvot tallentuvat)
def _auto_save():
    try:
        encoded = encode_state(collect_state())
        st.query_params["s"] = encoded
    except Exception:
        pass

_auto_save()

def save_to_url():
    """Encode current state into URL query param."""
    try:
        encoded = encode_state(collect_state())
        st.query_params["s"] = encoded
    except Exception:
        pass  # URL too long or other error — fail silently


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Malliasetukset")
    home_adv=st.slider("🏠 Kotietu (NR-pistettä)",0.0,5.0,3.0,0.25)
    variance=st.slider("🎲 Sarjan varianssi",0.0,1.0,0.0,0.05,
                        help="0=puhdas NR, 0.5=paljon satunnaisuutta")
    n_sim=st.select_slider("🔁 Simulaatiot",options=[10_000,50_000,100_000,200_000],
                            value=100_000,format_func=lambda x:f"{x:,}")
    st.divider()

    st.markdown("### 💾 Ratingit tallentuvat automaattisesti")
    st.caption("✅ Ratingit tallentuvat URL:iin automaattisesti — "
               "kopioi osoiterivin URL talteen niin arvot palautuvat seuraavalla kerralla.")

    st.divider()
    st.markdown("**Varmuuskopio (JSON)**")
    _sj=json.dumps(collect_state(),ensure_ascii=False,indent=2)
    st.download_button("⬇️ Lataa JSON",data=_sj,file_name="nba2026.json",
                       mime="application/json",use_container_width=True)
    _up=st.file_uploader("📂 Lataa JSON",type=["json"],label_visibility="collapsed")
    if _up:
        try:
            apply_state(json.load(_up))
            st.success("✅ Ladattu!"); st.rerun()
        except Exception as e:
            st.error(f"Virhe: {e}")

    st.divider()
    st.markdown("### 📊 Net Rating")
    st.markdown("""
**Net Rating** = OffRtg − DefRtg

| NR-ero | Pelin win% |
|--------|-----------|
| 0 | 58.6% (kotietu) |
| 3 | 64.8% |
| 5 | 69.5% |
| 10 | 79.8% |

**Marginaalilinjat:**
- **+3.5** = vain sweep (4-0)
- **+2.5** = 4-0 tai 4-1
- **+1.5** = 4-0, 4-1 tai 4-2
""")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#0f0f1a,#1a0a2e,#0f1a0a);
 border:1px solid #f97316;border-radius:12px;padding:20px 28px;margin-bottom:18px;text-align:center;">
 <h1 style="font-size:2.6rem;margin:0;color:#f97316;">🏀 NBA PLAYOFFS 2026</h1>
 <p style="color:#8888aa;margin:6px 0 0;font-size:0.82rem;letter-spacing:2px;">
  LIVE TRACKER · NET RATING · PATH-AWARE MONTE CARLO · KOTIETU · VARIANSSI
 </p>
</div>
""", unsafe_allow_html=True)

tab_bracket, tab_live, tab_series, tab_playin, tab_path = st.tabs([
    "🏆 BRACKET & MESTARUUS", "⚡ LIVE TILANNE", "🎯 SARJA-ANALYYSI", "🔮 PLAY-IN", "📈 REITTIANALYYSI"
])


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — build team list from session state
# ══════════════════════════════════════════════════════════════════════════════
def get_teams(conf_key):
    teams=[]
    for i,(_dn,_ds,_dnr) in enumerate(DEFAULTS[conf_key]):
        nm=st.session_state.get(f"{conf_key}{i}n",_dn)
        sd=st.session_state.get(f"{conf_key}{i}s",_ds)
        nr=st.session_state.get(f"{conf_key}{i}r",_dnr)
        teams.append({"name":nm,"seed":sd,"nr":nr})
    return teams


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – BRACKET (team inputs + simulation)
# ══════════════════════════════════════════════════════════════════════════════
with tab_bracket:

    def team_inputs(conf_key, conf_label):
        teams=[]
        st.markdown(f"**{conf_label}**")
        hc=st.columns([3,1,2])
        hc[0].markdown("Joukkue"); hc[1].markdown("Sija"); hc[2].markdown("Net Rating")
        for i,(_dn,_ds,_) in enumerate(DEFAULTS[conf_key]):
            if _ds==7:
                st.markdown('<div style="border-top:1px dashed #7c3aed;margin:3px 0;'
                            'color:#a78bfa;font-size:0.7rem;letter-spacing:1px;padding-top:3px;">'
                            '▼ PLAY-IN (7–10)</div>',unsafe_allow_html=True)
            c1,c2,c3=st.columns([3,1,2])
            nm=c1.text_input("n",key=f"{conf_key}{i}n",label_visibility="collapsed")
            sd=c2.number_input("s",key=f"{conf_key}{i}s",label_visibility="collapsed",min_value=1,max_value=10)
            nr=c3.number_input("r",key=f"{conf_key}{i}r",label_visibility="collapsed",step=0.1,format="%.1f")
            teams.append({"name":nm,"seed":sd,"nr":nr})
        return teams

    cw_col,ce_col=st.columns(2)
    with cw_col: west=team_inputs("west","🔵 Läntinen konferenssi")
    with ce_col: east=team_inputs("east","🟠 Itäinen konferenssi")

    locked_w=st.session_state.get("locked_west",{})
    locked_e=st.session_state.get("locked_east",{})
    scores=st.session_state.get("series_scores",{})
    total_locked=len(locked_w)+len(locked_e)
    total_live=len([k for k,v in scores.items() if v[0]+v[1]>0 and v[0]<4 and v[1]<4])

    if total_locked or total_live:
        parts=[]
        if total_locked: parts.append(f"🔒 {total_locked} sarjaa lukittu")
        if total_live:   parts.append(f"⚡ {total_live} sarjaa käynnissä")
        st.info(" · ".join(parts)+" → Live Tilanne -välilehti")

    run=st.button("🏆 SIMULOI KOKO BRACKET",use_container_width=True)

    if run:
        with st.spinner(f"Simuloidaan {n_sim:,} kautta…"):
            res=sim_full(east,west,home_adv,variance,n_sim,locked_e,locked_w,scores)
        st.session_state["res"]=res
        st.session_state["east_s"]=east; st.session_state["west_s"]=west
        st.session_state["n_sim_s"]=n_sim

    if "res" not in st.session_state:
        st.info("Syötä net ratingit ja paina **SIMULOI KOKO BRACKET**.")
    else:
        cw_e,cw_w,nba_w,rnd_e,rnd_w,pq_e,pq_w=st.session_state["res"]
        east_s=st.session_state["east_s"]; west_s=st.session_state["west_s"]
        n_s=st.session_state["n_sim_s"]

        st.divider()
        st.markdown("#### 🔮 Play-In")
        pc_w,pc_e=st.columns(2)
        for col,ts,pq_map,lbl in [(pc_w,west_s,pq_w,"Länsi"),(pc_e,east_s,pq_e,"Itä")]:
            with col:
                st.markdown(f"**{lbl}**")
                rows=[]
                for t in sorted(ts,key=lambda x:x['seed']):
                    if t['seed']<7: continue
                    q=pq_map[t['name']]
                    rows.append({"Joukkue":t['name'],"#":f"#{t['seed']}","NR":f"{t['nr']:+.1f}",
                                 "Sija #7":pct(q[7]/n_s),"Sija #8":pct(q[8]/n_s),
                                 "Pääsee":pct((q[7]+q[8])/n_s)})
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        st.divider()
        st.markdown("#### 🏆 Mestaruustodennäköisyydet")
        all_rows=[]
        for t in west_s+east_s:
            is_w=t in west_s
            rw=(rnd_w if is_w else rnd_e)[t['name']]
            cw_map=cw_w if is_w else cw_e
            all_rows.append({"Joukkue":t['name']+(" 🔮" if t['seed']>=7 else ""),
                "Konf.":"Länsi" if is_w else "Itä","#":t['seed'],"NR":f"{t['nr']:+.1f}",
                "R1":pct(rw[1]/n_s),"R2":pct(rw[2]/n_s),"CF":pct(rw[3]/n_s),
                "Konf.Mestari":pct(cw_map[t['name']]/n_s),
                "NBA Mestari":pct(nba_w[t['name']]/n_s),"_n":nba_w[t['name']]})
        df_all=pd.DataFrame(all_rows).sort_values("_n",ascending=False)
        st.dataframe(df_all[["Joukkue","Konf.","#","NR","R1","R2","CF","Konf.Mestari","NBA Mestari"]],
                     use_container_width=True,hide_index=True)
        st.bar_chart(df_all.head(12).set_index("Joukkue")["_n"].apply(lambda x:round(x/n_s*100,1)),
                     color="#f97316",height=260)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – LIVE TILANNE (peli kerrallaan)
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### ⚡ Live tilanne – peli kerrallaan")
    st.caption("Syötä jokaisen sarjan tämänhetkinen tilanne. Simulaatio laskee mestaruusprosat "
               "ottaen huomioon jo pelatut pelit.")

    west_live=get_teams("west"); east_live=get_teams("east")
    locked_w2=st.session_state.get("locked_west",{})
    locked_e2=st.session_state.get("locked_east",{})
    scores=st.session_state.get("series_scores",{})
    nr_live={t['name']:t['nr'] for t in west_live+east_live}
    e_seed_live={t['seed']:t['name'] for t in east_live}
    w_seed_live={t['seed']:t['name'] for t in west_live}

    def get_q(locked,seed_map,conf_prefix):
        q7=locked.get("playin7",seed_map.get(7,"?"))
        q8=locked.get("playin8",seed_map.get(8,"?"))
        return q7,q8

    q7_w,q8_w=get_q(locked_w2,w_seed_live,"west")
    q7_e,q8_e=get_q(locked_e2,e_seed_live,"east")

    # Series definitions: (key, home_name, away_name, label)
    def make_series_list(seed_map, locked, q7, q8, conf_prefix):
        s=seed_map
        return [
            (f"{conf_prefix}_playin_7v8", s[7],s[8], f"Play-In G1: #{7} {s[7]} vs #{8} {s[8]}"),
            (f"{conf_prefix}_playin_9v10",s[9],s[10],f"Play-In G2: #{9} {s[9]} vs #{10} {s[10]}"),
            (f"{conf_prefix}_r1_1v8",     s[1], q8,  f"R1: #{1} {s[1]} vs #{8} {q8}"),
            (f"{conf_prefix}_r1_4v5",     s[4], s[5],f"R1: #{4} {s[4]} vs #{5} {s[5]}"),
            (f"{conf_prefix}_r1_2v7",     s[2], q7,  f"R1: #{2} {s[2]} vs #{7} {q7}"),
            (f"{conf_prefix}_r1_3v6",     s[3], s[6],f"R1: #{3} {s[3]} vs #{6} {s[6]}"),
        ]

    west_series=make_series_list(w_seed_live,locked_w2,q7_w,q8_w,"west")
    east_series=make_series_list(e_seed_live,locked_e2,q7_e,q8_e,"east")

    def render_live_series(series_list, conf_label, conf_color):
        st.markdown(f'<div style="color:{conf_color};font-family:Bebas Neue,sans-serif;'
                    f'font-size:1.2rem;letter-spacing:2px;margin-bottom:8px;">{conf_label}</div>',
                    unsafe_allow_html=True)
        changed=False
        for key,home,away,label in series_list:
            cur=scores.get(key,[0,0])
            wh,wa=cur[0],cur[1]
            is_over=(wh==4 or wa==4)
            is_playin="playin" in key

            with st.expander(
                f"{'✅' if is_over else '⚡' if (wh+wa>0) else '🔲'} {label} — "
                f"**{wh}–{wa}**{'  (PÄÄTTYNYT)' if is_over else ''}",
                expanded=(not is_over and (wh+wa>0))
            ):
                if is_over:
                    winner=home if wh==4 else away
                    st.success(f"Voittaja: **{winner}**")
                    if wh+wa>0:
                        st.caption(f"Lopputulos: {home} {wh}–{wa} {away}")
                else:
                    c1,c2,c3=st.columns([2,1,2])
                    c1.markdown(f"**{home}**")
                    c3.markdown(f"**{away}**")

                    new_wh=c1.number_input("Voitot",min_value=0,max_value=4,value=wh,
                                           key=f"wh_{key}",label_visibility="visible")
                    new_wa=c3.number_input("Voitot",min_value=0,max_value=4,value=wa,
                                           key=f"wa_{key}",label_visibility="visible")
                    c2.markdown(f'<div class="series-score" style="margin-top:28px;">'
                                f'{new_wh}–{new_wa}</div>',unsafe_allow_html=True)

                    if new_wh!=wh or new_wa!=wa:
                        scores[key]=[new_wh,new_wa]; changed=True

                    # Live win probability
                    if new_wh+new_wa>0 or True:
                        if home in nr_live and away in nr_live:
                            nr_h=nr_live[home]; nr_a=nr_live[away]
                            r=series_probs(nr_h,nr_a,home_adv,variance,7,new_wh,new_wa)
                            p1=r['p_home']*100; p2=r['p_away']*100
                            st.markdown(f"""<div style="display:flex;gap:3px;margin:8px 0 4px;">
                                <div style="width:{p1:.0f}%;background:linear-gradient(90deg,#f97316,#fb923c);
                                 border-radius:5px 0 0 5px;padding:4px 8px;color:white;font-size:0.78rem;
                                 font-weight:700;white-space:nowrap;overflow:hidden;min-width:30px;">
                                 {home.split()[-1]} {p1:.0f}%</div>
                                <div style="width:{p2:.0f}%;background:linear-gradient(90deg,#3b82f6,#60a5fa);
                                 border-radius:0 5px 5px 0;padding:4px 8px;color:white;font-size:0.78rem;
                                 font-weight:700;text-align:right;white-space:nowrap;overflow:hidden;min-width:30px;">
                                 {p2:.0f}% {away.split()[-1]}</div>
                            </div>""",unsafe_allow_html=True)
                            if not is_playin:
                                st.caption(
                                    f"+3.5: {home.split()[-1]} {pct(r['p_home_p35'])} | "
                                    f"{away.split()[-1]} {pct(r['p_away_p35'])}   "
                                    f"+2.5: {pct(r['p_home_p25'])} | {pct(r['p_away_p25'])}   "
                                    f"+1.5: {pct(r['p_home_p15'])} | {pct(r['p_away_p15'])}"
                                )
        return changed

    changed_w=render_live_series(west_series,"🔵 LÄNTINEN KONFERENSSI","#3b82f6")
    st.divider()
    changed_e=render_live_series(east_series,"🟠 ITÄINEN KONFERENSSI","#f97316")

    if changed_w or changed_e:
        st.session_state["series_scores"]=scores

    st.divider()

    # Play-In winner lock
    st.markdown("#### 🔮 Play-In – Lukitse voittajat")
    pi_cols=st.columns(4)
    for col,(conf_k,slot,seed_map,lbl) in zip(pi_cols,[
        ("west","playin7",w_seed_live,"Länsi #7"),
        ("west","playin8",w_seed_live,"Länsi #8"),
        ("east","playin7",e_seed_live,"Itä #7"),
        ("east","playin8",e_seed_live,"Itä #8"),
    ]):
        lock_k=f"locked_{conf_k}"
        locked=st.session_state[lock_k]
        opts=["(simuloi)"]+[seed_map[s] for s in [7,8,9,10]]
        cur=locked.get(slot,"(simuloi)")
        sel=col.selectbox(lbl,opts,index=opts.index(cur) if cur in opts else 0,
                          key=f"pi_lock_{conf_k}_{slot}")
        if sel=="(simuloi)":
            locked.pop(slot,None)
        else:
            locked[slot]=sel
        st.session_state[lock_k]=locked

    st.divider()
    # Re-simulate with live scores
    if st.button("⚡ PÄIVITÄ MESTARUUSPROSAT LIVE-TILANTEELLA",use_container_width=True):
        west_cur=get_teams("west"); east_cur=get_teams("east")
        with st.spinner("Simuloidaan live-tilanne…"):
            res_live=sim_full(east_cur,west_cur,home_adv,variance,n_sim,
                              st.session_state["locked_east"],
                              st.session_state["locked_west"],
                              st.session_state["series_scores"])
        st.session_state["res"]=res_live
        st.session_state["east_s"]=east_cur; st.session_state["west_s"]=west_cur
        st.session_state["n_sim_s"]=n_sim
        st.success("✅ Mestaruusprosat päivitetty! Katso Bracket & Mestaruus -välilehti.")

    # Quick standings if sim done
    if "res" in st.session_state:
        _,_,nba_w2,_,_,_,_=st.session_state["res"]
        n_s2=st.session_state["n_sim_s"]
        all_t2=get_teams("west")+get_teams("east")
        st.markdown("#### 🏆 Mestaruusprosat (viimeisin simulaatio)")
        top_rows=[{"Joukkue":t['name'],"NR":f"{t['nr']:+.1f}",
                   "NBA Mestari %":pct(nba_w2[t['name']]/n_s2)}
                  for t in sorted(all_t2,key=lambda x:-nba_w2[x['name']])]
        st.dataframe(pd.DataFrame(top_rows[:10]),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SARJA-ANALYYSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_series:
    st.markdown("### Kahden joukkueen sarja-analyysi")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div style="color:#f97316;font-family:Bebas Neue,sans-serif;font-size:1.1rem;'
                    'letter-spacing:1px;">🟠 KORKEAMPI SIJOITUS – KOTIETU</div>',unsafe_allow_html=True)
        sn_a=st.text_input("Joukkue A","Oklahoma City Thunder",key="sa_n")
        nr_a=st.number_input("Net Rating A",value=12.8,step=0.1,format="%.1f",key="sa_nr")
        wh_a=st.number_input("Voitot tähän asti",min_value=0,max_value=4,value=0,key="sa_wh")
    with c2:
        st.markdown('<div style="color:#3b82f6;font-family:Bebas Neue,sans-serif;font-size:1.1rem;'
                    'letter-spacing:1px;">🔵 MATALAMPI SIJOITUS – VIERASJOUKKUE</div>',unsafe_allow_html=True)
        sn_b=st.text_input("Joukkue B","Los Angeles Lakers",key="sb_n")
        nr_b=st.number_input("Net Rating B",value=3.8,step=0.1,format="%.1f",key="sb_nr")
        wa_b=st.number_input("Voitot tähän asti",min_value=0,max_value=4,value=0,key="sb_wa")

    bo=st.radio("Sarjamuoto",[5,7],index=1,horizontal=True,format_func=lambda x:f"Best-of-{x}")

    if st.button("🔢 LASKE",use_container_width=True,key="calc_s"):
        res_s=series_probs(nr_a,nr_b,home_adv,variance,bo,wh_a,wa_b)
        wn=(bo+1)//2

        if wh_a>0 or wa_b>0:
            st.info(f"Lasketaan tilanteesta **{sn_a} {wh_a}–{wa_b} {sn_b}**")

        st.divider()
        ph=win_prob(nr_a,nr_b,home_adv); pa=win_prob(nr_a,nr_b,-home_adv)
        st.markdown("#### Yhden pelin voittotodennäköisyys")
        mc1,mc2,mc3,mc4=st.columns(4)
        mc1.markdown(f'<div class="mc"><p class="mv">{ph*100:.1f}%</p><p class="ml">{sn_a} kotona</p></div>',unsafe_allow_html=True)
        mc2.markdown(f'<div class="mc"><p class="mv" style="color:#3b82f6">{(1-ph)*100:.1f}%</p><p class="ml">{sn_b} vieraana</p></div>',unsafe_allow_html=True)
        mc3.markdown(f'<div class="mc"><p class="mv" style="color:#3b82f6">{(1-pa)*100:.1f}%</p><p class="ml">{sn_b} kotona</p></div>',unsafe_allow_html=True)
        mc4.markdown(f'<div class="mc"><p class="mv">{pa*100:.1f}%</p><p class="ml">{sn_a} vieraana</p></div>',unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Sarjan voittotodennäköisyys")
        bha,bba=res_s['p_home']*100,res_s['p_away']*100
        ca,cb=st.columns(2)
        ca.markdown(f'<div class="mc"><p class="mv">{bha:.1f}%</p><p class="ml">{sn_a} voittaa sarjan</p></div>',unsafe_allow_html=True)
        cb.markdown(f'<div class="mc"><p class="mv" style="color:#3b82f6">{bba:.1f}%</p><p class="ml">{sn_b} voittaa sarjan</p></div>',unsafe_allow_html=True)
        st.markdown(f"""<div style="display:flex;gap:3px;margin:8px 0;">
            <div style="width:{bha:.1f}%;background:linear-gradient(90deg,#f97316,#fb923c);
             border-radius:6px 0 0 6px;padding:5px 10px;color:white;font-weight:700;
             font-size:0.8rem;white-space:nowrap;overflow:hidden;">{sn_a} {bha:.1f}%</div>
            <div style="width:{bba:.1f}%;background:linear-gradient(90deg,#3b82f6,#60a5fa);
             border-radius:0 6px 6px 0;padding:5px 10px;color:white;font-weight:700;
             font-size:0.8rem;text-align:right;white-space:nowrap;overflow:hidden;">{bba:.1f}% {sn_b}</div>
        </div>""",unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Marginaalilinjat")
        st.caption("+3.5 = vain sweep (4-0) | +2.5 = 4-0 tai 4-1 | +1.5 = 4-0, 4-1 tai 4-2")
        mr=[
            {"Linja":"+3.5 (sweep 4-0)",f"{sn_a} kattaa":pct(res_s['p_home_p35']),f"{sn_b} kattaa":pct(res_s['p_away_p35'])},
            {"Linja":"+2.5 (4-0 / 4-1)", f"{sn_a} kattaa":pct(res_s['p_home_p25']),f"{sn_b} kattaa":pct(res_s['p_away_p25'])},
            {"Linja":"+1.5 (4-0/4-1/4-2)",f"{sn_a} kattaa":pct(res_s['p_home_p15']),f"{sn_b} kattaa":pct(res_s['p_away_p15'])},
        ]
        st.dataframe(pd.DataFrame(mr).set_index("Linja"),use_container_width=True)

        st.divider()
        st.markdown("#### Sarjan lopputulokset (jäljellä olevat pelit)")
        oc=[]
        for (wh,wa),prob in sorted(res_s['outcomes'].items(),key=lambda x:-x[1]):
            winner=sn_a if wh==wn else sn_b; loser=sn_b if wh==wn else sn_a
            score=f"{wh}–{wa}" if wh==wn else f"{wa}–{wh}"
            oc.append({"Tulos":f"{winner} {score} {loser}","Voittaja":winner,"Tn":pct(prob),"_p":round(prob*100,2)})
        df_oc=pd.DataFrame(oc)
        st.dataframe(df_oc[["Tulos","Voittaja","Tn"]],use_container_width=True,hide_index=True)
        st.bar_chart(df_oc.set_index("Tulos")["_p"],color="#f97316",height=220)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PLAY-IN
# ══════════════════════════════════════════════════════════════════════════════
with tab_playin:
    st.markdown("### Play-In turnauksen analyysi")
    st.markdown("""<div class="pibox"><b style="color:#a78bfa;font-family:Bebas Neue,sans-serif;
letter-spacing:2px;">🔮 PLAY-IN RAKENNE</b><br><br>
<b>Peli 1:</b> #7 vs #8 → <b>Voittaja = playoff #7</b><br>
<b>Peli 2:</b> #9 vs #10 → <b>Häviäjä putoaa</b><br>
<b>Peli 3:</b> Häviäjä(P1) vs Voittaja(P2) → <b>Voittaja = playoff #8</b></div>""",unsafe_allow_html=True)

    pi_conf=st.radio("Konferenssi",["Länsi","Itä"],horizontal=True,key="pi_conf")
    pi_defs=[(n,nr) for n,s,nr in DEFAULTS["west" if pi_conf=="Länsi" else "east"] if s>=7]
    pi_cols=st.columns(4); pi_names,pi_nrs=[],[]
    for i,(col,(dn,dnr)) in enumerate(zip(pi_cols,pi_defs)):
        with col:
            st.markdown(f'<div style="color:#a78bfa;font-family:Bebas Neue,sans-serif;">SIJA #{i+7}</div>',unsafe_allow_html=True)
            n=col.text_input("",value=dn,key=f"pi_n{i}{pi_conf}",label_visibility="collapsed")
            nr=col.number_input("",value=dnr,step=0.1,format="%.1f",key=f"pi_nr{i}{pi_conf}",label_visibility="collapsed")
            pi_names.append(n); pi_nrs.append(nr)

    if st.button("🔮 LASKE PLAY-IN",use_container_width=True):
        pm={pi_names[i]:pi_nrs[i] for i in range(4)}
        t7,t8,t9,t10=pi_names
        pr=playin_exact(t7,t8,t9,t10,pm,home_adv)
        g1=win_prob(pm[t7],pm[t8],home_adv); g2=win_prob(pm[t9],pm[t10],home_adv)

        st.divider()
        gc1,gc2,gc3=st.columns(3)
        gc1.markdown(f'<div class="mc"><p style="color:#a78bfa;font-family:Bebas Neue,sans-serif;margin:0 0 3px;">PELI 1</p><p style="margin:2px 0;font-size:0.77rem;">{t7} vs {t8}</p><p class="mv">{g1*100:.1f}%</p><p class="ml">{t7} voittaa</p></div>',unsafe_allow_html=True)
        gc2.markdown(f'<div class="mc"><p style="color:#a78bfa;font-family:Bebas Neue,sans-serif;margin:0 0 3px;">PELI 2</p><p style="margin:2px 0;font-size:0.77rem;">{t9} vs {t10}</p><p class="mv">{g2*100:.1f}%</p><p class="ml">{t9} voittaa</p></div>',unsafe_allow_html=True)
        gc3.markdown(f'<div class="mc"><p style="color:#a78bfa;font-family:Bebas Neue,sans-serif;margin:0 0 3px;">PELI 3</p><p style="margin:2px 0;font-size:0.77rem;">Riippuu P1&P2</p><p class="mv" style="font-size:1.6rem;">4 sk.</p></div>',unsafe_allow_html=True)

        st.divider()
        rows=[{"Joukkue":n,"#":f"#{i+7}","NR":f"{pi_nrs[i]:+.1f}",
               "Sija #7":pct(pr[n]['p7']),"Sija #8":pct(pr[n]['p8']),
               "Pääsee":pct(pr[n]['p_qualify']),"_q":pr[n]['p_qualify']}
              for i,n in enumerate(pi_names)]
        df_pi=pd.DataFrame(rows).sort_values("_q",ascending=False)
        st.dataframe(df_pi[["Joukkue","#","NR","Sija #7","Sija #8","Pääsee"]],use_container_width=True,hide_index=True)

        for _,row in df_pi.iterrows():
            nm=row["Joukkue"]; p7=pr[nm]['p7']*100; p8=pr[nm]['p8']*100; po=(1-pr[nm]['p_qualify'])*100
            st.markdown(f"**{nm}** ({row['#']})")
            st.markdown(f"""<div style="display:flex;gap:2px;margin:2px 0 8px;">
                <div style="width:{p7:.1f}%;background:linear-gradient(90deg,#f97316,#fb923c);
                 border-radius:5px 0 0 5px;padding:3px 7px;color:white;font-size:0.72rem;
                 font-weight:700;white-space:nowrap;overflow:hidden;min-width:0;">Sija #7: {p7:.1f}%</div>
                <div style="width:{p8:.1f}%;background:linear-gradient(90deg,#7c3aed,#a78bfa);
                 padding:3px 7px;color:white;font-size:0.72rem;font-weight:700;
                 white-space:nowrap;overflow:hidden;min-width:0;">Sija #8: {p8:.1f}%</div>
                <div style="width:{po:.1f}%;background:#1a1a2e;border-radius:0 5px 5px 0;
                 padding:3px 7px;color:#8888aa;font-size:0.72rem;
                 white-space:nowrap;overflow:hidden;min-width:0;">Putoaa: {po:.1f}%</div>
            </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – REITTIANALYYSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_path:
    st.markdown("### Reittianalyysi")
    if "res" not in st.session_state:
        st.info("Aja simulaatio ensin.")
    else:
        cw_e,cw_w,nba_w,rnd_e,rnd_w,pq_e,pq_w=st.session_state["res"]
        east_s=st.session_state["east_s"]; west_s=st.session_state["west_s"]
        n_s=st.session_state["n_sim_s"]; all_t=east_s+west_s

        sel=st.selectbox("Valitse joukkue",sorted([t['name'] for t in all_t],key=lambda x:-nba_w[x]))
        t_obj=next(t for t in all_t if t['name']==sel)
        is_e=t_obj in east_s
        rnd_m=rnd_e if is_e else rnd_w; cw_m=cw_e if is_e else cw_w
        pq_m=pq_e if is_e else pq_w; rw=rnd_m[sel]; playin=t_obj['seed']>=7

        st.divider()
        stages=[]
        if playin:
            q=pq_m[sel]
            stages+=[("🔮 Play-In: pääsee",(q[7]+q[8])/n_s,"#7c3aed"),
                     ("🔮 Sija #7",q[7]/n_s,"#a78bfa"),("🔮 Sija #8",q[8]/n_s,"#6d28d9")]
        stages+=[("✅ R1 voitto",rw[1]/n_s,"#f97316"),("✅ R2 voitto",rw[2]/n_s,"#fb923c"),
                 ("✅ Konf. finaali",rw[3]/n_s,"#fbbf24"),
                 ("🏆 Konf. mestari",cw_m[sel]/n_s,"#22c55e"),
                 ("🏆 NBA mestari",nba_w[sel]/n_s,"#16a34a")]
        for label,prob,color in stages:
            p=prob*100
            st.markdown(f"**{label}**")
            st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin:1px 0 8px;">
                <div style="flex:1;background:#1a1a2e;border-radius:6px;overflow:hidden;height:24px;">
                    <div style="width:{min(p,100):.1f}%;background:{color};height:100%;
                     display:flex;align-items:center;padding-left:10px;color:white;
                     font-weight:700;font-size:0.78rem;white-space:nowrap;">{p:.1f}%</div>
                </div></div>""",unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Sarjavoitto-% kaikkia vastustajia vastaan")
        nr_t=t_obj['nr']
        vs_rows=[]
        for opp in sorted(all_t,key=lambda x:-x['nr']):
            if opp['name']==sel: continue
            same=(opp in east_s)==is_e
            if nr_t>=opp['nr']:
                r=series_probs(nr_t,opp['nr'],home_adv,variance,7); p_ser=r['p_home']
            else:
                r=series_probs(opp['nr'],nr_t,home_adv,variance,7); p_ser=r['p_away']
            vs_rows.append({"Vastustaja":opp['name'],"Konf.":"Sama" if same else "Finals",
                            "Vast. NR":f"{opp['nr']:+.1f}","NR ero":f"{nr_t-opp['nr']:+.1f}",
                            "Sarjavoitto":pct(p_ser)})
        st.dataframe(pd.DataFrame(vs_rows),use_container_width=True,hide_index=True)
