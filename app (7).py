import streamlit as st
import pandas as pd
import numpy as np
import io, json

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
.save-tip{background:#1a2e1a;border:1px solid #22c55e;border-radius:8px;
          padding:10px 14px;font-size:0.82rem;color:#86efac;margin:8px 0;}
.elim{opacity:0.45;text-decoration:line-through;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MATH
# ══════════════════════════════════════════════════════════════════════════════

def win_prob(nr_a: float, nr_b: float, home_adv: float) -> float:
    """Win probability. k=0.116 → 10pt NR gap ≈ 76% per game."""
    return float(np.clip(1 / (1 + np.exp(-(nr_a + home_adv - nr_b) * 0.116)), 0.001, 0.999))


def series_probs(nr_home: float, nr_away: float, home_adv: float,
                 variance: float = 0.0, best_of: int = 7) -> dict:
    """
    Exact series probabilities via dynamic programming.
    NBA 2-2-1-1-1 schedule (games 1,2,5,7 at higher seed).

    Margin lines (+X.5) = voittaja voittaa sarjan vähintään X.5 voitolla enemmän:
      +3.5 → voittoero > 3.5 → vain sweep 4-0          (ero=4)
      +2.5 → voittoero > 2.5 → 4-0 tai 4-1             (ero=3 tai 4)
      +1.5 → voittoero > 1.5 → 4-0, 4-1 tai 4-2        (ero=2,3 tai 4)
    """
    wn = (best_of + 1) // 2
    schedule = {1:True,2:True,3:False,4:False,5:True,6:False,7:True}

    ph = win_prob(nr_home, nr_away,  home_adv) * (1-variance*0.5) + 0.5*variance*0.5
    pa = win_prob(nr_home, nr_away, -home_adv) * (1-variance*0.5) + 0.5*variance*0.5

    states = {(0,0): 1.0}
    outcomes = {}
    for g in range(1, best_of+1):
        pw = ph if schedule[g] else pa
        ns = {}
        for (wh,wa), prob in states.items():
            for dwh,dwa,p in [(1,0,pw),(0,1,1-pw)]:
                nwh,nwa = wh+dwh, wa+dwa
                if nwh==wn or nwa==wn:
                    outcomes[(nwh,nwa)] = outcomes.get((nwh,nwa),0) + prob*p
                else:
                    ns[(nwh,nwa)] = ns.get((nwh,nwa),0) + prob*p
        states = ns

    ph_win = sum(v for (wh,wa),v in outcomes.items() if wh==wn)
    pa_win = sum(v for (wh,wa),v in outcomes.items() if wa==wn)
    aw = {k:v for k,v in outcomes.items() if k[0]==wn}
    bw = {k:v for k,v in outcomes.items() if k[1]==wn}

    # margin: winner_wins - loser_wins > threshold
    # aw keys: (4, loser_wins) → home won → diff = wh - wa
    # bw keys: (home_wins, 4) → away won → diff = wa - wh
    def mgn_home(d, thresh):
        return sum(v for (wh,wa),v in d.items() if (wh - wa) > thresh)
    def mgn_away(d, thresh):
        return sum(v for (wh,wa),v in d.items() if (wa - wh) > thresh)

    return {
        'p_home': ph_win, 'p_away': pa_win, 'outcomes': outcomes,
        'p_home_p35': mgn_home(aw, 3.5),   # home sweeps 4-0
        'p_home_p25': mgn_home(aw, 2.5),   # home wins 4-0 or 4-1
        'p_home_p15': mgn_home(aw, 1.5),   # home wins 4-0, 4-1 or 4-2
        'p_away_p35': mgn_away(bw, 3.5),   # away sweeps 4-0
        'p_away_p25': mgn_away(bw, 2.5),   # away wins 4-0 or 4-1
        'p_away_p15': mgn_away(bw, 1.5),   # away wins 4-0, 4-1 or 4-2
    }


def playin_exact(t7, t8, t9, t10, nr, home_adv):
    p78  = win_prob(nr[t7], nr[t8],  home_adv)
    p910 = win_prob(nr[t9], nr[t10], home_adv)
    p_t8_t9  = win_prob(nr[t8], nr[t9],  home_adv)
    p_t8_t10 = win_prob(nr[t8], nr[t10], home_adv)
    p_t7_t9  = win_prob(nr[t7], nr[t9],  home_adv)
    p_t7_t10 = win_prob(nr[t7], nr[t10], home_adv)
    r = {t:{'p7':0.,'p8':0.} for t in [t7,t8,t9,t10]}
    r[t7]['p7'] += p78
    r[t8]['p8'] += p78*p910*p_t8_t9;      r[t9]['p8']  += p78*p910*(1-p_t8_t9)
    r[t8]['p8'] += p78*(1-p910)*p_t8_t10; r[t10]['p8'] += p78*(1-p910)*(1-p_t8_t10)
    r[t8]['p7'] += (1-p78)
    r[t7]['p8'] += (1-p78)*p910*p_t7_t9;      r[t9]['p8']  += (1-p78)*p910*(1-p_t7_t9)
    r[t7]['p8'] += (1-p78)*(1-p910)*p_t7_t10; r[t10]['p8'] += (1-p78)*(1-p910)*(1-p_t7_t10)
    for t in r: r[t]['p_qualify'] = r[t]['p7'] + r[t]['p8']
    return r


def sim_series(home, away, nr, home_adv, variance, rng, best_of=7):
    wn = (best_of+1)//2
    ph = win_prob(nr[home], nr[away],  home_adv) * (1-variance*0.5) + 0.5*variance*0.5
    pa = win_prob(nr[home], nr[away], -home_adv) * (1-variance*0.5) + 0.5*variance*0.5
    schedule = [True,True,False,False,True,False,True]
    wh = wa = 0
    for i in range(best_of):
        if rng.random() < (ph if schedule[i] else pa): wh += 1
        else: wa += 1
        if wh==wn: return home
        if wa==wn: return away
    return home


def sim_playin(t7, t8, t9, t10, nr, home_adv, rng):
    g1 = t7 if rng.random() < win_prob(nr[t7], nr[t8],  home_adv) else t8
    g1l = t8 if g1==t7 else t7
    g2 = t9 if rng.random() < win_prob(nr[t9], nr[t10], home_adv) else t10
    g3 = g1l if rng.random() < win_prob(nr[g1l], nr[g2], home_adv) else g2
    return g1, g3


def sim_full(east, west, home_adv, variance, n_sim,
             locked_e=None, locked_w=None):
    """
    Path-aware MC. locked_X = {round: {slot: winner}} for already-played series.
    east/west include all 10 teams (seeds 1-10).
    """
    rng = np.random.default_rng(42)
    nr = {t['name']: t['nr'] for t in east+west}
    locked_e = locked_e or {}
    locked_w = locked_w or {}

    cw_e  = {t['name']:0 for t in east}
    cw_w  = {t['name']:0 for t in west}
    nba_w = {t['name']:0 for t in east+west}
    rnd_e = {t['name']:{1:0,2:0,3:0} for t in east}
    rnd_w = {t['name']:{1:0,2:0,3:0} for t in west}
    pq_e  = {t['name']:{7:0,8:0} for t in east}
    pq_w  = {t['name']:{7:0,8:0} for t in west}

    e_seed = {t['seed']:t['name'] for t in east}
    w_seed = {t['seed']:t['name'] for t in west}

    def sim_conf(seed_map, cw, rnd, pq, locked):
        # Play-In
        pi7,pi8,pi9,pi10 = seed_map[7],seed_map[8],seed_map[9],seed_map[10]
        # Simulate Play-In once → get both qualifiers together
        if "playin7" in locked and "playin8" in locked:
            q7 = locked["playin7"]
            q8 = locked["playin8"]
        elif "playin7" in locked:
            q7 = locked["playin7"]
            # Simulate only G2 and G3 knowing q7 is locked
            # G3: whoever lost G1 vs G2 winner — but since q7 is fixed,
            # just simulate full playin and use the q8 result
            _, q8 = sim_playin(pi7,pi8,pi9,pi10,nr,home_adv,rng)
            # Ensure q8 != q7
            if q8 == q7:
                _, q8 = sim_playin(pi7,pi8,pi9,pi10,nr,home_adv,rng)
        elif "playin8" in locked:
            q8 = locked["playin8"]
            q7, _ = sim_playin(pi7,pi8,pi9,pi10,nr,home_adv,rng)
            if q7 == q8:
                q7, _ = sim_playin(pi7,pi8,pi9,pi10,nr,home_adv,rng)
        else:
            q7, q8 = sim_playin(pi7,pi8,pi9,pi10,nr,home_adv,rng)

        pq[q7][7] += 1
        pq[q8][8] += 1

        bseed = {seed_map[s]:s for s in range(1,7)}
        bseed[q7]=7; bseed[q8]=8

        def ps(a, b, lock_key=None):
            if lock_key and lock_key in locked:
                return locked[lock_key]
            h = a if bseed.get(a,99)<bseed.get(b,99) else b
            aw = b if h==a else a
            return sim_series(h, aw, nr, home_adv, variance, rng)

        s1,s2,s3,s4,s5,s6 = (seed_map[i] for i in range(1,7))
        w1v8 = ps(s1, q8, "r1_1v8")
        w4v5 = ps(s4, s5, "r1_4v5")
        w2v7 = ps(s2, q7, "r1_2v7")
        w3v6 = ps(s3, s6, "r1_3v6")
        for w in [w1v8,w4v5,w2v7,w3v6]: rnd[w][1] += 1

        wA = ps(w1v8, w4v5, "r2_A")
        wB = ps(w2v7, w3v6, "r2_B")
        rnd[wA][2] += 1; rnd[wB][2] += 1

        cf = ps(wA, wB, "cf")
        rnd[cf][3] += 1; cw[cf] += 1
        return cf

    for _ in range(n_sim):
        e_cf = sim_conf(e_seed, cw_e, rnd_e, pq_e, locked_e)
        w_cf = sim_conf(w_seed, cw_w, rnd_w, pq_w, locked_w)
        champ = sim_series(e_cf if nr[e_cf]>=nr[w_cf] else w_cf,
                           w_cf if nr[e_cf]>=nr[w_cf] else e_cf,
                           nr, home_adv, variance, rng)
        nba_w[champ] += 1

    return cw_e, cw_w, nba_w, rnd_e, rnd_w, pq_e, pq_w


def pct(v, n=1): return f"{v*100:.{n}f}%"


# ══════════════════════════════════════════════════════════════════════════════
# 2026 BRACKET DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "west": [
        ("Oklahoma City Thunder",  1, 0.0),
        ("San Antonio Spurs",      2, 0.0),
        ("Denver Nuggets",         3, 0.0),
        ("Los Angeles Lakers",     4, 0.0),
        ("Houston Rockets",        5, 0.0),
        ("Minnesota Timberwolves", 6, 0.0),
        ("Phoenix Suns",           7, 0.0),
        ("Portland Trail Blazers", 8, 0.0),
        ("LA Clippers",            9, 0.0),
        ("Golden State Warriors", 10, 0.0),
    ],
    "east": [
        ("Detroit Pistons",        1, 0.0),
        ("Boston Celtics",         2, 0.0),
        ("Atlanta Hawks",          3, 0.0),
        ("Cleveland Cavaliers",    4, 0.0),
        ("New York Knicks",        5, 0.0),
        ("Toronto Raptors",        6, 0.0),
        ("Philadelphia 76ers",     7, 0.0),
        ("Orlando Magic",          8, 0.0),
        ("Charlotte Hornets",      9, 0.0),
        ("Miami Heat",            10, 0.0),
    ],
}

ALL_KEYS = (
    [f"west{i}{s}" for i in range(10) for s in "nsr"] +
    [f"east{i}{s}" for i in range(10) for s in "nsr"]
)


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT STORAGE — JSON upload/download
# ══════════════════════════════════════════════════════════════════════════════

def get_save_data() -> dict:
    """Kerää kaikki NR-arvot session_statesta tallennettavaksi."""
    data = {}
    for ck in ("west","east"):
        for i in range(10):
            for s in "nsr":
                k = f"{ck}{i}{s}"
                if k in st.session_state:
                    data[k] = st.session_state[k]
    # Tallenna myös lukitut sarjat
    for k in ("locked_west","locked_east"):
        if k in st.session_state:
            data[k] = st.session_state[k]
    return data


def load_save_data(data: dict):
    """Lataa tallennetut arvot session_stateen."""
    for k,v in data.items():
        st.session_state[k] = v


# ── Alusta session_state oletuksilla ──────────────────────────────────────────
for _ck in ("west","east"):
    for _i,(_dn,_ds,_dnr) in enumerate(DEFAULTS[_ck]):
        if f"{_ck}{_i}n" not in st.session_state:
            st.session_state[f"{_ck}{_i}n"] = _dn
        if f"{_ck}{_i}s" not in st.session_state:
            st.session_state[f"{_ck}{_i}s"] = _ds
        if f"{_ck}{_i}r" not in st.session_state:
            st.session_state[f"{_ck}{_i}r"] = _dnr

if "locked_west" not in st.session_state:
    st.session_state["locked_west"] = {}
if "locked_east" not in st.session_state:
    st.session_state["locked_east"] = {}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Malliasetukset")
    home_adv = st.slider("🏠 Kotietu (NR-pistettä)", 0.0, 5.0, 3.0, 0.25)
    variance  = st.slider("🎲 Sarjan varianssi", 0.0, 1.0, 0.0, 0.05,
                           help="0=puhdas NR, 0.5=paljon satunnaisuutta")
    n_sim = st.select_slider("🔁 Simulaatiot",
                              options=[10_000,50_000,100_000,200_000], value=100_000,
                              format_func=lambda x:f"{x:,}")
    st.divider()

    # ── Tallenna / Lataa JSON ──────────────────────────────────────────────
    st.markdown("### 💾 Tallenna & Lataa ratingit")
    st.caption("Lataa JSON-tiedosto → tallentuu koneellesi. Avaa sovellus uudelleen ja lataa tiedosto takaisin.")

    save_data = get_save_data()
    save_json = json.dumps(save_data, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Lataa ratingit (JSON)",
        data=save_json,
        file_name="nba2026_ratingit.json",
        mime="application/json",
        use_container_width=True
    )

    uploaded = st.file_uploader("📂 Avaa tallennettu JSON", type=["json"], label_visibility="collapsed")
    if uploaded:
        try:
            loaded = json.load(uploaded)
            load_save_data(loaded)
            st.success("✅ Ratingit ladattu!")
            st.rerun()
        except Exception as e:
            st.error(f"Virhe: {e}")

    st.divider()
    st.markdown("### 📊 Net Rating")
    st.markdown("""
**Net Rating** = OffRtg − DefRtg.

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
  PATH-AWARE MONTE CARLO · NET RATING · PLAY-IN · VARIANSSI · KOTIETU
 </p>
</div>
""", unsafe_allow_html=True)

tab_bracket, tab_series, tab_playin, tab_path, tab_update = st.tabs([
    "🏆 BRACKET & MESTARUUS", "🎯 SARJA-ANALYYSI", "🔮 PLAY-IN", "📈 REITTIANALYYSI", "🔄 PÄIVITÄ BRACKET"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – BRACKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_bracket:

    def team_inputs(conf_key, conf_label):
        teams = []
        st.markdown(f"**{conf_label}**")
        hc = st.columns([3,1,2])
        hc[0].markdown("Joukkue"); hc[1].markdown("Sija"); hc[2].markdown("Net Rating")
        for i,(_dn,_ds,_) in enumerate(DEFAULTS[conf_key]):
            if _ds == 7:
                st.markdown('<div style="border-top:1px dashed #7c3aed;margin:3px 0;'
                            'color:#a78bfa;font-size:0.7rem;letter-spacing:1px;padding-top:3px;">'
                            '▼ PLAY-IN (7–10)</div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns([3,1,2])
            nm = c1.text_input("n", key=f"{conf_key}{i}n", label_visibility="collapsed")
            sd = c2.number_input("s",key=f"{conf_key}{i}s", label_visibility="collapsed",
                                 min_value=1,max_value=10)
            nr = c3.number_input("r",key=f"{conf_key}{i}r", label_visibility="collapsed",
                                 step=0.1, format="%.1f")
            teams.append({"name":nm,"seed":sd,"nr":nr})
        return teams

    cw_col, ce_col = st.columns(2)
    with cw_col:
        west = team_inputs("west", "🔵 Läntinen konferenssi")
    with ce_col:
        east = team_inputs("east", "🟠 Itäinen konferenssi")

    locked_w = st.session_state.get("locked_west", {})
    locked_e = st.session_state.get("locked_east", {})

    if locked_w or locked_e:
        total_locked = len(locked_w)+len(locked_e)
        st.info(f"🔒 {total_locked} sarjaa lukittu todellisilla tuloksilla (Päivitä Bracket -välilehti)")

    # ── Tallenna + Simuloi napit ──────────────────────────────────────────────
    btn1, btn2 = st.columns([1, 2])
    with btn1:
        # Tallenna-nappi — aina näkyvissä, lataa JSON-tiedoston koneelle
        _save = get_save_data()
        st.download_button(
            "💾 TALLENNA RATINGIT",
            data=json.dumps(_save, ensure_ascii=False, indent=2),
            file_name="nba2026_ratingit.json",
            mime="application/json",
            use_container_width=True,
            help="Lataa JSON-tiedosto koneellesi. Voit avata sen takaisin sivupalkista."
        )
    with btn2:
        run = st.button("🏆 SIMULOI KOKO BRACKET", use_container_width=True)

    if run:
        with st.spinner(f"Simuloidaan {n_sim:,} täyttä playoff-kautta…"):
            res = sim_full(east, west, home_adv, variance, n_sim, locked_e, locked_w)
        st.session_state["res"] = res
        st.session_state["east_s"] = east
        st.session_state["west_s"] = west
        st.session_state["n_sim_s"] = n_sim

    if "res" not in st.session_state:
        st.info("Syötä net ratingit ja paina **SIMULOI KOKO BRACKET**.")
    else:
        cw_e,cw_w,nba_w,rnd_e,rnd_w,pq_e,pq_w = st.session_state["res"]
        east_s = st.session_state["east_s"]
        west_s = st.session_state["west_s"]
        n_s    = st.session_state["n_sim_s"]

        st.divider()
        st.markdown("#### 🔮 Play-In – Pääsy playoffseihin")
        pc_w,pc_e = st.columns(2)
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
        st.markdown("#### 🏆 Bracket-tilastot")
        all_rows=[]
        for t in west_s:
            rw=rnd_w[t['name']]
            all_rows.append({"Joukkue":t['name']+(" 🔮" if t['seed']>=7 else ""),
                "Konf.":"Länsi","#":t['seed'],"NR":f"{t['nr']:+.1f}",
                "R1":pct(rw[1]/n_s),"R2":pct(rw[2]/n_s),"CF":pct(rw[3]/n_s),
                "Konf.Mestari":pct(cw_w[t['name']]/n_s),
                "NBA Mestari":pct(nba_w[t['name']]/n_s),"_n":nba_w[t['name']]})
        for t in east_s:
            rw=rnd_e[t['name']]
            all_rows.append({"Joukkue":t['name']+(" 🔮" if t['seed']>=7 else ""),
                "Konf.":"Itä","#":t['seed'],"NR":f"{t['nr']:+.1f}",
                "R1":pct(rw[1]/n_s),"R2":pct(rw[2]/n_s),"CF":pct(rw[3]/n_s),
                "Konf.Mestari":pct(cw_e[t['name']]/n_s),
                "NBA Mestari":pct(nba_w[t['name']]/n_s),"_n":nba_w[t['name']]})

        df_all = pd.DataFrame(all_rows).sort_values("_n",ascending=False)
        st.dataframe(df_all[["Joukkue","Konf.","#","NR","R1","R2","CF","Konf.Mestari","NBA Mestari"]],
                     use_container_width=True,hide_index=True)
        st.bar_chart(df_all.head(12).set_index("Joukkue")["_n"]
                     .apply(lambda x:round(x/n_s*100,1)),
                     color="#f97316",height=280)

        rc_w,rc_e = st.columns(2)
        for col,ts,cw_map,rnd_map,lbl,clr in [
            (rc_w,west_s,cw_w,rnd_w,"Länsi","#3b82f6"),
            (rc_e,east_s,cw_e,rnd_e,"Itä","#f97316")]:
            with col:
                st.markdown(f"**{lbl}inen konferenssi**")
                rows=[]
                for t in sorted(ts,key=lambda x:x['seed']):
                    rw=rnd_map[t['name']]
                    rows.append({"Joukkue":t['name']+(" 🔮" if t['seed']>=7 else ""),
                        "NR":f"{t['nr']:+.1f}","R1":pct(rw[1]/n_s),
                        "R2":pct(rw[2]/n_s),"CF":pct(rw[3]/n_s),
                        "Mestari":pct(cw_map[t['name']]/n_s),"_m":cw_map[t['name']]})
                df_c=pd.DataFrame(rows)
                st.dataframe(df_c[["Joukkue","NR","R1","R2","CF","Mestari"]],
                             use_container_width=True,hide_index=True)
                st.bar_chart(df_c.set_index("Joukkue")["_m"]
                             .apply(lambda x:round(x/n_s*100,1)),
                             color=clr,height=180)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – SARJA-ANALYYSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_series:
    st.markdown("### Kahden joukkueen sarja-analyysi")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div style="color:#f97316;font-family:Bebas Neue,sans-serif;font-size:1.1rem;'
                    'letter-spacing:1px;">🟠 KORKEAMPI SIJOITUS – KOTIETU</div>',unsafe_allow_html=True)
        sn_a = st.text_input("Joukkue A","Oklahoma City Thunder",key="sa_n")
        nr_a = st.number_input("Net Rating A",value=12.8,step=0.1,format="%.1f",key="sa_nr")
    with c2:
        st.markdown('<div style="color:#3b82f6;font-family:Bebas Neue,sans-serif;font-size:1.1rem;'
                    'letter-spacing:1px;">🔵 MATALAMPI SIJOITUS – VIERASJOUKKUE</div>',unsafe_allow_html=True)
        sn_b = st.text_input("Joukkue B","Los Angeles Lakers",key="sb_n")
        nr_b = st.number_input("Net Rating B",value=3.8,step=0.1,format="%.1f",key="sb_nr")

    bo = st.radio("Sarjamuoto",[5,7],index=1,horizontal=True,format_func=lambda x:f"Best-of-{x}")

    if st.button("🔢 LASKE",use_container_width=True,key="calc_s"):
        res_s = series_probs(nr_a, nr_b, home_adv, variance, bo)
        wn = (bo+1)//2

        st.divider()
        ph = win_prob(nr_a,nr_b, home_adv)
        pa = win_prob(nr_a,nr_b,-home_adv)
        st.markdown("#### Yhden pelin voittotodennäköisyys")
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.markdown(f'<div class="mc"><p class="mv">{ph*100:.1f}%</p><p class="ml">{sn_a} kotona</p></div>',unsafe_allow_html=True)
        mc2.markdown(f'<div class="mc"><p class="mv" style="color:#3b82f6">{(1-ph)*100:.1f}%</p><p class="ml">{sn_b} vieraana</p></div>',unsafe_allow_html=True)
        mc3.markdown(f'<div class="mc"><p class="mv" style="color:#3b82f6">{(1-pa)*100:.1f}%</p><p class="ml">{sn_b} kotona</p></div>',unsafe_allow_html=True)
        mc4.markdown(f'<div class="mc"><p class="mv">{pa*100:.1f}%</p><p class="ml">{sn_a} vieraana</p></div>',unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Sarjan voittotodennäköisyys")
        bha,bba = res_s['p_home']*100,res_s['p_away']*100
        ca,cb = st.columns(2)
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
        mr = [
            {"Linja":"+3.5 (sweep)",
             f"{sn_a} kattaa":pct(res_s['p_home_p35']),
             f"{sn_b} kattaa":pct(res_s['p_away_p35'])},
            {"Linja":"+2.5 (4-0 / 4-1)",
             f"{sn_a} kattaa":pct(res_s['p_home_p25']),
             f"{sn_b} kattaa":pct(res_s['p_away_p25'])},
            {"Linja":"+1.5 (4-0 / 4-1 / 4-2)",
             f"{sn_a} kattaa":pct(res_s['p_home_p15']),
             f"{sn_b} kattaa":pct(res_s['p_away_p15'])},
        ]
        st.dataframe(pd.DataFrame(mr).set_index("Linja"),use_container_width=True)

        st.divider()
        st.markdown("#### Sarjan lopputulokset")
        oc=[]
        for (wh,wa),prob in sorted(res_s['outcomes'].items(),key=lambda x:-x[1]):
            winner = sn_a if wh==wn else sn_b
            loser  = sn_b if wh==wn else sn_a
            score  = f"{wh}–{wa}" if wh==wn else f"{wa}–{wh}"
            oc.append({"Tulos":f"{winner} {score} {loser}","Voittaja":winner,
                       "Tn":pct(prob),"_p":round(prob*100,2)})
        df_oc=pd.DataFrame(oc)
        st.dataframe(df_oc[["Tulos","Voittaja","Tn"]],use_container_width=True,hide_index=True)
        st.bar_chart(df_oc.set_index("Tulos")["_p"],color="#f97316",height=230)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – PLAY-IN
# ══════════════════════════════════════════════════════════════════════════════
with tab_playin:
    st.markdown("### Play-In turnauksen analyysi")
    st.markdown("""<div class="pibox">
<b style="color:#a78bfa;font-family:Bebas Neue,sans-serif;letter-spacing:2px;">🔮 PLAY-IN RAKENNE</b><br><br>
<b>Peli 1:</b> #7 vs #8 (kotietu #7) → <b>Voittaja = playoff sija #7</b><br>
<b>Peli 2:</b> #9 vs #10 (kotietu #9) → <b>Häviäjä putoaa kaudelta</b><br>
<b>Peli 3:</b> Häviäjä(P1) vs Voittaja(P2), kotietu häviäjälle(P1) → <b>Voittaja = playoff sija #8</b>
</div>""",unsafe_allow_html=True)

    pi_conf = st.radio("Konferenssi",["Länsi","Itä"],horizontal=True,key="pi_conf")
    pi_defs = [(n,nr) for n,s,nr in DEFAULTS["west" if pi_conf=="Länsi" else "east"] if s>=7]

    pi_cols = st.columns(4)
    pi_names,pi_nrs=[],[]
    for i,(col,(dn,dnr)) in enumerate(zip(pi_cols,pi_defs)):
        with col:
            st.markdown(f'<div style="color:#a78bfa;font-family:Bebas Neue,sans-serif;">SIJA #{i+7}</div>',unsafe_allow_html=True)
            n  = col.text_input("",value=dn,  key=f"pi_n{i}{pi_conf}",label_visibility="collapsed")
            nr = col.number_input("",value=dnr,step=0.1,format="%.1f",key=f"pi_nr{i}{pi_conf}",label_visibility="collapsed")
            pi_names.append(n); pi_nrs.append(nr)

    if st.button("🔮 LASKE PLAY-IN",use_container_width=True):
        pm={pi_names[i]:pi_nrs[i] for i in range(4)}
        t7,t8,t9,t10=pi_names
        pr=playin_exact(t7,t8,t9,t10,pm,home_adv)

        g1=win_prob(pm[t7],pm[t8],home_adv)
        g2=win_prob(pm[t9],pm[t10],home_adv)

        st.divider()
        gc1,gc2,gc3=st.columns(3)
        gc1.markdown(f'<div class="mc"><p style="color:#a78bfa;font-family:Bebas Neue,sans-serif;margin:0 0 3px;">PELI 1</p><p style="margin:2px 0;font-size:0.77rem;">{t7} vs {t8}</p><p class="mv">{g1*100:.1f}%</p><p class="ml">{t7} voittaa</p></div>',unsafe_allow_html=True)
        gc2.markdown(f'<div class="mc"><p style="color:#a78bfa;font-family:Bebas Neue,sans-serif;margin:0 0 3px;">PELI 2</p><p style="margin:2px 0;font-size:0.77rem;">{t9} vs {t10}</p><p class="mv">{g2*100:.1f}%</p><p class="ml">{t9} voittaa</p></div>',unsafe_allow_html=True)
        gc3.markdown(f'<div class="mc"><p style="color:#a78bfa;font-family:Bebas Neue,sans-serif;margin:0 0 3px;">PELI 3</p><p style="margin:2px 0;font-size:0.77rem;">Riippuu P1 & P2</p><p class="mv" style="font-size:1.6rem;">4 sk.</p></div>',unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Pääsy playoffseihin")
        rows=[{"Joukkue":n,"#":f"#{i+7}","NR":f"{pi_nrs[i]:+.1f}",
               "Sija #7":pct(pr[n]['p7']),"Sija #8":pct(pr[n]['p8']),
               "Pääsee":pct(pr[n]['p_qualify']),"_q":pr[n]['p_qualify']}
              for i,n in enumerate(pi_names)]
        df_pi=pd.DataFrame(rows).sort_values("_q",ascending=False)
        st.dataframe(df_pi[["Joukkue","#","NR","Sija #7","Sija #8","Pääsee"]],
                     use_container_width=True,hide_index=True)

        for _,row in df_pi.iterrows():
            nm=row["Joukkue"]
            p7=pr[nm]['p7']*100; p8=pr[nm]['p8']*100; po=(1-pr[nm]['p_qualify'])*100
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
# TAB 4 – REITTIANALYYSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_path:
    st.markdown("### Reittianalyysi")
    if "res" not in st.session_state:
        st.info("Aja simulaatio ensin Bracket-välilehdellä.")
    else:
        cw_e,cw_w,nba_w,rnd_e,rnd_w,pq_e,pq_w = st.session_state["res"]
        east_s = st.session_state["east_s"]
        west_s = st.session_state["west_s"]
        n_s    = st.session_state["n_sim_s"]
        all_t  = east_s+west_s

        sel = st.selectbox("Valitse joukkue",
                           sorted([t['name'] for t in all_t],key=lambda x:-nba_w[x]))
        t_obj = next(t for t in all_t if t['name']==sel)
        is_e  = t_obj in east_s
        rnd_m = rnd_e if is_e else rnd_w
        cw_m  = cw_e  if is_e else cw_w
        pq_m  = pq_e  if is_e else pq_w
        rw    = rnd_m[sel]
        playin= t_obj['seed']>=7

        st.divider()
        st.markdown(f"#### {sel} – todennäköisyydet vaiheittain")
        stages=[]
        if playin:
            q=pq_m[sel]
            stages+=[("🔮 Play-In: pääsee playoffseihin",(q[7]+q[8])/n_s,"#7c3aed"),
                     ("🔮 Play-In: sija #7",q[7]/n_s,"#a78bfa"),
                     ("🔮 Play-In: sija #8",q[8]/n_s,"#6d28d9")]
        stages+=[
            ("✅ Voittaa 1. kierroksen",rw[1]/n_s,"#f97316"),
            ("✅ Voittaa 2. kierroksen",rw[2]/n_s,"#fb923c"),
            ("✅ Voittaa konferenssifinalin",rw[3]/n_s,"#fbbf24"),
            ("🏆 Konferenssimestari",cw_m[sel]/n_s,"#22c55e"),
            ("🏆 NBA-mestari",nba_w[sel]/n_s,"#16a34a")]

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
        st.markdown("#### Ehdolliset todennäköisyydet")
        cond_rows=[]; prev=1.0
        for label,prob,_ in stages:
            if "sija #" in label: continue
            cond=(prob/prev) if prev>0.001 else 0
            cond_rows.append({"Vaihe":label,"Absoluuttinen":f"{prob*100:.1f}%","Ehdollinen":f"{cond*100:.1f}%"})
            prev=prob
        st.dataframe(pd.DataFrame(cond_rows),use_container_width=True,hide_index=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – PÄIVITÄ BRACKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_update:
    st.markdown("### 🔄 Päivitä bracket todellisilla tuloksilla")
    st.caption("Kun sarja on pelattu, lukitse voittaja. Simulaatio käyttää oikeita tuloksia "
               "ja laskee jäljellä olevat todennäköisyydet uudelleen.")

    # Hae joukkuenimet session_statesta
    def get_conf_teams(conf_key):
        teams = {}
        for i in range(10):
            nm = st.session_state.get(f"{conf_key}{i}n", DEFAULTS[conf_key][i][0])
            sd = st.session_state.get(f"{conf_key}{i}s", DEFAULTS[conf_key][i][1])
            nr_ = st.session_state.get(f"{conf_key}{i}r", DEFAULTS[conf_key][i][2])
            teams[sd] = {"name":nm,"seed":sd,"nr":nr_}
        return teams

    conf_upd = st.radio("Konferenssi",["Länsi","Itä"],horizontal=True,key="upd_conf")
    conf_key_upd = "west" if conf_upd=="Länsi" else "east"
    lock_key = f"locked_{conf_key_upd}"
    locked = st.session_state[lock_key]
    teams_by_seed = get_conf_teams(conf_key_upd)

    def tname(s): return teams_by_seed.get(s,{}).get("name",f"Sija {s}")

    st.divider()
    st.markdown("#### Play-In tulokset")
    pi_c1, pi_c2 = st.columns(2)
    with pi_c1:
        st.markdown(f"**Sija 7 (Play-In voittaja)**")
        pi7_opts = ["— Ei lukittu —", tname(7), tname(8), tname(9), tname(10)]
        pi7_cur = locked.get("playin7","— Ei lukittu —")
        pi7_sel = st.selectbox("Sija 7 qualifier", pi7_opts,
                               index=pi7_opts.index(pi7_cur) if pi7_cur in pi7_opts else 0,
                               key=f"pi7_{conf_key_upd}")
    with pi_c2:
        st.markdown(f"**Sija 8 (Play-In voittaja)**")
        pi8_opts = ["— Ei lukittu —", tname(7), tname(8), tname(9), tname(10)]
        pi8_cur = locked.get("playin8","— Ei lukittu —")
        pi8_sel = st.selectbox("Sija 8 qualifier", pi8_opts,
                               index=pi8_opts.index(pi8_cur) if pi8_cur in pi8_opts else 0,
                               key=f"pi8_{conf_key_upd}")

    st.divider()
    st.markdown("#### 1. kierros (R1)")

    # Play-In qualified teams (tai oletusseeds)
    q7_name = locked.get("playin7", tname(7))
    q8_name = locked.get("playin8", tname(8))

    r1_matchups = [
        ("r1_1v8", tname(1), q8_name,  "1 vs 8"),
        ("r1_4v5", tname(4), tname(5), "4 vs 5"),
        ("r1_2v7", tname(2), q7_name,  "2 vs 7"),
        ("r1_3v6", tname(3), tname(6), "3 vs 6"),
    ]
    r1_cols = st.columns(2)
    for idx,(lock_k, team_h, team_a, lbl) in enumerate(r1_matchups):
        with r1_cols[idx%2]:
            st.markdown(f"**{lbl}: {team_h} vs {team_a}**")
            opts = ["— Ei lukittu —", team_h, team_a]
            cur  = locked.get(lock_k,"— Ei lukittu —")
            sel  = st.selectbox(lbl, opts,
                                index=opts.index(cur) if cur in opts else 0,
                                key=f"{lock_k}_{conf_key_upd}", label_visibility="collapsed")
            locked[lock_k] = sel if sel != "— Ei lukittu —" else None

    st.divider()
    st.markdown("#### 2. kierros (R2) – Puolivälierät")

    w1v8 = locked.get("r1_1v8") or f"{tname(1)}/{q8_name}"
    w4v5 = locked.get("r1_4v5") or f"{tname(4)}/{tname(5)}"
    w2v7 = locked.get("r1_2v7") or f"{tname(2)}/{q7_name}"
    w3v6 = locked.get("r1_3v6") or f"{tname(3)}/{tname(6)}"

    r2_cols = st.columns(2)
    with r2_cols[0]:
        st.markdown(f"**Yläpuolisko: W(1v8) vs W(4v5)**")
        r2a_opts = ["— Ei lukittu —",
                    locked.get("r1_1v8","?"), locked.get("r1_4v5","?")] \
                   if locked.get("r1_1v8") and locked.get("r1_4v5") else \
                   ["— Ei lukittu —", w1v8, w4v5]
        r2a_cur = locked.get("r2_A","— Ei lukittu —")
        r2a_sel = st.selectbox("R2A",r2a_opts,
                               index=r2a_opts.index(r2a_cur) if r2a_cur in r2a_opts else 0,
                               key=f"r2a_{conf_key_upd}",label_visibility="collapsed")
        locked["r2_A"] = r2a_sel if r2a_sel != "— Ei lukittu —" else None

    with r2_cols[1]:
        st.markdown(f"**Alapuolisko: W(2v7) vs W(3v6)**")
        r2b_opts = ["— Ei lukittu —",
                    locked.get("r1_2v7","?"), locked.get("r1_3v6","?")] \
                   if locked.get("r1_2v7") and locked.get("r1_3v6") else \
                   ["— Ei lukittu —", w2v7, w3v6]
        r2b_cur = locked.get("r2_B","— Ei lukittu —")
        r2b_sel = st.selectbox("R2B",r2b_opts,
                               index=r2b_opts.index(r2b_cur) if r2b_cur in r2b_opts else 0,
                               key=f"r2b_{conf_key_upd}",label_visibility="collapsed")
        locked["r2_B"] = r2b_sel if r2b_sel != "— Ei lukittu —" else None

    st.divider()
    st.markdown("#### Konferenssifinaali (CF)")
    wA = locked.get("r2_A") or "Yläpuolisko voittaja"
    wB = locked.get("r2_B") or "Alapuolisko voittaja"
    st.markdown(f"**{wA} vs {wB}**")
    cf_opts = ["— Ei lukittu —",
               locked.get("r2_A","?"), locked.get("r2_B","?")] \
              if locked.get("r2_A") and locked.get("r2_B") else \
              ["— Ei lukittu —", wA, wB]
    cf_cur = locked.get("cf","— Ei lukittu —")
    cf_sel = st.selectbox("CF",cf_opts,
                          index=cf_opts.index(cf_cur) if cf_cur in cf_opts else 0,
                          key=f"cf_{conf_key_upd}",label_visibility="collapsed")
    locked["cf"] = cf_sel if cf_sel != "— Ei lukittu —" else None

    # Tallenna Play-In valinnat
    if pi7_sel != "— Ei lukittu —": locked["playin7"] = pi7_sel
    elif "playin7" in locked: del locked["playin7"]
    if pi8_sel != "— Ei lukittu —": locked["playin8"] = pi8_sel
    elif "playin8" in locked: del locked["playin8"]

    # Poista None-arvot
    st.session_state[lock_key] = {k:v for k,v in locked.items() if v}

    st.divider()
    locked_count = len(st.session_state[lock_key])
    if locked_count:
        st.success(f"🔒 {locked_count} vaihetta lukittu {conf_upd}essa. "
                   f"Mene Bracket-välilehdelle ja paina Simuloi uudelleen.")
        st.json(st.session_state[lock_key])
    else:
        st.info("Ei lukittuja sarjoja. Valitse sarjojen voittajat yllä.")

    if st.button("🗑️ Tyhjennä kaikki lukitukset", key=f"clear_{conf_key_upd}"):
        st.session_state[lock_key] = {}
        st.rerun()
