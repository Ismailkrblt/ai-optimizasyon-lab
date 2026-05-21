"""
app.py – Meta-Sezgisel Optimizasyon Görselleştiricisi  (Gelişmiş Sürüm)
5 Sekme: Görselleştirici · Karşılaştırma · Analiz · Algoritma Rehberi · Fonksiyon Galerisi
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import io
import csv

from optimizer import (
    BENCHMARK_FUNCTIONS, ALGORITHM_REGISTRY, BenchmarkFunction,
    OptimizationResult, OptimizationState, get_diversity_history,
)

# ══════════════════════════════════════════════════════════════
# SAYFA YAPISI
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Meta-Sezgisel Optimizasyon Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS — PROFESYONELEŞTİRME
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Genel ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Başlık ──────────────────────────────── */
.hero-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 0.6rem 0 0 0; margin-bottom: 0;
}
.hero-sub {
    text-align: center; color: #64748b; font-size: 0.95rem;
    font-weight: 400; margin-bottom: 1.2rem;
}

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0f0e17 0%, #1a1040 100%);
    padding: 1.2rem 1rem;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h2 { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { color: #e2e8f0; }
section[data-testid="stSidebar"] hr { border-color: #334155; }

/* ── Kartlar ─────────────────────────────── */
.kart {
    background: white; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1rem 1.2rem; box-shadow: 0 4px 16px rgba(99,102,241,0.07);
    transition: box-shadow .2s;
}
.kart:hover { box-shadow: 0 8px 28px rgba(99,102,241,0.14); }

.metric-kart {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 0.85rem 0.7rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-etiket { font-size: 0.68rem; color: #94a3b8; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 0.2rem; }
.metric-deger  { font-size: 1.2rem; font-weight: 700; color: #1e293b; }

/* ── Algo açıklama kartı ─────────────────── */
.algo-card {
    background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
    border-left: 4px solid #6366f1; border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem; font-size: 0.85rem; color: #475569;
    margin: 0.4rem 0 0.8rem 0;
}

/* ── Badge ───────────────────────────────── */
.badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; margin: 0.15rem;
}
.badge-indigo { background:#eef2ff; color:#4f46e5; }
.badge-green  { background:#f0fdf4; color:#16a34a; }
.badge-amber  { background:#fffbeb; color:#d97706; }
.badge-rose   { background:#fff1f2; color:#e11d48; }
.badge-violet { background:#f5f3ff; color:#7c3aed; }
.badge-sky    { background:#f0f9ff; color:#0284c7; }

/* ── Rehber kartları ─────────────────────── */
.guide-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.guide-header { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.8rem; }
.guide-icon { font-size: 2rem; }
.guide-title { font-size: 1.15rem; font-weight:700; color:#1e293b; }
.guide-sub   { font-size: 0.8rem; color:#64748b; }

/* ── Sekme stili ─────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600; font-size: 0.88rem;
    padding: 0.5rem 1.1rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
}

/* ── Buton ───────────────────────────────── */
.stButton > button {
    border-radius: 10px; font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    border: none; color: white;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.45);
}

/* ── Ekstra bilgi şeridi ─────────────────── */
.info-strip {
    background: linear-gradient(135deg, #f8faff, #f1f5ff);
    border: 1px solid #dbeafe; border-radius: 10px;
    padding: 0.5rem 1.2rem; font-size: 0.84rem;
    color: #475569; text-align: center; margin: 0.4rem 0;
}

/* ── Footer ──────────────────────────────── */
.footer {
    text-align:center; color:#94a3b8; font-size:0.78rem;
    padding: 1.5rem 0 0.5rem 0;
}

/* ── Başarı / Hata renkleri ──────────────── */
.ok   { color: #16a34a; font-weight:700; }
.fail { color: #dc2626; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ALGORİTMA REHBER İÇERİĞİ
# ══════════════════════════════════════════════════════════════

ALGO_GUIDE = {
    "Parçacık Sürü Optimizasyonu (PSO)": {
        "icon": "✈️", "year": 1995, "authors": "Kennedy & Eberhart",
        "category": "Sürü Zekası", "color": "#6366f1",
        "inspired": "Kuş sürüleri ve balık okulu davranışı",
        "description": "Her parçacık, hem kendi geçmişteki en iyi konumunu (pbest) hem de sürünün küresel en iyi konumunu (gbest) izleyerek hızını günceller. Basit ama güçlü; sürekli optimizasyonda sıkça tercih edilir.",
        "steps": [
            "Rastgele konum ve hız ile parçacıkları başlat",
            "Her parçacık için f(x) hesapla",
            "Kişisel ve küresel en iyileri güncelle",
            "v = w·v + c₁·r₁·(pbest−x) + c₂·r₂·(gbest−x)",
            "x = x + v  →  Sonra 2'ye git",
        ],
        "pros":  ["Az parametre", "Hızlı yakınsama", "Uygulaması kolay"],
        "cons":  ["Erken yakınsama riski", "Sürü çeşitliliği azalabilir"],
        "best_for": ["Sürekli optimizasyon", "Çok modlu yüzeyler"],
        "formula": "vᵢ = w·vᵢ + c₁·r₁·(pbestᵢ−xᵢ) + c₂·r₂·(gbest−xᵢ)",
    },
    "Genetik Algoritma (GA)": {
        "icon": "🧬", "year": 1975, "authors": "John Holland",
        "category": "Evrimsel Hesaplama", "color": "#10b981",
        "inspired": "Doğal seçim ve genetik miras",
        "description": "Popülasyondaki bireyler, turnuva seçimi, SBX çaprazlama ve polinomial mutasyon operatörleriyle nesiller boyu evrilerek daha iyi çözümlere yaklaşır. Elitizm ile en iyi birey korunur.",
        "steps": [
            "Rastgele popülasyon oluştur ve fitness hesapla",
            "Turnuva ile ebeveynleri seç",
            "SBX çaprazlama ile çocuk üret",
            "Polinomial mutasyon uygula",
            "Elitizm: en iyiyi yeni nesle aktar → 2'ye git",
        ],
        "pros":  ["Genel amaçlı", "Çeşitlilik yüksek", "Paralel arama"],
        "cons":  ["Yavaş yakınsama", "Çok sayıda parametre", "Erken yakınsama"],
        "best_for": ["Kombinatoryal problemler", "Çok boyutlu arama"],
        "formula": "Child = SBX(Parent₁, Parent₂) + PolMutation",
    },
    "Isıl İşlem (SA)": {
        "icon": "❄️", "year": 1983, "authors": "Kirkpatrick, Gelatt, Vecchi",
        "category": "Fizik Tabanlı", "color": "#0ea5e9",
        "inspired": "Metallerin kontrollü soğutulması (tavlama) süreci",
        "description": "Tek bir çözümü iteratif olarak iyileştirir. Yüksek sıcaklıkta kötü çözümleri de kabul ederek yerel minimumdan kurtulur; düşen sıcaklıkla arama daraltılır.",
        "steps": [
            "Rastgele bir başlangıç çözümü seç, T = T_init",
            "Komşu çözüm üret (Gauss gürültüsü)",
            "Δf < 0 ise kabul et; aksi halde P = e^(-Δf/T) olasılığıyla kabul et",
            "T = T · α  (soğutma)",
            "T < T_min ise dur; yoksa 2'ye git",
        ],
        "pros":  ["Teorik yakınsama garantisi", "Basit uygulama", "Yerel minimumdan kaçış"],
        "cons":  ["Tek çözüm (yavaş)", "Soğuma takvimi hassas", "Paralel arama yok"],
        "best_for": ["Kombinezonlar (TSP)", "Az sayıda parametre", "Keşif gerektiren problemler"],
        "formula": "P(kabul) = exp(−ΔE / T)",
    },
    "Tabu Araştırma (TS)": {
        "icon": "🚫", "year": 1986, "authors": "Fred Glover",
        "category": "Yerel Arama / Hafıza Tabanlı", "color": "#f59e0b",
        "inspired": "İnsan problem çözme davranışı ve kısa süreli hafıza",
        "description": "Komşuluk aramasında son ziyaret edilen konumları 'tabu listesi'nde saklar; bu konumlara geri dönmez. Böylece yerel optimumlarda döngüye girmez ve arama alanında ilerlemeye devam eder.",
        "steps": [
            "Başlangıç çözümü seç",
            "N adet komşu üret",
            "Tabu olmayan (veya aspirasyon kriteri sağlayan) en iyi komşuyu seç",
            "Seçilen komşuyu tabu listesine ekle",
            "Tabu listesi uzunluğu aşılırsa en eskiyi sil → 2'ye git",
        ],
        "pros":  ["Yerel minimumdan güçlü kaçış", "Hafıza mekanizması", "Deterministik"],
        "cons":  ["Parametre ayarı kritik", "Hesaplama maliyeti yüksek", "Komşuluk tanımı önemli"],
        "best_for": ["Çizelgeleme", "Kombinatoryal optimizasyon", "Rota problemleri"],
        "formula": "x* = argmin{f(N(x)) : x ∉ TabuListesi}",
    },
    "Karınca Koloni (ACOR)": {
        "icon": "🐜", "year": "1992 (ACOR: 2008)", "authors": "Dorigo / Socha & Dorigo",
        "category": "Sürü Zekası", "color": "#84cc16",
        "inspired": "Karıncaların feromon bırakarak en kısa yolu bulması",
        "description": "Sürekli uzay için ACOR varyantı kullanılır. En iyi çözümlerin oluşturduğu arşivden Gauss dağılımları türetilir; yeni karıncalar bu dağılımlardan örneklenir.",
        "steps": [
            "Arşivi rastgele çözümlerle başlat ve sırala",
            "Ağırlıklı olasılıkla arşivden bir çözüm seç",
            "Seçilen çözüm etrafında Gauss'tan örnekle",
            "Yeni çözümleri arşive ekle, en iyileri tut",
            "Yakınsama koşulu sağlanana kadar 2'ye git",
        ],
        "pros":  ["Sürekli uzayda etkili (ACOR)", "Arşiv tabanlı çeşitlilik", "Güçlü keşif"],
        "cons":  ["Ayrık problemler için uyarlanması gerekir", "Arşiv boyutu kritik"],
        "best_for": ["Rota optimizasyonu", "Çizelgeleme", "Sürekli fonksiyon optimizasyonu"],
        "formula": "Gₗ(x) = (1/σₗ√2π) · exp(−(x−mₗ)²/2σₗ²)",
    },
    "Yapay Bağışıklık (CLONALG)": {
        "icon": "🦠", "year": 2002, "authors": "De Castro & Von Zuben",
        "category": "Bağışıklık Tabanlı", "color": "#ec4899",
        "inspired": "Bağışıklık sisteminin klonal seçim ilkesi",
        "description": "En iyi antikorlar (çözümler), afiniteleriyle orantılı sayıda klonlanır. Klon sayısı iyi çözümler için fazla, mutasyon oranı ise düşük tutulur. Böylece iyi bölgeler derinlemesine araştırılır.",
        "steps": [
            "Popülasyonu başlat, afiniteleri (f değerleri) hesapla",
            "En iyi antikoru seç ve klonla (sayı afiniteyle orantılı)",
            "Klonları hipermutasyona uğrat (afin olanı az mutasyona)",
            "En iyi klonlarla popülasyonu güncelle",
            "En kötüleri rastgele bireylerle değiştir (çeşitlilik) → 2'ye git",
        ],
        "pros":  ["Güçlü çeşitlilik", "Yerel + küresel denge", "Çok tepe noktası için iyi"],
        "cons":  ["Çok parametre", "Hesaplama yoğun", "Yavaş yakınsama"],
        "best_for": ["Çok modlu yüzeyler", "Anomali tespiti", "Örüntü tanıma"],
        "formula": "σ = exp(−β · (1 − affinity)) × σ_max",
    },
    "Diferansiyel Gelişim (DE)": {
        "icon": "🌀", "year": 1997, "authors": "Storn & Price",
        "category": "Evrimsel Hesaplama", "color": "#8b5cf6",
        "inspired": "Vektör farkı tabanlı arama stratejisi",
        "description": "Popülasyondan rastgele seçilen üç bireyin fark vektöründen mutant oluşturur. Binomial çaprazlama ile trial vektörü türetilir; eğer daha iyiyse mevcut bireyin yerini alır.",
        "steps": [
            "Rastgele popülasyon başlat",
            "Her birey için: 3 rastgele birey seç (r1, r2, r3)",
            "Mutant: v = x_r1 + F·(x_r2 − x_r3)",
            "Binomial çaprazlama: CR oranında mutant genlerini al",
            "Eğer trial daha iyiyse mevcut bireyi değiştir → 2'ye git",
        ],
        "pros":  ["Az parametre (F, CR)", "Hızlı ve güçlü", "Sürekli problemlerde üstün"],
        "cons":  ["Popülasyon çeşitliliği düşebilir", "Çok modlu sorunlarda zorlanabilir"],
        "best_for": ["Sürekli optimizasyon", "Mühendislik tasarımı", "Hiperparametre ayarı"],
        "formula": "vᵢ = x_r1 + F·(x_r2 − x_r3)",
    },
    "Yapay Arı Koloni (ABC)": {
        "icon": "🐝", "year": 2005, "authors": "Derviş Karaboğa",
        "category": "Sürü Zekası", "color": "#f97316",
        "inspired": "Bal arılarının besin kaynağı arama davranışı",
        "description": "Üç arı rolü vardır: İşçi arılar mevcut kaynakları sömürür, izci arılar kaliteli kaynakları tercih eder, kaşif arılar tükenen kaynakları rastgele yeniler. Bu denge keşif-sömürü ikileminini çözer.",
        "steps": [
            "Besin kaynaklarını (çözümleri) rastgele başlat",
            "İşçi arılar: her kaynak için komşu üret, iyiyse güncelle",
            "İzci arılar: fitness olasılığıyla kaynak seç ve iyileştir",
            "Tükenme sayacı aşılan kaynakları yenile (kaşif arı)",
            "En iyi kaynağı kaydet → 2'ye git",
        ],
        "pros":  ["Dengeli keşif-sömürü", "Az kontrol parametresi", "Türkiye kaynaklı 🇹🇷"],
        "cons":  ["Yüksek boyutlarda yavaşlar", "Yakınsama hızı orta"],
        "best_for": ["Çok modlu optimizasyon", "Mühendislik problemleri"],
        "formula": "xₖₙₑᵥ = xⱼ + φ·(xⱼ − xₖ)",
    },
}

ALGO_COLORS = {
    name: info["color"]
    for name, info in ALGO_GUIDE.items()
    if name in ALGORITHM_REGISTRY
}

# ══════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════

_CONTOUR_CACHE: dict = {}

def _contour_data(bf: BenchmarkFunction, n: int = 220):
    key = bf.name
    if key not in _CONTOUR_CACHE:
        lb, ub = bf.bounds
        x = np.linspace(lb, ub, n)
        y = np.linspace(lb, ub, n)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda xi, yi: bf.func(np.array([xi, yi])))(X, Y)
        _CONTOUR_CACHE[key] = (x, y, Z)
    return _CONTOUR_CACHE[key]

def _plotly_theme() -> dict:
    return dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        margin=dict(l=8, r=8, t=8, b=8),
    )


def draw_contour_map(
    bf, state, result, colorscale, show_traj, show_known, n_levels, step_idx,
    agent_color="#6366f1", height=480,
):
    x, y, Z = _contour_data(bf)
    Z_plot = np.log1p(Z - Z.min())

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=x, y=y, z=Z_plot,
        ncontours=n_levels, colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="log(f+1)", thickness=10, len=0.75, x=1.01,
                      tickfont=dict(size=9)),
        contours=dict(showlines=True, coloring="heatmap"),
        opacity=0.9, name="f(x,y)",
        hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
    ))

    pos = state.positions
    if pos.ndim == 1:
        pos = pos.reshape(1, 2)
    scores_disp = np.array([bf.func(np.array([p[0], p[1]])) for p in pos])

    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode="markers",
        marker=dict(
            size=9, color=scores_disp, colorscale="Turbo",
            showscale=False,
            line=dict(width=1.2, color="rgba(255,255,255,0.8)"),
            opacity=0.9,
        ),
        name="Ajanlar",
        hovertemplate="x:%{x:.3f}  y:%{y:.3f}<extra>Ajan</extra>",
    ))

    if show_traj and step_idx > 1:
        bx = [result.states[i].best_position[0] for i in range(step_idx + 1)]
        by = [result.states[i].best_position[1] for i in range(step_idx + 1)]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(color="rgba(255,255,255,0.55)", width=1.5, dash="dot"),
            name="En iyi yol", hoverinfo="skip",
        ))

    bp = state.best_position
    fig.add_trace(go.Scatter(
        x=[bp[0]], y=[bp[1]], mode="markers",
        marker=dict(size=18, color="gold", symbol="star",
                    line=dict(width=2, color="#1e293b")),
        name="En İyi",
        hovertemplate=f"x:{bp[0]:.4f}<br>y:{bp[1]:.4f}<br>f:{state.best_score:.6f}<extra>En İyi</extra>",
    ))

    if show_known:
        for km in bf.global_minima:
            fig.add_trace(go.Scatter(
                x=[km[0]], y=[km[1]], mode="markers",
                marker=dict(size=14, color="#ef4444", symbol="x",
                            line=dict(width=3, color="white")),
                name="Küresel Min.",
                hovertemplate=f"x:{km[0]:.4f}  y:{km[1]:.4f}<extra>Küresel Min.</extra>",
            ))

    lb, ub = bf.bounds
    fig.update_layout(
        height=height,
        xaxis=dict(range=[lb, ub], title="x", showgrid=False, zeroline=False),
        yaxis=dict(range=[lb, ub], title="y", showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(size=10)),
        plot_bgcolor="#0a0a12", paper_bgcolor="white",
        **{k: v for k, v in _plotly_theme().items() if k not in ("plot_bgcolor","paper_bgcolor")},
    )
    return fig


def draw_convergence(result, step_idx, known_min, color="#6366f1", height=480):
    history = [s.best_score for s in result.states]
    iters = list(range(len(history)))

    fig = go.Figure()

    # Tüm süreç (soluk)
    fig.add_trace(go.Scatter(
        x=iters, y=history, mode="lines",
        line=dict(color=color, width=1), opacity=0.2,
        name="Tüm süreç", hoverinfo="skip",
    ))
    # Tamamlanan kısım
    fig.add_trace(go.Scatter(
        x=iters[:step_idx + 1], y=history[:step_idx + 1], mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.07,)}",
        name="İlerleme",
        hovertemplate="İter:%{x}<br>Skor:%{y:.6f}<extra></extra>",
    ))
    # Şu anki nokta
    fig.add_trace(go.Scatter(
        x=[step_idx], y=[history[step_idx]], mode="markers",
        marker=dict(size=13, color="gold", symbol="diamond",
                    line=dict(width=2, color="#1e293b")),
        name="Şu an",
        hovertemplate=f"İter {step_idx}: {history[step_idx]:.6f}<extra></extra>",
    ))
    # Hedef çizgisi
    if known_min is not None:
        fig.add_hline(y=known_min, line_dash="dash", line_color="#ef4444",
                      opacity=0.6,
                      annotation_text=f" Hedef: {known_min:.4f}",
                      annotation_position="top left",
                      annotation_font_color="#ef4444")

    # İyileşme notu
    if step_idx > 0 and abs(history[0]) > 1e-12:
        pct = (history[0] - history[step_idx]) / abs(history[0]) * 100
        fig.add_annotation(
            x=step_idx * 0.7, y=history[0] * 0.92,
            text=f"↓ {pct:.1f}% iyileşme",
            showarrow=False, font=dict(color=color, size=11, family="Inter"),
        )

    fig.update_layout(
        height=height,
        xaxis=dict(title="İterasyon", showgrid=True,
                   gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(title="En İyi Skor", showgrid=True,
                   gridcolor="#f1f5f9", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        **_plotly_theme(),
    )
    return fig


def draw_diversity(result, step_idx, color="#6366f1", height=220):
    divs = get_diversity_history(result.states)
    iters = list(range(len(divs)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters[:step_idx + 1], y=divs[:step_idx + 1],
        mode="lines", line=dict(color="#f59e0b", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
        name="Çeşitlilik",
        hovertemplate="İter:%{x}<br>Çeşitlilik:%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=height,
        xaxis=dict(title="İterasyon", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Çeşitlilik (σ)", showgrid=True, gridcolor="#f1f5f9"),
        **_plotly_theme(),
    )
    return fig


def draw_surface_3d(bf, state, height=480):
    x, y, Z = _contour_data(bf, n=80)
    pos = state.positions
    if pos.ndim == 1:
        pos = pos.reshape(1, 2)
    z_agents = np.array([bf.func(np.array([p[0], p[1]])) for p in pos])

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=Z,
        colorscale="Viridis", opacity=0.82, showscale=False,
        name="f(x,y)",
    ))
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=z_agents + (Z.max() - Z.min()) * 0.01,
        mode="markers",
        marker=dict(size=5, color="#f97316",
                    line=dict(color="white", width=0.5)),
        name="Ajanlar",
    ))
    bp = state.best_position
    fig.add_trace(go.Scatter3d(
        x=[bp[0]], y=[bp[1]], z=[bf.func(bp) + (Z.max()-Z.min())*0.015],
        mode="markers",
        marker=dict(size=10, color="gold", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="En İyi",
    ))
    fig.update_layout(
        height=height, scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="f(x,y)",
            bgcolor="#0a0a12",
        ),
        **_plotly_theme(),
    )
    return fig


def draw_comparison(results_dict, known_min, height=420):
    fig = go.Figure()
    palette = ["#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#ec4899"]
    for i, (name, result) in enumerate(results_dict.items()):
        history = [s.best_score for s in result.states]
        color = ALGO_COLORS.get(name, palette[i % len(palette)])
        fig.add_trace(go.Scatter(
            x=list(range(len(history))), y=history,
            mode="lines", name=name,
            line=dict(color=color, width=2.5),
            hovertemplate=f"{name}<br>İter:%{{x}}<br>Skor:%{{y:.6f}}<extra></extra>",
        ))
    if known_min is not None:
        fig.add_hline(y=known_min, line_dash="dash", line_color="#94a3b8",
                      annotation_text=f" Hedef {known_min:.4f}",
                      annotation_position="bottom left")
    fig.update_layout(
        height=height,
        xaxis=dict(title="İterasyon", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="En İyi Skor", showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        **_plotly_theme(),
    )
    return fig


def export_csv(result: OptimizationResult) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["iterasyon", "best_score", "best_x", "best_y", "diversity"])
    divs = get_diversity_history(result.states)
    for s, d in zip(result.states, divs):
        writer.writerow([s.iteration, s.best_score,
                         s.best_position[0], s.best_position[1], d])
    return buf.getvalue().encode("utf-8")


def metric_card(label: str, value: str) -> str:
    return (f'<div class="metric-kart">'
            f'<div class="metric-etiket">{label}</div>'
            f'<div class="metric-deger">{value}</div>'
            f'</div>')

# ══════════════════════════════════════════════════════════════
# BAŞLIK
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">🔬 Meta-Sezgisel Optimizasyon Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">8 algoritma · 10 benchmark fonksiyonu · Adım adım görselleştirme · Karşılaştırma & Analiz</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 Kontrol Paneli")
    st.markdown("---")

    # Algoritma seçimi
    st.markdown("### 🤖 Algoritma")
    algo_name = st.selectbox(
        "alg", list(ALGORITHM_REGISTRY.keys()), index=0,
        label_visibility="collapsed",
        format_func=lambda n: f"{ALGO_GUIDE.get(n,{}).get('icon','●')} {n}",
    )
    algo_info = ALGORITHM_REGISTRY[algo_name]
    guide_entry = ALGO_GUIDE.get(algo_name, {})
    if guide_entry:
        st.markdown(
            f'<div class="algo-card">'
            f'<b>{guide_entry.get("category","")}</b> · {guide_entry.get("year","")}<br>'
            f'{guide_entry.get("description","")[:120]}…</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Benchmark
    st.markdown("### 📐 Test Fonksiyonu")
    func_name = st.selectbox(
        "fn", list(BENCHMARK_FUNCTIONS.keys()), index=0,
        label_visibility="collapsed",
    )
    bf = BENCHMARK_FUNCTIONS[func_name]
    st.caption(f"📌 {bf.description}")
    st.caption(f"🎯 Min = **{bf.min_value:.4f}**  @  {bf.global_minima[0]}")

    st.markdown("---")

    # Parametreler
    st.markdown("### 🎛️ Parametreler")
    params: dict = {}
    for key, cfg in algo_info["params"].items():
        if cfg["type"] == "int":
            params[key] = st.slider(cfg["label"], cfg["min"], cfg["max"],
                                    cfg["default"], step=1)
        elif cfg["type"] == "float":
            params[key] = st.slider(cfg["label"], float(cfg["min"]),
                                    float(cfg["max"]), float(cfg["default"]),
                                    step=float(cfg.get("step", 0.01)))
        elif cfg["type"] == "select":
            params[key] = st.selectbox(cfg["label"], cfg["options"],
                                       index=cfg["options"].index(cfg["default"]))

    st.markdown("---")

    # Görsel ayarlar
    st.markdown("### 🎨 Görsel Ayarlar")
    colorscale   = st.selectbox("Renk Paleti",
                                ["Viridis","Plasma","Turbo","Hot_r","Cividis","RdYlGn_r"], index=0)
    show_traj    = st.checkbox("En iyi yolu göster", value=True)
    show_known   = st.checkbox("Bilinen minimumları göster", value=True)
    n_levels     = st.slider("Kontur seviyesi", 10, 60, 30, step=5)
    show_3d      = st.checkbox("3D Yüzey Görünümü", value=False)
    show_div     = st.checkbox("Çeşitlilik (Diversity) Grafiği", value=True)

    st.markdown("---")

    # Seed
    st.markdown("### 🎲 Tekrar Edilebilirlik")
    use_seed = st.checkbox("Sabit seed kullan", value=False)
    seed_val = st.number_input("Seed", 0, 9999, 42, step=1,
                               disabled=not use_seed)

    st.markdown("---")
    run_btn = st.button("▶  Optimizasyonu Başlat", type="primary",
                        use_container_width=True)

# ══════════════════════════════════════════════════════════════
# OTURUM DURUMU
# ══════════════════════════════════════════════════════════════

for k, v in [("result", None), ("step", 0), ("playing", False),
             ("cmp_results", {}), ("analysis_done", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# OPTİMİZASYONU ÇALIŞTIR
# ══════════════════════════════════════════════════════════════

if run_btn:
    st.session_state.playing = False
    _CONTOUR_CACHE.clear()
    if use_seed:
        np.random.seed(int(seed_val))
    with st.spinner(f"⏳ {algo_name} çalışıyor…"):
        t0 = time.time()
        result = algo_info["func"](bf, **params)
        elapsed = time.time() - t0
    st.session_state.result = result
    st.session_state.step = 0
    st.toast(f"✅ Tamamlandı · {elapsed:.2f}s · Skor: {result.best_score:.6f}", icon="🏆")

# ══════════════════════════════════════════════════════════════
# SEKMELEr
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Görselleştirici",
    "⚔️ Karşılaştırma",
    "📊 Analiz",
    "📚 Algoritma Rehberi",
    "🗂️ Fonksiyon Galerisi",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 – GÖRSELLEŞTİRİCİ
# ──────────────────────────────────────────────────────────────

with tab1:
    result: OptimizationResult | None = st.session_state.result

    if result is None:
        st.markdown("---")
        st.info("👈 Sol panelden bir algoritma ve test fonksiyonu seçin, ardından **▶ Optimizasyonu Başlat** butonuna tıklayın.")
        cols = st.columns(4)
        icons = ["✈️","🧬","❄️","🚫","🐜","🦠","🌀","🐝"]
        names = list(ALGORITHM_REGISTRY.keys())
        for i, col in enumerate(cols):
            n = names[i]
            g = ALGO_GUIDE.get(n, {})
            with col:
                st.markdown(
                    f'<div class="kart" style="text-align:center;padding:1.2rem">'
                    f'<div style="font-size:2rem">{g.get("icon","●")}</div>'
                    f'<div style="font-weight:700;font-size:0.85rem;margin:.4rem 0;color:#1e293b">{n.split("(")[0].strip()}</div>'
                    f'<div style="font-size:0.72rem;color:#64748b">{g.get("category","")}</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("")
        cols2 = st.columns(4)
        for i, col in enumerate(cols2):
            n = names[i + 4]
            g = ALGO_GUIDE.get(n, {})
            with col:
                st.markdown(
                    f'<div class="kart" style="text-align:center;padding:1.2rem">'
                    f'<div style="font-size:2rem">{g.get("icon","●")}</div>'
                    f'<div style="font-weight:700;font-size:0.85rem;margin:.4rem 0;color:#1e293b">{n.split("(")[0].strip()}</div>'
                    f'<div style="font-size:0.72rem;color:#64748b">{g.get("category","")}</div>'
                    f'</div>', unsafe_allow_html=True)
    else:
        total_steps = len(result.states) - 1

        # ── Simülasyon Kontrol Çubuğu ──
        st.markdown("#### 🎬 Simülasyon Kontrolü")
        bc1, bc2, bc3, bc4, _, bslider = st.columns([1,1,1,1,0.3,6])
        with bc1:
            if st.button("⏮", help="Başa sar", use_container_width=True):
                st.session_state.step = 0; st.session_state.playing = False
        with bc2:
            if st.button("◀", help="Geri", use_container_width=True):
                st.session_state.step = max(0, st.session_state.step - 1)
                st.session_state.playing = False
        with bc3:
            if st.button("▶", help="İleri", use_container_width=True):
                st.session_state.step = min(total_steps, st.session_state.step + 1)
                st.session_state.playing = False
        with bc4:
            if st.button("⏭", help="Sona git", use_container_width=True):
                st.session_state.step = total_steps; st.session_state.playing = False
        with bslider:
            new_step = st.slider("İterasyon", 0, total_steps,
                                 st.session_state.step,
                                 label_visibility="collapsed")
            if new_step != st.session_state.step:
                st.session_state.step = new_step

        pc1, pc2, pc3 = st.columns([1, 1, 4])
        with pc1:
            if st.button("▶️ Oynat", use_container_width=True):
                st.session_state.playing = True
        with pc2:
            if st.button("⏸ Durdur", use_container_width=True):
                st.session_state.playing = False
        with pc3:
            play_speed = st.select_slider(
                "Hız", options=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0], value=2.0,
                format_func=lambda v: f"{v}×", label_visibility="collapsed",
            )

        # ── Metrikler ──
        st.markdown("---")
        cur: OptimizationState = result.states[st.session_state.step]
        divs = get_diversity_history(result.states)
        mc = st.columns(6)
        met = [
            ("🏆 En İyi Skor",   f"{cur.best_score:.6f}"),
            ("📍 X",             f"{cur.best_position[0]:.4f}"),
            ("📍 Y",             f"{cur.best_position[1]:.4f}"),
            ("🔁 İterasyon",     f"{cur.iteration} / {total_steps}"),
            ("📊 Çeşitlilik",    f"{divs[st.session_state.step]:.4f}"),
            ("🎯 Hedef Hata",    f"{abs(cur.best_score - bf.min_value):.2e}"),
        ]
        for col, (lbl, val) in zip(mc, met):
            with col:
                st.markdown(metric_card(lbl, val), unsafe_allow_html=True)

        # Ekstra bilgi (SA sıcaklık, TS tabu listesi)
        if cur.extra:
            parts = []
            if "temperature" in cur.extra:
                parts.append(f"🌡️ Sıcaklık: <b>{cur.extra['temperature']:.4f}</b>")
            if "tabu_size" in cur.extra:
                parts.append(f"🚫 Tabu Listesi: <b>{cur.extra['tabu_size']}</b>")
            if "current_score" in cur.extra:
                parts.append(f"📈 Geçerli Skor: <b>{cur.extra['current_score']:.6f}</b>")
            if parts:
                st.markdown(
                    '<div class="info-strip">' + "&nbsp;·&nbsp;".join(parts) + "</div>",
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Ana grafikler ──
        if show_3d:
            st.markdown("#### 🌐 3D Yüzey Görünümü")
            st.plotly_chart(
                draw_surface_3d(bf, cur, height=520),
                use_container_width=True, config={"displayModeBar": True},
            )
        else:
            gl, gr = st.columns([3, 2])
            with gl:
                st.markdown("#### 🗺️ Arama Uzayı")
                st.plotly_chart(
                    draw_contour_map(
                        bf, cur, result, colorscale,
                        show_traj, show_known, n_levels,
                        st.session_state.step,
                        agent_color=ALGO_COLORS.get(algo_name, "#6366f1"),
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with gr:
                st.markdown("#### 📈 Yakınsama Grafiği")
                st.plotly_chart(
                    draw_convergence(result, st.session_state.step,
                                     bf.min_value,
                                     color=ALGO_COLORS.get(algo_name, "#6366f1")),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        # ── Çeşitlilik grafiği ──
        if show_div:
            st.markdown("#### 🌈 Popülasyon Çeşitliliği (Arama Alanı Yayılımı)")
            st.plotly_chart(
                draw_diversity(result, st.session_state.step,
                               color=ALGO_COLORS.get(algo_name, "#6366f1"),
                               height=200),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Export ──
        st.markdown("---")
        exp_col1, exp_col2 = st.columns([1, 4])
        with exp_col1:
            csv_data = export_csv(result)
            st.download_button(
                "⬇️ CSV İndir", data=csv_data,
                file_name=f"{algo_name}_{func_name}_results.csv",
                mime="text/csv", use_container_width=True,
            )
        with exp_col2:
            st.caption(f"📁 İndirilen dosya: iterasyon, en iyi skor, konum ve çeşitlilik içerir.")

        # ── Detaylar ──
        with st.expander("ℹ️ Sonuç Özeti & Parametre Bilgisi", expanded=False):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown("**Optimizasyon Özeti**")
                st.markdown(f"- Algoritma: `{result.algorithm}`")
                st.markdown(f"- Fonksiyon: `{result.function_name}`")
                st.markdown(f"- Toplam değerlendirme: `{result.n_evaluations:,}`")
                st.markdown(f"- Bulunan min: `{result.best_score:.10f}`")
                st.markdown(f"- Konum: `({result.best_position[0]:.6f}, {result.best_position[1]:.6f})`")
                st.markdown(f"- Hedefe hata: `{abs(result.best_score - bf.min_value):.2e}`")
            with dc2:
                st.markdown("**Kullanılan Parametreler**")
                for k, v in params.items():
                    label = algo_info["params"][k]["label"]
                    st.markdown(f"- {label}: `{v}`")

        # ── Auto-play ──
        if st.session_state.playing:
            if st.session_state.step < total_steps:
                time.sleep(1.0 / play_speed)
                st.session_state.step += 1
                st.rerun()
            else:
                st.session_state.playing = False

# ──────────────────────────────────────────────────────────────
# TAB 2 – KARŞILAŞTIRMA
# ──────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### ⚔️ Algoritma Karşılaştırması")
    st.markdown("Aynı test fonksiyonu üzerinde birden fazla algoritma çalıştırın, sonuçları karşılaştırın.")
    st.markdown("---")

    cmp_c1, cmp_c2 = st.columns([2, 3])
    with cmp_c1:
        st.markdown("**Karşılaştırmak istediğiniz algoritmalar:**")
        cmp_algos = st.multiselect(
            "Algoritmalar",
            list(ALGORITHM_REGISTRY.keys()),
            default=list(ALGORITHM_REGISTRY.keys())[:3],
            label_visibility="collapsed",
            format_func=lambda n: f"{ALGO_GUIDE.get(n,{}).get('icon','●')} {n}",
        )
        cmp_func = st.selectbox("Test Fonksiyonu", list(BENCHMARK_FUNCTIONS.keys()),
                                index=0, key="cmp_func")
        cmp_iter = st.slider("İterasyon Sayısı", 20, 300, 100, step=10, key="cmp_iter")
        cmp_pop  = st.slider("Popülasyon / Ajan", 10, 100, 30, step=5, key="cmp_pop")
        cmp_seed_on = st.checkbox("Sabit seed (adil karşılaştırma)", value=True, key="cmp_seed_on")
        cmp_seed = st.number_input("Seed", 0, 9999, 42, key="cmp_seed_val",
                                   disabled=not cmp_seed_on)

    with cmp_c2:
        run_cmp = st.button("⚔️ Karşılaştırmayı Başlat", type="primary",
                            use_container_width=True, key="run_cmp")

        if run_cmp and len(cmp_algos) >= 2:
            cmp_bf = BENCHMARK_FUNCTIONS[cmp_func]
            cmp_results = {}
            prog = st.progress(0, text="Karşılaştırma başlatılıyor…")
            for idx, aname in enumerate(cmp_algos):
                if cmp_seed_on:
                    np.random.seed(int(cmp_seed))
                a_info = ALGORITHM_REGISTRY[aname]
                # Parametre setini adaptif oluştur
                p = {}
                for k, cfg in a_info["params"].items():
                    if k == "n_iter":
                        p[k] = cmp_iter
                    elif k in ("n_particles", "pop_size", "n_ants",
                               "colony_size", "archive_size"):
                        p[k] = max(cfg["min"],
                                   min(cmp_pop, cfg["max"]))
                    elif cfg["type"] == "float":
                        p[k] = cfg["default"]
                    elif cfg["type"] == "select":
                        p[k] = cfg["default"]
                    else:
                        p[k] = cfg["default"]
                with st.spinner(f"  {aname} çalışıyor…"):
                    cmp_results[aname] = a_info["func"](cmp_bf, **p)
                prog.progress((idx + 1) / len(cmp_algos),
                              text=f"{aname} tamamlandı")
            prog.empty()
            st.session_state.cmp_results = cmp_results
        elif run_cmp and len(cmp_algos) < 2:
            st.warning("En az 2 algoritma seçin.")

    if st.session_state.cmp_results:
        cmp_results = st.session_state.cmp_results
        cmp_bf = BENCHMARK_FUNCTIONS.get(
            st.session_state.get("cmp_func", func_name),
            bf
        )

        st.markdown("---")
        st.markdown("#### 📈 Yakınsama Karşılaştırması")
        st.plotly_chart(
            draw_comparison(cmp_results, cmp_bf.min_value, height=450),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Sonuç tablosu
        st.markdown("#### 📋 Sonuç Tablosu")
        rows = []
        for aname, res in cmp_results.items():
            history = [s.best_score for s in res.states]
            divs_c = get_diversity_history(res.states)
            # Yakınsama hızı: %99 iyileşmeye kaç iterasyon?
            target = history[0] - 0.99 * (history[0] - cmp_bf.min_value)
            conv_iter = next((i for i, h in enumerate(history) if h <= target + 1e-9), len(history))
            rows.append({
                "Algoritma": f"{ALGO_GUIDE.get(aname,{}).get('icon','●')} {aname}",
                "En İyi Skor": f"{res.best_score:.6f}",
                "Hedefe Hata": f"{abs(res.best_score - cmp_bf.min_value):.2e}",
                "Değerlendirme": f"{res.n_evaluations:,}",
                "Yakınsama İter.": conv_iter if conv_iter < len(history) else "—",
                "Son Çeşitlilik": f"{divs_c[-1]:.4f}",
            })
        # Hataya göre sırala
        rows.sort(key=lambda r: float(r["Hedefe Hata"].replace("e","E")))
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Bar chart: final scores
        st.markdown("#### 🏅 Final Skor Karşılaştırması")
        names_c = [r["Algoritma"] for r in rows]
        scores_c = [cmp_results[aname].best_score
                    for aname in cmp_results
                    if f"{ALGO_GUIDE.get(aname,{}).get('icon','●')} {aname}" in names_c]
        colors_c = [ALGO_COLORS.get(aname, "#6366f1") for aname in cmp_results]
        bar_fig = go.Figure(go.Bar(
            x=names_c,
            y=[cmp_results[list(cmp_results.keys())[i]].best_score
               for i in range(len(cmp_results))],
            marker_color=colors_c,
            text=[f"{v:.4f}" for v in
                  [cmp_results[k].best_score for k in cmp_results]],
            textposition="outside",
        ))
        bar_fig.update_layout(
            height=320,
            xaxis_title="Algoritma",
            yaxis_title="En İyi Skor",
            **_plotly_theme(),
        )
        st.plotly_chart(bar_fig, use_container_width=True,
                        config={"displayModeBar": False})

# ──────────────────────────────────────────────────────────────
# TAB 3 – ANALİZ
# ──────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### 📊 İstatistiksel Analiz")
    st.markdown("Aynı algoritmayı birden fazla kez çalıştırarak istatistiksel güvenilirlik ölçün.")
    st.markdown("---")

    an_c1, an_c2 = st.columns([2, 3])
    with an_c1:
        an_algo = st.selectbox(
            "Algoritma", list(ALGORITHM_REGISTRY.keys()),
            format_func=lambda n: f"{ALGO_GUIDE.get(n,{}).get('icon','●')} {n}",
            key="an_algo",
        )
        an_func = st.selectbox("Fonksiyon", list(BENCHMARK_FUNCTIONS.keys()),
                               key="an_func")
        an_runs = st.slider("Koşu Sayısı", 3, 30, 10, step=1, key="an_runs")
        an_iter = st.slider("İterasyon", 20, 300, 100, step=10, key="an_iter")
        an_pop  = st.slider("Popülasyon", 10, 100, 30, step=5, key="an_pop")

    with an_c2:
        run_an = st.button("📊 Analizi Başlat", type="primary",
                           use_container_width=True, key="run_an")
        if run_an:
            an_bf_obj = BENCHMARK_FUNCTIONS[an_func]
            a_info = ALGORITHM_REGISTRY[an_algo]
            p = {}
            for k, cfg in a_info["params"].items():
                if k == "n_iter":
                    p[k] = an_iter
                elif k in ("n_particles","pop_size","n_ants","colony_size","archive_size"):
                    p[k] = max(cfg["min"], min(an_pop, cfg["max"]))
                elif cfg["type"] == "float":
                    p[k] = cfg["default"]
                elif cfg["type"] == "select":
                    p[k] = cfg["default"]
                else:
                    p[k] = cfg["default"]

            all_histories = []
            final_scores = []
            prog2 = st.progress(0, text="Analiz başlatılıyor…")
            for run_i in range(an_runs):
                r = a_info["func"](an_bf_obj, **p)
                all_histories.append([s.best_score for s in r.states])
                final_scores.append(r.best_score)
                prog2.progress((run_i + 1) / an_runs,
                               text=f"Koşu {run_i+1}/{an_runs}")
            prog2.empty()
            st.session_state["an_histories"] = all_histories
            st.session_state["an_scores"]    = final_scores
            st.session_state["an_func_obj"]  = an_bf_obj
            st.session_state["an_algo_name"] = an_algo
            st.session_state["analysis_done"] = True

    if st.session_state.get("analysis_done"):
        all_histories = st.session_state["an_histories"]
        final_scores  = st.session_state["an_scores"]
        an_bf_obj     = st.session_state["an_func_obj"]
        an_algo_disp  = st.session_state["an_algo_name"]
        color_an = ALGO_COLORS.get(an_algo_disp, "#6366f1")

        st.markdown("---")

        # Özet metrikler
        sm = st.columns(5)
        stats_vals = [
            ("En İyi",     f"{min(final_scores):.6f}"),
            ("En Kötü",    f"{max(final_scores):.6f}"),
            ("Ortalama",   f"{np.mean(final_scores):.6f}"),
            ("Std Sapma",  f"{np.std(final_scores):.6f}"),
            ("Başarı Oranı", f"{sum(1 for s in final_scores if abs(s - an_bf_obj.min_value) < 0.1) / len(final_scores) * 100:.0f}%"),
        ]
        for col, (lbl, val) in zip(sm, stats_vals):
            with col:
                st.markdown(metric_card(lbl, val), unsafe_allow_html=True)

        st.markdown("---")
        left_a, right_a = st.columns(2)

        with left_a:
            st.markdown("#### 📉 Yakınsama Yelpazesi")
            fan_fig = go.Figure()
            arr = np.array(all_histories)
            iters_a = list(range(arr.shape[1]))
            med = np.median(arr, axis=0)
            q25 = np.percentile(arr, 25, axis=0)
            q75 = np.percentile(arr, 75, axis=0)
            mn  = np.min(arr, axis=0)
            mx  = np.max(arr, axis=0)

            # Min-max bant
            fan_fig.add_trace(go.Scatter(
                x=iters_a + iters_a[::-1],
                y=list(mx) + list(mn[::-1]),
                fill="toself",
                fillcolor=f"rgba{tuple(int(color_an.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.07,)}",
                line=dict(color="rgba(0,0,0,0)"),
                name="Min-Max",
            ))
            # IQR bant
            fan_fig.add_trace(go.Scatter(
                x=iters_a + iters_a[::-1],
                y=list(q75) + list(q25[::-1]),
                fill="toself",
                fillcolor=f"rgba{tuple(int(color_an.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.2,)}",
                line=dict(color="rgba(0,0,0,0)"),
                name="IQR",
            ))
            # Medyan
            fan_fig.add_trace(go.Scatter(
                x=iters_a, y=med,
                line=dict(color=color_an, width=2.5),
                name="Medyan",
            ))
            # Hedef
            fan_fig.add_hline(y=an_bf_obj.min_value, line_dash="dash",
                              line_color="#ef4444", opacity=0.6,
                              annotation_text=f" Min={an_bf_obj.min_value:.4f}")
            fan_fig.update_layout(
                height=350,
                xaxis=dict(title="İterasyon", showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(title="En İyi Skor", showgrid=True, gridcolor="#f1f5f9"),
                **_plotly_theme(),
            )
            st.plotly_chart(fan_fig, use_container_width=True,
                            config={"displayModeBar": False})

        with right_a:
            st.markdown("#### 📦 Final Skor Dağılımı (Box Plot)")
            box_fig = go.Figure()
            box_fig.add_trace(go.Box(
                y=final_scores,
                name=an_algo_disp.split("(")[0].strip(),
                marker_color=color_an,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                line_width=2,
            ))
            box_fig.add_hline(y=an_bf_obj.min_value, line_dash="dash",
                              line_color="#ef4444", opacity=0.6)
            box_fig.update_layout(
                height=350,
                yaxis=dict(title="Final Skor", showgrid=True, gridcolor="#f1f5f9"),
                showlegend=False,
                **_plotly_theme(),
            )
            st.plotly_chart(box_fig, use_container_width=True,
                            config={"displayModeBar": False})

        # Histogram
        st.markdown("#### 📊 Final Skor Histogramı")
        hist_fig = px.histogram(
            x=final_scores, nbins=min(15, len(final_scores)),
            color_discrete_sequence=[color_an],
            labels={"x": "Final Skor", "count": "Frekans"},
        )
        hist_fig.add_vline(x=np.mean(final_scores), line_dash="dash",
                           line_color="#f59e0b",
                           annotation_text="Ort.")
        hist_fig.add_vline(x=an_bf_obj.min_value, line_dash="dot",
                           line_color="#ef4444",
                           annotation_text="Hedef")
        hist_fig.update_layout(height=250, **_plotly_theme())
        st.plotly_chart(hist_fig, use_container_width=True,
                        config={"displayModeBar": False})

# ──────────────────────────────────────────────────────────────
# TAB 4 – ALGORİTMA REHBERİ
# ──────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### 📚 Algoritma Rehberi")
    st.markdown("Her algoritmanın teorik temeli, adımları, avantaj ve dezavantajları.")
    st.markdown("---")

    cat_filter = st.multiselect(
        "Kategoriye göre filtrele",
        sorted(set(g["category"] for g in ALGO_GUIDE.values())),
        default=[],
        placeholder="Tümü göster",
    )

    for aname, g in ALGO_GUIDE.items():
        if cat_filter and g["category"] not in cat_filter:
            continue

        color = g["color"]
        with st.expander(
            f'{g["icon"]}  {aname}  —  {g["category"]}', expanded=False
        ):
            g1, g2 = st.columns([3, 2])
            with g1:
                st.markdown(f"**{g['description']}**")
                st.markdown("")

                # Adımlar
                st.markdown("**🔢 Algoritma Adımları:**")
                for i, step in enumerate(g["steps"], 1):
                    st.markdown(f"&nbsp;&nbsp;**{i}.** {step}")
                st.markdown("")

                # Formül
                if g.get("formula"):
                    st.markdown("**📐 Temel Formül:**")
                    st.code(g["formula"], language=None)

            with g2:
                st.markdown("**📅 Yayın Bilgisi:**")
                st.markdown(
                    f'<span class="badge badge-indigo">📅 {g["year"]}</span>'
                    f'<span class="badge badge-violet">👤 {g["authors"]}</span>'
                    f'<span class="badge badge-sky">🏷️ {g["category"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                st.markdown(f"**💡 İlham Kaynağı:** _{g['inspired']}_")
                st.markdown("")

                st.markdown("**✅ Avantajlar:**")
                for pro in g["pros"]:
                    st.markdown(
                        f'<span class="badge badge-green">+ {pro}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown("")
                st.markdown("**❌ Dezavantajlar:**")
                for con in g["cons"]:
                    st.markdown(
                        f'<span class="badge badge-rose">− {con}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown("")
                st.markdown("**🎯 En iyi kullanım alanları:**")
                for b in g["best_for"]:
                    st.markdown(
                        f'<span class="badge badge-amber">→ {b}</span>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    st.markdown("#### ⚡ Hızlı Karşılaştırma Tablosu")
    comp_rows = []
    for aname, g in ALGO_GUIDE.items():
        comp_rows.append({
            "İkon": g["icon"],
            "Algoritma": aname.split("(")[0].strip(),
            "Kategori": g["category"],
            "Yıl": str(g["year"]),
            "Parametre Sayısı": len(ALGORITHM_REGISTRY.get(aname, {}).get("params", {})),
            "İlham": g["inspired"][:40] + "…",
        })
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
# TAB 5 – FONKSİYON GALERİSİ
# ──────────────────────────────────────────────────────────────

with tab5:
    st.markdown("### 🗂️ Benchmark Fonksiyonu Galerisi")
    st.markdown("10 test fonksiyonunun 2D kontur görünümleri, minimum noktaları ve özellikleri.")
    st.markdown("---")

    gal_n = st.slider("Çözünürlük (grid)", 60, 180, 100, step=20,
                      key="gal_n", help="Yüksek değer = daha keskin ama daha yavaş")

    func_names = list(BENCHMARK_FUNCTIONS.keys())
    for row_start in range(0, len(func_names), 2):
        cols = st.columns(2)
        for ci, fname in enumerate(func_names[row_start:row_start + 2]):
            bf_g = BENCHMARK_FUNCTIONS[fname]
            with cols[ci]:
                lb_g, ub_g = bf_g.bounds
                x_g = np.linspace(lb_g, ub_g, gal_n)
                y_g = np.linspace(lb_g, ub_g, gal_n)
                X_g, Y_g = np.meshgrid(x_g, y_g)
                Z_g = np.vectorize(
                    lambda xi, yi: bf_g.func(np.array([xi, yi]))
                )(X_g, Y_g)
                Z_plot_g = np.log1p(Z_g - Z_g.min())

                fig_g = go.Figure()
                fig_g.add_trace(go.Contour(
                    x=x_g, y=y_g, z=Z_plot_g,
                    ncontours=25, colorscale="Viridis",
                    showscale=False,
                    contours=dict(coloring="heatmap", showlines=True),
                ))
                for km in bf_g.global_minima:
                    fig_g.add_trace(go.Scatter(
                        x=[km[0]], y=[km[1]], mode="markers",
                        marker=dict(size=12, color="red", symbol="star",
                                    line=dict(width=1.5, color="white")),
                        name="Min", showlegend=False,
                        hovertemplate=f"Min: ({km[0]:.3f},{km[1]:.3f})<extra></extra>",
                    ))
                fig_g.update_layout(
                    height=280,
                    title=dict(text=f"<b>{fname}</b>", x=0.5,
                               font=dict(size=14, color="#1e293b")),
                    xaxis=dict(title="x", showgrid=False),
                    yaxis=dict(title="y", showgrid=False, scaleanchor="x", scaleratio=1),
                    plot_bgcolor="#0a0a12", paper_bgcolor="white",
                    margin=dict(l=5, r=5, t=40, b=5),
                )
                st.plotly_chart(fig_g, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#64748b;text-align:center;'
                    f'margin-top:-0.5rem;margin-bottom:1rem">'
                    f'Min={bf_g.min_value:.4f} &nbsp;|&nbsp; '
                    f'Alan:[{lb_g},{ub_g}] &nbsp;|&nbsp; '
                    f'{len(bf_g.global_minima)} minimum &nbsp;|&nbsp; '
                    f'{bf_g.description[:60]}…</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    Meta-Sezgisel Optimizasyon Lab &nbsp;•&nbsp;
    PSO &nbsp;·&nbsp; GA &nbsp;·&nbsp; SA &nbsp;·&nbsp; TS &nbsp;·&nbsp;
    ACOR &nbsp;·&nbsp; CLONALG &nbsp;·&nbsp; DE &nbsp;·&nbsp; ABC
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Görselleştirici · Karşılaştırma · Analiz · Rehber · Galeri
</div>
""", unsafe_allow_html=True)
