"""
optimizer.py – Benchmark fonksiyonları ve tüm meta-sezgisel algoritmalar.
Her algoritma, adım adım görselleştirme için OptimizationState listesi döndürür.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable
import copy

# ─────────────────────────────────────────────────────────────
# BENCHMARK FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

def _rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def _ackley(x: np.ndarray) -> float:
    d = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
        + np.e + 20
    )

def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def _himmelblau(x: np.ndarray) -> float:
    return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)

def _beale(x: np.ndarray) -> float:
    return float(
        (1.5   - x[0] + x[0]*x[1]   )**2
      + (2.25  - x[0] + x[0]*x[1]**2)**2
      + (2.625 - x[0] + x[0]*x[1]**3)**2
    )

def _booth(x: np.ndarray) -> float:
    return float((x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2)

def _eggholder(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return float(
        -(x2 + 47) * np.sin(np.sqrt(abs(x1/2 + (x2 + 47))))
        - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))
    )

def _levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    return float(term1 + term2 + term3)

def _griewank(x: np.ndarray) -> float:
    i = np.arange(1, len(x)+1)
    return float(np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(i))) + 1)


@dataclass
class BenchmarkFunction:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: Tuple[float, float]
    global_minima: List[Tuple[float, float]]
    min_value: float
    description: str = ""

BENCHMARK_FUNCTIONS: Dict[str, BenchmarkFunction] = {
    "Sphere": BenchmarkFunction(
        "Sphere", _sphere, (-5.12, 5.12), [(0.0, 0.0)], 0.0,
        "f(x,y) = x² + y²  |  Tek küresel minimum, basit kase şekli"
    ),
    "Rastrigin": BenchmarkFunction(
        "Rastrigin", _rastrigin, (-5.12, 5.12), [(0.0, 0.0)], 0.0,
        "Çok sayıda yerel minimum → algoritmalar için zorlu"
    ),
    "Ackley": BenchmarkFunction(
        "Ackley", _ackley, (-5.0, 5.0), [(0.0, 0.0)], 0.0,
        "Düz yüzey + derin küresel minimum; yerel minimumlara tuzak kurabilir"
    ),
    "Rosenbrock": BenchmarkFunction(
        "Rosenbrock", _rosenbrock, (-2.0, 2.0), [(1.0, 1.0)], 0.0,
        "Muz şekilli vadi; minimum bulmak kolay, vadi boyunca ilerlemek zor"
    ),
    "Himmelblau": BenchmarkFunction(
        "Himmelblau", _himmelblau, (-5.0, 5.0),
        [(3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)], 0.0,
        "4 eşdeğer global minimum; simetrik ve çok modlu"
    ),
    "Beale": BenchmarkFunction(
        "Beale", _beale, (-4.5, 4.5), [(3.0, 0.5)], 0.0,
        "Keskin köşeler ve derin vadiler"
    ),
    "Booth": BenchmarkFunction(
        "Booth", _booth, (-10.0, 10.0), [(1.0, 3.0)], 0.0,
        "Pürüzsüz, uzun eliptik vadi"
    ),
    "Eggholder": BenchmarkFunction(
        "Eggholder", _eggholder, (-512.0, 512.0), [(512.0, 404.2319)], -959.6407,
        "Çok dalgalı, zorlu; geniş arama alanı gerektirir"
    ),
    "Levy": BenchmarkFunction(
        "Levy", _levy, (-10.0, 10.0), [(1.0, 1.0)], 0.0,
        "Düzensiz dalgalı yüzey"
    ),
    "Griewank": BenchmarkFunction(
        "Griewank", _griewank, (-600.0, 600.0), [(0.0, 0.0)], 0.0,
        "Geniş dalgalı; sinüs ve karesel terimler kombinasyonu"
    ),
}

# ─────────────────────────────────────────────────────────────
# SONUÇ VERİ YAPISI
# ─────────────────────────────────────────────────────────────

@dataclass
class OptimizationState:
    """Tek bir iterasyondaki durum anlık görüntüsü."""
    iteration: int
    positions: np.ndarray        # (n, 2) — geçerli ajan konumları
    best_position: np.ndarray    # (2,)
    best_score: float
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    algorithm: str
    function_name: str
    states: List[OptimizationState]
    best_position: np.ndarray
    best_score: float
    n_evaluations: int

# ─────────────────────────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────────────────────────

def _clip(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    return np.clip(x, lb, ub)

def _eval(func: Callable, pop: np.ndarray) -> np.ndarray:
    return np.array([func(ind) for ind in pop])

def _random_pop(n: int, lb: float, ub: float) -> np.ndarray:
    return np.random.uniform(lb, ub, (n, 2))

def compute_diversity(positions: np.ndarray) -> float:
    """Popülasyon çeşitliliği: pozisyonların ortalama standart sapması."""
    pos = positions if positions.ndim == 2 else positions.reshape(1, 2)
    if len(pos) < 2:
        return 0.0
    return float(np.mean(np.std(pos, axis=0)))

def get_diversity_history(states: List[OptimizationState]) -> List[float]:
    """Tüm state'ler için çeşitlilik geçmişi döndür."""
    return [compute_diversity(s.positions) for s in states]

# ─────────────────────────────────────────────────────────────
# 1. PARÇACİK SÜRÜ OPTİMİZASYONU (PSO)
# ─────────────────────────────────────────────────────────────

def run_pso(
    bf: BenchmarkFunction,
    n_particles: int = 30,
    n_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func
    rng = np.random.default_rng()

    pos = _random_pop(n_particles, lb, ub)
    vel = rng.uniform(-(ub-lb)*0.1, (ub-lb)*0.1, (n_particles, 2))
    scores = _eval(func, pos)

    pbest = pos.copy()
    pbest_scores = scores.copy()
    g_idx = int(np.argmin(pbest_scores))
    gbest = pbest[g_idx].copy()
    gbest_score = float(pbest_scores[g_idx])

    states = [OptimizationState(0, pos.copy(), gbest.copy(), gbest_score)]
    n_evals = n_particles

    for i in range(1, n_iter + 1):
        r1 = rng.random((n_particles, 2))
        r2 = rng.random((n_particles, 2))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -(ub-lb), ub-lb)
        pos = _clip(pos + vel, lb, ub)

        scores = _eval(func, pos)
        n_evals += n_particles

        mask = scores < pbest_scores
        pbest[mask] = pos[mask]
        pbest_scores[mask] = scores[mask]

        g_idx = int(np.argmin(pbest_scores))
        if pbest_scores[g_idx] < gbest_score:
            gbest = pbest[g_idx].copy()
            gbest_score = float(pbest_scores[g_idx])

        states.append(OptimizationState(
            i, pos.copy(), gbest.copy(), gbest_score,
            {"velocities": vel.copy()}
        ))

    return OptimizationResult("Parçacık Sürü (PSO)", bf.name, states, gbest, gbest_score, n_evals)

# ─────────────────────────────────────────────────────────────
# 2. GENETİK ALGORİTMA (GA)
# ─────────────────────────────────────────────────────────────

def run_genetic(
    bf: BenchmarkFunction,
    pop_size: int = 50,
    n_iter: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.05,
    tournament_k: int = 3,
    eta_c: float = 15.0,    # SBX dağılım indeksi
    eta_m: float = 20.0,    # polinomial mutasyon indeksi
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func

    pop = _random_pop(pop_size, lb, ub)
    scores = _eval(func, pop)
    best_idx = int(np.argmin(scores))
    best = pop[best_idx].copy()
    best_score = float(scores[best_idx])

    states = [OptimizationState(0, pop.copy(), best.copy(), best_score)]
    n_evals = pop_size

    def tournament_select(pop, scores, k):
        idx = np.random.choice(len(pop), k, replace=False)
        return pop[idx[np.argmin(scores[idx])]].copy()

    def sbx_crossover(p1, p2, eta):
        child1, child2 = p1.copy(), p2.copy()
        for i in range(2):
            if np.random.random() < 0.5:
                if abs(p2[i] - p1[i]) > 1e-10:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2*u)**(1/(eta+1))
                    else:
                        beta = (1/(2*(1-u)))**(1/(eta+1))
                    child1[i] = 0.5*((1+beta)*p1[i] + (1-beta)*p2[i])
                    child2[i] = 0.5*((1-beta)*p1[i] + (1+beta)*p2[i])
        return child1, child2

    def poly_mutate(x, eta, lb, ub):
        x = x.copy()
        for i in range(2):
            if np.random.random() < mutation_rate:
                u = np.random.random()
                delta = ub - lb
                if u < 0.5:
                    delta_q = (2*u)**(1/(eta+1)) - 1
                else:
                    delta_q = 1 - (2*(1-u))**(1/(eta+1))
                x[i] += delta_q * delta
        return np.clip(x, lb, ub)

    for i in range(1, n_iter + 1):
        new_pop = []
        # Elitizm: en iyi bireyi koru
        new_pop.append(pop[np.argmin(scores)].copy())

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, scores, tournament_k)
            p2 = tournament_select(pop, scores, tournament_k)
            if np.random.random() < crossover_rate:
                c1, c2 = sbx_crossover(p1, p2, eta_c)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = poly_mutate(c1, eta_m, lb, ub)
            c2 = poly_mutate(c2, eta_m, lb, ub)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop[:pop_size])
        scores = _eval(func, pop)
        n_evals += pop_size

        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best = pop[best_idx].copy()
            best_score = float(scores[best_idx])

        states.append(OptimizationState(i, pop.copy(), best.copy(), best_score))

    return OptimizationResult("Genetik Algoritma (GA)", bf.name, states, best, best_score, n_evals)

# ─────────────────────────────────────────────────────────────
# 3. ISIL İŞLEM (Simulated Annealing)
# ─────────────────────────────────────────────────────────────

def run_simulated_annealing(
    bf: BenchmarkFunction,
    n_iter: int = 500,
    T_init: float = 100.0,
    T_min: float = 0.01,
    cooling: float = 0.97,
    step_size: float = 0.5,
    n_restarts: int = 5,
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func
    range_ = ub - lb

    current = np.random.uniform(lb, ub, 2)
    current_score = func(current)
    best = current.copy()
    best_score = current_score

    states = [OptimizationState(0, current.reshape(1, 2).copy(), best.copy(), best_score,
                                 {"temperature": T_init})]
    n_evals = 1
    T = T_init
    restart_interval = max(1, n_iter // (n_restarts + 1))
    adaptive_step = step_size

    for i in range(1, n_iter + 1):
        T = max(T * cooling, T_min)

        # Adaptif adım: sıcaklıkla orantılı
        adaptive_step = step_size * (T / T_init) * 0.7 + step_size * 0.3

        # Gauss komşu üret (Uniform yerine — daha iyi yerel arama)
        neighbor = current + np.random.normal(0, adaptive_step * range_ * 0.5, 2)
        neighbor = np.clip(neighbor, lb, ub)
        neighbor_score = func(neighbor)
        n_evals += 1

        delta = neighbor_score - current_score
        accept_prob = 1.0 if delta < 0 else np.exp(-delta / max(T, 1e-300))
        if np.random.random() < accept_prob:
            current = neighbor
            current_score = neighbor_score

        if current_score < best_score:
            best = current.copy()
            best_score = current_score

        # Yeniden başlatma: en iyi pozisyondan başla (random değil)
        if i % restart_interval == 0 and i < n_iter * 0.9:
            current = best.copy() + np.random.normal(0, range_ * 0.1, 2)
            current = np.clip(current, lb, ub)
            current_score = func(current)
            T = T_init * (0.5 ** (i / restart_interval))
            T = max(T, T_min * 10)
            n_evals += 1

        states.append(OptimizationState(
            i,
            np.vstack([current, best]),
            best.copy(),
            best_score,
            {"temperature": T, "current_score": current_score}
        ))

    return OptimizationResult("Isıl İşlem (SA)", bf.name, states, best, best_score, n_evals)

# ─────────────────────────────────────────────────────────────
# 4. TABU ARAŞTIRMA (Tabu Search)
# ─────────────────────────────────────────────────────────────

def run_tabu_search(
    bf: BenchmarkFunction,
    n_iter: int = 200,
    tabu_tenure: int = 15,
    n_neighbors: int = 20,
    step_size: float = 0.3,
    diversification_interval: int = 50,
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func
    range_ = ub - lb

    current = np.random.uniform(lb, ub, 2)
    current_score = func(current)
    best = current.copy()
    best_score = current_score

    tabu_list: List[np.ndarray] = []
    states = [OptimizationState(0, current.reshape(1, 2).copy(), best.copy(), best_score,
                                 {"tabu_size": 0})]
    n_evals = 1
    no_improve = 0

    for i in range(1, n_iter + 1):
        neighbors = []
        for _ in range(n_neighbors):
            nb = current + np.random.uniform(-step_size * range_, step_size * range_, 2)
            nb = np.clip(nb, lb, ub)
            neighbors.append(nb)

        neighbors.sort(key=lambda nb: func(nb))
        n_evals += n_neighbors

        # Tabu olmayan en iyi komşuyu seç (aspiration kriteriyle)
        chosen = None
        chosen_score = np.inf
        for nb in neighbors:
            nb_score = func(nb)
            n_evals += 1
            is_tabu = any(np.linalg.norm(nb - t) < step_size * range_ * 0.1 for t in tabu_list)
            if not is_tabu or nb_score < best_score:  # aspiration
                if nb_score < chosen_score:
                    chosen = nb
                    chosen_score = nb_score

        if chosen is None:
            chosen = neighbors[0]
            chosen_score = func(chosen)
            n_evals += 1

        current = chosen
        current_score = chosen_score
        tabu_list.append(current.copy())
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if current_score < best_score:
            best = current.copy()
            best_score = current_score
            no_improve = 0
        else:
            no_improve += 1

        # Çeşitlendirme: belirli adımda takılırsa uzak noktaya sıçra
        if no_improve >= diversification_interval:
            current = np.random.uniform(lb, ub, 2)
            current_score = func(current)
            n_evals += 1
            tabu_list.clear()
            no_improve = 0

        tabu_positions = np.array(tabu_list) if tabu_list else current.reshape(1, 2)
        states.append(OptimizationState(
            i,
            np.vstack([current, tabu_positions[-min(5, len(tabu_list)):]]),
            best.copy(),
            best_score,
            {"tabu_size": len(tabu_list), "current_score": current_score}
        ))

    return OptimizationResult("Tabu Araştırma (TS)", bf.name, states, best, best_score, n_evals)

# ─────────────────────────────────────────────────────────────
# 5. KARINCA KOLONİSİ OPTİMİZASYONU – sürekli (ACOR)
# ─────────────────────────────────────────────────────────────

def run_ant_colony(
    bf: BenchmarkFunction,
    n_ants: int = 30,
    n_iter: int = 100,
    archive_size: int = 10,
    q: float = 0.5,          # seçim baskısı (küçük = elitist)
    xi: float = 0.85,        # hız (yerel arama aralığı azaltma)
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func

    # Arşivi başlat
    archive = _random_pop(archive_size, lb, ub)
    archive_scores = _eval(func, archive)
    sort_idx = np.argsort(archive_scores)
    archive = archive[sort_idx]
    archive_scores = archive_scores[sort_idx]

    best = archive[0].copy()
    best_score = float(archive_scores[0])

    states = [OptimizationState(0, archive.copy(), best.copy(), best_score)]
    n_evals = archive_size

    # Gauss ağırlıkları (sıra tabanlı)
    k = archive_size
    ranks = np.arange(1, k + 1)
    weights = np.exp(-((ranks - 1)**2) / (2 * q**2 * k**2))
    weights /= weights.sum()

    for i in range(1, n_iter + 1):
        new_solutions = []
        for _ in range(n_ants):
            # Arşivden bir çözüm seç (ağırlıklı olasılık)
            l = np.random.choice(k, p=weights)
            sol_l = archive[l]

            # Seçilen çözüm etrafında Gauss'tan örnekle
            # Standart sapma: mesafe ortalaması * xi
            dists = np.mean([np.abs(archive[j] - sol_l) for j in range(k)], axis=0)
            sigma = xi * dists + 1e-6

            new_sol = sol_l + np.random.normal(0, sigma, 2)
            new_sol = np.clip(new_sol, lb, ub)
            new_solutions.append(new_sol)

        new_solutions = np.array(new_solutions)
        new_scores = _eval(func, new_solutions)
        n_evals += n_ants

        # Arşivi güncelle: tüm adayları ekle, en iyileri tut
        combined = np.vstack([archive, new_solutions])
        combined_scores = np.concatenate([archive_scores, new_scores])
        sort_idx = np.argsort(combined_scores)[:k]
        archive = combined[sort_idx]
        archive_scores = combined_scores[sort_idx]

        if archive_scores[0] < best_score:
            best = archive[0].copy()
            best_score = float(archive_scores[0])

        states.append(OptimizationState(
            i, new_solutions.copy(), best.copy(), best_score,
            {"archive": archive.copy()}
        ))

    return OptimizationResult("Karınca Koloni (ACOR)", bf.name, states, best, best_score, n_evals)

# ─────────────────────────────────────────────────────────────
# 6. YAPAY BAĞIŞIKLIK (Clonal Selection – CLONALG)
# ─────────────────────────────────────────────────────────────

def run_artificial_immune(
    bf: BenchmarkFunction,
    pop_size: int = 30,
    n_iter: int = 100,
    n_clones_factor: int = 5,
    mutation_decay: float = 2.5,
    n_replace: int = 5,
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func
    range_ = ub - lb

    pop = _random_pop(pop_size, lb, ub)
    scores = _eval(func, pop)
    sort_idx = np.argsort(scores)
    pop = pop[sort_idx]
    scores = scores[sort_idx]

    best = pop[0].copy()
    best_score = float(scores[0])

    states = [OptimizationState(0, pop.copy(), best.copy(), best_score)]
    n_evals = pop_size

    for i in range(1, n_iter + 1):
        clones_all = []
        clones_scores_all = []

        for j, ab in enumerate(pop):
            n_clones = int(np.ceil(n_clones_factor * pop_size / (j + 1)))
            # Mutasyon oranı: daha iyi antikorlar daha az mutasyona uğrar
            norm_score = (scores[j] - scores[0]) / (scores[-1] - scores[0] + 1e-12)
            sigma = np.exp(-mutation_decay * (1 - norm_score)) * range_ * 0.5

            clones = ab + np.random.normal(0, sigma, (n_clones, 2))
            clones = np.clip(clones, lb, ub)
            c_scores = _eval(func, clones)
            n_evals += n_clones

            # Her antikordaki en iyi klonu sakla
            best_clone_idx = np.argmin(c_scores)
            clones_all.append(clones[best_clone_idx])
            clones_scores_all.append(c_scores[best_clone_idx])

        clones_all = np.array(clones_all)
        clones_scores_all = np.array(clones_scores_all)

        # Popülasyonu güncelle
        combined = np.vstack([pop, clones_all])
        combined_scores = np.concatenate([scores, clones_scores_all])
        sort_idx = np.argsort(combined_scores)[:pop_size]
        pop = combined[sort_idx]
        scores = combined_scores[sort_idx]

        # En kötü bireyleri rastgele yeni bireylerle değiştir (çeşitlilik)
        new_random = _random_pop(n_replace, lb, ub)
        new_scores = _eval(func, new_random)
        n_evals += n_replace
        pop[-n_replace:] = new_random
        scores[-n_replace:] = new_scores
        sort_idx = np.argsort(scores)
        pop = pop[sort_idx]
        scores = scores[sort_idx]

        if scores[0] < best_score:
            best = pop[0].copy()
            best_score = float(scores[0])

        states.append(OptimizationState(
            i,
            np.vstack([clones_all, pop[:5]]),
            best.copy(),
            best_score,
            {"pop": pop.copy()}
        ))

    return OptimizationResult("Yapay Bağışıklık (CLONALG)", bf.name, states, best, best_score, n_evals)

# ─────────────────────────────────────────────────────────────
# 7. DİFERANSİYEL GELİŞİM (Differential Evolution)
# ─────────────────────────────────────────────────────────────

def run_differential_evolution(
    bf: BenchmarkFunction,
    pop_size: int = 30,
    n_iter: int = 100,
    F: float = 0.8,
    CR: float = 0.9,
    strategy: str = "DE/rand/1/bin",
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func

    pop = _random_pop(pop_size, lb, ub)
    scores = _eval(func, pop)
    best_idx = int(np.argmin(scores))
    best = pop[best_idx].copy()
    best_score = float(scores[best_idx])

    states = [OptimizationState(0, pop.copy(), best.copy(), best_score)]
    n_evals = pop_size

    for i in range(1, n_iter + 1):
        new_pop = pop.copy()

        for j in range(pop_size):
            idxs = [k for k in range(pop_size) if k != j]

            if strategy == "DE/rand/1/bin":
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
            elif strategy == "DE/best/1/bin":
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                mutant = best + F * (pop[r1] - pop[r2])
            elif strategy == "DE/rand/2/bin":
                r1, r2, r3, r4, r5 = np.random.choice(idxs, 5, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])
            elif strategy == "DE/current-to-best/1/bin":
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                mutant = pop[j] + F * (best - pop[j]) + F * (pop[r1] - pop[r2])
            else:
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

            mutant = np.clip(mutant, lb, ub)

            # Çaprazlama (binomial)
            cross_points = np.random.random(2) < CR
            if not cross_points.any():
                cross_points[np.random.randint(0, 2)] = True
            trial = np.where(cross_points, mutant, pop[j])

            trial_score = func(trial)
            n_evals += 1

            if trial_score <= scores[j]:
                new_pop[j] = trial
                scores[j] = trial_score

        pop = new_pop
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best = pop[best_idx].copy()
            best_score = float(scores[best_idx])

        states.append(OptimizationState(i, pop.copy(), best.copy(), best_score))

    return OptimizationResult("Diferansiyel Gelişim (DE)", bf.name, states, best, best_score, n_evals)

# ─────────────────────────────────────────────────────────────
# 8. YAPAY ARI KOLONİSİ (Artificial Bee Colony – ABC)
# ─────────────────────────────────────────────────────────────

def run_abc(
    bf: BenchmarkFunction,
    colony_size: int = 40,
    n_iter: int = 100,
    limit_factor: int = 5,
) -> OptimizationResult:
    lb, ub = bf.bounds
    func = bf.func

    n_employed = colony_size // 2
    n_onlooker = colony_size - n_employed
    limit = limit_factor * n_employed * 2

    # Besin kaynakları
    sources = _random_pop(n_employed, lb, ub)
    source_scores = _eval(func, sources)
    trial_counts = np.zeros(n_employed, dtype=int)

    best_idx = int(np.argmin(source_scores))
    best = sources[best_idx].copy()
    best_score = float(source_scores[best_idx])

    states = [OptimizationState(0, sources.copy(), best.copy(), best_score)]
    n_evals = n_employed

    def fitness(score):
        return 1 / (1 + score) if score >= 0 else 1 + abs(score)

    for i in range(1, n_iter + 1):
        # — İşçi arılar: besin kaynaklarını sömür —
        for j in range(n_employed):
            k = np.random.choice([m for m in range(n_employed) if m != j])
            phi = np.random.uniform(-1, 1, 2)
            d = np.random.randint(0, 2)  # rastgele boyut
            candidate = sources[j].copy()
            candidate[d] = sources[j][d] + phi[d] * (sources[j][d] - sources[k][d])
            candidate = np.clip(candidate, lb, ub)
            c_score = func(candidate)
            n_evals += 1

            if c_score < source_scores[j]:
                sources[j] = candidate
                source_scores[j] = c_score
                trial_counts[j] = 0
            else:
                trial_counts[j] += 1

        # — İzci arılar: olasılıkla kaynak seç —
        fit_vals = np.array([fitness(s) for s in source_scores])
        probs = fit_vals / fit_vals.sum()

        for _ in range(n_onlooker):
            j = np.random.choice(n_employed, p=probs)
            k = np.random.choice([m for m in range(n_employed) if m != j])
            phi = np.random.uniform(-1, 1, 2)
            d = np.random.randint(0, 2)
            candidate = sources[j].copy()
            candidate[d] = sources[j][d] + phi[d] * (sources[j][d] - sources[k][d])
            candidate = np.clip(candidate, lb, ub)
            c_score = func(candidate)
            n_evals += 1

            if c_score < source_scores[j]:
                sources[j] = candidate
                source_scores[j] = c_score
                trial_counts[j] = 0
            else:
                trial_counts[j] += 1

        # — Kaşif arılar: tükenmiş kaynakları yenile —
        for j in range(n_employed):
            if trial_counts[j] > limit:
                sources[j] = np.random.uniform(lb, ub, 2)
                source_scores[j] = func(sources[j])
                n_evals += 1
                trial_counts[j] = 0

        best_idx = int(np.argmin(source_scores))
        if source_scores[best_idx] < best_score:
            best = sources[best_idx].copy()
            best_score = float(source_scores[best_idx])

        states.append(OptimizationState(
            i, sources.copy(), best.copy(), best_score,
            {"trials": trial_counts.copy()}
        ))

    return OptimizationResult("Yapay Arı Koloni (ABC)", bf.name, states, best, best_score, n_evals)


# ─────────────────────────────────────────────────────────────
# ALGORİTMA KATALOĞu
# ─────────────────────────────────────────────────────────────

ALGORITHM_REGISTRY = {
    "Parçacık Sürü Optimizasyonu (PSO)": {
        "func": run_pso,
        "description": "Her parçacık hem kendi en iyisini hem de sürünün en iyisini izler.",
        "params": {
            "n_particles": {"label": "Parçacık Sayısı", "type": "int",  "default": 30, "min": 5,   "max": 200},
            "n_iter":      {"label": "İterasyon",       "type": "int",  "default": 100,"min": 10,  "max": 500},
            "w":           {"label": "Atalet (w)",      "type": "float","default": 0.7,"min": 0.1, "max": 1.0, "step": 0.05},
            "c1":          {"label": "Bilişsel (c1)",   "type": "float","default": 1.5,"min": 0.5, "max": 3.0, "step": 0.1},
            "c2":          {"label": "Sosyal (c2)",     "type": "float","default": 1.5,"min": 0.5, "max": 3.0, "step": 0.1},
        },
    },
    "Genetik Algoritma (GA)": {
        "func": run_genetic,
        "description": "Seçim, çaprazlama ve mutasyon ile nesiller boyu evrim.",
        "params": {
            "pop_size":       {"label": "Popülasyon",         "type": "int",  "default": 50,  "min": 10,  "max": 300},
            "n_iter":         {"label": "Nesil Sayısı",       "type": "int",  "default": 100, "min": 10,  "max": 500},
            "crossover_rate": {"label": "Çaprazlama Oranı",  "type": "float","default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05},
            "mutation_rate":  {"label": "Mutasyon Oranı",    "type": "float","default": 0.05,"min": 0.0, "max": 0.5, "step": 0.01},
            "tournament_k":   {"label": "Turnuva Büyüklüğü", "type": "int",  "default": 3,   "min": 2,   "max": 10},
        },
    },
    "Isıl İşlem (SA)": {
        "func": run_simulated_annealing,
        "description": "Soğuma sürecini taklit eder; başlangıçta kötü çözümleri de kabul eder.",
        "params": {
            "n_iter":      {"label": "İterasyon",        "type": "int",  "default": 500, "min": 50,  "max": 2000},
            "T_init":      {"label": "Başlangıç T",      "type": "float","default": 100.,"min": 1.,  "max": 1000.,"step": 10.},
            "T_min":       {"label": "Minimum T",        "type": "float","default": 0.01,"min": 1e-4,"max": 1.,  "step": 0.01},
            "cooling":     {"label": "Soğuma Katsayısı", "type": "float","default": 0.97,"min": 0.8, "max": 0.999,"step": 0.005},
            "step_size":   {"label": "Adım Boyutu",      "type": "float","default": 0.3, "min": 0.01,"max": 1.0, "step": 0.01},
            "n_restarts":  {"label": "Yeniden Başlatma", "type": "int",  "default": 5,   "min": 0,   "max": 20},
        },
    },
    "Tabu Araştırma (TS)": {
        "func": run_tabu_search,
        "description": "Son ziyaret edilen konumları hafızada tutarak döngüden kaçınır.",
        "params": {
            "n_iter":      {"label": "İterasyon",         "type": "int",  "default": 200, "min": 20,  "max": 1000},
            "tabu_tenure": {"label": "Tabu Listesi Uzunluğu","type":"int","default": 15,  "min": 2,   "max": 100},
            "n_neighbors": {"label": "Komşu Sayısı",      "type": "int",  "default": 20,  "min": 5,   "max": 100},
            "step_size":   {"label": "Adım Boyutu (oran)","type": "float","default": 0.3, "min": 0.01,"max": 1.0, "step": 0.01},
        },
    },
    "Karınca Koloni (ACOR)": {
        "func": run_ant_colony,
        "description": "Karıncaların yol iziyle arama; sürekli uzayda Gauss dağılımlı örnekleme.",
        "params": {
            "n_ants":       {"label": "Karınca Sayısı",     "type": "int",  "default": 30, "min": 5,   "max": 200},
            "n_iter":       {"label": "İterasyon",          "type": "int",  "default": 100,"min": 10,  "max": 500},
            "archive_size": {"label": "Arşiv Büyüklüğü",   "type": "int",  "default": 10, "min": 2,   "max": 50},
            "q":            {"label": "Seçim Baskısı (q)",  "type": "float","default": 0.5,"min": 0.01,"max": 2.0, "step": 0.05},
            "xi":           {"label": "Yerel Arama Faktörü (ξ)","type":"float","default":0.85,"min":0.1,"max":1.5,"step":0.05},
        },
    },
    "Yapay Bağışıklık (CLONALG)": {
        "func": run_artificial_immune,
        "description": "Klonal seçim ilkesi: iyi antikoru kopyala, az afin olanı çok mutasyona uğrat.",
        "params": {
            "pop_size":        {"label": "Antikor Sayısı",    "type": "int",  "default": 30, "min": 5,  "max": 200},
            "n_iter":          {"label": "İterasyon",         "type": "int",  "default": 100,"min": 10, "max": 500},
            "n_clones_factor": {"label": "Klon Çarpanı",      "type": "int",  "default": 5,  "min": 1,  "max": 20},
            "mutation_decay":  {"label": "Mutasyon Sönümü",   "type": "float","default": 2.5,"min": 0.5,"max": 10.,"step": 0.5},
            "n_replace":       {"label": "Yenileme Sayısı",   "type": "int",  "default": 5,  "min": 1,  "max": 20},
        },
    },
    "Diferansiyel Gelişim (DE)": {
        "func": run_differential_evolution,
        "description": "Üç rastgele bireyin farkından mutant vektör üretir; verimli ve güçlü.",
        "params": {
            "pop_size": {"label": "Popülasyon",   "type": "int",    "default": 30,  "min": 5,   "max": 200},
            "n_iter":   {"label": "İterasyon",    "type": "int",    "default": 100, "min": 10,  "max": 500},
            "F":        {"label": "Ölçek (F)",    "type": "float",  "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05},
            "CR":       {"label": "Çaprazlama (CR)","type":"float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
            "strategy": {"label": "Strateji",     "type": "select",
                         "options": ["DE/rand/1/bin","DE/best/1/bin","DE/rand/2/bin","DE/current-to-best/1/bin"],
                         "default": "DE/rand/1/bin"},
        },
    },
    "Yapay Arı Koloni (ABC)": {
        "func": run_abc,
        "description": "İşçi, izci ve kaşif arı rolleriyle besin kaynaklarını (çözümler) keşfeder.",
        "params": {
            "colony_size":   {"label": "Koloni Büyüklüğü",  "type": "int", "default": 40,  "min": 4,  "max": 200},
            "n_iter":        {"label": "İterasyon",          "type": "int", "default": 100, "min": 10, "max": 500},
            "limit_factor":  {"label": "Tükenme Çarpanı",   "type": "int", "default": 5,   "min": 1,  "max": 20},
        },
    },
}
