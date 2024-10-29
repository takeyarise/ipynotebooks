from numpy.typing import NDArray
import numpy as np
import scipy.special
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

# - [x] sinusoidal
# - [x] weierstrass
# - [x] brownian motion
# - [x] koch curve
# - [ ] midpoint displacement
# - [x] takagi curve
# - [ ] gaussian noise
# - [x] levy flight
#   - [ ] 確認
# - [x] perlin noise
#   - [ ] 確認
# - [ ] sigmoid
# - [ ] chirp function
# - [ ] morlet wavelet
# - [x] bessel function
#   - [ ] 確認
# - [ ] airy function
# - [ ] random walk
# - [ ] gabor function
# - [x] lorenz
#   - [ ] 確認
# - [x] duffing
#   - [ ] 確認


def sinusoidal(x: NDArray, a: float, f: float, p: float) -> NDArray:
    """sinusoidal

    .. math:: $a \sin(2\pi x f + p)$

    Parameters
    ----------
    x : NDArray
        _description_
    a : float
        _description_
    f : float
        _description_
    p : float
        _description_

    Returns
    -------
    NDArray
        _description_
    """    
    return a * np.sin(x * 2 * np.pi * f + p)


def weierstrass_function(x: NDArray, a: float, b: int, n_terms: int) -> NDArray:
    """weierstrass_function

    .. math:: $f(x)=\sum_{n=0}^{\mathrm{n_terms}} a^n \cos(b^n\pi x)$

    本当は `n_terms -> ∞`

    Parameters
    ----------
    x : axis
    a : float
        0 < a < 1
    b : int
        b は奇数で、$ab > 1 + 1.5\pi$ (= 5.7123...) を満たす必要があります。これは関数が連続で至る所微分不可能であるための条件です。
    n_terms : int
        級数の項数で、値を増やすと波形の細部がより詳細になります。

    Returns
    -------
    signal
    """
    assert 0 < a < 1, "パラメータ 'a' は0と1の間である必要があります"
    assert b % 2 == 1 and b * a > 1 + 1.5 * np.pi, "パラメータ 'b' は奇数で、b * a > 1 + 1.5π (5.7123...) を満たす必要があります"
    y = np.zeros_like(x)
    for n in range(n_terms):
        y += a**n * np.cos(b**n * np.pi * x)
    return y


def fractional_brownian_motion(t: NDArray, dt: float, H: float) -> NDArray:
    """fractional Brownian motion

    - if H = 1/2 then the process is in fact a Brownian motion or Wiener process;
    - if H > 1/2 then the increments of the process are positively correlated;
    - if H < 1/2 then the increments of the process are negatively correlated.

    Parameters
    ----------
    t : time
        _description_
    dt : float
        time interval
    H : float
        Hurst index, 0 < H < 1
    """
    delta = np.random.randn(len(t)) * np.sqrt(dt**(2*H))
    return np.cumsum(delta)


def koch_curve(iterations: int) -> tuple[NDArray, NDArray]:
    """koch_curve

    コッホ曲線を生成

    Parameters
    ----------
    iterations : int
        反復回数

    Returns
    -------
    real, imag
    """    
    def koch_segment(p1, p2, iteration):
        if iteration == 0:
            return [p1, p2]
        else:
            delta = (p2 - p1) / 3
            p3 = p1 + delta
            p4 = p1 + 2 * delta
            p5 = p3 + np.exp(np.pi / 3 * 1j) * delta
            return (
                koch_segment(p1, p3, iteration - 1)[:-1] +
                koch_segment(p3, p5, iteration - 1)[:-1] +
                koch_segment(p5, p4, iteration - 1)[:-1] +
                koch_segment(p4, p2, iteration - 1)
            )

    p1 = 0 + 0j
    p2 = 1 + 0j
    points = koch_segment(p1, p2, iterations)
    x = np.array([p.real for p in points])
    y = np.array([p.imag for p in points])
    return x, y


def takagi_function(x: NDArray, n_terms: int) -> NDArray:
    """Takagi function

    高木関数（ブランマンジュ関数）

    .. math:: $T(x)=\sum_{n=0}^\mathrm{n_term}\frac{s(2^nx)}{2^n}$

    コードだと $\phi(x)$ が整数に近ければ 0.5 で 0.5 だけ離れれば 0 になるため，下に膨らんだ形になる

    Parameters
    ----------
    x : _type_
        _description_
    n_terms : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    y = np.zeros_like(x)
    for n in range(n_terms):
        y += (2 ** -n) * np.abs(2 ** n * x % 1 - 0.5)
    return y


def levy_flight(n, alpha=1.5):
    """levy_flight

    レヴィフライトは、重尾分布に従うステップ長を持つランダムウォークです。
    大きな飛躍が自己相似なパターンを生み出します。

    Parameters
    ----------
    n : step 数
        _description_
    alpha : float, optional
        _description_, by default 1.5

    Returns
    -------
    signal
    """
    step_lengths = np.random.pareto(alpha, n)
    directions = np.random.choice([-1, 1], n)
    steps = directions * step_lengths
    positions = np.cumsum(steps)
    return positions


def perlin_noise(x):
    """perlin_noise

    パーリンノイズは、滑らかな疑似乱数ノイズで、自然なテクスチャの生成によく使われます。
    異なるスケールのノイズを重ねることで自己相似性が得られます。

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    signal
    """
    # グラデーションベクトルを初期化
    grad = np.random.randn(len(x) + 1)
    # 整数部分と小数部分を計算
    xi = x.astype(int)
    xf = x - xi
    # フェード関数を適用
    u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
    # 線形補間
    n0 = grad[xi] * xf
    n1 = grad[xi + 1] * (xf - 1)
    return (1 - u) * n0 + u * n1


def Bessel_function(x):
    """Bessel_function

    ベッセル関数は、円筒座標系や球座標系での波動方程式の解として現れ、振動系の解析に利用されます。
    第一種ベッセル関数: $$J_n(x) = \sum_{k=0}^\inf\frac{(-1)^k}{k!\Gamma(k + n + 1)}(\frac{x}{2})^{2k+n}$$

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    J_0(x)
    """
    y = scipy.special.jn(
        0,  # ベッセル関数の次数
        x
    )
    return y


def chaos_signal_by_lorenz(t):
    def lorenz(t, state, sigma=10, beta=8/3, rho=28):
        """lorenz

        ローレンツ方程式は、気象学における対流の簡易モデルであり、カオス的な挙動を示すことで有名です。
        これを利用して複雑な波形を生成できます。

        Parameters
        ----------
        t : _type_
            _description_
        state : _type_
            _description_
        sigma : int, optional
            _description_, by default 10
        beta : _type_, optional
            _description_, by default 8/3
        rho : int, optional
            _description_, by default 28
        """
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    # t_span = (0, 40)
    # t_eval = np.linspace(t_span[0], t_span[1], 10000)
    t_span = t[0], t[-1]
    t_eval = t
    initial_state = [1, 1, 1]
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
    x, y, z = sol.y

    return x


def duffing_signal(t):
    def duffing(state, t, delta=0.2, alpha=-1, beta=1, gamma=0.3, omega=1.2):
        """duffing

        ダフィング方程式は、非線形ばね特性を持つ振動子をモデル化する微分方程式で、カオス的な挙動も示します。

        Parameters
        ----------
        state : _type_
            _description_
        t : _type_
            _description_
        delta : float, optional
            _description_, by default 0.2
        alpha : int, optional
            _description_, by default -1
        beta : int, optional
            _description_, by default 1
        gamma : float, optional
            _description_, by default 0.3
        omega : float, optional
            _description_, by default 1.2

        Returns
        -------
        _type_
            _description_
        """    
        x, y = state
        dx = y
        dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
        return [dx, dy]

    # t = np.linspace(0, 100, 10000)
    initial_state = [0, 0]

    sol = odeint(duffing, initial_state, t)
    x = sol[:, 0]
    return x
