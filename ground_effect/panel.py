"""Inviscid surface pressure on the airfoil surface that groundGen.py meshes.

gmsh joins the points of an airfoil with its built-in 'Spline', a uniform Catmull-Rom curve
(gmsh_curve), from the first point to the last, and groundGen closes the trailing edge with
a straight line (cfd_surface). A panel method with constant-strength sources and one vortex
strength (Hess & Smith 1967) gives the inviscid Cp on that surface (inviscid_cp). Kinks in
the surface show up as spikes in Cp; wiggle() measures them. Used by shape_check.py,
smooth_airfoils.py (to screen the smoothed shapes) and smoothing_test.py.
"""
import numpy as np

from groundGen import tidy_trailing_edge

PANELS = 1600  # about this many panels around the airfoil


def gmsh_curve(ar, per_segment):
    """Points along the uniform Catmull-Rom spline through ar, as gmsh's built-in kernel
    builds it (end segments: mirrored end points)."""
    p = np.vstack([2 * ar[0] - ar[1], ar, 2 * ar[-1] - ar[-2]])
    t = np.linspace(0, 1, per_segment, endpoint=False)[:, None]
    out = [0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
           for p0, p1, p2, p3 in zip(p[:-3], p[1:-2], p[2:-1], p[3:])]
    return np.vstack(out + [ar[-1:]])


def cfd_surface(ar, per_segment):
    """The airfoil surface as groundGen.py meshes it, from the points as load_airfoil reads
    them: gmsh's spline from the first point to the last, then a straight line back to the
    first. If that line is the base of a blunt trailing edge, it is left out (the panel
    method keeps a thick trailing edge open)."""
    ar, blunt = tidy_trailing_edge(ar)
    curve = gmsh_curve(ar, per_segment)
    if not blunt:
        step = np.hypot(*np.diff(curve, axis=0).T).mean()
        k = max(1, int(np.ceil(np.hypot(*(ar[0] - ar[-1])) / step)))
        curve = np.vstack([curve, ar[-1] + (ar[0] - ar[-1]) * np.linspace(0, 1, k + 1)[1:, None]])
    return curve[np.r_[True, np.hypot(*np.diff(curve, axis=0).T) > 1e-12]]  # no zero-length panels


def panel_cp(pts, alpha_deg):
    """Inviscid Cp at the panel midpoints of the contour pts (UIUC order: trailing edge ->
    upper surface -> leading edge -> lower surface; a thick trailing edge stays open).
    Returns the midpoints and Cp, in the same order."""
    pts = pts[::-1]  # clockwise, so that the left normal points into the flow
    p1, p2 = pts[:-1], pts[1:]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1])
    t = d / length[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    mid = 0.5 * (p1 + p2)
    rx, ry = mid[:, None, 0] - p1[None, :, 0], mid[:, None, 1] - p1[None, :, 1]
    xi = rx * t[None, :, 0] + ry * t[None, :, 1]   # midpoint i in the frame of panel j
    eta = rx * n[None, :, 0] + ry * n[None, :, 1]
    log_r = np.log(np.hypot(xi, eta) / np.hypot(xi - length[None, :], eta))
    angle = np.arctan2(eta, xi - length[None, :]) - np.arctan2(eta, xi)
    np.fill_diagonal(log_r, 0.0)
    np.fill_diagonal(angle, np.pi)  # on the panel itself, seen from the flow side
    k = 1 / (2 * np.pi)
    # velocity at midpoint i from a unit source on panel j: k*log_r along panel j and
    # k*angle across it; turned into the frame of panel i (angle difference th_i - th_j)
    th = np.arctan2(t[:, 1], t[:, 0])
    s, c = np.sin(th[:, None] - th[None, :]), np.cos(th[:, None] - th[None, :])
    src_n = k * (c * angle - s * log_r)
    src_t = k * (c * log_r + s * angle)
    vor_n, vor_t = -src_t.sum(1), src_n.sum(1)  # a vortex of unit strength on every panel
    a = np.radians(alpha_deg)
    v_inf = np.array([np.cos(a), np.sin(a)])
    m = len(length)
    lhs, rhs = np.zeros((m + 1, m + 1)), np.zeros(m + 1)
    lhs[:m, :m], lhs[:m, m], rhs[:m] = src_n, vor_n, -(n @ v_inf)        # no flow through the wall
    lhs[m, :m], lhs[m, m] = src_t[0] + src_t[-1], vor_t[0] + vor_t[-1]  # Kutta condition
    rhs[m] = -(t[0] + t[-1]) @ v_inf
    sol = np.linalg.solve(lhs, rhs)
    vt = src_t @ sol[:m] + vor_t * sol[m] + t @ v_inf
    return mid[::-1], (1 - vt ** 2)[::-1]


def surfaces(mid, cp):
    """{side: (x, cp)} sorted by x; upper = from the trailing edge to the leading edge."""
    le = np.argmin(mid[:, 0])
    out = {}
    for side, sl in [("upper", slice(0, le + 1)), ("lower", slice(le, None))]:
        o = np.argsort(mid[sl, 0])
        out[side] = (mid[sl, 0][o], cp[sl][o])
    return out


def inviscid_cp(ar, alpha):
    """Inviscid Cp of both surfaces of the surface groundGen meshes from the points ar."""
    return surfaces(*panel_cp(cfd_surface(ar, max(2, PANELS // len(ar))), alpha))


def wiggle(x, cp, lo=0.02, hi=0.95, half_width=0.01):
    """Largest departure of cp from a parabola fitted over +-half_width around each point."""
    best = (0.0, np.nan)
    for i in np.where((x > lo) & (x < hi))[0]:
        s = np.abs(x - x[i]) < half_width
        dev = abs(cp[i] - np.polyfit(x[s] - x[i], cp[s], 2)[-1])  # the fit's value at x[i]
        if dev > best[0]:
            best = (dev, float(x[i]))
    return best


def cp_change(a, b, lo=0.02, hi=0.95):
    """Largest difference between two Cp distributions of one surface (b interpolated to
    the points of a), and where."""
    x, cp = a
    s = (x > lo) & (x < hi)
    d = np.abs(cp[s] - np.interp(x[s], *b))
    i = int(np.argmax(d))
    return float(d[i]), float(x[s][i])
