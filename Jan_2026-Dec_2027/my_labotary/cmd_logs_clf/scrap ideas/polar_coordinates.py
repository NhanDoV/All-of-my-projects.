
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Polar Coordinates Lab",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Polar Coordinates Lab")
st.caption(
    "Explore polar equations, understand their parameters, "
    "and discover how mathematical curves are constructed."
)

# =========================================================
# CURVE DEFINITIONS
# =========================================================
CURVES = {
    "Circle": {
        "equation": r"r = a",
        "description": (
            "A circle centered at the origin. Every point has "
            "the same distance a from the origin."
        ),
        "formula": r"x = a\cos\theta,\quad y = a\sin\theta",
        "default_a": 3.0,
    },
    "Ellipse": {
        "equation": (
            r"r = \frac{ab}"
            r"{\sqrt{b^2\cos^2\theta+a^2\sin^2\theta}}"
        ),
        "description": (
            "An ellipse centered at the origin. Parameter a is "
            "the semi-axis along x, while b is the semi-axis "
            "along y."
        ),
        "formula": (
            r"x=r\cos\theta,\quad y=r\sin\theta"
        ),
        "default_a": 4.0,
    },
    "Cardioid (Heart-like)": {
        "equation": r"r = a(1+\cos\theta)",
        "description": (
            "A cardioid is a heart-like curve with a cusp at "
            "the origin. The parameter a controls its size."
        ),
        "formula": (
            r"x=a(1+\cos\theta)\cos\theta"
            r",\quad y=a(1+\cos\theta)\sin\theta"
        ),
        "default_a": 2.0,
    },
    "Rose Curve (Flower)": {
        "equation": r"r = a\cos(k\theta)",
        "description": (
            "A rose curve forms flower petals. Parameter a "
            "controls the petal length. For positive integer k, "
            "there are k petals when k is odd and 2k petals "
            "when k is even."
        ),
        "formula": (
            r"x=a\cos(k\theta)\cos\theta"
            r",\quad y=a\cos(k\theta)\sin\theta"
        ),
        "default_a": 3.0,
    },
    "Spiral": {
        "equation": r"r = a\theta",
        "description": (
            "An Archimedean spiral. The radius grows linearly "
            "with the angle, so the curve moves farther away "
            "from the origin as theta increases."
        ),
        "formula": (
            r"x=a\theta\cos\theta"
            r",\quad y=a\theta\sin\theta"
        ),
        "default_a": 0.25,
    },
    "Lemniscate": {
        "equation": r"r^2 = a^2\cos(2\theta)",
        "description": (
            "A two-loop, figure-eight-like curve. Real-valued "
            "r exists only where cos(2 theta) is non-negative."
        ),
        "formula": (
            r"x=r\cos\theta,\quad y=r\sin\theta"
        ),
        "default_a": 3.0,
    },
}

# =========================================================
# LAYOUT
# =========================================================
left_col, right_col = st.columns([5, 3], gap="large",)

with left_col:
    st.subheader("🎛️ Learning Controls")
    param_col, descr_col = st.columns([1, 2], gap='medium')
    with param_col:
        curve_name = st.selectbox(
            "1. Select a curve",
            options=list(CURVES.keys()),
        )

    curve = CURVES[curve_name]

    with descr_col:
        st.markdown("##### 2. Mathematical equation")
        st.latex(curve["equation"])

        st.markdown("##### 3. Understand the curve")
        st.write(curve["description"])

        st.markdown("**Cartesian conversion**")
        st.latex(curve["formula"])

    with param_col:
        a = st.slider(
            "Scale parameter a",
            min_value=0.5,
            max_value=5.0,
            value=CURVES[curve_name]["default_a"],
            step=0.5,
        )

        b = 2.0
        k = 3

        if curve_name == "Ellipse":
            b = st.slider(
                "Semi-axis b",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
            )

        if curve_name == "Rose Curve (Flower)":
            k = st.slider(
                "Number parameter k",
                min_value=1,
                max_value=8,
                value=3,
                step=1,
            )

        domain_options = {
            r"$[0,\pi/2]$ — First quadrant": (0, np.pi / 2),
            r"$[0,\pi]$ — Upper half": (0, np.pi),
            r"$[0,2\pi]$ — Full revolution": (0, 2 * np.pi),
            r"$[-\pi,\pi]$ — Symmetric interval": (-np.pi, np.pi),
            r"$[0,4\pi]$ — Two revolutions": (0, 4 * np.pi),
        }

        domain_label = st.selectbox(
            "Angle interval",
            options=list(domain_options.keys()),
            index=2,
        )

        theta_min, theta_max = domain_options[domain_label]

        st.caption(
            f"θ starts at {theta_min / np.pi:.2f}π "
            f"and ends at {theta_max / np.pi:.2f}π."
        )

        n_points = st.slider(
            "Curve resolution",
            min_value=100,
            max_value=2000,
            value=800,
            step=100,
            help="More points produce a smoother curve.",
        )

    st.markdown("#### 💡 Key concepts to remember")

    concept_col1, concept_col2, concept_col3 = st.columns(3)

    with concept_col1:
        st.markdown("**Radius — r**")
        st.write("Distance from the origin.")

    with concept_col2:
        st.markdown("**Angle — θ**")
        st.write("Rotation measured from the positive x-axis.")

    with concept_col3:
        st.markdown("**Domain — θ interval**")
        st.write(
            "The chosen interval determines which part of the "
            "curve is traced."
        )

# =========================================================
# MATHEMATICAL COMPUTATION
# =========================================================
theta = np.linspace(theta_min, theta_max, n_points)

if curve_name == "Circle":
    r = np.full_like(theta, a)

elif curve_name == "Ellipse":
    denominator = np.sqrt(
        (b * np.cos(theta)) ** 2
        + (a * np.sin(theta)) ** 2
    )
    r = (a * b) / denominator

elif curve_name == "Cardioid (Heart-like)":
    r = a * (1 + np.cos(theta))

elif curve_name == "Rose Curve (Flower)":
    r = a * np.cos(k * theta)

elif curve_name == "Spiral":
    r = a * theta

elif curve_name == "Lemniscate":
    # Mask invalid regions rather than taking sqrt of negatives.
    inside = np.cos(2 * theta) >= 0
    r = np.full_like(theta, np.nan)
    r[inside] = a * np.sqrt(np.cos(2 * theta[inside]))

# =========================================================
# VISUALIZATION
# =========================================================
with right_col:
    st.subheader("📊 Interactive Visualization")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="polar")

    ax.plot(
        theta,
        r,
        linewidth=2.5,
        color="#2878B5",
        label=curve_name,
    )

    ax.set_title(
        f"{curve_name}\n"
        rf"$\theta \in [{theta_min / np.pi:.2g}\pi,"
        rf"\ {theta_max / np.pi:.2g}\pi]$",
        fontsize=14,
        pad=25,
    )

    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Cartesian interpretation
    x = r * np.cos(theta)
    y = r * np.sin(theta)

with left_col:
    st.markdown("##### 🔎 Cartesian coordinates")

    st.latex(
        r"x=r\cos\theta,\qquad y=r\sin\theta"
    )

    valid = np.isfinite(x) & np.isfinite(y)

    if np.any(valid):
        st.write(
            f"**Valid points:** {valid.sum():,} / {len(theta):,}"
        )

        st.dataframe(
            {
                "θ (radians)": theta[valid][:8],
                "r": r[valid][:8],
                "x": x[valid][:8],
                "y": y[valid][:8],
            },
            hide_index=True,
            use_container_width=True,
        )