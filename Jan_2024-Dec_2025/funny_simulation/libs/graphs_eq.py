import warnings, math
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.patches as patches

warnings.filterwarnings("ignore", category=RuntimeWarning)

def graph_render_wrt_background(bg_img):
    st.markdown(f"""
        <style>
        .graphs {{
            background-image: url("data:image/jpg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }}
        </style>
        <div class="graphs">
            <h2 style="color: #003366;">Funny Graphs 📈</h2>
            <p>Draw shapes using equations: heart, spiral, batman logo…</p>
        </div>
    """, unsafe_allow_html=True)

# ----------------------------- HEARTS ---------------------------------- #
def heart_eq1(n_pat=101, epsilon=1e-6):
    t = np.linspace(-1, 1, num=n_pat)
    x = np.sin(t) * np.cos(t) * np.log(np.abs(t) + epsilon) # make the tail of heart better
    y = np.sqrt(np.abs(t)) * np.cos(t)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(x, y)
    st.pyplot(fig)

def heart_eq2(n_pat=500):
    # Define grid
    x = np.linspace(-1.25, 1.25, n_pat)
    y = np.linspace(-1.1, 1.25, n_pat)
    X, Y = np.meshgrid(x, y)

    # Define implicit function
    F = (X**2 + Y**2 - 1)**3 - (X**2) * (Y**3)

    # Plot contour where F = 0
    fig = plt.figure(figsize=(6,6))
    plt.contour(X, Y, F, levels=[0], colors="red")
    plt.gca().set_aspect("equal")
    st.pyplot(fig)

def heart_eq3(n_pat=500):
    x = np.linspace(-1.1, 1.1, n_pat)
    y = np.linspace(-0.8, 1.1, n_pat)
    X, Y = np.meshgrid(x, y)
    F = X**2 + 2*(0.6*np.cbrt(X**2) - Y)**2 - 1  # safer form for x^(2/3)
    fig = plt.figure(figsize=(6, 6))
    plt.contour(X, Y, F, levels=[0], colors="#9400D3")
    st.pyplot(fig)

def heart_eq4(n_pat=500):
    t = np.linspace(0, 60, n_pat)
    fig = plt.figure(figsize=(6, 6))
    plt.plot([-0.01*(-_**2 + 40*_ + 1200)*np.sin( np.pi * _ /180 ) for _ in t],
            [0.01*(-_**2 + 40*_ + 1200)*np.cos( np.pi * _ /180 ) for _ in t], "#BA55D3"
            )
    plt.plot([0.01*(-_**2 + 40*_ + 1200)*np.sin( np.pi * _ /180 ) for _ in t],
            [0.01*(-_**2 + 40*_ + 1200)*np.cos( np.pi * _ /180 ) for _ in t], "#BA55D3"
            )
    st.pyplot(fig)

def heart_eq5(n_pat=500):
    t1 = np.linspace(-np.pi, -np.pi/2, n_pat)
    t2 = np.linspace(np.pi/2, np.pi, n_pat)
    r1 = (np.sin(t1))**7*np.exp(2*np.abs(t1))
    r2 = (np.sin(t2))**7*np.exp(2*np.abs(t2))
    fig = plt.figure(figsize=(6, 6))
    plt.plot(r1*np.cos(t1), r1*np.sin(t1), 'magenta')
    plt.plot(r2*np.cos(t2), r2*np.sin(t2), 'magenta')
    st.pyplot(fig)

def heart_eq6(n_deg = 3, n_pat=500):
    r = (n_deg + 5) / 3  # scale factor
    # expand domain relative to r
    x = np.linspace(-r*0.8, r*0.8, n_pat)
    y = np.linspace(-r*0.8, r, n_pat)
    X, Y = np.meshgrid(x, y)
    F = X**2 + (Y - np.abs(X)**(1/2) )**2 - n_deg
    fig = plt.figure(figsize=(6, 6))
    plt.contour(X, Y, F, levels=[0], colors="purple")
    st.pyplot(fig)

def all_hearts():
    sns.set_theme()
    fig, ax = plt.subplots(2, 3, figsize=(12, 9))
    ax = ax.ravel()
    # Type 1
    t = np.linspace(-1, 1, num=1111)
    x = np.sin(t) * np.cos(t) * np.log(np.abs(t) + 1e-8) # make the tail of heart better
    y = np.sqrt(np.abs(t)) * np.cos(t)
    ax[0].plot(x, y, color="red")

    # Type 2
    x = np.linspace(-1.5, 1.5, 1001)
    y = np.linspace(-1.1, 1.35, 1001)
    X, Y = np.meshgrid(x, y)
    F = (X**2 + Y**2 - 1)**3 - (X**2) * (Y**3)
    ax[1].contour(X, Y, F, levels=[0], colors="#800000")

    # Type 3
    x = np.linspace(-1.1, 1.1, 1001)
    y = np.linspace(-0.8, 1.1, 1001)
    X, Y = np.meshgrid(x, y)
    F = X**2 + 2*(0.6*np.cbrt(X**2) - Y)**2 - 1  # safer form for x^(2/3)
    ax[2].contour(X, Y, F, levels=[0], colors="#9400D3")

    # Type 4
    t = np.linspace(0, 60, 1001)
    ax[3].plot([-0.01*(-_**2 + 40*_ + 1200)*np.sin( np.pi * _ /180 ) for _ in t],
            [0.01*(-_**2 + 40*_ + 1200)*np.cos( np.pi * _ /180 ) for _ in t], "#BA55D3"
            )
    ax[3].plot([0.01*(-_**2 + 40*_ + 1200)*np.sin( np.pi * _ /180 ) for _ in t],
            [0.01*(-_**2 + 40*_ + 1200)*np.cos( np.pi * _ /180 ) for _ in t], "#BA55D3"
            )

    # Type 5
    t1 = np.linspace(-np.pi, -np.pi/2, 1001)
    t2 = np.linspace(np.pi/2, np.pi, 1001)
    r1 = (np.sin(t1))**7*np.exp(2*np.abs(t1))
    r2 = (np.sin(t2))**7*np.exp(2*np.abs(t2))
    ax[4].plot(r1*np.cos(t1), r1*np.sin(t1), 'magenta')
    ax[4].plot(r2*np.cos(t2), r2*np.sin(t2), 'magenta')

    # Type 6
    x = np.linspace(-2., 2, 1001)
    y = np.linspace(-2, 2.5, 1001)
    X, Y = np.meshgrid(x, y)
    F = X**2 + (Y - np.abs(X)**(1/2) )**2 - 3
    ax[5].contour(X, Y, F, levels=[0], colors="purple")

    for idx in range(6):
        ax[idx].set_aspect("equal")  # keep proportions
        ax[idx].set_title(f"type_{idx + 1}")
    
    plt.subplots_adjust(
        left=0.05,   # space from left edge of figure
        right=0.95,  # space from right edge
        top=0.92,    # space from top
        bottom=0.08, # space from bottom
        wspace=0.3,  # horizontal space between subplots
        hspace=0.1   # vertical space between subplots
    )
    st.pyplot(fig)

def heart_plot(sel_eq):
    c1, _, c2 = st.columns([15, 1, 10])
    with c1:
        c12, c11 = st.columns([9, 4])
        with c12:
            all_hearts()
    if sel_eq == "type_1":
        with c11:
            st.write(r"""
                    This shape is constructed from the following equations:
                    - $x = \sin t \cdot \cos t \cdot \ln \left( |t| + \epsilon \right) $
                    - $y = \sqrt{ |t| } \cdot \cos t $

                    where $t \in [-1, 1]$ and $\epsilon > 0$
                    """)            
        with c2:
            c21, c22 = st.columns(2)
            with c21:
                eps = st.number_input("Select `epsilon`", 
                                      value=0.0001, min_value=1e-29, max_value=0.1, format="%.19f")
            with c22:
                n_patt = st.number_input("Select the number of points [how much we `partitioned the heart`]",
                        value=101, min_value=7, max_value=9999)
            _, c, _ = st.columns([2, 9, 1])
            with c:
                heart_eq1(n_patt, eps)
    
    elif sel_eq == "type_2":
        with c11:
            st.write(r"`Type 2` is totally defined by this equation")
            st.latex(r"(x^2+y^2-1)^3 - x^2 y^3 = 0")
            st.write(r"""
                     where $y \in [-1, 1]$
                     """)
        with c2:
            n_patt = st.number_input("Select the number of points [how much we `partitioned the heart`]",
                            value=11, min_value=7, max_value=9999)
            _, c, _ = st.columns([2, 9, 1])
            with c:
                heart_eq2(n_patt)
    
    elif sel_eq == "type_3":
        with c11:
            st.write(r"`Type 3` is totally defined by this equation")
            st.latex(r" x^2 + 2\cdot(0.6\cdot x^{2/3} - y)^2 - 1 = 0")
            st.write(r"""
                     where $y \in [-1, 1]$
                     """)
        with c2:
            n_patt = st.number_input("Select the number of points [how much we `partitioned the heart`]",
                            value=11, min_value=7, max_value=9999)
            _, c, _ = st.columns([2, 9, 1])
            with c:
                heart_eq3(n_patt)
    
    elif sel_eq == "type_4":
        with c11:
            st.write(r"`Type 4` is totally defined by these equations")
            st.latex(r"""
                        \begin{array}{l}
                            x \, = \, \pm \tfrac{\left( -t^2 + 40 t + 1200 \right)}{100} \cdot \sin \left( \tfrac{\pi t}{180} \right) 
                            \\ \\
                            y \, = \, \tfrac{\left( -t^2 + 40 t + 1200 \right)}{100} \cdot \cos \left( \tfrac{\pi t}{180} \right)
                        \end{array}
                     """)
            st.write(r"""
                     where $t \in [0, 60]$
                     """)
        with c2:
            n_patt = st.number_input("Select the number of points [how much we `partitioned the heart`]",
                            value=11, min_value=7, max_value=9999)
            _, c, _ = st.columns([2, 9, 1])
            with c:
                heart_eq4(n_patt)

    elif sel_eq == "type_5":
        with c11:
            st.write(r"`Type 5` is totally defined by this equation")
            st.latex(r""" 
                      x \, = \, r \cos t \; , \quad y \, = \, r \sin t
                      """)
            st.write(r"""
                  where 
                     - $r = \sin^7 t \cdot e^{|2t|} $ 
                     - $t \in \left[ -\pi, - \tfrac{\pi}{2} \right] \cup \left[ \tfrac{\pi}{2}, \pi \right] $
                     """)
        with c2:
            n_patt = st.number_input("Select the number of points [how much we `partitioned the heart`]",
                            value=11, min_value=7, max_value=9999)
            _, c, _ = st.columns([2, 9, 1])
            with c:
                heart_eq5(n_patt)

    else:
        with c11:
            st.write(r"`Type 6` is totally defined by this equation")
            st.latex(r""" 
                      x^2 - \left( y - \sqrt{ \Big \vert x \Big \vert } \right)^2 = 3
                      """)
        with c2:
            n_patt = st.number_input("Select the number of points [how much we `partitioned the heart`]",
                            value=11, min_value=7, max_value=9999)
            _, c, _ = st.columns([2, 9, 1])
            with c:
                heart_eq6(3, n_patt)

# ----------------------------- BAT-MAN ---------------------------------- #
def batman_eq1():
    eps = 1e-6
    xs = np.arange(-7.25, 7.25, 0.005)
    ys = np.arange(-3, 3, 0.005)
    x, y = np.meshgrid(xs, ys)

    with np.errstate(invalid='ignore', divide='ignore'):
        eq1 = ((x/7)**2*np.sqrt(abs(abs(x)-3)/(abs(x)-3)) +
               (y/3)**2*np.sqrt(abs(y + eps + 3/7*np.sqrt(33))/(y+3/7*np.sqrt(33))) - 1)
        eq2 = (abs(x/2) + eps - ((3*np.sqrt(33)-7)/112 + eps)*x**2 - 3 +
               np.sqrt(1 - (abs(abs(x)-2) - 1)**2) + eps - y)
        eq3 = (9*np.sqrt(abs((abs(x)-1)*(abs(x)-.75)) /
                         ((1-abs(x))*(abs(x)-.75))) - 8*abs(x) - y + eps)
        eq4 = (3*abs(x) + .75*np.sqrt(abs((abs(x)-.75)*(abs(x)-.5)) /
                                      ((.75-abs(x))*(abs(x)-.5))) - y)
        eq5 = (2.25*np.sqrt(abs((x-.5)*(x+.5)) /
                            ((.5-x)*(.5+x))) - y)
        eq6 = (6*np.sqrt(10)/7 + (1.5-.5*abs(x)+ eps) *
               np.sqrt(abs(abs(x)-1)/(abs(x) - 1)+ eps) -
               (6*np.sqrt(10)/14)*np.sqrt(4-(abs(x)-1)**2 + eps) - y)

    fig = plt.figure(figsize=(10, 6))
    colors = ['#FFD700', '#FF4500', '#1E90FF',
              '#32CD32', '#8A2BE2', '#000000']  # gold, orange-red, blue, green, violet, black
    eqs = [eq1, eq2, eq3, eq4, eq5, eq6]

    # plot contours
    for f, c in zip(eqs, colors):
        plt.contour(x, y, f, [0], colors=c, linewidths=1.5)

    # proxy artists for legend    
    legend_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [f"Eq {i}" for i in range(1, 7)]
    plt.legend(legend_lines, labels, loc="upper right", fontsize=10)

    plt.axis("equal")
    st.pyplot(fig)

def batman_eq2(step=0.001):
    eps = 1e-6
    def plot_segment(ax, xs, ys, **kwargs):
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.any():
            ln, = ax.plot(xs[mask], ys[mask], **kwargs)
            return ln
        return None

    # segments (same x ranges as turtle code, in "user" coordinates -7..7)
    xs_seg1 = np.arange(-7.0, -3.0, step)    # f: -7 -> -3
    xs_seg2 = np.arange(-3.0, -1.0, step)    # i: -3 -> -1
    xs_seg3 = np.arange(1.0, 3.0, step)      # i: 1 -> 3
    xs_seg4 = np.arange(3.0, 7.0, step)      # f: 3 -> 7
    xs_seg5 = np.arange(4.0, 7.0, step)      # g: 4 -> 7 (right)
    xs_seg6 = np.arange(-4.0, 4.0, step)     # h: -4 -> 4 (bottom)
    xs_seg7 = np.arange(-7.0, -4.0, step)    # g: -7 -> -4 (left)

    with np.errstate(invalid='ignore', divide='ignore'):
        # Segment f (left top)
        x = xs_seg1
        absx = np.abs(x)
        f1 = (
            1.5 * np.sqrt((-np.abs(absx - 1)) * np.abs(3 - absx) / ((absx - 1) * (3 - absx))) *
            (1 + np.abs(absx - 3) / (absx - 3)) * np.sqrt(1 - (x / 7) ** 2)
            + (4.5 + 0.75 * (np.abs(x - 0.5) + np.abs(x + 0.5))
               - 2.75 * (np.abs(x - 0.75) + np.abs(x + 0.75))) *
            (1 + np.abs(1 - absx) / (1 - absx))
        )

        # Segment i (inner curves)
        x = xs_seg2
        absx = np.abs(x)
        i_left = ((2.71052 + 1.5 - 0.5 * absx - 1.35526 * np.sqrt(4 - (absx - 1) ** 2)) *
                  np.sqrt(np.abs(absx - 1) / (absx - 1)))

        x = xs_seg3
        absx = np.abs(x)
        i_right = ((2.71052 + 1.5 - 0.5 * absx - 1.35526 * np.sqrt(4 - (absx - 1) ** 2)) *
                   np.sqrt(np.abs(absx - 1) / (absx - 1)))

        # Segment f on right
        x = xs_seg4
        absx = np.abs(x)
        f2 = (
            1.5 * np.sqrt((-np.abs(absx - 1)) * np.abs(3 - absx) / ((absx - 1) * (3 - absx))) *
            (1 + np.abs(absx - 3) / (absx - 3)) * np.sqrt(1 - (x / 7) ** 2)
            + (4.5 + 0.75 * (np.abs(x - 0.5) + np.abs(x + 0.5))
               - 2.75 * (np.abs(x - 0.75) + np.abs(x + 0.75))) *
            (1 + np.abs(1 - absx) / (1 - absx))
        )

        # g(x) right and left
        x = xs_seg5
        absx = np.abs(x + eps)
        g_right = (-3) * np.sqrt(1 - (x / 7) ** 2) * np.sqrt(np.abs(absx - 4) / (absx - 4))

        x = xs_seg7
        absx = np.abs(x + eps)
        g_left = (-3) * np.sqrt(1 - (x / 7) ** 2) * np.sqrt(np.abs(absx - 4) / (absx - 4))

        # h(x) bottom
        x = xs_seg6
        absx = np.abs(x)
        h = np.abs(x / 2) - 0.0913722 * x ** 2 - 3 + np.sqrt(1 - (np.abs(absx - 2) - 1) ** 2)

    # create figure (black background), yellow pen
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")

    color = "#FFD700"  # yellow
    lw = 1.6

    # plot pieces and capture handles (if present)
    h_f1 = plot_segment(ax, xs_seg1, f1, color="red", linewidth=lw)
    h_i_left = plot_segment(ax, xs_seg2, i_left, color="blue", linewidth=lw)

    # top connector (explicit polygon)
    top_x = np.array([-0.99, -1.0, -0.5, 0.5, 1.0, 0.99])
    top_y = np.array([1, 3.0, 2.2, 2.2, 3.0, 1])
    if top_x.size and top_y.size:
        h_conn, = ax.plot(top_x, top_y, color="green", linewidth=lw)
    else:
        h_conn = None

    h_i_right = plot_segment(ax, xs_seg3, i_right, color="blue", linewidth=lw)
    h_f2 = plot_segment(ax, xs_seg4, f2, color="red", linewidth=lw)

    # right g curve (4..7)
    h_g_right = plot_segment(ax, xs_seg5, g_right, color="red", linewidth=lw)

    # bottom central curve
    h_h = plot_segment(ax, xs_seg6, h, color="violet", linewidth=lw)

    # left g curve
    h_g_left = plot_segment(ax, xs_seg7, g_left, color="red", linewidth=lw)

    # choose representative handles for legend (prefer non-None)
    # f : f1 or f2
    h_f = h_f1 if h_f1 is not None else h_f2
    # i : left or right
    h_i = h_i_left if h_i_left is not None else h_i_right
    # g : combine left/right; prefer right then left
    h_g = h_g_right if h_g_right is not None else h_g_left

    # build handles & labels list only for existing handles
    handles = []
    labels = []
    if h_f is not None:
        handles.append(h_f); labels.append("f(x) — wings")
    if h_i is not None:
        handles.append(h_i); labels.append("i(x) — inner arms")
    if h_conn is not None:
        handles.append(h_conn); labels.append("g(x): head")
    if h_h is not None:
        handles.append(h_h); labels.append("h(x) — bottom")

    # add legend and style it for dark background
    if handles:
        leg = ax.legend(handles, labels, loc="upper right", fontsize=10)
        leg.get_frame().set_facecolor("black")
        leg.get_frame().set_edgecolor("white")
        leg.get_frame().set_alpha(0.8)
        for txt in leg.get_texts():
            txt.set_color("white")

    # final styling
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-4.5, 4.5)
    ax.axis("off")

    st.pyplot(fig)

def batman_plot(sel_eq):
    c1, _, c2 = st.columns([15, 1, 10])
    if sel_eq == "type_1":
        with c1:
            st.write("This shape is constructed from the following equations:")
            st.latex(r"""
                \begin{array}{ccl} 
                    f_1(x,y) &=& \left(\frac{x}{7}\right)^2 \cdot \left| \vert x \vert - 3 \right| \cdot \sqrt{\frac{1}{|x| \,- \, 3}} \; + \; \left(\frac{y}{3}\right)^2 \sqrt{\frac{ \left| y \, + \, \frac{3 \sqrt{33}}{7} \right|}{y \, + \,\tfrac{3 \sqrt{33}}{7}}} \; - \; 1 
                    \\ \\
                    f_2(x,y) &=& \left|\frac{x}{2}\right| - \left( \frac{3\sqrt{33}-7}{112} \right) x^2 \; - \; 3 \; + \; \sqrt{1 - ( 1 - \left| 2 - |x| \right|)^2} \; - \; y 
                    \\ \\
                    f_3(x,y) &=& 9\sqrt{\dfrac{|(|x|-1)(|x|-0.75)|}{(1-|x|)(|x|-0.75)}} \; - \; 8|x| \; - \; y 
                    \\ \\
                    f_4(x,y) &=& 3|x| + \tfrac{3}{4}\sqrt{\tfrac{|(|x|-0.75)(|x|-0.5)|}{(0.75-|x|)(|x|-0.5)}} - y
                    \\ \\
                    f_5(x,y) &=& 2.25\sqrt{\tfrac{|(x-0.5)(x+0.5)|}{(0.5-x)(0.5+x)}} - y
                    \\ \\
                    f_6(x,y) &=& \tfrac{6\sqrt{10}}{7} + \left(1.5 - 0.5|x|\right)\sqrt{\tfrac{||x|-1|}{|x|-1}} - \tfrac{3\sqrt{10}}{7}\sqrt{4 - (|x|-1)^2} - y
                \end{array}
            """)
        with c2:
            batman_eq1()
    else:
        with c1:
            st.write(r"Also, this is constructed from the following equations:")
            st.latex(r"""
                    \begin{array}{ccl}
                        f(x) &=& 1.5 \, \sqrt{\frac{-\left|\,|x|-1\,\right| \,\cdot\, \left|3-|x|\right|}{(|x|-1)(3-|x|)}} \cdot \left(1 + \frac{\left||x|-3\right|}{|x|-3}\right) \cdot \sqrt{1 - \left(\tfrac{x}{7}\right)^2} \\
                            & & \quad + \; \Bigl( 4.5 \; + \; 0.75 \bigl(|x-0.5|+|x+0.5|\bigr) \\
                            & & \qquad \qquad - 2.75 \bigl(|x-0.75|+|x+0.75|\bigr)\Bigr) \cdot \left(1 + \frac{|1-|x||}{1-|x|}\right)
                        \\[1.2em] 
                        g(x) &=& -3 \, \sqrt{1-\left(\tfrac{x}{7}\right)^2} \cdot \sqrt{\frac{\,||x|-4|}{|x|-4}}
                        \\[1.2em] 
                        h(x) &=& \tfrac{|x|}{2} - 0.0913722 \, x^2 - 3 + \sqrt{1 - \bigl(|\,|x|-2\,|-1\bigr)^2}
                        \\[1.2em] 
                        i(x) &=& \Bigl(2.71052 + 1.5 - 0.5|x| - 1.35526 \sqrt{\,4-(|x|-1)^2}\Bigr) \cdot \sqrt{\frac{\,||x|-1|}{|x|-1}}
                    \end{array} 
                     """)
        with c2:
            batman_eq2()

# ---------------------------- SPIRAL ------------------------------------ #
def archimedean_spiral(a=0, b=0.2, n_round=6, n_points=2000):
    theta_max = n_round*np.pi
    theta = np.linspace(0, theta_max, n_points)
    r = a + b*theta   # Archimedean spiral equation

    # Convert polar (r, theta) to Cartesian (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig = plt.figure(figsize=(6,6))
    plt.plot(x, y, color="crimson")
    plt.gca().set_aspect("equal")
    plt.title(f"Archimedean Spiral (a={a}, b={b})")
    st.pyplot(fig)

def golden_spiral(a=0, n_round=9, n_points=2000):
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    b = np.log(phi) / (np.pi / 2)  # growth factor
    
    theta = np.linspace(0, n_round * np.pi, n_points)
    rho = a * np.exp(b * theta)

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, color="orange")
    # ax.set_aspect("equal")
    ax.set_title("Golden Spiral")
    st.pyplot(fig)

def sprial_plot(sel_eq):
    c1, _, c2 = st.columns([15, 1, 10])
    if sel_eq == "type_1":
        with c1:
            st.write("This `spiral` is generated by")
            st.latex(r" x \, = \, \rho \cdot \cos(\theta) \; , \qquad y \, = \, \rho \cdot \sin(\theta) ")
            st.write(r"always known as `Archimedean spiral`, where")
            st.latex(r"\rho =  a \, + \, b \theta")
            st.write(r"""
                     is the `radial coordinate` in `polar coordinates`, and

                     - $\theta \in [0, k_r \cdot \pi] $ and $k_r > 0$ is the number of turns from the spiral
                     - $a$ is the `starting radius` (where the spiral begins); 

                       $\qquad \diamond$ as $\theta = 0$ then $\rho = a$; 
                     
                       $\qquad \diamond$ moreover if $a=0$ then the spiral starts at the origin.
                     
                     - $b$ controls how tightly or loosely the spiral winds (small `b` turns are close together, otherwise)
                     """)
            st.warning(r"As $\theta$ increases that implies $\rho$ increases linearly with slope $b$. This makes the spiral expand outward with each turn")
        with c2:
            c21, c22, c23, c24 = st.columns([1,1,2,2])
            with c21:
                a = st.number_input(
                    "Select `a`", 
                    value=0.0, min_value=0.0, max_value=5.0, step=0.1,
                    help="Controls the starting radius (offset)"
                )
            with c22:
                b = st.number_input(
                    "Select `b`", 
                    value=0.2, min_value=0.01, max_value=2.0, step=0.01,
                    help="Controls the spacing between successive turns"
                )
            with c23:
                n_round = st.number_input(
                    "Select `n_round`", 
                    value=6.0, min_value=1.0, max_value=10.0, step=0.25,
                    help="defines how many turns the spiral makes"
                )
            with c24:
                n_points = st.number_input(
                    "Select `n_points`", 
                    value=1000, min_value=100, max_value=10000, step=100,
                    help="Number of points to render the spiral"
                )
            _, c, _ = st.columns([2, 9, 1])
            with c:
                archimedean_spiral(a, b, n_round, n_points)
    
    else:
        with c1:
            st.write("This `spiral` is also generated by")
            st.latex(r" x \, = \, \rho \cdot \cos(\theta) \; , \qquad y \, = \, \rho \cdot \sin(\theta) ")
            st.write(r"always known as `Golden spiral`, where")
            st.latex(r"\rho =  a \, \cdot \, e^{b \theta}")
            st.write("but here")
            st.latex(r" b = \frac{2}{\pi} \cdot \ln \varphi ")
            st.write(r"""
                      is a constant, and
                     - $\varphi = \dfrac{1 + \sqrt{5}}{2}$ is the `golden-ratio`
                     - $a, \rho$ play the same roles as in `Archimedean spiral` 
                     """)
            st.warning(r"""
                       $\quad \diamond$ In the Archimedean spiral, $\rho$ increases linearly with $\theta$.

                       $\quad \diamond$ In the Golden spiral, $\rho$ increases exponentially with $\theta$ (expands faster, linked to the golden ratio).
                       """)
        with c2:
            c21, c22, c23 = st.columns([2,3,3])
            with c21:
                a = st.number_input("Select `a`",
                                    value=0.1, min_value=0.0, max_value=10.0)
            with c22:
                n_round = st.number_input("Select `n_rounds`",
                                           value=10.0, min_value=0.25, max_value=20.0)
            with c23:
                n_points = st.number_input("Select `n_points`",
                                           value=int(12*n_round), min_value=int(n_round*2), max_value=10000)
            _, c, _ = st.columns([2, 9, 1])
            with c:
                golden_spiral(a, n_round, n_points)

# ---------------------------- FLOWER ------------------------------------ #
def flower_eq1(x0=0, y0=0, r_x=0.6, r_y=0.8, N=20, n_patt = 1001, core_tp="cosine"):
    """
        Eq 1a: Modified Cosine Flowers
        Eq 1b: Modified Sine Flowers        
    """
    t = np.linspace(0, 2*np.pi, n_patt)
    # if core is cosine in the polar-coordinate
    if core_tp == "cosine":
        x = np.cos(t)*(x0 + r_x * np.cos(N*t))
        y = np.sin(t)*(y0 + r_y * np.cos(N*t))
    else:
        x = np.cos(t)*(x0 + r_x * np.sin(N*t))
        y = np.sin(t)*(y0 + r_y * np.sin(N*t))
    # plot & show
    fig = plt.figure(figsize=(6, 6*r_y /r_x))
    plt.plot(x, y, 'pink')
    plt.scatter(x, y, c=t, cmap="spring", s=2)
    plt.axis("equal")
    st.pyplot(fig)

def flower_eq2(a, n, d, n_patt):
    """
        Eq2: Rose (Rhodonea curves)
        a : amplitude
        n, d : rational parameters (k = n/d)
        n_patt : number of sample points
    """
    k = n / d
    t = np.linspace(0, 2*np.pi*d, n_patt)  # <-- important fix
    r = a * np.cos(k * t)
    x = r * np.cos(t)
    y = r * np.sin(t)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'violet')
    plt.scatter(x, y, c=t, cmap="spring", s=2)
    plt.axis("equal")
    st.pyplot(fig)

def flower_eq3(R, r, d, n_points=2000):
    """
    Eq3: Epicycloid / Hypocycloid (Spirograph-like flowers)

    Parameters
    ----------
    R : float
        Radius of fixed circle
    r : float
        Radius of rolling circle
    d : float
        Distance from rolling circle center to drawing point
    n_points : int
        Number of sample points
    """
    from math import gcd
    theta_max = 2 * np.pi * R // gcd(int(R), int(r))
    
    theta = np.linspace(0, theta_max, n_points)

    x = (R + r) * np.cos(theta) - d * np.cos((R + r) / r * theta)
    y = (R + r) * np.sin(theta) - d * np.sin((R + r) / r * theta)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(x, y, color="violet")
    plt.scatter(x, y, c=theta, cmap="plasma", s=1)
    plt.axis("equal")
    st.pyplot(fig)

def flower_eq4(inner_scale, out_radius, n_wings, delta=0.2, n_points=200):
    """
        Curvy-flower version: edges replaced by sinusoidal arcs.        
            inner_scale: ratio r/R
            out_radius: outer radius R
            n_wings: number of wings
            delta: curve strength
            n_points: resolution per edge
    """
    R = out_radius
    r = inner_scale * out_radius

    # angles
    if n_wings % 2:
        t = np.linspace(0, 2*np.pi, n_wings*2, endpoint=False)
    else:
        t = np.linspace(-np.pi, np.pi, n_wings*2, endpoint=False)

    # radii: alternate outer/inner
    radius = np.where(np.arange(len(t)) % 2 == 0, R, r)
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    # close polygon
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    Xc, Yc = [], []

    for i in range(len(x)-1):
        A = np.array([x[i], y[i]])
        B = np.array([x[i+1], y[i+1]])
        AB = B - A

        # normal vector (rotate AB by 90°)
        n = np.array([-AB[1], AB[0]])
        n = n / np.linalg.norm(n)

        u = np.linspace(0, 1, n_points)
        curve = (1-u)[:,None]*A + u[:,None]*B + delta*np.sin(np.pi*u)[:,None]*n

        Xc.extend(curve[:,0])
        Yc.extend(curve[:,1])

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(Xc, Yc, color="purple")
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig)

def flower_plot(sel_eq):
    c1, _, c2 = st.columns([15, 1, 10])
    if sel_eq == "type_1":        
        with c2:
            cleft, _, cright = st.columns([6,0.5,19])
            with cleft:
                core_tp = st.selectbox("`cosine` or `sine`",
                                        ["cosine", "sine"], help="read the description in the left")
                r_x     = st.number_input("`x_axes-radius`", 
                                            value = 0.6, min_value = 0.1, max_value = 0.99)
                r_y     = st.number_input("`y_axes-radius`", 
                                            value = 0.5, min_value = 0.1, max_value = 0.99 )
                N       = st.number_input("`angular frequency`",
                                            value = 5, min_value = 2, max_value = 99)
                n_patt  = st.number_input("`n-points`", 
                                            value = 1001,  min_value = 11, max_value = 9999)                
            with cright:
                flower_eq1(0, 0, r_x, r_y, N, n_patt, core_tp)
        with c1:
            st.write("This form is generated by")
            if core_tp == "cosine":
                st.latex(r"""
                            \begin{array}{ccl}
                                x &=& \cos t \cdot \left( x_0 + r_x * \cos(N*t) \right) \\
                                y &=& \sin t \cdot \left( y_0 + r_y * \cos(N*t) \right)
                            \end{array}
                          """)
            else:
                st.latex(r"""
                            \begin{array}{ccl}
                                x &=& \cos t \cdot \left( x_0 + r_x * \sin(N*t) \right) \\
                                y &=& \sin t \cdot \left( y_0 + r_y * \sin(N*t) \right)
                            \end{array}
                          """)
            st.write(r"""
                        where
                        - $t \in [0, 2\pi]$
                        - $x_0, y_0$ are the center's coordinate of the flower
                        - $r_x, r_y \in (0, 1)$ is the radius-ratio on x-axes and y-axes respectively
                        - $N$ controls the number of petals                   
                    """)
            st.warning(r"""
                       As $N$ is even, the number of petals is $2N$
                        - When the **core** is `sine`, the *petals* do not fall directly on the coordinate axes ($x = 0$, and $y = 0$); rather, they appear in a symmetric arrangement around them
                        - In other cases (here is `cosine`), four petals align exactly with the axes
                       
                       Otherwise, the number of petals is $N$
                       """)
            
    elif sel_eq == "type_2":
        with c2:
            cleft, _, cright = st.columns([6,0.5,19])
            with cleft:
                a      = st.number_input("amplitude of petal",
                                         value = 1.0, min_value = 0.1, max_value = 9.9)
                n      = st.number_input("`numerator of the frequency ratio`", 
                                         value = 6, min_value = 1, max_value = 99)
                d      = st.number_input("`denominator of the frequency ratio`", 
                                         value = 4, min_value = 1, max_value = 99 )
                n_patt = st.number_input("`n-points`",
                                         value = 99, min_value = 11, max_value = 9999)
            with cright:
                flower_eq2(a, n, d, n_patt)
        with c1:
            st.write("This form is generated by")
            st.latex(r"""
                        \begin{array}{l}
                            x &=& r \cdot \cos t 
                            & , &
                            y &=& r \cdot \sin t
                        \end{array}
                      """)
            st.write("where")
            st.latex(r" r \, = \, A \cdot \cos \left( k t \right)  ")
            st.write(r"""
                      and
                        - $k = \dfrac{n}{d}$ is a rational number which represent the `angular-speed` or `frequency ratio` 
                            
                            with ($n, d$ is the *numerator* and *denominator* of the frequency ratio)
                        - $t \in [0, 2d \pi]$
                        - $A$ limits the `amplitude` of petal
                      """)

    elif sel_eq == "type_3":
        with c2:
            cleft, _, cright = st.columns([6,0.5,19])
            with cleft:
                R      = st.number_input("radius of fixed circle",
                                         value = 8, min_value = 2, max_value = 99)
                r      = st.number_input("`radius of rolling circle`", 
                                         value = 3, min_value = 1, max_value = 99)
                d      = st.number_input("`d : distance`", 
                                         value = 7, min_value = 1, max_value = 99, 
                                         help = "Distance from rolling circle center to drawing point")
                n_patt = st.number_input("`Number of sample points`",
                                         value = 999, min_value = 11, max_value = 9999)
            with cright:
                flower_eq3(R, r, d, n_patt)
        with c1:
            st.write("This form is generated by")
            st.latex(r"""
                        \begin{array}{l}
                            \\ 
                            x &=& (R + r) \cdot \cos(\theta) - d \cdot \cos\left( \tfrac{R + r}{r} \cdot \theta \right)
                            \\ \\
                            y &=& (R + r) \cdot \sin(\theta) - d \cdot \sin\left( \tfrac{R + r}{r} \cdot \theta \right)
                        \end{array}
                    """)
            st.write(r"""
                        where,
                         - $R, r$ : radius of `fixed circle` and `rolling circle` respectively
                         - $d$ is the distance from rolling circle center to drawing point
                     """)

    elif sel_eq == "type_5":
        with c2:
            cleft, _, cright = st.columns([6,0.5,19])
            with cleft:
                number_of_levels = st.slider("Number of flower levels", 
                                            value=3, min_value=3, max_value=5)
                display_leaf = st.checkbox("Show leaves", value=True)
            with cright:
                flower_eq5(display_leaf, number_of_levels)

        with c1:
            st.write("This form is generated by")
            st.latex(r"""
                r(\theta) = R + A \cdot 
                \frac{|\cos(3\theta)| + 2\Big(0.25 - |\cos(3\theta + \tfrac{\pi}{2})|\Big)}
                {2 + 8|\cos(6\theta + \tfrac{\pi}{2})|}
            """)

    else:
        with c1:
            st.write(r"""
                     You can refer the idea how to plot the stars in this tab to understand how `out_radius, n_wings, inner_scale` works
                     
                     The *curvature coefficient* (δ) controls how much each edge bends. 
                    - δ = 0 → straight edges (classic star); remember to adjust `inner-scale` greater (than $0.1$) when $\delta = 0$ to see the `star` clearer
                    - δ > 0 → edges bulge outward.  
                    - δ < 0 → edges bend inward.  
                     
                    Larger |δ| → stronger curve.                     
                     """)
        with c2:
            cleft, _, cright = st.columns([6,0.5,19])
            with cleft:
                out_radius  = st.number_input("outer_radius",
                                         value = 2, min_value = 2, max_value = 99)
                n_wings     = st.number_input("`n_wings`", 
                                         value = 5, min_value = 1, max_value = 99)
                delta       = st.number_input("`delta`", 
                                         value = 0.4, min_value = 0.0, max_value = 0.9, 
                                         help = "Distance from rolling circle center to drawing point")
                inner_scale = st.number_input("`inner_scale`",
                                         value = 0.005, min_value = 0.001, max_value = 0.2)
            with cright:
                flower_eq4(inner_scale, out_radius, n_wings, delta)

def flower_eq5(display_leaf=True, number_of_levels=3):
    pi = np.pi
    theta = np.linspace(-pi, pi, 10001)
    fig = plt.figure(figsize=(5, 5))

    # ---------- Core (fixed) ----------
    core_R = [0.9, 0.95, 1]
    core_colors = ["magenta", "violet", "red"]

    # ---------- Leaves (optional) ----------
    leaf_R = [1.5, 1.52, 1.54, 1.57]
    leaf_colors = ["#7fff00", "green", "#00ff00", "darkgreen"]

    # ---------- Flower (variable levels) ----------
    if number_of_levels == 3:
        flower_colors = ["#ff0090", "#cf3476", "purple"]
        step = 0.05
    elif number_of_levels == 4:
        flower_colors = ["#f4bbff", "#ff0090", "#cf3476", "purple"]
        step = 0.04
    elif number_of_levels == 5:
        flower_colors = ["#f4bbff", "#fe4eda", "#ff0090", "#cf3476", "purple"]
        step = 0.03
    else:
        raise ValueError("number_of_levels must be 3, 4, or 5")

    flower_R = [2 + i*step for i in range(number_of_levels)]

    # ---------- Plot Core ----------
    for R, col in zip(core_R, core_colors):
        r = R + (np.abs(np.cos(3*theta)) 
                 + 2*(0.25 - np.abs(np.cos(3*theta + pi/2)))) \
                 / (2 + 8 * np.abs(np.cos(6*theta + pi/2)))
        plt.plot(r*np.cos(theta), r*np.sin(theta), col)

    # ---------- Plot Leaves ----------
    if display_leaf:
        for R, col in zip(leaf_R, leaf_colors):
            r = R + (np.abs(np.cos(3*theta)) 
                     + 2*(0.25 - np.abs(np.cos(3*theta + pi/2)))) \
                     / (2 + 8 * np.abs(np.cos(6*theta + pi/2)))
            plt.plot(r*np.cos(theta), r*np.sin(theta), col)

    # ---------- Plot Flower ----------
    for R, col in zip(flower_R, flower_colors):
        r = R + (np.abs(np.cos(3*theta)) 
                 + 2*(0.25 - np.abs(np.cos(3*theta + pi/2)))) \
                 / (2 + 8 * np.abs(np.cos(6*theta + pi/2)))
        plt.plot(r*np.cos(theta), r*np.sin(theta), col)

    plt.axis("equal")
    plt.axis("off")
    st.pyplot(fig)

# -------------------------- CARD-SUITS ---------------------------------- #
def diamond_curves(diamond_rt=2/3):
    y_max = 1.1 * (2 / diamond_rt) 
    x = np.linspace(-3, 3, 1001)
    y = np.linspace(-y_max, y_max, 1001)
    X, Y = np.meshgrid(x, y)
    scale = 2 / np.log(2)
    R = 2.75   # "radius" parameter, adjust to resize diamond
    F = scale * (np.log(np.abs(X) + diamond_rt) + np.log(np.abs(Y) + 1)) - R

    # plot & show
    fig = plt.figure()
    plt.contour(X, Y, F, levels=[0], colors="purple")
    plt.gca().set_aspect("equal")
    st.pyplot(fig)

def club_curves(alpha = 0.15, y_bottom = -1.9, y_top = 3.75, w0 = 0.3):
    """
        alpha (in [0, 1]): widening factor
        y_bottom : bottom base
        y_top : position which connect to the leaf
        w0 : half-width at top    
    """
    x = np.linspace(-3, 3, 1001)
    y = np.linspace(-2.05, 3, 1001)
    X, Y = np.meshgrid(x, y)
    F = np.minimum.reduce([
        X**2 + (Y-1.5)**2 - 1,    # top circle
        (X+1.2)**2 + Y**2 - 1,    # right circle
        (X-1.2)**2 + Y**2 - 1     # left circle
    ])

    # Stem trapezoid for CLUB: width at each Y
    stem_width = w0 + alpha * (Y - y_top)

    # inside condition: |X| < stem_width,  y between [y_bottom, y_top]
    cond_x = np.abs(X) + stem_width
    cond_y_top = Y - y_top
    cond_y_bottom = y_bottom - Y

    stem = np.maximum.reduce([cond_x, cond_y_top, cond_y_bottom])
    F = np.minimum(F, stem)

    # plot & show
    fig = plt.figure(figsize=(6, 6))
    plt.contour(X, Y, F, levels=[0], colors="purple")
    st.pyplot(fig)

def spade_curves(alpha = 0.25, y_bottom = -3.75, y_top = -.25, w0 = 0.3):
    x = np.linspace(-3, 3, 1001)
    y = np.linspace(-4, 2.2, 1001)
    X, Y = np.meshgrid(x, y)
    F = X**2 + (Y + np.abs(X)**0.5)**2 - 5
    
    # Stem trapezoid for CLUB : width at each Y
    stem_width = w0 + alpha * (Y - y_top)

    # inside condition: |X| < stem_width,  y between [y_bottom, y_top]
    cond_x = np.abs(X) + stem_width
    cond_y_top = Y - y_top
    cond_y_bottom = y_bottom - Y

    stem = np.maximum.reduce([cond_x, cond_y_top, cond_y_bottom])
    F = np.minimum(F, stem)

    # plot & show
    fig = plt.figure(figsize=(6, 6))
    plt.contour(X, Y, F, levels=[0], colors="purple")
    st.pyplot(fig)

def card_suits_plot(sel_eq):
    c1, _, c2 = st.columns([15, 1, 10])

    if sel_eq == "heart":
        with c2:
            c21, _, c22 = st.columns([5, 1, 17])
            with c21:
                n_deg = st.number_input("Select the degree", value=3, min_value=3, max_value=15)
                n_patt = st.number_input("Select the number-of-points", value=99, min_value=21, max_value=9999)
            with c22:
                heart_eq6(n_deg, n_patt)
        with c1:
            st.success("You can select section `heart` to see more information")
            c11, c12, c13, c14 = st.columns(4)
            with c11:
                st.write("**HEART**")
                heart_eq6(5, 999)
            with c12:
                st.write("**DIAMOND**")
                diamond_curves(diamond_rt=0.72)
            with c13:
                st.write("**CLUB**")
                club_curves()
            with c14:
                st.write("**SPADE**")
                spade_curves()

    elif sel_eq == "diamond":
        with c2:
            c21, _, c22 = st.columns([5, 1, 17])
            with c21:
                st.write("")
                diamond_rt = st.number_input("Select `diamond_rt`",
                                             value=1/3, min_value=0.1, max_value=0.99)
            with c22:
                diamond_curves(diamond_rt)
        with c1:
            st.write("The `diamond-curve` comes from")
            st.latex(r"""
                     \begin{array}{ccl} 
                         F(x,y) &=& \dfrac{2}{\ln 2} \, \big( \ln(|x| + r ) \; + \; \ln(|y|+1) \big) - R \\
                     \end{array}
                     """)
            st.write(r"""                    
                    where
                     - $r \in (0, 1)$ meant `diamond_rt`; adjusts curvature offset — ↓ sharper points, ↑ rounder shape
                     - $y \in [-1, 1]$ and $x \in \left[ -r^{-1}, r^{-1} \right]$

                     In this simulation, I restrict the range of $r$ into $\left[ \tfrac{1}{3}, 1 \right)$ to have the best intuition result
                    """)

    elif sel_eq == "club":
        with c2:
            c21, _, c22 = st.columns([5, 1, 17])
            with c21:
                st.write("")
                alpha    = st.number_input("Select `alpha`",
                                            value=0.15, min_value=0.1, max_value=0.99)
                y_bottom = st.number_input("Select `alpha`",
                                           value=-1.9, min_value=-9.1, max_value=9.99)
                y_top    = st.number_input("Select `alpha`",
                                             value=3.75, min_value=0.1, max_value=9.99)
                w0       = st.number_input("Select `w0`",
                                             value=0.3, min_value=0.1, max_value=0.99)                
            with c22:
                club_curves(alpha, y_bottom, y_top, w0)
        with c1:
            st.write(r"Firstly, the range of the `club` above is denoted $F_u$, defined by")
            st.latex(r"""
            F_u(x, y) = 
                \min \left\{
                    \underbrace{x^2 + (y - 1.5)^2 - 1}_{\text{top circle}} \, , \;
                    \underbrace{(x \pm 1.2)^2 + y^2 - 1}_{\text{right \& left circle}}
                \right\}
            """)
            st.write("and the bottom is the `trapezoid`, which limited in")
            st.latex(r" |x| \le w_0 + \alpha \,(y - y_{top}) ")
            st.write(r"""
                        where
                        - `\alpha`: stem widening (larger → wider base)  
                        - `y_{top}, y_{bottom}`: stem top / stem base positions  
                        - `w_0`: half-width at stem top  
                    """)
    
    else:
        with c2:
            c21, _, c22 = st.columns([5, 1, 17])
            with c21:
                st.write("")
                alpha    = st.number_input("Select `alpha`",
                                            value = 0.25, min_value=0.1, max_value=0.99)
                y_bottom = st.number_input("Select `alpha`",
                                           value = -3.75, min_value=-9.1, max_value=9.99)
                y_top    = st.number_input("Select `alpha`",
                                             value = -0.25, min_value=-0.9, max_value=9.99)
                w0       = st.number_input("Select `w0`",
                                             value = 0.3, min_value=0.1, max_value=0.99)                
            with c22:
                spade_curves(alpha, y_bottom, y_top, w0)
        with c1:
            st.write(r"Firstly, the range of the `club` above is denoted $F_u$, defined by")
            st.latex(r"""
                     \begin{array}{ccl} 
                         \\
                         F_u(x, y) &=& x^2 \, + \, \left( y + \sqrt{ \vert x \vert } \right)^2 \, - \, 5
                     \end{array}
                     """)
            st.write("and the bottom is the `trapezoid` by the same method as in the `club`")

# ----------------------------- STARS ------------------------------------ #
def stars_eq1(inner_scale, out_radius, n_wings):
    R = out_radius
    r = inner_scale * out_radius

    if n_wings % 2:
        t = np.linspace(0, 2*np.pi, n_wings*2, endpoint=False)
    else:
        t = np.linspace(-np.pi, np.pi, n_wings*2, endpoint=False)

    radius = np.where(np.arange(len(t)) % 2 == 0, R, r)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    
    # CLOSE the polygon by repeating the first point
    x_closed = np.concatenate([x, x[:1]])
    y_closed = np.concatenate([y, y[:1]])

    fig, ax = plt.subplots(figsize=(5,5))
    poly = Polygon(np.column_stack((x, y)), closed=True,
                facecolor='violet', edgecolor='purple', alpha=0.3)
    ax.add_patch(poly)
    ax.plot(x_closed, y_closed, marker='o', color='purple', linestyle='-')  # markers + outline
    ax.set_aspect("equal")
    ax.axis("off")

    st.pyplot(fig)

def stars_plot(sel_eq):
    c1, _, c2 = st.columns([15, 1, 10])    
    with c2:
        # interactive inputs
        c21, _, c22 = st.columns([6, 0.1, 9])
        with c21:
            inner_scale = st.number_input(
                "Inner scale (ratio `r`/`R`)", value=0.5, min_value=0.1, max_value=0.95, step=0.05
            )
            out_radius = st.number_input(
                "Outer radius", value=2.0, min_value=0.5, max_value=5.0, step=0.1
            )
            n_wings = st.number_input(
                "Number of wings", value=5, min_value=3, max_value=20, step=1
            )
        with c22:
            stars_eq1(inner_scale, out_radius, n_wings)
    with c1:
        st.write("To construct this type of star (straight-line stars), we use `polar coordinates`:")
        st.latex(r" x \, = \, r^{*} \cos (t) \; , \quad y \, = \, r^{*} \sin (t) ")
        st.write("where")
        st.latex(r"""
                    r^{*} \, = \, \left \lbrace \begin{array}{crl}
                                    r & , & \text{ even index for outer vertices } \\
                                    R & , & \text{ otherwise }
                                \end{array} \right.                      
                    """)
        st.write("")
        c11, c12 = st.columns([3, 2])
        with c11:
            st.write(r"""
                        and,
                        - $R, r$ respectively the outer / inner radius of the stars
                        - $n_w$ is number of wings
                    """)
        with c12:
            st.write(r"""
                        Beside that,
                        - As $n_w$ is even, $t \in \left[ 0, 2\pi \right]$
                        - As $n_w$ is odd, $t \in \left[ -\pi, \pi \right]$
                        """)                

# ---------------------------- LOGOs ------------------------------------ #
def window_plot(facecolor_, edgecolor_, 
                vert_line_width=8, horz_line_width=8, lean_flag=False, lean_offset=40):
    """
    Draws a window with adjustable vertical/horizontal line thickness
    and an optional leaning (as trapezoid instead of rectangle).

    Parameters
    ----------
    vert_line_width : float
        Thickness of vertical line.
    horz_line_width : float
        Thickness of horizontal line.
    lean_flag : bool
        If True, draw a trapezoid (leaning window).
    lean_offset : float
        Horizontal shift applied to top side (controls the leaning).
    """

    # Create a figure and axes
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    color_ = "black"

    if not lean_flag:
        # === Normal Rectangle Window ===
        rect = patches.Rectangle((-50, -100), 150, 200, 
                                 facecolor = facecolor_, edgecolor = edgecolor_)
        ax.add_patch(rect)

        # Vertical line
        ax.plot([15, 15], [-100, 100], color = color_, linewidth=vert_line_width)

        # Horizontal line
        ax.plot([-50, 100], [0, 0], color = color_, linewidth=horz_line_width)

    else:
        # === Leaning Trapezoid Window ===
        # Bottom coordinates
        x1, y1 = -50, -100
        x2, y2 = 100, -100
        # Top coordinates (shifted by lean_offset)
        x3, y3 = 100 + lean_offset, 100
        x4, y4 = -50 + lean_offset, 100

        trapezoid = patches.Polygon(
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
            closed = True, facecolor = facecolor_, edgecolor = edgecolor_
        )
        ax.add_patch(trapezoid)

        # Vertical "split" line (connect midpoint bottom to midpoint top)
        xb_bot = (x1 + x2)/2
        yb_bot = y1
        xt_top = (x3 + x4)/2
        yt_top = y3
        ax.plot([xb_bot, xt_top], [yb_bot, yt_top], 
                color = color_, linewidth = vert_line_width)

        # Horizontal "split" line (connect left-mid to right-mid)
        xl_left = (x1 + x4)/2
        yl_left = (y1 + y4)/2
        xr_right = (x2 + x3)/2
        yr_right = (y2 + y3)/2
        ax.plot([xl_left, xr_right], [yl_left, yr_right], 
                color = color_, linewidth = horz_line_width)

    # Set limits to keep view stable
    ax.set_xlim(-150, 250)
    ax.set_ylim(-150, 150)
    ax.axis("off")

    # Show in Streamlit
    st.pyplot(fig)

class TurtleSim:
    def __init__(self, x=0.0, y=0.0, heading_deg=0.0):
        self.x = float(x)
        self.y = float(y)
        self.h = float(heading_deg)  # heading in degrees, 0 = to the right (east)
        self.pen = True
        self.filling = False
        self.current_fill = []
        self.filled = []  # list of (coords_list, fillcolor)
        self.fillcolor = "#000000"
        self.pencolor = "black"
        self.dots = []  # list of (x,y,diam,color)

    # state-control
    def penup(self):
        self.pen = False

    def pendown(self):
        self.pen = True

    def set_fillcolor(self, color):
        self.fillcolor = color

    def set_pencolor(self, color):
        self.pencolor = color

    def begin_fill(self):
        self.filling = True
        # start polygon from current position
        self.current_fill = [(self.x, self.y)]

    def end_fill(self):
        if self.filling and len(self.current_fill) > 0:
            self.filled.append((self.current_fill.copy(), self.fillcolor))
        self.filling = False
        self.current_fill = []

    def dot(self, diameter):
        # record a dot at current position (diameter is turtle 'pixels' — we treat as units)
        self.dots.append((self.x, self.y, float(diameter), self.pencolor))

    # motions
    def forward(self, dist):
        # move forward by dist in direction h
        rad = math.radians(self.h)
        self.x += dist * math.cos(rad)
        self.y += dist * math.sin(rad)
        if self.filling:
            self.current_fill.append((self.x, self.y))

    def left(self, angle_deg):
        self.h = (self.h + angle_deg) % 360

    def right(self, angle_deg):
        self.h = (self.h - angle_deg) % 360

    def position(self):
        return (self.x, self.y, self.h)

def s_curve(sim: TurtleSim):
    # 90 small left(1)+forward(1) steps
    for _ in range(90):
        sim.left(1)
        sim.forward(1)

def r_curve(sim: TurtleSim):
    for _ in range(90):
        sim.right(1)
        sim.forward(1)

def l_curve(sim: TurtleSim):
    s_curve(sim)
    sim.forward(80)
    s_curve(sim)

def l_curve1(sim: TurtleSim):
    s_curve(sim)
    sim.forward(90)
    s_curve(sim)

def half(sim: TurtleSim):
    sim.forward(50)
    s_curve(sim)
    sim.forward(90)
    l_curve(sim)
    sim.forward(40)
    sim.left(90)
    sim.forward(80)
    sim.right(90)
    sim.forward(10)
    sim.right(90)
    sim.forward(120)  # on test
    l_curve1(sim)
    sim.forward(30)
    sim.left(90)
    sim.forward(50)
    r_curve(sim)
    sim.forward(40)
    # end_fill should be called by caller (as in original script)

def get_pos(sim: TurtleSim):
    sim.penup()
    sim.forward(20)
    sim.right(90)
    sim.forward(10)
    sim.right(90)
    sim.pendown()

def eye(sim: TurtleSim):
    sim.penup()
    sim.right(90)
    sim.forward(160)
    sim.left(90)
    sim.forward(70)
    sim.set_pencolor("black")
    sim.dot(35)
    # position doesn't get restored (same as original turtle script)

def sec_dot(sim: TurtleSim):
    sim.left(90)
    sim.penup()
    sim.forward(310)
    sim.left(90)
    sim.forward(120)
    sim.pendown()
    sim.dot(35)

# ---------- Simulation runner (reproduces the order in your turtle file) ----------
def run_turtle_script(sim: TurtleSim):
    # First half (blue)
    sim.set_fillcolor("#306998")
    sim.begin_fill()
    half(sim)
    sim.end_fill()

    # reposition like get_pos()
    get_pos(sim)

    # Second half (yellow)
    sim.set_fillcolor("#FFD43B")
    sim.begin_fill()
    half(sim)
    sim.end_fill()

    # draw dots
    eye(sim)
    sec_dot(sim)

    # small "pause" loop in original does rotation; ignore visual animation
    # (we don't reproduce the pause because Streamlit shows a static image)

    return sim

def plot_simulation(sim: TurtleSim, show_outline=False, figsize=6):
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_facecolor("white")  # you can change to black if you want that background
    ax.set_aspect("equal")

    # Draw filled polygons (in order)
    for poly_pts, color in sim.filled:
        poly = patches.Polygon(poly_pts, closed=True, facecolor=color, edgecolor='none')
        ax.add_patch(poly)
        if show_outline:
            ax.add_patch(patches.Polygon(poly_pts, closed=True, facecolor='none', edgecolor='k', linewidth=0.5))

    # Draw dots
    for (x, y, diam, color) in sim.dots:
        # turtle.dot size historically measured in pixels; we treat diam as data-units
        circ = patches.Circle((x, y), radius=diam/2.0, facecolor=color, edgecolor='none')
        ax.add_patch(circ)

    # Autoscale view with margin
    all_x = []
    all_y = []
    for poly_pts, _ in sim.filled:
        for (px, py) in poly_pts:
            all_x.append(px); all_y.append(py)
    for (x, y, _, _) in sim.dots:
        all_x.append(x); all_y.append(y)

    if len(all_x) == 0:
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
    else:
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        dx = maxx - minx
        dy = maxy - miny
        pad = max(dx, dy) * 0.12 + 10
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

    ax.axis("off")
    plt.tight_layout()
    return fig, ax

def python_plot(bg_color, show_outline, figsize_):
    sim = TurtleSim(x=0.0, y=0.0, heading_deg=0.0)  # same initial state as turtle
    sim = run_turtle_script(sim)
    # Plot
    fig, ax = plot_simulation(sim, show_outline=show_outline, figsize=figsize_)
    if bg_color == "black":
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

    st.pyplot(fig)

def google_plot(theta1_w1=50, theta2_w3=360):
    # Create figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.set_facecolor("white")  # background

    # === Google Chrome colors ===
    blue = "#4285F4"
    green = "#0F9D58"
    yellow = "#F4B400"
    red = "#DB4437"

    # Outer circle wedges
    # total span
    delta = theta2_w3 - theta1_w1
    step = delta / 3

    # Wedges
    w1 = patches.Wedge(center=(0,0), r=150,
                    theta1=theta1_w1,
                    theta2=theta1_w1 + step,
                    facecolor=red, edgecolor="none")

    w2 = patches.Wedge(center=(0,0), r=150,
                    theta1=theta1_w1 + step,
                    theta2=theta1_w1 + 2*step,
                    facecolor=yellow, edgecolor="none")

    w3 = patches.Wedge(center=(0,0), r=150,
                    theta1=theta1_w1 + 2*step,
                    theta2=theta2_w3,
                    facecolor=green, edgecolor="none")

    # Inner white circle (to create the gap)
    inner_white = patches.Circle((0,0), radius=110, facecolor="white", edgecolor="none")
    cutout = patches.Rectangle((45, -20),   # start point (x,y)
                            110, 40,     # width, height
                            facecolor="white", edgecolor="none")
    ax.add_patch(cutout)

    # Inner blue circle
    inner_blue = patches.Circle((0,0), radius=45, facecolor=blue, edgecolor="none")

    # Add patches
    ax.add_patch(w1)
    ax.add_patch(w2)
    ax.add_patch(w3)
    ax.add_patch(inner_white)
    ax.add_patch(inner_blue)

    # === Horizontal bar ===
    theta_end = np.deg2rad(theta2_w3)
    x_end = 150 * np.cos(theta_end)
    y_end = 150 * np.sin(theta_end)

    # Draw horizontal bar aligned with the wedge end
    ax.hlines(y=y_end, xmin=60, xmax=x_end, colors=green, linewidth=10)
    ax.hlines(y=y_end, xmin=60, xmax=x_end-1, colors=green, linewidth=20)
    ax.hlines(y=y_end, xmin=60, xmax=x_end-2, colors=green, linewidth=30)

    # Remove axes
    ax.set_xlim(-160,160)
    ax.set_ylim(-160,160)
    ax.axis("off")

    # display in streamlit
    st.pyplot(fig)

def logo_plot(sel_eq):
    c1, _, c2 = st.columns([15, 1, 10])
    if sel_eq == "Window":
        with c2:
            cl, _, cr = st.columns([5,1,5])
            with cl:
                vert_line_width = st.number_input("width of the vertical-line", 
                                    value=8, min_value=1, max_value=100, 
                                    help="Thickness of vertical divider")
            with cr:
                horz_line_width = st.number_input("width of the horizontal-line", 
                                                  value=8, min_value=1, max_value=100, 
                                                  help="Thickness of horizontal divider")            
            c21, _, c22 = st.columns([3, 0.2, 9])
            with c21:
                st.write(" ")
                lean_flag       = st.checkbox("Leaning window?", value=False, 
                                              help="Check to make trapezoid instead of rectangle")
                lean_offset     = st.slider("Leaning offset", 
                                            value=40, min_value=-100, max_value=100, 
                                            help="Horizontal shift at top side when leaning")
                facecolor_      = st.selectbox("face_color",
                                               ["red", "green", "blue", "magenta", "purple"])
                edgecolor_      = st.selectbox("edge_color",
                                               ["red", "green", "blue", "magenta", "purple"])
            with c22:
                window_plot(facecolor_, edgecolor_, vert_line_width, horz_line_width, lean_flag, lean_offset)
        with c1:
            st.write(r"""
                     Idea:
                     - When `Leaning-flag` is choosen, the window is created by adding the horizontal-line and vertical-line 
                        inside the given **Trapezoid**:
                        
                        $ \qquad T = \operatorname{Conv}\{(-50,-100),\,(100,-100),\,(100+\Delta,100),\,(-50+\Delta,100)\} $
                        
                        where $\Delta$ = `lean_offset`.
                     
                        The vertical and horizontal lines are drawn by connecting midpoints of opposite edges.
                                          
                     - Otherwise, the window is created by adding the horizontal-line and vertical-line 
                        inside the given **Rectangle**:

                        $\qquad R = \{(x,y) \mid -50 \leq x \leq 100,\; -100 \leq y \leq 100\}$

                        with the cross formed at its geometric center.
                     """)

    elif sel_eq == "Python":
        with c2:
            c21, _, c22 = st.columns([3, 0.2, 9])
            with c21:      
                bg_color = st.selectbox("Background color", options=["white", "black"], index=0)
                show_outline = st.checkbox("Show polygon outlines", value=False)
            with c22:
                python_plot(bg_color, show_outline, figsize_ = 6)
    
    else:
        with c2:
            c21, _, c22 = st.columns([3, 0.2, 9])
            with c21:
                theta1_w1 = st.number_input("Theta1 (red wedge start)", 
                                            value=50, min_value=30, max_value=90)
                theta2_w3 = st.number_input("Theta2 (green wedge end)", 
                                            value=360, min_value=300, max_value=390)
            with c22:
                google_plot(theta1_w1, theta2_w3)

# ---------------------------- LAUNCH ------------------------------------ #
def graph_plot(sel_graph, sel_eq):
    if sel_graph == "heart":
        heart_plot(sel_eq)
    elif sel_graph == "batman-logo":
        batman_plot(sel_eq)
    elif sel_graph == "spiral":
        sprial_plot(sel_eq)
    elif sel_graph == "flower":
        flower_plot(sel_eq)
    elif sel_graph == "card-suits":
        card_suits_plot(sel_eq)
    elif sel_graph == "stars":
        stars_plot(sel_eq)
    else:
        logo_plot(sel_eq)

def run():
    c1, _, c2 = st.columns([15, 1, 10])
    with c1:
        sel_graph = st.selectbox("Select graph", # move card-suits to first
                                 ["card-suits", "heart", "batman-logo", "spiral", "flower",  
                                  "stars", "other-wellknown-logo"])
    with c2:
        if sel_graph == "heart":
            sel_eq = st.selectbox("Select equation", ["type_1", "type_2", "type_3",
                                                      "type_4", "type_5", "type_6"
                                                      ])
        elif sel_graph == "batman-logo":
            sel_eq = st.selectbox("Select equation", ["type_1", "type_2"])            
        elif sel_graph == "spiral":
            sel_eq = st.selectbox("Select equation", ["type_1", "type_2"])
        elif sel_graph == "flower":
            sel_eq = st.selectbox("Select equation", ["type_1", "type_2", "type_3", "type_4", "type_5"])
        elif sel_graph == "card-suits":
            sel_eq = st.selectbox("Select suits", ["heart", "diamond", "club", "spade"])
        elif sel_graph == "stars":
            sel_eq = st.selectbox("Select equation", ["type_1"])
        else:
            sel_eq = st.selectbox("Select an icon", ["Window", "Python", "Google"])

    graph_plot(sel_graph, sel_eq)