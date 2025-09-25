import time
import math
import inspect
import numpy as np
import streamlit as st
import pandas as pd
import streamlit_flow as sfl
import matplotlib.pyplot as plt
from collections import defaultdict
# =========================================================================================================
#--------------------------------------------- Decoration -------------------------------------------------
# .........................................................................................................
def nums_render_wrt_background(bg_img):
    st.markdown(f"""
        <style>
        .sequences {{
            background-image: url("data:image/jpg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }}
        </style>
        <div class="sequences">
            <h2 style="color: #003366;">Funny Serial & Sequence 🔢</h2>
            <p>Explore Fibonacci sequence, Smith numbers, Collatz conjecture…</p>
        </div>
    """, unsafe_allow_html=True)

def streamlit_format_text_to_cols_(nums_ls):
    """
        Pretty-print a list of numbers into aligned columns in Streamlit.
        Args:
            nums_ls (list[int]): list of numbers
    """
    # Handle empty list
    if not nums_ls:
        st.text("(empty list)")
        return

    try:
        cols_per_row = cal_cols_per_row(nums_ls[-1])
        if cols_per_row <= 0:
            st.text("(invalid column width)")
            return
    except Exception as e:
        st.text(f"(error calculating cols_per_row: {e})")
        return

    formatted_rows = []
    for i in range(0, len(nums_ls), cols_per_row):
        row = "\t\t".join(str(x) for x in nums_ls[i : i + cols_per_row])
        formatted_rows.append(f"\t{row}")  # indent each row with one tab

    output_text = "\n".join(formatted_rows)
    st.text(output_text)

def cal_cols_per_row(n_max):
    """
        Calculate number of columns per row depending on max digit length.
        Args:
            n_max (int): maximum number in the list
    """
    n_digits_max = len(str(abs(int(n_max))))  # safe: works if n_max is negative too
    if n_digits_max == 0:
        return 1
    return max(1, 1 + 36 // n_digits_max)

def streamlit_allocate_text_to_cols(data_ls, mode="tuple"):
    """
    Allocate list elements into columns in Streamlit.

    Args:
        data_ls (list): list of tuples (e.g. [(3, 5), (5, 7), ...]) 
                        or list of numbers (e.g. [4, 22, 58, 94]).
        mode (str): "tuple" for list of tuples, "num" for list of numbers.
    """
    if not data_ls:
        st.write("(empty list)")
        return

    if mode == "tuple":
        last_item = str(data_ls[-1])
        # get number of digits of the last element in the tuple
        max_digits = len(last_item.split(", ")[-1]) - 1
    elif mode == "num":
        max_digits = len(str(data_ls[-1]))
    else:
        raise ValueError("mode must be 'tuple' or 'num'")

    n_cols = max(1, 25 // max_digits)  # ensure at least 1 column
    cols = st.columns(n_cols)

    for idx, val in enumerate(data_ls):
        col_order = idx % n_cols
        with cols[col_order]:
            st.write(val)

# ========================================================================================================
#--------------------------------------------- Sequences -------------------------------------------------
# ===========================================  Fibonacci =================================================
def Fibonacci_fact():
    nodes = [
        sfl.elements.StreamlitFlowNode('1', (20, 100), {'content': 'F(n)'},
                                       'input', 'right', draggable=False,
                                       style={'color': 'white', 'backgroundColor': '#00c04b', 'border': '2px solid white'}),

        # intermediate nodes: use 'default' and pass ONE position (positional) like the root
        sfl.elements.StreamlitFlowNode('2', (150, 25), {'content': 'F(n-1)'}, 'default', 'right', draggable=False),
        sfl.elements.StreamlitFlowNode('3', (150, 175), {'content': 'F(n-2)'}, 'default', 'right', draggable=False),

        # layer 3 (same pattern)
        sfl.elements.StreamlitFlowNode('4', (300, 25), {'content': 'F(n-2)'}, 'default', 'right', draggable=False,
                                       style={'fontSize': '8px', 'padding': 0, 'width': '40px'}),
        sfl.elements.StreamlitFlowNode('5', (300, 85), {'content': 'F(n-3)'}, 'default', 'right', draggable=False,
                                       style={'fontSize': '8px', 'padding': 0, 'width': '40px'}),
        sfl.elements.StreamlitFlowNode('6', (300, 165), {'content': 'F(n-3)'}, 'default', 'right', draggable=False,
                                       style={'fontSize': '8px', 'padding': 0, 'width': '40px'}),
        sfl.elements.StreamlitFlowNode('7', (300, 205), {'content': 'F(n-4)'}, 'default', 'right', draggable=False,
                                       style={'fontSize': '8px', 'padding': 0, 'width': '40px'}),
    ]

    edges = [
        sfl.elements.StreamlitFlowEdge('1-2', '1', '2', animated=True, label="+", label_show_bg=True,
                                       label_bg_style={'stroke': 'red', 'fill': 'gray'}),
        sfl.elements.StreamlitFlowEdge('1-3', '1', '3', animated=True, label="+", label_show_bg=True),
        sfl.elements.StreamlitFlowEdge('2-4', '2', '4', animated=True),
        sfl.elements.StreamlitFlowEdge('2-5', '2', '5', animated=True),
        sfl.elements.StreamlitFlowEdge('3-6', '3', '6', animated=True),
        sfl.elements.StreamlitFlowEdge('3-7', '3', '7', animated=True),
    ]

    st.session_state.custom_styles_state = sfl.state.StreamlitFlowState(nodes, edges)
    sfl.streamlit_flow('flow_a', st.session_state.custom_styles_state,
                       fit_view=True, show_minimap=False, show_controls=False,
                       pan_on_drag=False, allow_zoom=False)

class Fibo:
    def using_ordinary_recursive(self, a: float, b: float, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be greater than 0") 
        elif n == 1:
            return a
        elif n == 2:
            return b
        else:
            return self.using_ordinary_recursive(a, b, n - 1) + self.using_ordinary_recursive(a, b, n - 2)
        
    def Top_Down_DP(self, a: float, b: float, n: int):
        memo = {1: a, 2: b}
        def fibo(a, b, n):
            if n in memo:
                return memo[n]
            memo[n] = fibo(a, b, n - 1) + fibo(a, b, n - 2)
            return memo[n]
        
        return fibo(a, b, n)
    
    def Bottom_Up_DP_Fibo(self, a: float, b: float, n: int): # Tabulation
        dp = [a, b]
        for idx in range(2, n):
            new = dp[idx - 1] + dp[idx - 2]
            dp.append(new)
        return dp[-1]

    def Bottom_Up_no_memo_DP_Fibo(self, a: float, b: float, n: int): # Memory space: O(1)
        if n <= 0:
            raise ValueError("n must be greater than 0")
        if n == 1:
            return a
        if n == 2:
            return b        
        prev, curr = a, b
        for i in range(3, n + 1):
            prev, curr = curr, prev + curr
        return curr

def Fibo_seq():
    with st.expander("Illustration"):
        st.write("We know")
        st.latex(r" F(n) = F(n-1) + F(n-2)")
        Fibonacci_fact()

def Fibo_deploy():    
    c1, c2 = st.columns([3, 2])
    with c1:
        st.write("### `Simulation`")
        c11, c12, c13 = st.columns(3)
        with c11:
            F0 = st.number_input("a := F(0) = ", value=1, min_value=-1, max_value=99999999)
        with c12:
            F1 = st.number_input("b := F(1) = ", value=1, min_value=-1, max_value=99999999)
        with c13:
            n  = st.number_input("n = ", value=10, min_value=1, max_value=9999)  # keep limit small for recursive
        
        my_experiment = Fibo()

        # Run experiments
        results = {}
        methods = {
            "Ordinary Recursive": my_experiment.using_ordinary_recursive,
            "Top-Down DP": my_experiment.Top_Down_DP,
            "Bottom-Up DP": my_experiment.Bottom_Up_DP_Fibo,
            "Bottom-Up O(1)": my_experiment.Bottom_Up_no_memo_DP_Fibo,
        }

        for name, func in methods.items():
            try:
                start = time.perf_counter()
                res = func(F0, F1, n)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                results[name] = [F0, F1, n, res, f"{elapsed:.3f} ms"]
            except Exception as e:
                results[name] = [F0, F1, n, str(e), "-"]

        df = pd.DataFrame.from_dict(
            results,
            orient="index",
            columns=["F0", "F1", "n", "Result", "Time"]
        )

        st.dataframe(df)

    with c2:
        Fibo_seq()
# ========================================================================================================
# ===========================================  Sylvester =================================================
def Sylvester_gen(num):    
    sylvester_seqs = [0] * (num + 1)
    sylvester_seqs[0] = 2
    for deg in range(1, num):
        sylvester_seqs[deg] = sylvester_seqs[deg - 1] * (sylvester_seqs[deg - 1] - 1) + 1
    sylvester_seqs[-1] = int(np.ceil(1 / (1 - sum([ 1/x for x in sylvester_seqs[:num] ]))))
    return sylvester_seqs
    
def sylvester_show(levels=2):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["lightblue", "salmon", "#98fb98", "violet", "gold"]
    sylvester_seqs = Sylvester_gen(levels)    
    y_offset = 1.0
    for k in range(levels+1):
        n_cells = sylvester_seqs[k]     # số ô vuông ở hàng k
        side = 1 / n_cells              # độ rộng/cao mỗi ô vuông
        y = y_offset                    # vị trí hàng hiện tại

        for i in range(n_cells):
            square = plt.Rectangle((i*side, y), side, -side,
                                   facecolor=colors[k % len(colors)],
                                   edgecolor="black")
            ax.add_patch(square)

        y_offset -= side   # hạ xuống cho hàng tiếp theo

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    st.pyplot(fig)

def Sylvester_deploy():
    c1, _, c2, = st.columns([9, 1, 9])
    with c1:
        st.write("Formally, Sylvester's sequence can be defined by the formula")
        st.latex(r" \displaystyle s_{n} = 1 + \prod_{i=0}^{n-1} s_{i}. ")
        st.write("where $s_0 = 2$; moreover")
        st.latex(r" s_n \; = \; s_{n-1} \cdot \left( s_{n-1} - 1 \right) \, + \, 1 ")
        with st.expander("code simulation"):
            c11, c12 = st.columns([2, 1])
            with c11:
                # how to move the code in the function `Sylvester_gen` here
                code_str = inspect.getsource(Sylvester_gen)
                st.code(code_str, language="python")
            with c12:
                deg = st.number_input("degree:", value=2, min_value=2, max_value=100)
                st.write(f"The first {deg+1} elements are : {Sylvester_gen(deg)} ")
    with c2:
        st.success(r"""
                   Some first elements in this sequences are
                   
                   $ \qquad \qquad 2, 3, 7, 43, 1807, 3263443, 10650056950807, $ etc.

                   Values derived from this sequence have also been used to construct finite Egyptian fraction representations of 1,
                   """)
        st.latex(r"""
                    \begin{array}{ccl}
                        1 &=& \dfrac{1}{2} + \dfrac{1}{3} + \dfrac{1}{6} \\ \\
                        1 &=& \dfrac{1}{2} + \dfrac{1}{3} + \dfrac{1}{7} + \dfrac{1}{42} \\ \\
                        1 &=& \dfrac{1}{2} + \dfrac{1}{3} + \dfrac{1}{7} + \dfrac{1}{43} + \dfrac{1}{1806}
                    \end{array}
                """)
        with st.expander("show"):
            levels = st.select_slider("levels", [2,3,4])
            sylvester_show(levels)

# ===========================================  Geometric =================================================
def geo_sims():
    c1, _, c2 = st.columns([4, 1, 4])
    with c1:
        s_base = st.number_input("Input base", value = 0.2, 
                                               min_value = 1e-6, max_value = 1 - 1e-6)
    with c2:
        n_len = st.number_input("number of seq", value = 10, 
                                               min_value = 5, max_value = 100)
    # Continuous domain
    t = np.linspace(0, n_len, 400)
    y_curve = s_base ** t

    # Make a meshgrid
    X = np.linspace(0, n_len, 400)
    Y = np.linspace(0, 1, 400)   # y in [0,1] because max of 0.8^0 = 1
    XX, YY = np.meshgrid(X, Y)

    # Mask region above the curve
    mask = YY <= (s_base ** XX)

    fig, ax = plt.subplots()

    # Plot shaded mesh under curve
    ax.pcolormesh(XX, YY, mask, shading='auto', cmap="RdPu", alpha=0.3)

    # Note on the first 2 elements
    ax.text(-.1, 1.1, '$s_0$', color='darkgreen', fontsize=12)
    ax.text( .8, 1.1*s_base, '$s_1$'+f"={s_base}", color='red', fontsize=11)
    ax.text( n_len-0.5, 1, "$s_k := a^k $", color='blue', fontsize=12)

    # Plot curve
    ax.plot(t, y_curve, '--', color='#dc143c', label = "y = 1 / x")
    ax.plot([0, 0], [0, 1], '--', color='#dc143c')

    # Plot discrete bars + points
    for x in range(n_len + 1):
        y = s_base ** x
        square = plt.Rectangle((x - 0.25, 0), 0.5, y,
                            facecolor="#4169e1", edgecolor="blue", alpha = max(y, 0.4))
        ax.add_patch(square)
        ax.plot(x, y, 'ro')

    ax.plot([x, x], [0, y], '--', color='#dc143c')
    ax.plot([0, x], [0, 0], '--', color='#dc143c')
    ax.set_ylim([-0.01, 1.2])
    ax.set_axis_off()
    ax.legend()

    _, c, _ = st.columns([1,9,1])
    with c:
        st.pyplot(fig)

def perfpow_sims():
    c1, c2 = st.columns([1, 4])
    with c1:
        n_min = st.slider("n_min", 1, 100, 1)
        n_max = st.slider("n_max", n_min + 1, 1000, n_min + 10)

    with c2:
        perfect_num_dict = defaultdict(list)
        for m in range(2, 1 + int(math.sqrt(n_max))):
            for k in range(2, int(math.log2(n_max)) + 1):
                n = m**k
                if n_min <= n <= n_max:
                    perfect_num_dict[n].append(f"{m}^{k}")
        st.json(perfect_num_dict)

def Geometric_deploy():
    c1, _, c2, = st.columns([9, 1, 9])
    with c1:
        st.write(r"Any sequence $ \left( s_k \right)_{k \in \mathbb{N}} $ where $s_k \in (0, 1) \quad \forall k$ is called **`geometric sequence`**, then")
        st.latex(r" \sum_{k=0}^{\infty} s_k \; = \; \dfrac{1}{1 - s_0} ")
        st.write(r"- $n$ is called **`perfect k-th power`** if there exist natural numbers $m > 1$, and $k > 1$ such that")
        st.latex(r" n \; = \; m^k ")
        st.write(r"""
                    we have,

                    $\qquad \diamond$ If $k = 2$ or $k = 3$, then $n$ is called a `perfect square` or `perfect cube`, respectively.

                    $\qquad \diamond$ The first few ascending perfect powers in numerical order (showing duplicate powers) are
                 
                    $\qquad \qquad \begin{array}{ccl}
                        4 &=& 2^2 && 16 &=& 2^4 & & 49 &=& 7^1 & & 100 &=& 10^2 \\
                        8 &=& 2^3 && 25 &=& 5^2 & & 64 &=& 2^6 & & 121 &=& 11^2 \\
                        9 &=& 3^2 && 27 &=& 3^3 & & 64 &=& 8^2 & & 125 &=& 5^3 \\
                          & &     && 32 &=& 2^5 & & 81 &=& 3^4 & & 128 &=& 2^7\\
                          & &     && 36 &=& 6^2 & & 81 &=& 9^2 & & \ldots
                    \end{array}
                    $

                  - The sum of the reciprocals of the perfect powers (including duplicates such as $3^4$ and $9^2$, both of which equal 81) is 1
                  """)
        st.latex(r" \displaystyle \sum_{m=2}^{\infty }\sum _{k=2}^{\infty }{\frac {1}{m^{k}}}=1. ")
    with c2:
        with st.expander("Show proof"):
            st.write(r"""
                        We have

                        $\begin{array}{l}
                        \displaystyle \sum_{m=2}^{\infty }\sum_{k=2}^{\infty }{\frac {1}{m^{k}}} \; = \; \displaystyle \sum_{m=2}^{\infty }{\frac {1}{m^{2}}}\sum _{k=0}^{\infty }{\frac {1}{m^{k}}} 
                                    \; = \; \displaystyle \sum_{m=2}^{\infty }{\frac {1}{m^{2}}}\left({\frac {m}{m-1}}\right)
                                    &=&\displaystyle \sum_{m=2}^{\infty }{\frac {1}{m(m-1)}} \\ \\
                                    &=& \displaystyle \sum_{m=2}^{\infty } \left({\frac {1}{m-1}}-{\frac {1}{m}} \right) \\ \\ &=& 1
                        \end{array}
                        $
                     """)
        with st.expander("**Simulation 1**: `geometrics sequence intuition`"):
            geo_sims()        
        with st.expander("**Simulation 2**: `perfect k-th power`"):
            perfpow_sims()
# .........................................................................................................
def seq_deploy(sub_sel):
    st.subheader(f"{sub_sel} sequence")
    if sub_sel == "Fibonacci":
        Fibo_deploy()
    elif sub_sel == "Sylvester":
        Sylvester_deploy()
    else:
        Geometric_deploy()
# ========================================================================================================
#--------------------------------------------- Numerics --------------------------------------------------
# ========================================== prime number ================================================
class is_prime:
    def Bruce_Force_all_prime_interval(self, n_min: int, n_max: int) -> list:
        prime_ls = []
        for num in range(n_min, n_max + 1):
            cnt = 0
            # count if there exists any divisors from 2 to sqrt(N)
            for divisor in range(2, int(num ** 0.5) + 1):
                if num % divisor == 0:
                    cnt += 1
            # append if there is no divisors
            if cnt == 0:
                prime_ls.append(num)

        return prime_ls

    def Sieve_of_Era_all_prime_interval(self, n_min: int, n_max: int) -> list:
        sieve = [True] * (n_max + 1)
        sieve[0:2] = [False, False]  # 0 and 1 are not prime
        
        for p in range(2, int(n_max**0.5) + 1):
            if sieve[p]:
                for multiple in range(p * p, n_max + 1, p):
                    sieve[multiple] = False

        return [i for i in range(n_min, n_max + 1) if sieve[i]]
    
    def brute_force(self, nums: int) -> bool:
        st.markdown(
            r"""
            - **Idea:** 🔎 Check every divisor from $1$ → $N$  
            - **Complexity:** $O(N)$
            """
        )
        count = 0
        for divisor in range(1, nums + 1):
            if nums % divisor == 0:
                count += 1
        return True if count == 2 else False

    def prime_factorization(self, nums: int) -> int:
        primes_ls = self.Sieve_of_Era_all_prime_interval(2, nums)
        prime_dict = { val: 0 for val in primes_ls}
        while nums > 1:
            for num in primes_ls:
                if nums % num == 0:
                    prime_dict[num] += 1
                    nums = nums // num
        return {k: v for k, v in prime_dict.items() if v > 0}
    
    def division_method(self, nums: int) -> bool:
        st.markdown(
            r"""
            - **Idea:** Only check divisors from $2$ → $\sqrt{N}$ ⚡ 
            - **Complexity:** $O(\sqrt{N})$
            """
        )
        if nums == 2:
            return True
        for divisor in range(2, int(nums ** 0.5) + 1):
            if nums % divisor == 0:
                return False
        return True

    def reasoning(self, nums: int) -> int:
        for divisor in range(2, int(nums ** 0.5) + 1):
            if nums % divisor == 0:
                st.warning(f"{nums} % {divisor} = {int(nums / divisor)}")
                break

    def all_twin_primes_from_interval(self, n_min: int, n_max: int) -> list:
        primes_ls = self.Sieve_of_Era_all_prime_interval(n_min, n_max)
        twin_primes_ls = []
        for idx in range(len(primes_ls) - 1):
            if primes_ls[idx + 1] - primes_ls[idx] == 2:
                twin_primes_ls.append((primes_ls[idx], primes_ls[idx + 1]))
        return twin_primes_ls

    def sum_digit(self, n : int) -> int:
        """
            Calculate sum of all digits from a given number            
        """
        str_num = str(n)
        digit_of_num = sum([ int(str_num[idx]) for idx in range(len(str_num))])
        
        return digit_of_num

    def is_Smith_num(self, nums: int) -> bool:
        prime_factors = self.prime_factorization(nums)
        # if the input nums is a `composite number`
        if nums != max(prime_factors, key = prime_factors.get):
            sum_prime_factors = sum([ self.sum_digit(k)*v for k, v in prime_factors.items()])        
            total_digit_itself = self.sum_digit(nums)

            return sum_prime_factors == total_digit_itself
        else:            
            return False

    def reasoning_Smith(self, nums):
        st.write("##### Reason")
        prime_factors = self.prime_factorization(nums)
        # Firstly check if nums is PRIME or not
        if nums == max(prime_factors, key = prime_factors.get):
            st.warning(f"{nums} itself is a **PRIME**!!")
        else:
            # convert dictionary to dict
            all_primes_ls = [ [k]*v for k, v in prime_factors.items()]
            # flatten
            all_primes_ls = [x for xs in all_primes_ls for x in xs]
            nums_total_digit = self.sum_digit(nums)
            nums_in_indigit = ','.join(_ for _ in str(nums))
            nums_prime_indigits = ','.join([str(_) for _ in all_primes_ls])
            st.write(f"- {nums} has sum of all digits `{nums_in_indigit}` are {nums_total_digit}")
            total_digit_sum_of_prime = sum([self.sum_digit(_) for _ in all_primes_ls])
            st.write(f"- {nums} has these following primes: `{nums_prime_indigits}` and this give a sum equals to {total_digit_sum_of_prime}")

    def find_all_Smith_num_from_interval(self, n_min: int, n_max: int) -> list:
        Smith_ls = []
        for num in range(n_min, n_max + 1):
            if self.is_Smith_num(num):
                Smith_ls.append(num)
        return Smith_ls

def prime_deploy():  
    c1, _, c2, = st.columns([5, 0.2, 9])
    with c1:
        st.write(r"""
                    - A `prime number` (or a *prime*) is a natural number greater than $1$ that is not a product of two smaller natural numbers. 
                    - A natural number greater than 1 that is not prime is called a **`composite number`**.
                    - A **twin prime** is a prime number that is either 2 less or 2 more than another prime number
                  """)
    with c2:
        with st.expander("Is-prime"):
            c21, c22, c23 = st.columns([2, 3, 2])
            with c23:
                st.write("#### Explanation")
            with c21:
                st.subheader("`simulation-params`")
                method = st.selectbox("method", ["brute_force", "division_method"])
                nums = st.number_input("input a number", value=2, min_value=1, max_value = int(1e9))
            with c22:
                st.write("*Please select the large number to see the different between these 2 methods*")
                t0 = time.time()
                sol = is_prime()
                func = getattr(sol, method)
                res = func(nums)
                ellapsed_time = (time.time() - t0)
                if res:
                    st.success(f"""
                                    {nums} is a prime 

                                    [ Ellapsed time: {ellapsed_time:.5f} seconds]
                                """)
                    with c23:
                        st.write("")
                else:
                    st.warning(f"""
                                    {nums} is NOT a prime 

                                    [ Ellapsed time: {ellapsed_time:.5f} seconds]
                                    """)
                    with c23:
                        reasoning = st.checkbox("Reasoning")
                        if reasoning:
                            sol.reasoning(nums)

        with st.expander("Find prime in an specific interval"):
            c21_, _, c22_ = st.columns([2, 0.1, 5])
            with c21_:
                method = st.selectbox("method", 
                                      ["Bruce-Force-sqrt", "Sieve-of-Eratosthenes"],
                                      help = "in `Bruce-Force-sqrt` DO NOT TRY FOR LARGE number; `Sieve of Eratosthenes` will be better")
                if method == "Sieve-of-Eratosthenes":
                    n_min = st.number_input("n_min", min_value=1, max_value=int(1e6), value=1)
                    n_max = st.number_input("n_max:", min_value=n_min + 1, max_value=int(1e7), value=n_min + 1000)
                else:
                    n_min = st.slider("n_min", 1, int(1e4), 1)
                    n_max = st.number_input("n_max", min_value=n_min + 1, max_value=int(1e5), value=n_min + 20)                    
            with c22_:
                t0 = time.time()
                sol = is_prime()
                if method == "Bruce-Force-sqrt":
                    prime_ls = sol.Bruce_Force_all_prime_interval(n_min, n_max)
                    with c21_:
                        st.success(f"[Finished after {(time.time() - t0):.4f} seconds ]")
                    st.write(f"All primes in the interval ({n_min}, {n_max}) are:")
                else:
                    prime_ls = sol.Sieve_of_Era_all_prime_interval(n_min, n_max)
                    with c21_:
                        st.success(f"[Finished after {(time.time() - t0):.2f} seconds ]")
                    st.write(f"All primes in the interval ({n_min}, {n_max}) are:")
                streamlit_format_text_to_cols_(prime_ls)

        with st.expander("Prime Factorization"):
            c_left, _, c_right_a, _, c_right_b = st.columns([2, 1, 2, 1, 7])
            with c_left:
                nums = st.number_input("nums: ", 2, int(1e8), 10)
            with c_right_a:
                sol = is_prime()
                prime_dict = sol.prime_factorization(nums)
                st.write(prime_dict)
            with c_right_b:
                st.write("#### Results:")
                my_fstring = ' * '.join([f"{key}^{val}" for key, val in prime_dict.items()])
                st.markdown(f" \t\t {my_fstring}", unsafe_allow_html=True)
        
        with st.expander("Twin-prime in an specific interval"):
            _c21, _c22 = st.columns(2)
            with _c21:
                n_min = st.number_input("n_min", min_value=1, max_value=int(1e5), value=1)                
            with _c22:
                n_max = st.number_input("n_max:", min_value=n_min + 1, max_value=int(1e6), value=n_min + 1000)
            sol = is_prime()
            all_twin_primes = sol.all_twin_primes_from_interval(n_min, n_max)
            n_twin = len(all_twin_primes)
            st.success(f"There are {n_twin} twin-primes here: ")
            streamlit_allocate_text_to_cols(all_twin_primes)

def smith_deploy():
    c1, _, c2, = st.columns([5, 0.2, 9])
    with c1:
        st.write(r"""
                    - A **Smith number** is a `composite number`, the sum of whose digits is the sum of the digits of its prime factors obtained as a result of prime factorization (excluding $1$)
                    
                    The first few such numbers are
                 
                            4        22       27         58          94
                            121      etc
                 
                    - Example; $22$ is a Smith because $22 = 2 \times 11$; we have 

                    $\qquad \diamond$ The sum of its digits is $2 + 2 = 4$                
                
                    $\qquad \diamond \; 2 + 11 = 13$ is the sum of its factors; and 
                 
                    $\qquad \diamond$ sum of the digits of these prime factors is $1 + 3 = 4$

                """)

    with c2:
        with st.expander("is_Smith number"):
            c21, _, c22, _, c23 = st.columns([2, 0.1, 3, 0.1, 6])
            with c21:
                nums = st.number_input("number", min_value=1, max_value=int(1e6), value=2)
                st.write("-----")
                see_explain = st.checkbox("See explaination")
            with c22:
                st.write("##### Results")
                t0 = time.time()
                sol = is_prime()
                res = sol.is_Smith_num(nums)
                if res:
                    st.success(f"{nums} is really a SMITH number")
                else:
                    st.warning(f"{nums} is **NOT** a SMITH number")
                st.write(f"Ellapsed time: \t {(time.time() - t0):.2f} seconds")

                # reasoning
                if see_explain:
                    with c23:
                        sol.reasoning_Smith(nums)

        with st.expander("Find all Smith number in an interval"):
            cleft, _, cright = st.columns([9, 1, 9])
            with cleft:
                n_min = st.number_input("n_min", value=4, min_value=1, max_value=10000)
            with cright:
                n_max = st.number_input("n_max", value=n_min + 5, min_value=n_min + 1, max_value=100000)
            sol = is_prime()
            all_Smith_nums = sol.find_all_Smith_num_from_interval(n_min, n_max)
            st.success(f"There are {len(all_Smith_nums)} Smith-numbers from {n_min} to {n_max}")
            streamlit_allocate_text_to_cols(all_Smith_nums, "num")

        with st.expander("show all_scripts"):
            st.warning("all functions of this numeric-section is from the following class")
            code_str = inspect.getsource(is_prime)
            _, c = st.columns([1, 9])
            with c:
                st.code(code_str, language="python")

def num_deploy(sub_sel):
    st.write("----------------")
    c1, _, c2, = st.columns([5, 0.2, 9])
    with c1:
        st.subheader(f"{sub_sel}")
    with c2:
        st.subheader("Examples & simulations")    
    if sub_sel == "Prime":
        prime_deploy()
    elif sub_sel == "Smith":
        smith_deploy()

# ========================================================================================================
#-------------------------------------------- Set-theory -------------------------------------------------
# ========================================================================================================
def set_deploy(sub_sel):
    st.write("----------------")
    if sub_sel == "De-Moorgan":
        st.write("**De Morgan's Laws**")
        st.latex(r"(A \cup B)^c = A^c \cap B^c")
        st.latex(r"(A \cap B)^c = A^c \cup B^c")
        st.write("These laws describe how complements distribute over unions and intersections.")

    elif sub_sel == "Inclusion-Exclusion Principle":
        st.write("**Inclusion–Exclusion Principle**")
        st.latex(r"|A \cup B| = |A| + |B| - |A \cap B|")
        st.latex(r"|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |B \cap C| - |C \cap A| + |A \cap B \cap C|")
        st.write("This principle allows us to count elements in unions of sets by correcting for overcounting.")
    
    else:
        st.write("📌 Please select a valid sub-section in Set Theory.")

# ========================================================================================================
#------------------------------------------- LAUNCH / RUN ------------------------------------------------
# ========================================================================================================

def deploy(sel_topic, sub_sel):
    if sel_topic == "Sequences":
        seq_deploy(sub_sel)
    elif sel_topic == "Numerics":
        num_deploy(sub_sel)
    else:
        set_deploy(sub_sel)

def run():
    c1, _, c2, = st.columns([9, 1, 9])
    with c1:
        sel_topic = st.selectbox("Select topic", ["Numerics", "Set", "Sequences"])
    with c2:
        if sel_topic == "Numerics":
            sub_sel = st.selectbox("section", ["Prime", "Smith"])
        elif sel_topic == "Set":
            sub_sel = st.selectbox("section", ["De-Moorgan", "Inclusion-Exclusion Principle"])
        else:
            sub_sel = st.selectbox("section", ["Fibonacci", "Sylvester", "Geometric"])
        
            # ================== if you prefer type the context
            # sub_sel = st.text_input("Type or select an option:")
            # if sub_sel == "Fibonacci":
            #    Fibonacci_fact()
               
    deploy(sel_topic, sub_sel)