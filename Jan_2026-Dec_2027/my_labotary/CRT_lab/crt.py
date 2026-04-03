import math
from itertools import combinations
from collections import defaultdict
import pandas as pd
import streamlit as st

# ======================== 1. Modulo inverse =============================== 
modulo_inverse_ps = """
- Find the inverse of $A$ while modulo $B$

- For example: 

    - inverse of $31$ is $26$ while modulo $35$
    - inverse of $7$ is $5$ while modulo $17$                
"""

modulo_inverse_script = """
for d in range(B):
    if (d * A) % B == 1:
        print(d)
"""

def modulo_inv_explaination(method):
    if method == "fermat's litte theorem":
        st.write("##### Idea (Fermat's litte theorem)")
        st.write(r"- if $B$ is prime then $A^{-1} \; \equiv \; A^{B - 2}$ mod $B$")
    else:
        st.write("##### Brute force")
        st.code(modulo_inverse_script)

def is_prime(num):
    cnt = 0
    for d in range(1, int(num**0.5) + 1):
        if num % d == 0:
            cnt += 1

    return True if (cnt == 1) and (num >= 2) else False

def modulo_inv_launch(method, num_A, num_B):
    if method == "brute force":
        res = 1
        gcd_ab = math.gcd(num_A, num_B)
        if gcd_ab == 1:
            for inv in range(1, num_B + 1):
                remainder = inv * num_A % num_B
                st.write(rf"- ${inv}$ * ${num_A} \equiv {inv * num_A}$ % ${num_B} \equiv {remainder}$")
                if remainder == 1:
                    res = inv
                    break
            st.success(rf"Inverse of ${num_A}$ is $\boxed{{{res}}}$ while modulo ${num_B}$")
        else:
            st.warning(rf"Since gcd$({num_A}, {num_B})$ = ${gcd_ab} \neq 1$ hence NOT EXIST the INVERSE of ${num_A}$ while modulo ${num_B}$.")
    else:
        if is_prime(num_B):
            st.warning(rf"We have $B = {num_B}$ is a prime, hence")
            temp = int(math.pow(num_A, num_B - 2))
            res = temp % num_B
            st.latex(rf"""
                        \begin{{array}}{{ccl}}
                            {num_A}^{{-1}} & \equiv & {num_A}^{{{num_B} - 2}} \, \% \, {num_B} \\ \\
                            & = & {temp} \, \% \, {num_B} \\ \\
                            & = & \qquad {res}
                        \end{{array}}
                    """)
            with st.expander("Final result", expanded = True):
                st.success(rf"Inverse of ${num_A}$ is $\boxed{{{res}}}$ while modulo ${num_B}$")
        else:
            st.warning(rf"Number ${num_B}$ is NOT A PRIME")

# =========================== 2. Smallest number of system of CRT with coprimes =============================== 
find_smallest_integer_ps = """
Find $x$ such that :
    
- $x$ mod $n_1$ $=$ $d_1$

- $x$ mod $n_2$ $=$ $d_2$

- $x$ mod $n_3$ $=$ $d_3$

where $\\boxed{ \\gcd(n_1, n_2, n_3) = 1}$
"""

def is_pairwise_coprime(arr):
    return all(math.gcd(a, b) == 1 for a, b in combinations(arr, 2))

def classic_coprimes_equation(ns: list[int], ds: list[int]):
    S = 0
    if is_pairwise_coprime(ns):
        M = math.lcm(*ns)
        st.write(f"All these coefficient $({', '.join(map(str, ns))})$ are coprimes, with $M = {M}$.")

        for idx in range(len(ns)):
            temp = M // ns[idx]
            r = temp % ns[idx]

            # modular inverse (clean way)
            r_inv = pow(r, -1, ns[idx])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(rf"- $M_{{{idx+1}}} = {temp}$")
            with c2:
                st.write(rf"$r_{{{idx+1}}} = {r}$")
            with c3:
                st.latex(rf"r_{{{idx+1}}}^{{-1}} \equiv {r_inv} \pmod{{{ns[idx]}}}")

            S += ds[idx] * temp * r_inv

        res = S % M
        res_alt = res + M
        st.write("Applying the formula:")
        st.latex(r"x \equiv \sum_{j=1}^{k} d_j M_j r_j^{-1} \pmod{M}")
        st.success(rf"Smallest integer answer is $\boxed{{{res}}}$, next solution is ${res_alt}$.")
    else:
        not_coprimes = ','.join([  f"({a} vs {b})" for a, b in combinations(ns, 2) if math.gcd(a, b) != 1])
        st.warning(f"These coefficients ({', '.join(map(str, ns))}) are not coprimes because these pair {not_coprimes} are not coprimes pairwise.")


# =========================== 3. Smallest number of system of CRT in general case =============================== 
find_smallest_integer_GeneralCase_ps = """
Find $x$ such that :
    
- $x$ mod $n_1$ $=$ $d_1$

- $x$ mod $n_2$ $=$ $d_2$

- $x$ mod $n_k$ $=$ $d_k$

"""

example_1_latex = r"""
\left\lbrace 
    \begin{array}{ccl}
        x & \equiv & 1  & & (\text{mod } 2 ) \\
        x & \equiv & 3  & & (\text{mod } 4 )
    \end{array}
    \right. \qquad \Rightarrow \quad x = 3 + 8k
"""

example_2_latex = r"""
\left\lbrace 
    \begin{array}{ccl}
        x & \equiv & 1  & & (\text{mod } 2 ) \\
        x & \equiv & 2  & & (\text{mod } 4 ) \\
        x & \equiv & 3  & & (\text{mod } 5 )
    \end{array}
\right. \qquad \Rightarrow \quad x = \emptyset
"""

def existence_example():
    with st.expander("Existence Condition", expanded = True):
        st.latex(r"d_i \; \equiv \; d_j \quad \gcd(n_j, n_j) \quad \forall i, j")

    with st.expander("Exists", expanded = True):
        st.latex(example_1_latex)
        
    with st.expander("Not exists", expanded = True):
        st.latex(example_2_latex)
        st.write(r"Because $x \equiv 1$ (mod $2$), and $x \equiv 2$ (mod $4$) are contradictory. Here $n_1 = 2, n_2 = 4, n_3 = 5, d_1 = 1, d_2 = 2, d_3 = 3$")
        st.latex(r"d_1 \; \text{mod} \; \gcd(n_1, n_2) \quad \neq \quad d_2 \; \text{mod} \; \gcd(n_1, n_2)")

def merge_congruence(a1, n1, a2, n2):
    g = math.gcd(n1, n2)
    diff = a2 - a1

    # No solution
    if (a2 - a1) % g != 0:
        st.warning(rf"Since $\gcd({n2}, {n1}) = {g}$ and ${a2} - {a1} = {diff}$ which NOT modulo ${g}$.")
        return None, None

    # Reduce
    n1_ = n1 // g
    n2_ = n2 // g
    b = (a2 - a1) // g

    # Solve: n1_ * k ≡ b (mod n2_)
    k = (b * pow(n1_, -1, n2_)) % n2_

    # Construct solution
    x = a1 + k * n1
    mod = n1 * n2_   # = lcm(n1, n2)

    return x % mod, mod

def classic_general_equation(ns, ds):
    x = ds[0]
    mod = ns[0]

    for i in range(1, len(ns)):
        x, mod = merge_congruence(x, mod, ds[i], ns[i])

        if x is None:
            st.warning("Hence, NO EXISTS ANY SOLUTION")
        else:
            n1, n2 = x, ds[i]
            a1, a2 = mod, ns[i]
            # st.write(rf"- $n_1$: {n1}, $n_2$: {n2} \t $a1$: {a1}, $a_2$: {a2}")
            st.write(rf"- Merging: $x \, \equiv \, {a1}$ mod ${n1}$ with $x$ ≡ ${a2}$ mod ${n2}$")

    if x != None:
        st.write(rf"LCM : ${mod}$")
        st.success(rf"Final result: $\boxed{{{x}}}$")

# ============================= 4. Smallest number of system of CRT with coprimes =============================== 
last_k_digits_pow_n_ps_ = """
Find last $k$ digits of power of any number. For example:

- $k = 1$,  num $= 2$ then period $= 4$. Indeed,

$\\qquad \\diamond \\quad 2^1 =  2 \\Rightarrow$  last digit ended by $2$

$\\qquad \\diamond \\quad 2^2 =  4 \\Rightarrow$  last digit ended by $4$

$\\qquad \\diamond \\quad 2^3 =  8 \\Rightarrow$  last digit ended by $8$

$\\qquad \\diamond \\quad 2^4 = 16 \\Rightarrow$  last digit ended by $6$

$\\qquad \\diamond \\quad 2^5 = 32 \\Rightarrow$  last digit ended by $2$

$\\qquad \\diamond \\quad 2^6 = 64 \\Rightarrow$  last digit ended by $4$

- $k = 2$,  num $= 2015$ then period $= 2$ for degree greater than $3$

$\\qquad \\diamond \\quad 2015^n$  last 2-digits ended by $25$ iff $n = 2, 4, 6, \\ldots$

$\\qquad \\diamond \\quad 2015^n$  last 2-digits ended by $75$ iff $n = 3, 5, 7, \\ldots$
"""

last_kdigits_scripts = """
seen = {}
cur = num % mod
step = 1

while cur not in seen:
    seen[cur] = step
    cur = (cur * num) % mod
    step += 1

start = seen[cur]
cycle_len = step - start
"""

latex_last_kdigits_trick_CRT = r"""
\left \lbrace 
    \begin{array}{ccl}
        x & \equiv & a^n \qquad \left( \text{mod } 2^k \right) \\
        x & \equiv & a^n \qquad \left( \text{mod } 5^k \right) 
    \end{array}
\right.
"""

def last_k_digits_powered_by_n(method):
    if method == "brute force":
        st.code(last_kdigits_scripts)
    else:
        st.write(r"Now, the problem is find $k$ last digits in $a^n$. It meant the remainder after modulo $10^k$, i.e.")
        st.latex(r" a^n \; \equiv \; x \qquad \text{mod} \quad 10^k ")
        st.write(r"this equivalent to find $x$ such that")
        st.latex(latex_last_kdigits_trick_CRT)
        st.write(r"since $10^k = 2^k \cdot 5^k$")

def find_cycle(num, mod):
    seen = {}
    cur = num % mod
    step = 1

    while cur not in seen:
        seen[cur] = step
        cur = (cur * num) % mod
        step += 1

    start = seen[cur]
    cycle_len = step - start

    return start, cycle_len, seen

def last_k_digits_powered_by_n(num, n, k):
    mod_2 = 2**k
    mod_5 = 5**k

    st.write("### Step 1: Split modulus")
    st.latex(rf"10^{k} = 2^{k} \cdot 5^{k}")
    st.write(f"→ mod 2^k = {mod_2}, mod 5^k = {mod_5}")

    # ===== Compute each side =====
    r2 = pow(num, n, mod_2)
    r5 = pow(num, n, mod_5)

    st.write("### Step 2: Solve sub-problems")
    st.latex(rf"x \equiv {num}^{n} \equiv {r2} \pmod{{{mod_2}}}")
    st.latex(rf"x \equiv {num}^{n} \equiv {r5} \pmod{{{mod_5}}}")

    # ===== Merge via CRT =====
    st.write("### Step 3: Merge using CRT")

    x, mod = merge_congruence(r2, mod_2, r5, mod_5)

    if x is None:
        st.error("No solution (should not happen in this case)")
        return

    st.latex(rf"x \equiv {x} \pmod{{{mod}}}")
    st.success(f"Last {k} digits = {str(x).zfill(k)}")

def get_cycle_table(a, k):
    mod = 10 ** k
    _, cycle_len, _ = find_cycle(a, mod)
    degrees = cycle_len * 4
    st.write(f"deg: {degrees} \t cycle_len : {cycle_len} \t k: {k}")
    seen = defaultdict(list)
    cur = 1  # num^0
    for step in range(1, degrees + 1):
        cur = (cur * a) % mod   # num^step mod mod
        seen[str(cur).zfill(k)].append(step)

    df = pd.DataFrame([
        {
            "pattern": k,
            "degrees appearance": ", ".join(map(str, v))
        }
        for k, v in seen.items()
    ])
    st.table(df.set_index('pattern'))

def get_pre_solution(a, k, mod):
    st.write(r"With these numbers, we have to find $x$ such that:")
    st.latex(rf" x \; \equiv \; a^n \quad \% \quad 10^k  ")
    st.write("where")
    st.write(rf"- $a$: ${a}$")
    st.write(rf"- $k$: ${k} \quad \Rightarrow \quad$  mod: ${mod}$")

def get_result_last_kdigits(a, n, method):
    
    if method == "brute force":
        start, cycle_len, seen = find_cycle(a, n)
        st.success(rf"Cycle: $\boxed{{{cycle_len}}}$")
        sorted_cycle_res = dict(sorted(seen.items(), 
                                        key = lambda item: item[1]))
        st.json(sorted_cycle_res)

def get_problem_statement(crt_type):
    st.subheader("Problem statement")

    if crt_type == "Modulo inverse":
        st.write(modulo_inverse_ps)

    elif crt_type == "Classic with coprimes pairwise":
        st.write(find_smallest_integer_ps)

    elif crt_type == "Classic generalized":
        st.write(find_smallest_integer_GeneralCase_ps)

    elif crt_type == "Last k digits by power of n":
        st.write(last_k_digits_pow_n_ps_)