from crt import *
import streamlit as st

# ======= Title of the page =======
st.set_page_config(layout="wide")
st.title("CRT Explaination")

crt_type = st.selectbox("Select CRT problem type", 
                    [
                        "Modulo inverse",
                        "Classic with coprimes pairwise", 
                        "Classic generalized",
                        "Last k digits by power of n",
                        "Period of power for last k digits"
                    ])

c1, _, c2, _, c3 = st.columns([2, 0.1, 1, 0.2, 4])
with c1:
    get_problem_statement(crt_type)

with c2:
    st.write("#### Params.")

if crt_type == "Modulo inverse":
    with c2:
        num_A = st.number_input("Number A", value = 3,
                                min_value = 1, max_value = 10000, step = 1)
        num_B = st.number_input("Number B", value = 5,
                                min_value = 1, max_value = 10000, step = 1)
        method = st.selectbox("Method", 
                            ["brute force", "fermat's litte theorem"])
        with c1:
            st.write("-----------")
            with st.expander("Show idea", expanded = True):
                modulo_inv_explaination(method)

    with c3:
        a, b, c = st.columns(3)
        with a:
            st.write("#### Results")
        with b:
            submit = st.button("Run")
        if submit:
            modulo_inv_launch(method, num_A, num_B)

elif crt_type == "Classic with coprimes pairwise":
    with c1:
        with st.expander("Necessary Condition", expanded = True):
            st.latex(r"\gcd(n_j, n_j) \; = \; 1 \quad \forall i, j")
        with st.expander("Uniqueness", expanded = True):
            st.latex(r" d_i \equiv d_j \quad \text{mod} \; \gcd(n_j, n_j) ")
    with c2:
        n_eq = st.number_input("Number of equations", value = 3,
                               min_value = 2, max_value = 10, step = 1)
        ns = []
        ds = []
        left, right = st.columns(2)
        for idx in range(n_eq):
            with left:
                n_i = st.number_input(rf"$n_{idx + 1}$", value = 2, 
                                      min_value = 2, max_value = 100, step = 1)
            with right:
                d_i = st.number_input(rf"$d_{idx + 1}$", value = 1, 
                                      min_value = -10, max_value = 100, step = 1)
            ns.append(n_i)
            ds.append(d_i)
    with c3:
        a, b = st.columns([2, 3])
        with a:
            st.write("#### Results")
            st.write(r"With these numbers, we have to find $x$ such that:")
            for idx in range(len(ns)):
                st.write(rf"$\qquad \diamond \quad x \equiv {ds[idx]} \pmod{{{ns[idx]}}}$")
        with b:
            submit = st.button("Run")
            if submit:
                classic_coprimes_equation(ns, ds)

elif crt_type == "Classic generalized":
    with c2:
        n_eq = st.number_input("Number of equations", value = 2,
                               min_value = 2, max_value = 10, step = 1)
        ns = []
        ds = []
        left, right = st.columns(2)
        for idx in range(n_eq):
            with left:
                n_i = st.number_input(rf"$n_{idx + 1}$", value = 2, 
                                      min_value = 2, max_value = 100, step = 1)
            with right:
                d_i = st.number_input(rf"$d_{idx + 1}$", value = 1, 
                                      min_value = -10, max_value = 100, step = 1)
            ns.append(n_i)
            ds.append(d_i)
    with c1:
        existence_example()

    with c3:
        a, b = st.columns([2, 3])
        with a:
            st.write("#### Results")
            st.write(r"With these numbers, we have to find $x$ such that:")
            for idx in range(len(ns)):
                st.write(rf"$\qquad \diamond \quad x \equiv {ds[idx]} \pmod{{{ns[idx]}}}$")
        with b:
            submit = st.button("Run")
            if submit:
                classic_general_equation(ns, ds)

elif crt_type == "Last k digits by power of n":
    with c2:
        a = st.number_input(r"Input number (base): $a$", value = 2,
                              min_value = 2, max_value = 9999, step = 1)
        k = st.number_input(r"$k$ last digits", value = 2,
                            min_value = 1, max_value = 10, step = 1)
        mod = 10**k
        method = st.selectbox("Method", 
                            ["brute force", "Tricky"])
        with c1:
            st.write("-----------")
            with st.expander("Show idea", expanded = True):
                last_k_digits_powered_by_n(method)

    with c3:
        st.write("#### Results")
        l, r = st.columns([2, 3])
        with l:
            get_pre_solution(a, k, mod)

        with r:
            submit = st.button("Run")
            if submit:
                get_result_last_kdigits(a, mod, method)
                with l:
                    if method == "brute force":
                        with st.expander("Table", expanded = True):
                            get_cycle_table(a, k)