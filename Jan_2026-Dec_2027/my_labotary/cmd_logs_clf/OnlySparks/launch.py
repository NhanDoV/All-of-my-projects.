import streamlit as st
import pandas as pd

topic_title_dict = {
    'Basic': [
            'Bs01. Top High Salary Employees', 
            'Bs02. Big Economy but small area',
            'Bs03. BroadCast small dataset joinning',
            'Bs04. Sort Merge Join',
            'Bs05. Anti-join',
            'Bs06. Salt_key for skewed data',
        ],
    'EDA': [
            'Eda01. Schema, columns & partition count',
            'Eda02. Null profiling',
            'Eda03. Numeric distribution',
            'Eda04. Quantiles & outlier screening',
            'Eda05. Drop duplicates then Repartition',
            'Eda06. Skewness notes'
    ],
    'GroupAgg': [
            'GA01. Monthly Transaction',
            'GA02. Sliding Event-Time Window',
            'GA03. Percentile Approximation'
        ],
    'GroupPivot': [
            'GP01. Pivot Student Subject Marks',
            'GP02. Purchase per category for each customer',
            'GP03. Dynamic High-Cardinality'
        ],
    'Timestamp Processor': [
            'TP01. Extract daypart',
            'TP02. Previous & Next date',
            "TP03. Transactions's statistics over 30 days"
        ],
    'Regex Clause': [
            'RC01. Find spending-behaviors pattern',
            'RC02. Customers used E-wallet & gmail',
            'RC03. Valid-IP & confused transaction-description'
        ],
    'Window': [
            'Wd01. Rising Temperatures',
            'Wd02. Distinguish rank, dense_rank & row_num',
            'Wd03. End-to-end Sales Analytic Pipeline'
    ]
}

# Cột slidebar để chọn topic và title
topic_titles = list(topic_title_dict.keys())

with st.sidebar:
    topic = st.selectbox('Chọn topic', topic_titles)
    title = st.selectbox('Chọn title', topic_title_dict[topic])

# Phần app
st.title('Pyspark only')
st.set_page_config(page_title="OnlySpark", page_icon="⚡", layout="wide")

if topic == 'Basic':
    if title == 'Bs01. Top High Salary Employees':

        df = pd.DataFrame({
            'employee_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'name': ['Alice', 'Anna', 'Ben', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
            'department': ['HR', 'Digital', 'Design', 'HR', 'Finance', 'Digial', 'HR', 'Finance', 'Design'],
            'hire_date': ['2023-01-23', '2022-07-15', '2021-03-08', '2020-11-02', '2019-06-24', '2022-02-14', '2023-05-09', '2018-09-17', '2021-12-01'],
            'salary': [98000, 62500, 73200, 69500, 121000, 86500, 59000, 118000, 92000]
        })

        c1, c2 = st.columns(2)
        with c1:
            st.write(f'##### :blue[{title}]')
            st.write('Nội dung bài toán: Tìm 5 nhân viên có mức lương cao nhất trong công ty.')
            st.write('Dữ liệu đầu vào: Bảng `employees_df` với các cột: `employee_id, name, department, hire_date, salary`.')
        with c2:
            st.code(
"""
  from pyspark.sql import functions as F
    
  employees_df.orderBy(F.col('salary').desc())\\
              .limit(5) \\
              .show()
""", language="python")

        st.divider()
        ic, _, oc = st.columns([9,1,9], gap='large')
        with ic:
            st.write("##### :orange[Input]")
            st.dataframe(df, hide_index=True)
        with oc:
            st.write("##### :orange[Output]")
            res = df.copy()
            res = df.sort_values(by='salary', ascending=False).head(5)
            st.dataframe(res, hide_index=True)

    if title == "Bs02. Big Economy but small area":
        df = pd.DataFrame({
            "country": ["Netherlands", "Belgium", "Switzerland", "Israel", "Singapore", "Vietnam", "Brazil", "Russia", "Australia", "Japan", "Denmark", "Norway", "Ireland", "Austria", "Czechia", "Portugal", "Greece", "Malaysia", "Thailand", "Indonesia"],
            "continent": ["Europe", "Europe", "Europe", "Asia", "Asia", "Asia", "South America", "Europe", "Oceania", "Asia", "Europe", "Europe", "Europe", "Europe", "Europe", "Europe", "Europe", "Asia", "Asia", "Asia"],
            "GDP_Billion_USD": [1150, 660, 940, 530, 530, 470, 2170, 2180, 1750, 4200, 420, 490, 560, 520, 340, 300, 250, 430, 540, 1370],
            "area_km2": [41543, 30528, 41284, 22072, 728, 331212, 8514877, 17098246, 7692024, 377975, 42933, 385207, 70273, 83879, 78865, 92090, 131957, 330803, 513120, 1904569]
        })
        df = df.sort_values(by='country')

        c1, c2 = st.columns(2)
        with c1:
            st.write(f'##### :blue[{title}]')
            st.write('''
                        Nội dung bài toán: 
                        - Tìm các quốc gia có diện tích dưới 50000 $km^2$ nhưng có GDP trên 200 tỷ USD.
                        - Sắp xếp theo kinh tế giảm dần, diện tích giảm dần và theo thứ tự tăng dần alphabet tên quốc gia
                    ''')

        with c2:
            st.code(
"""
  world_df.filter(
            (F.col('GDP_Billion_USD') > 200) & (F.col('area_km2') < 50000)
           ).orderBy(
                F.col('GDP_Billion_USD').desc(),
                F.col('area_km2').desc(),
                F.col('country').asc() 
        ).show()
"""                
            )
        st.divider()

        ic, _, oc = st.columns([6, 1, 6], gap='large')
        with ic:
            st.write("##### :orange[Input]")
            st.dataframe(df, hide_index=True)

        with oc:
            res = df.copy()
            st.write("##### :orange[Output]")
            res = res[(res['GDP_Billion_USD'] > 200) & (res['area_km2'] < 50000)]\
                    .sort_values(
                        by=['GDP_Billion_USD', 'area_km2', 'country'],
                        ascending=[False, False, True]
                    )
            st.dataframe(res, hide_index=True)

    if title == "Bs03. BroadCast small dataset joinning":
        c1, c2 = st.columns([3, 2], gap='large')
        with c1:
            st.write(f'##### :blue[{title}]')
            cl, cr = st.columns([4, 3], gap='medium')
            cl.write('''
  Chỉ sử dụng khi join `fact-table` vào `dim-table`

        FACT            (join)               DIM
                                           /  |  \\
         x                                (nhân bản)
         |             Executor         exe1  ...  exe4
      shuffle            (exe)            

      
    Trong ví dụ dưới đây, `sales_df` là `FACT`  và `stores_df` là `DIM`
                    ''')
            cr.write('''
    trong đó,

    - `FACT table`: chứa hàng triệu dữ liệu trở lên
    - `DIM table`: chứa từ hàng nghìn dữ liệu trở xuống (không quá 10-50 Mb)

    Khi Broadcast Hash Join
    - `FACT table`: **không shuffle** để phục vụ join
    - `DIM table`: được `broadcast` đến các `executor` 
    - `JOIN`: thực hiện `local` trên các `executor`
            ''')

            sales_df = pd.DataFrame({
                'sale_id': ['s_00000001', 's_00000002', 's_0000000', '...', 's_99999998', 's_99999999'],
                'store_id': ['a1', 'a1', 'b2', '...', 'a9', 'b9'],
                'product_id': ['tzua1', 'rsxyb', 'iksjr2t', '...', 'ultcmn01', 'dkmnxlvl'],
                'amount': ['90', '82', '37', '...', '231', '15']
            })

            stores_df = pd.DataFrame({
                'store_id': ['a1', 'a2', 'a3', '...', 'b8', 'b9'],
                'store_name': ['Uptown Flagship', 'Mint & Market', 'NeonJoky', '...', 'Velvet & Sage', 'Heart & Haven'],
                'city': ['Chicago', 'Seattle', 'Seattle', '...', 'San Francisco', 'London']
            })

            st.write("##### :orange[Input]")
            _, fact, _, dim, _ = st.columns([1, 8, 1, 6, 1])
            fact.write("**:blue[sales_df]**")
            fact.dataframe(sales_df, hide_index=True)
            dim.write("**:blue[stores_df]**")
            dim.dataframe(stores_df, hide_index=True)

        with c2:
            st.code(
"""
  sales_df.join(
            F.broadcast(stores_df), 
            on = 'store_id', 
            how = 'inner'
        ).select('sale_id', 'store_name', 'city', 'amount')
""", language='python'
            )
            st.divider()
            st.write("##### :green[Output]")
            _, oc, _ = st.columns([1, 9, 1])
            oc.dataframe(pd.DataFrame({
                'sale_id': ['s_00000001', 's_00000002', 's_00000003', '...', 's_99999998', 's_99999999'],
                'store_name': ['Uptown Flagship', 'Uptown Flagship', 'Aurelia', '...', 'The Gilded Room', 'Heart & Haven'],
                'city': ['Chicago', 'Chicago', 'Amsterdam', '...', 'New York', 'London'],
                'amount': ['90', '82', '37', '...', '231', '15']
            }), hide_index=True)

    elif title == 'Bs04. Sort Merge Join':
        c1, c2 = st.columns([7, 4], gap='large')
        with c1:
            st.write(f'##### :blue[{title}]')
            cl, cr = st.columns([4, 3], gap='medium')
            cl.write('''
  Khác với `broadcast` join (bảng lớn [fact] join với bảng nhỏ [dim]), `Sort Merge Join (SMJ)` sử dụng khi cả hai dataset đều lớn. 

  Không có giới hạn về đến mức nào có thể thực hiện `SMJ`, tùy thuộc vào các `clusters` có đủ tài nguyên và các điều kiện thực thi phù hợp

  SMJ thường được cân nhắc khi:
- Cả hai bảng đều lớn, không thể broadcast an toàn.
- Join key có thể được xử lý thông qua shuffle và sort.
- Loại join và kiểu dữ liệu của join key được Spark hỗ trợ.
- Cluster có đủ disk, memory và network capacity để xử lý shuffle.
                      ''')
            cr.write('''
    Ví dụ, khi join `transactions_df` với `customers_df`; khi đó `[SMJ]` 
    `transactions_df.join(customers_df, on = 'customer_id')` sẽ hoạt động như sau:

          transactions_df             customers_df
                |                         |
             shuffle                   shuffle
                |                         |
              sort                      sort 

                     \\    merge    / 
                           join
    
            ''')
            st.write("##### :orange[Input]")
            transactions_df = pd.DataFrame({
                'tx_id': ['100000000', '100000001', '...', '999999998', '999999999'],
                'customer_id': ['10000002', '10000001', '...', '99239967', '10000001'],
                'tx_date': ['2023-01-08 10:12:25', '2023-01-08 10:35:12', '...', '2026-10-22 20:46:38', '2026-10-22 20:48:52'],
                'amount': ['80,000', '620', '...', '8,200', '925']
            })
            customers_df = pd.DataFrame({
                'customer_id': ['10000000', '10000001', '...', '99999998', '99999999'],
                'name': ['Adam', 'Alice', '...', 'Zoey', 'Zuck'],
                'tier': ['Silver', 'Bronze', '...', 'Platinum', 'Gold']
            })
            _, txcol, _, cuscol = st.columns([1, 8, 1, 6])
            txcol.write("**:blue[transactions_df]**")
            txcol.dataframe(transactions_df, hide_index=True)
            cuscol.write("**:blue[customers_df]**")
            cuscol.dataframe(customers_df, hide_index=True)

    elif title == 'Bs05. Anti-join':
        c1, c2 = st.columns([7, 5], gap='large')
        with c1:
            st.write(f"""
            ##### :blue[{title}]
            Tùy thuộc các business-requirements (e.g. upsell dựa trên các existing users),

            sau đây là một số nguyên tắc:
            - `left_anti`: những `rows` có ở bên `left_df` nhưng không ở `right_df` dựa trên `join_key`
            - `right_anti`: những `rows` có ở bên `right_df` nhưng không ở `left_df` dựa trên `join_key`.

            Trong bài toán sau, bạn được cho `customers_df` (chứa dữ liệu khách hàng) và `cards_df` (chứa dữ liệu thẻ của khách hàng)
            
            Hãy tìm những khách hàng chưa có thẻ.
            """)
            st.write("##### :orange[Input]")
            customers_df = pd.DataFrame({
                "customer_id": [101, 102, 103, 104, 105],
                "customer_name": ["An", "Binh", "Chi", "Dung", "Huy"],
                "age": [25, 32, 28, 40, 35]
            })
            cards_df = pd.DataFrame({
                "card_id": ["C001", "C002", "C003"],
                "customer_id": [101, 103, 105],
                "card_type": ["Gold", "Platinum", "Silver"]
            })
            _, cuscol, _, crdcol = st.columns([1, 3, 1, 3])
            cuscol.write("**:blue[customers_df]**")
            cuscol.dataframe(customers_df, hide_index=True)
            crdcol.write("**:blue[cards_df]**")
            crdcol.dataframe(cards_df, hide_index=True)

        with c2:
            st.code(
'''
  customers_df.join(
                    cards_df, 
                    on = 'customer_id', 
                    how = 'left_anti'
                ).select(
                        'customer_id', 'customer_name', 'age'
                    )
'''
            )
            left_anti_df = customers_df[
                ~customers_df["customer_id"].isin(
                    cards_df["customer_id"]
                )
            ]
            st.divider()
            st.write("##### :green[Output]")
            _, oc, _ = st.columns([1, 4, 1])
            oc.dataframe(left_anti_df, hide_index=True)

    elif title == "Bs06. Salt_key for skewed data":
        c1, c2 = st.columns(2, gap='large')
        with c1:
            st.write(f"##### :blue[{title}]")
            st.write("**Khi nào nên dùng đến**")
            st.dataframe(pd.DataFrame({
                'Tình huống': ['Một vài customer_id có hàng triệu giao dịch, trong khi đa số chỉ có vài giao dịch',
                               'Join hai bảng lớn và một số join key xuất hiện cực kỳ thường xuyên',
                               'Một partition chứa lượng dữ liệu lớn bất thường, gây spill hoặc OOM do skewness data',
                               'Dữ liệu phân bố tương đối đồng đều',
                               'OOM do mỗi executor giữ quá nhiều dữ liệu hoặc do cache toàn bộ DataFrame'
                            ],
                'Quyết định': ['Có thể xài', 'Cần cân nhắc', 'Nên dùng', 'không cần', 'Salting không xử lý được gốc rễ']
            }), hide_index=True)
            st.write(f"""
                        Ví dụ trong banking, một `merchant_id` lớn hoặc một `customer_id` có lịch sử giao dịch đặc biệt dày có thể tạo ra skew. 
                        Nếu tất cả records của key đó bị gom về một partition trong shuffle, một task có thể mất nhiều thời gian hoặc hết bộ nhớ, trong khi các task khác đã hoàn thành.

                        Trong ví dụ sau đây, `transactions_df` có hơn 100 triệu giao dịch (`tx_id`) nhưng 
                        - `customer_id = 121` chiếm gần 35 triệu rows, các KH còn lại chia sẻ 60 hơn triệu rows
                        - Khi shuffle partition theo `customer_id`, các records có cùng key thường được hash vào cùng một partition. Vì vậy, nếu key `121` chiếm 35 triệu rows, partition tương ứng có thể quá lớn và gây `data skew`.

                        → repartition phân phối theo key không đồng nghĩa với chia nhỏ một hot key thành nhiều partition
                    """,)
        with c2:
            st.write(""" 
                        **Salt-key giải quyết ntn??**

                        Ta thêm một salt có nhiều giá trị cho các records mang hot key,
                    """)
            st.code(
"""
  # 1. Chia hot key trong fact thành 10 salt groups
  salted_tx = transactions_df.withColumn("salt_key",
                                          F.when(
                                            F.col("customer_id") == 121, 
                                            F.floor(F.rand(seed=42) * 10).cast("int")
                                        ).otherwise( F.lit(0) ) )

  # 2. Replicate customer 121 thành 10 records tương ứng salt 0-9
  salted_customers = customers_df.withColumn("salt_key",
                                             F.explode(
                                                F.when(
                                                  F.col("customer_id") == 121,
                                                  F.sequence(F.lit(0), F.lit(9))
                                              ).otherwise(F.array(F.lit(0))) ) 
                                            )

  # 3. Join theo customer_id + salt_key
  result = salted_tx.join(salted_customers, 
                          on = ["customer_id", "salt_key"], how = "inner")
"""
            )
            st.write("""
                    Cơ chế lúc này:
                    - Fact có customer_id = 121 được chia ngẫu nhiên vào salt 0–9.
                    - Customer 121 ở bảng dimension được nhân thành 10 bản sao, mỗi bản mang một salt tương ứng.
                    - Các customer còn lại vẫn dùng salt 0 ở cả hai bảng.
                    - Khi join, mỗi giao dịch vẫn khớp với customer tương ứng, nhưng workload của hot key có cơ hội được phân phối qua nhiều partition.
                    Lưu ý: salting làm tăng số bản ghi của bảng dimension đối với hot key, và không đảm bảo tuyệt đối workload cân bằng. Số salt cần chọn dựa trên mức độ skew và tài nguyên cluster.
                    """)


elif topic == "Window":
    if title == "Wd01. Rising Temperatures":
        c1, c2 = st.columns([3, 4], gap='large')
        with c1:
            st.write(f"""
                    ##### :blue[{title}]
                    Ví dụ minh họa này được lấy từ `LeetCode 197` nên sẽ giữ nguyên tiêu đề `Rising Temperatures`
                    
                    Trong bài toán này, chúng ta phải trích xuất được những ngày mà nhiệt độ ngày hôm đó cao hơn ngày hôm trước
                    """)
            st.write("##### :green[Input]")
            inp_df = pd.DataFrame({
                'record_date': ['20260930', '20261001', '20261002', '20261003', '20261004', '20261005', '20261006'],
                'temperatue': [29, 25, 30, 32, 24, 20, 22]
            })
            _, ic, _ = st.columns([1,2,1], gap='large')
            ic.dataframe(inp_df, hide_index=True)

        with c2:
            st.code(
'''
  from pyspark.sql.window import Window

  window_roll = Window.orderBy('record_date')
  filters_df = weather_df.withColumn('record_date', to_date('record_date', 'yyyyMMdd'))
                         .withColumn('prev_date', F.lag('record_date', 1).over(window_roll))\\
                         .withColumn('prev_date', F.lag('record_date', 1).over(window_roll))\\
                         .filter(
                                (F.date_diff('record_date', 'prev_date') == 1) &
                                (F.col('temperature') > F.col('prev_temp'))
                            )

  res = filters_df.select('record_date', 'temperature')
''', language='python'
            )
            out_df = pd.DataFrame({
                'record_date': ['2026-10-02', '2026-10-03', '2026-10-06'],
                'temperature': [30, 32, 22],
                'prev_date': ['2026-10-01', '2026-10-02', '2026-10-05'],
                'prev_temp': [25, 30, 20]
            })
            ocol, fcol, rcol = st.columns([1, 4, 2], gap='large')
            ocol.write("##### :orange[Output]")
            fcol.write('**:green[filters_df]**')
            fcol.dataframe(out_df, hide_index=True)
            rcol.write('**:green[res]**')
            rcol.dataframe(out_df[['record_date', 'temperature']], hide_index=True)

    elif title == "Wd02. Distinguish rank, dense_rank & row_num":
        c1, c2 = st.columns([2, 3], gap='large')
        with c1:
            st.write(f"""
                    ##### :blue[{title}]
                    Các thuật ngữ `rank`, `dense_rank` và `row_num` rất dễ nhầm lẫn khi viết các hàm `ranking`
                    (e,g. `rank`, `dense_rank` và `row_number`).

                    Trong dữ liệu `employees_df` của bài toán này, bạn được yêu cầu phân hoạch (`partition`) theo `department` và sắp xếp (`orderBy`) 
                    - theo `salary` DESC; và theo `emp_id` ASC với `row_number`
                    - theo `salary` DESC với `rank` và `dense_rank`
                    """)
            st.write("##### :orange[Input]")
            inp_df = pd.DataFrame({
                'emp_id': range(101, 108, 1),
                'department': ['Finance', 'HR', 'HR', 'Finance', 'Finance', 'HR', 'Finance'],
                'salary': ['98,000', '62,000', '62,000', '92,000', '121,000', '95,000', '121,000']
            })
            _, ic, _ = st.columns([1,4,1], gap='large')
            ic.dataframe(inp_df, hide_index=True)

        with c2:
            st.code(
'''
  from pyspark.sql.window import Window

  wd_row = Window.partitionBy('department').orderBy(F.col('salary').desc(), F.col('emp_id').asc())
  wd_rnk = Window.partitionBy('department').orderBy(F.col('salary').desc())

  employees_df.withColumn('row_num', F.row_number().over(wd_row)) \\
              .withColumn('rank', F.rank().over(wd_rnk)) \\  
              .withColumn('dense_rank', F.dense_rank().over(wd_rnk)) 
''', language='python'
            )
            st.write("##### :orange[Output]")
            _, oc, _ = st.columns([1, 6, 1], gap='large')
            oc.dataframe(pd.DataFrame({
                'emp_id': [105, 107, 101, 104, 106, 102, 103],
                'department': ['Finance', 'Finance', 'Finance', 'Finance', 'HR', 'HR', 'HR'],
                'salary': ['121,000', '121,000', '98,000', '92,000', '95,000', '62,000', '62,000'],
                'row_num': [1, 2, 3, 4, 1, 2, 3],
                'rank': [1, 1, 3, 4, 1, 2, 2],
                'dense_rank': [1, 1, 2, 3, 1, 2, 2]
            }), hide_index=True)

elif topic == "Regex Clause":
    if title == "RC01. Find spending-behaviors pattern":
        c1, c2 = st.columns([2, 3], gap='large')
        with c1:
            st.write(f'##### :blue[{title}]')
            st.write("""
                    Trong các dữ liệu giao dịch trong bảng `transactions_df`, người ta mong muốn có thể trích xuất được các thông tin về hành vi như
                    - `travel` nếu cột `tx_description` có chứa các từ khóa như `vé máy bay, khách sạn,...`
                    - `shopping` nếu `tx_description` có chứa `mua sắm, quần áo, thời trang, ...`
                    - `education` nếu `tx_description` có chứa `khóa học, học phí, đào tạo, ngoại ngữ, ...`
                    - `investment` nếu `tx_description` có chứa `chứng khoán, cổ phiếu, đầu tư, ...`
                    - `gamming` nếu `tx_description` có chứa `game, nạp, vật phẩm, ...`                    
                    """)
            st.write("**:green[SYNTAX NOTES]**")
            st.code('''
    F.col('_').rlike(pattern)         --> return True/False
    F.regexp_extract(F.col('_'),
                     pattern, replacement
                    )                 --> pull a pice out   
    F.split(F.col('_'), pattern)      --> split by a regex-delimeter
                    ''', language='python')
            st.write('**:green[Patterns rules]**')
            st.dataframe(
                pd.DataFrame({
                    'pattern': ['^/$', '.', '\d / \D', '\w / \W', '\s / \S', '[abc]', '[^abc]', '(...)'],
                    'read it as': ['start/end of string', 
                                   'any single char', 'digit/ non digit', 
                                   'word (letter/digits/underscore) / not',
                                   'whitespace or not', 'one of a,b or c',
                                   'anything except a/b/c', 'capture group'],
                    'examples': ['^hi$ matches exactly "hi"', 
                                 'a.c matches "abc", "axc"', "\d\d is 2 digits",
                                 '\w+ is word chunk', '\s+ is one or more spaces',
                                 '[ae] in tea hits e', '[^0-9] is not a digit', 
                                 '(momo|shopee|lazada) one of them'
                                 ]
                }), hide_index=True
            )
        with c2:
            st.code(
'''
  from pyspark.sql import functions as F

  travel_rgx = r"(?i)(vé máy bay|khách sạn|du lịch|resort|tour)"
  gamming_rgx = r"(?i)(game|vật phẩm|vltk|free fire|lmht)"
  shopping_rgx = r"(?i)(mua sắm|quần áo|thời trang|điện thoại)"
  education_rgx = r"(?i)(khóa học|học phí|đào tạo|ngoại ngữ)"
  investment_rgx = r"(?i)(chứng khoán|cổ phiếu|đầu tư|chứng chỉ quỹ)"

  transactions_df.withColumn('is_travel', F.col('tx_description').rlike(travel_rgx)) \\
                .withColumn('is_shopping', F.col('tx_description').rlike(shopping_rgx)) \\
                .withColumn('is_education', F.col('tx_description').rlike(education_rgx)) \\
                .withColumn('is_gamming', F.col('tx_description').rlike(gamming_rgx)) \\
                .withColumn('is_invest', F.col('tx_description').rlike(investment_rgx))
''', language='python'
            )
            st.write("##### :orange[Output]")
            st.dataframe(pd.DataFrame({
                'tx_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
                'tx_date': ['2024-01-03', '2024-12-02', '2024-12-09', '2025-01-04', '2025-02-10', '2025-10-06', '2026-03-29'],
                'payment_method': ['Credit Card', 'Bank Transfer', 'Debit Card', 'Bank Transfer', 'Debit Card', 'Debit Card', 'Credit Card'],
                'tx_description': [
                    'Thanh toán vé máy bay...', 'Đặt phòng khách sạn...', 'Mua điện thoại...', 'Tạm ứng lương kỳ T01...',
                    'Đóng học phí HK1 ....', 'Mua vật phẩm VLTK...', 'Lệnh mua 200 cổ phiếu FPT...'
                ],
                'is_travel': [True, True, False, False, False, False, False],
                'is_shopping': [False, False, True, False, False, False, False],
                'is_education': [False, False, False, False, True, False, False],
                'is_gamming': [False, False, False, False, False, True, False],
                'is_invest': [False, False, False, False, False, False, True]
            }), hide_index=True)

    if title == "RC02. Customers used E-wallet & gmail":
        c1, c2 = st.columns([6, 5],  gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong dữ liệu `transactions_df`, có các trường: `name`, `payment_content`, `email` và `amount`
                        
                        Người ta muốn check xem có bao nhiêu \% số giao dịch cũng như \% tổng số tiền giao dịch đến từ những nhóm:
                        - users sử dụng `gmail` đăng ký tài khoản
                        - users sử dụng `ví điện tử (E-wallet)` như zalo, momo, ...
                    ''')
            ipc, opc = st.columns([5, 3], gap='large')
            with ipc:
                st.write('**:green[Input dataframe]**')
                inp_df = pd.DataFrame({
                    'name': ['Alice', 'Ben', 'Bob', 'Charlie', 'Diana', 'Evan', 'Frank'],
                    'payment_content': [
                            'MOMO Topup from...', 'ZaloPay transfer...', '(VCB) phát sinh giao dịch...',
                            'ShopeePay payment...', 'zalopay transfer', '(TPBank): 08/12/22 19:45..', 'MOMO...'
                        ],
                    'email': ['alice@gmail.com', 'ben10@gmail.com', 'obob1@fpt.com', 'lecharle@vng.com', 
                              'diana411@gmail.com', 'evankig@gmail.com', 'frankenstn19x@outlook.com'],
                    'amount': [75000, 92000, 111000, 68000, 80000, 69000, 89000]
                })
                st.dataframe(inp_df, hide_index=True)
            with opc:
                st.write('**:green[Expected output schema]**')
                st.dataframe(
                    pd.DataFrame({
                        'measure_by': ['STRING'],
                        'perc_tx': ['DOUBLE'],
                        'perc_amount': ['DOUBLE']
                    }), hide_index=True
                )
                st.write('''
                            Trong đó, cột `measure_by` sẽ có 2 giá trị là :
                            - `use_Ewallet` và 
                            - `had_gmail`        
                        ''')
                st.divider()
                st.write("##### :orange[Output]")
                st.dataframe(pd.DataFrame({
                    'measure_by': ['use_Ewallet', 'had_gmail'],
                    'perc_tx': [71.43, 42.86],
                    'perc_amount': [69.3, 38.25]
                }), hide_index=True)

        with c2:
            st.code(
'''
  e_wallet_patt = r"(?i)(momo|zalopay|vnpay|shopeepay|viettel money|vtc pay)"
  main_stats = transactions_df.agg(
            F.count('*).alias('n_trans'), 
            F.sum('amount').alias('t_amount')
        )
  n_trans, t_amount = main_stats['n_trans'], main_stats['t_amount']

  EWal_df = transactions_df.withColumn(
                    'is_wallet',
                    F.when(F.col('payment_content').rlike(e_wallet_patt),
                        'use_Ewallet').otherwise('None')
                ).filter(F.col('is_wallet')=='use_Ewallet')\\
                .groupBy('is_wallet').agg(
                    F.round(100*(F.count('*')/ n_trans)).alias('perc_tx'),
                    F.round(100*(F.sum('amount')/ t_amount)).alias('perc_amount'')
                ).withColumnRenamed('is_wallet', 'measure_by')

  GMail_df = transactions_df.withColumn(
                    'sign_gmail',
                    F.when(F.col('email').rlike(r'@gmail.com$'),
                        'had_gmail').otherwise('None')
                ).filter(F.col('sign_gmail')=='had_gmail')\\
                .groupBy('sign_gmail').agg(
                    F.round(100*(F.count('*')/ n_trans)).alias('perc_tx'),
                    F.round(100*(F.sum('amount')/ t_amount)).alias('perc_amount'')
                ).withColumnRenamed('sign_gmail', 'measure_by')

  EWal_df.union(GMail_df).show()
''', language='python'
            )

    elif title == "RC03. Valid-IP & confused transaction-description":
        c1, c2 = st.columns([3, 7],  gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong dữ liệu ground truth của `transactions_df`, có các trường: `tx_id`, `tx_content`, `IP_address` và `is_fraud`
                        
                        Người ta muốn sử dụng Feature Engineering để tạo thêm các biến quan sát:
                        - `IP_valid`: xem `IPv4` có hợp lệ hay k?
                        - `is_ctx_suspicious`: nếu `tx_content` chứa các thông tin gây nghi ngờ (như `Team Member Discount`, `Bypass`. etc)

                        Ngoài ra, tạo `pivot table` in ra `IP_valid` (index) theo `is_ctx_suspicious` (column)
                    ''')
            st.write('**:green[Input dataframe]**')
            inp_df = pd.DataFrame({
                        'tx_id': range(1000, 1010, 1),
                        'tx_content': ['Payment Phuc Long Coffee', 'TeamMember Discount purchase',
                                    'Employee Benifit Adjustment', 'Amazone Order', 'VIP Internal Transfer',
                                    'Payroll Override Payment', 'Normal Grocery Purchase', 'Test Transaction Bypass',
                                    'Internal Staff Purchase', 'Shopee Payment'
                                    ],
                        'IP_address': ['192.168.1.100', '10.20.30.40', '172.16.5.200', '8.8.8.8',
                                    '999.999.999.999', 'abc.def.1.1', '123.45.67.89',
                                    '172.300.1.2', '192.168.0.50', '14.225.12.20'
                                    ],
                        'is_fraud': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
                    })
            st.dataframe(inp_df, hide_index=True)
        with c2:
            st.code('''
  suspicious_patt = r"(?i)(Team\s*Member\s*Discount|Employee\s*Benifit|Internal\s*Staff|Payroll\s*Override|VIP\s*Internal|Bypass|Test\s*Transaction)"
  ipv4_patt = r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"

  enhanced_df = transactions_df.withColumn(
                                'is_ctx_suspicious', 
                                F.when(F.col('tx_content').rlike(suspicious_patt), 'suspicious_context')\\
                                 .otherwise('clear_context') 
                            ).withColumn('IP_valid', F.col('IP_address').rlike(ipv4_patt))

  pivoted_df = enhanced_df.groupBy('IP_valid').pivot('is_ctx_suspicious').sum('is_fraud').fillna(0)
                    ''', 
                    language='python', width='content', wrap_lines=True)
            op_c, eh_c, pv_c = st.columns(3, gap='large')
            op_c.write("##### :orange[Output]")
            eh_c.write("**:blue[enhanced_df]**")
            pv_c.write("**:blue[pivoted_df]**")
            
            fe_col, piv_col = st.columns([2, 1], gap='large')
            with fe_col:
                out_df = inp_df.copy()
                out_df['is_ctx_suspicious'] = ['clear_context', 'suspicious_context', 'suspicious_context',
                                               'clear_context', 'suspicious_context', 'suspicious_context',
                                               'clear_context', 'suspicious_context', 'suspicious_context', 'clear_context']
                out_df['IP_valid'] = ['true', 'true', 'true', 'true', 'false', 'false', 'true', 'false', 'true', 'true']
                st.dataframe(out_df, hide_index=True)
            with piv_col:
                st.dataframe(pd.DataFrame({
                    'IP_valid': ['true', 'false'],
                    'clear_context': [0, 0],
                    'suspicious_context': [3, 3]
                }), hide_index=True)

elif topic == "Timestamp Processor":
    if title == "TP01. Extract daypart":
        c1, c2 = st.columns([8, 7],  gap='large')
        c1.write(f"##### :blue[{title}]")
        with c1:
            ic, dc = st.columns([3, 8], gap='medium')
            with dc:
                st.write(f'''
                            Trong một số bài toán phân tích chuỗi thời gian, các yếu tố thời gian khác như `hour, day, month`, etc cũng góp phần phân tích 
                            các đặc tính như `seasonality`, `trend` hay thậm chí phục vụ cho một sô phân tích thống kê theo các đặc trưng này.
                            
                            Bên cạnh đó, người ta cũng mong muốn sứ dụng `window-function` lên các đặc trưng thời gian để tìm ra các đặc trưng 
                            (e.g, `avg_revenue_30d`, `std_revenue_7d`, etc). Để phục vụ được mục đích đó, chúng ta cần một công cụ có thể tính toán
                            quy đổi được dữ liệu thời gian về dạng hiệu (theo ngày/ giây) tùy thuộc độ rộng của `window-size`

                            Trong bài toán này, chúng ta sẽ tìm hiểu làm thế nào để 
                            - Chuyển đổi định dạng `STRING` -> `TIMESTAMP` 
                            - Trích xuất được các yếu tố `year, month, week, hour` từ dữ liệu `timestamp`
                            - Tính được `diff_days, diff_secs` dựa trên dữ liệu `timestamp` hiện tại tới ngày tiêu chuẩn `1970-01-01`
                        ''')
            ic.write('**:green[Input dataframe]**')
            inp_df = pd.DataFrame({
                        'tx_id': range(100, 108, 1),
                        'tx_date': ['19700101000023', '19700101060005', '19700108235959', '20220220081237',
                                    '20230214230153', '20231229082347', '20240229152634', '20250630132248'] 
                    })
            ic.dataframe(inp_df, hide_index=True)

        with c2:
            st.code(
'''
  transactions_df.withColumn('tx_ts', F.to_timestamp('tx_date', 'yyyyMMddHHmmss')) \\
                 .withColumn('tx_year', F.year('tx_ts') ) \\
                 .withColumn('tx_month', F.month('tx_ts') ) \\
                 .withColumn('tx_week', F.weekofyear('tx_ts') ) \\
                 .withColumn('diff_days', F.unix_date(F.to_date('tx_ts')) ) \\
                 .withColumn('diff_secs', F.unix_seconds('tx_ts') )
''', language='python'
            )

            st.write('**:green[Output]**')
            out_df = inp_df.copy()
            out_df['tx_ts'] = pd.to_datetime(out_df['tx_date'])
            out_df['tx_year'] = out_df['tx_ts'].dt.year
            out_df['tx_month'] = out_df['tx_ts'].dt.month
            out_df['tx_week'] = out_df['tx_ts'].dt.isocalendar().week
            out_df['diff_days'] = (out_df['tx_ts'] - pd.Timestamp('1970-01-01 00:00:00')).dt.days
            out_df['diff_secs'] = (out_df['tx_ts'] - pd.Timestamp('1970-01-01 00:00:00')).dt.seconds

            st.dataframe(out_df, hide_index=True)

    elif title == 'TP02. Previous & Next date':
        c1, c2 = st.columns([3, 5], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Sau khi đã hiểu về `diff_days` với so sánh về một ngày cố định `1970-01-01`, 
                        - chúng ta sẽ tiếp tục với `date_add(date_col[ts], n_day[int])` và `date_diff(date_col_1, date_col_2)` nếu ta muốn
                        trả về ngày cụ thể (sau bao nhiêu ngày) hoặc hiệu của 2 cột ngày giờ được chọn
                        - và `lag(date_col, n_days)` nếu muốn truy xuất về bao nhiêu dòng trước / sau trên cùng cột `date_col`

                        Chúng ta đang có dữ liệu `employees_df` với
                    ''')
            st.write('**:green[Input dataframe]**')
            _, ic, _ = st.columns([1, 2, 1], gap='small')
            ic.dataframe(pd.DataFrame({
                'emp_id': [101, 103, 102],
                'department': ['Finance', 'HR', 'HR'],
                'start_date': ['2024-09-29', '2025-01-01', '2026-10-12']
            }), hide_index=True)
        with c2:
            st.code(
'''
  from pyspark.sql.window import Window

  wd = Window.orderBy('start_date')

  employees_df.withColumn('start_date', F.to_date('start_date', 'yyyy-MM-dd')) \\
              .withColumn('prev_1row', F.lag('start_date', 1).over(wd)) \\
              .withColumn('next_1row', F.lag('start_date', -1).over(wd)) \\
              .withColumn('next_1day', F.date_add('start_date', 1)) \\
              .withColumn('prev_1day', F.date_add('start_date', -1)) \\
              .withColumn('diff(start, prev_1row)', F.date_diff('start_date', 'prev_1row'))
''', language='python'
            )
            st.write('**:green[Output]**')
            _, oc, _ = st.columns([1,9,1])
            oc.dataframe(pd.DataFrame({
                'emp_id': [101, 103, 192],
                'department': ['Finance', 'HR', 'HR'],
                'start_date': ['2024-09-29', '2025-01-01', '2026-10-12'],
                'prev_1row': ['NULL', '2024-09-29', '2025-01-01'],
                'next_1row': ['2025-01-01', '2026-10-12', 'NULL'],
                'next_1day': ['2024-09-30', '2025-01-02', '2026-10-13'],
                'diff(start_date, prev_1row)': ['NULL', '94', '649']
            }), hide_index=True, width='content')

    elif title == "TP03. Transactions's statistics over 30 days":
        c1, c2 = st.columns([3, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Tiếp theo, sau khi đã nắm rõ các kỹ thuật xử lý `timestamp` cũng như `window / partition`, ở bài toán này
                        chúng ta mong muốn có thể tính được

                        - `n_transactions`: số giao dịch trong 30 ngày tính đến ngày hiện tại
                        - `std_amount, avg_amount, med_amount, t_amount`: lần lượt là các giá trị `standard-deviation, average, median` và `total` của biến `amount` trong 30 ngày gần nhất kể từ ngày giao dịch hiện tại

                        theo đó, ta sẽ phân hoạch (`partitionBy`) mỗi `user_id`
                    ''')
            st.write('**:green[Input dataframe]**')
            inp_df = pd.DataFrame({
                    'tx_id': ['axx..001', 'axz..012', 'axy..011', 'txy..010', 'wxa..002', 'uxp..101', 'xnx..102', 'nxn..121', 'ntn..244'],
                    'user_id': ['A01', 'G02', 'G02', 'A01', 'A01', 'G02', 'A01', 'A01', 'G02'],
                    'tx_date': ['20230901', '20230903', '20230904', '20230905', '20230919', '20230923', '20231006', '20231026', '20231026'],
                    'amount': [29000, 43000, 17000, 21500, 92500, 61200, 96000, 83000, 69000]
                    })
            _, ic, _ = st.columns([1,6,1])
            ic.dataframe(inp_df, hide_index=True)
        with c2:
            st.code(
'''
  from pyspark.sql.window import Window

  transactions_df.withColumn('tx_date', F.to_date('tx_date', 'yyyyMMdd')) \\
                 .withColumn('unix_date', F.unix_timestamp('tx_date'))
  wd_30d = Window.partitionBy('user_id') \\
                 .orderBy('unix_date') \\ 
                 .rangeBetween(-30*86400, 0)  

  transactions_df.withColumn('n_trans_30d', F.count('*').over(wd_30d)) \\
                 .withColumn('t_amount_30d', F.sum('amount').over(wd_30d)) \\
                 .withColumn('std_amount_30d', F.round(F.std('amount').over(wd_30d), 2) )\\
                 .withColumn('med_amount_30d', F.percentile_approx('amount', F.lit(0.5)).over(wd_30d))\\
                 .drop(F.col('unix_date'))       

''', language='python'
            )
            st.write('**:green[Output]**')
            out_df = pd.DataFrame({
                'tx_id': ['axx..001', 'txy..010', 'wxa..002', 'xnx..102', 'nxn..121', 'axz..012', 'axy..011', 'uxp..101', 'ntn..244'],
                'user_id': ['A01','A01','A01','A01','A01', 'G02','G02','G02','G02'],
                'tx_date': ['2023-09-01', '2023-09-05', '2023-09-19', '2023-10-06', '2023-10-26',
                            '2023-09-03', '2023-09-04', '2023-09-23', '2023-10-26'],
                'amount': ['29,000', '21,500', '92,500', '96,000', '83,000', '43,000', '17,000', '61,200', '69,000'],
                'n_trans_30d': [1, 2, 3, 2, 2, 1, 2, 3, 1],
                't_amount_30d': ['29,000', '50,500', '143,000', '188,500', '179,000', '43,000', '60,000', '121,200', '69,000'],
                'std_amount_30d': ['NULL', '5,303.3', '39,007.48', '2,474.87', '9,192.39', 'NULL', '18,384.78', '22,214.41', 'NULL'],
                'med_amount_30d': ['29,000', '21,500', '29,000', '92,500', '83,000', '43,000', '17,000', '43,000', '69,000']
            })
            _, oc, _ = st.columns([1, 9, 1])
            oc.dataframe(out_df, hide_index=True)

elif topic == "GroupAgg":
    if title == "GA01. Monthly Transaction":
        c1, c2 = st.columns([4, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Bạn được nhận dữ liệu `transactions_df` với các trường như bên dưới.

                        Hãy viết một đoạn script để nhận được các thông tin sau:
                        - `tx_count`: số lượng các giao dịch
                        - `approved_cnt`: số giao dịch được `approved` 
                        - `total_amount` : tổng số tiền giao dịch mỗi tháng
                        - `approved_t_amount`: số tiền giao dịch mỗi tháng được `approved`

                    ''')
            st.write('**:green[Input dataframe]**')
            inp_df = pd.DataFrame({
                'id': [121, 122, 123, 124, 244, 288, 411],
                'country': ['US', 'US', 'US', 'DE', 'DE', 'US', 'DE'],
                'status': ['approved', 'declined', 'approved', 'approved', 'approved', 'declined', 'approved'],
                'amount': [1000, 2000, 2000, 2000, 1200, 1800, 1400],
                'tx_date': ['2025-11-27', '2025-11-29', '2025-12-09', '2025-12-23', '2025-12-28', '2026-01-08', '2026-01-14']
            })
            _, ic, _ = st.columns([1, 4, 1], gap='small')
            ic.dataframe(inp_df, hide_index=True)

        with c2:
            st.code(
'''
  transactions_df.withColumn('tx_date', F.to_date('tx_date')) \\
                 .withColumn('tx_YM', F.date_format('tx_date', 'yyyy-MM')) \\
                 .groupBy('tx_YM', 'country') \\
                 .agg(
                    F.count('*').alias('trans_count'),
                    F.count(F.when(F.col('status') == 'approved', 1)).alias('approved_cnt'),
                    F.sum('amount').alias('total_amount'),
                    F.sum(F.when(F.col('status') == 'approved', F.col('amount')) ).alias('approved_t_amount')
                 )
'''
            )

            out_df = pd.DataFrame({
                'tx_YM': ['2025-11', '2025-12', '2025-12', '2026-01', '2026-01'],
                'country': ['US', 'US', 'DE', 'US', 'DE'],
                'tx_count': [2, 1, 2, 1, 1],
                'approved_cnt': [1, 1, 2, 0, 1],
                'total_amount': ['3,000', '2,000', '3,200', '1,800', '1,400'],
                'approved_t_amount': ['1,000', '2,000', '3,200', 'NULL', '1,400']
            })

            st.write('**:green[Output]**')
            _, oc, _ = st.columns([1, 9, 1], gap='medium')
            oc.dataframe(out_df, hide_index=True)

elif topic == "GroupPivot":
    if title == "GP01. Pivot Student Subject Marks":
        c1, c2 = st.columns([5, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Bạn được nhận dữ liệu `marks_df` với các thông tin về MSSV, tên môn học và điểm thi của SV theo kiểu long table. 
                        Hãy sử dụng `pivot` để có thể phân phối 
                        - MSSV (`student_id`) vào các `index`
                        - các môn học (`subject`) vào các `column`

                        Sau đó, tính điểm trung bình với
                        - `AVG_main_subj`: 3 môn học chính gồm `Math, Physic, Chemistry`
                        - `AVG_all_subj`: tất cả các môn học
                    ''')
            st.write('**:green[Input dataframe]**')
            _, r5, _, r9 = st.columns([1, 9, 1, 9], gap='small')
            r5.write("Row 1-5")
            r5.dataframe(pd.DataFrame({
                'subject': ['Chemistry', 'Chemistry', 'English', 'English', 'History'],
                'student_id': [1, 2, 1, 2, 1],
                'marks': [9.8, 9.5, 9.2, 7.3, 6.1]
            }), hide_index=True)
            r9.write("Row 6-10")
            r9.dataframe(pd.DataFrame({
                'subject': ['History', 'Maths', 'Maths', 'Physics', 'Physics'],
                'student_id': [2, 1, 2, 1, 2],
                'marks': [7.1, 10.0, 9.5, 8.0, 9.8]
            }), hide_index=True)

        with c2:
            st.code(
'''
  main_subjects = ['Maths', 'Physics', 'Chemistry']
  all_subjects = ['Chemistry', 'English', 'History', 'Maths', 'Physics']

  pivoted = marks_df.groupBy('student_id')\\
                      .pivot('subject', all_subjects)\\
                      .sum('marks').fillna(0)

  pivoted.withColumn('AVG_main_subj', F.round( sum(F.col(sub) for sub in main_subjects)/3 ), 2 ) \\
         .withColumn('AVG_all_subj', F.round( sum(F.col(sub) for sub in all_subjects)/5 ), 2 )
'''
            )
            st.write('**:green[Output]**')
            _, oc, _ = st.columns([1, 9, 1], gap='medium')
            oc.dataframe(pd.DataFrame({
                'student_id': [1, 2],
                'Chemistry': [9.8, 9.5],
                'English': [9.2, 7.3],
                'History': [6.1, 7.1],
                'Maths': [10.0, 9.5],
                'Physics': [8.0, 9.8],
                'AVG_main_subj': [9.27, 9.6],
                'AVG_all_subj': [8.62, 8.64]
            }), hide_index=True) 

    elif title == 'GP02. Purchase per category for each customer':
        c1, c2 = st.columns([6, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Bạn được nhận dữ liệu `products_df` với các thông tin về mã giao dịch (`tx_id`), tên sản phẩm (`product`), mã khách hàng (`customer_id`) và giá thành (`amount`)

                        Hãy sử dụng `pivot` với
                        - `index` được `groupBy` mã khách hàng
                        - `column` là các sản phẩm
                        - `aggfunc` được tính bằng cách đếm số sản phẩm được bán và tổng số tiền tương ứng
                    ''')
            st.write('**:green[Input dataframe]**')
            r5, r9 = st.columns(2, gap='small')
            r5.write("Row 1-7")
            r5.dataframe(pd.DataFrame({
                'tx_id': ['x101', 'n121', 'a139', 'p140', 'z025', 'x206', 'n244'],
                'customer_id': ['U001', 'U001', 'U001', 'U002', 'U002', 'U002', 'U003'],
                'product': ['Laptop', 'Mouse', 'Laptop', 'Laptop', 'Keyboard', 'Mouse', 'Laptop'],
                'amount': ['1,200', '30', '1,500', '900', '80', '25', '2,000']
            }), hide_index=True)

            r9.write("Row 8-14")
            r9.dataframe(pd.DataFrame({
                'tx_id': ['n288', 'x103', 't333', 'n411', 'x019', 'x029', 'n023'],
                'customer_id': ['U003', 'U003', 'U004', 'U004', 'U004', 'U005', 'U005'],
                'product': ['Keyboard', 'Keyboard', 'Mouse', 'Laptop', 'Laptop', 'Mouse', 'Laptop'],
                'amount': ['100', '120', '20', '1,100', '1,400', '15', '800']
            }), hide_index=True)            
        with c2:
            st.code(
'''
  products_df.groupBy('customer_id')\\
             .pivot('product') \\
             .agg(
                F.sum('amount').alias('t.amount'),
                F.count('*').alias('cnt')
             ).fillna(0)\\
             .orderBy('customer_id')
''', language='python'
            )
            st.write('**:green[Output]**')
            _, oc, _ = st.columns([1, 29, 1], gap='small')
            oc.dataframe(pd.DataFrame({
                'customer_id': ['U001', 'U002', 'U003', 'U004', 'U005'],
                'Keyboard_t.amount': [0, 80, 220, 0, 0],
                'Keyboard_cnt': [0, 1, 2, 0, 0],
                'Laptop_t.amount': ['2,700', '900', '2,000', '2,500', '800'],
                'Laptop_cnt': [2, 1, 1, 2, 1],
                'Mouse_t.amount': [30, 25, 0, 20, 15],
                'Mouse_cnt': [1, 1, 0, 1, 1]
            }), hide_index=True) 

elif topic == "EDA":
    if title == 'Eda01. Schema, columns & partition count':
        c1, c2 = st.columns([9, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong EDA (chỉ bàn về Performance & Data Quality), chứ không chỉ EDA thông thường; mỗi phép phân tích phải trả lời được ít nhất một câu hỏi:
                        - Data có vấn đề gì: null, duplicate, outlier, skew?
                        - Phép xử lý nào có thể gây shuffle, spill hoặc OOM?
                        - Có cách nào giảm số lần scan, giảm lượng dữ liệu và rút ngắn execution time không?

                        Đầu tiên, với dữ liệu FACT table `transactions_df`, Hãy
                        - kiểm tra số cột, data types, file formats, số partitions, kích thước dữ liệu đầu vào. 
                        - Tìm các cột không cần thiết và các kiểu dữ liệu quá tốn bộ nhớ.
                    ''')
            st.write('**:green[Input dataframe]**')
            st.dataframe(pd.DataFrame({
                'InvoiceNo': ['536365', '536365', '536365', '536365', '536365', '...', '544812', 'C538847'],
                'StockCode': ['85123A', '71053', '84406B', '84029G', '84029E', '...', '22087', '85232B'],
                'Description': ['WHITE HANGING HEART T-LIGH..',
                                'WHITE METAL LANTERN',
                                'CREAM CUPID HEARTS COAT HA..',
                                'KNITTED UNION FLAG HOT WAT...',
                                'RED WOOLLY HOTTIE WHITE HE...', '...',
                                'PAPER BUNTING WHITE LACE', 'SET OF 3 BABUSHKA...'
                                ],
                'Quantity': ['6', '6', '8', '6', '6', '...', '3', '-240'],
                'InvoiceDate': ['12/1/2010 8:26',
                                '12/1/2010 8:26',
                                '12/1/2010 8:26',
                                '12/1/2010 8:26',
                                '12/1/2010 8:26', '...', '2/23/2011 15:58', '12/14/2010 13:27'                         
                                ],
                'UnitPrice': ['2.55', '3.39', '2.75', '3.39', '3.39', '...', '5.79', '4,95'],
                'CustomerID': ['17850', '17850', '17850', '17850', '17850', '...', 'NULL', '17404'], 
                'Country': ['United Kingdom',
                            'United Kingdom',
                            'United Kingdom',
                            'United Kingdom',
                            'United Kingdom', '...', 'United Kingdom', 'Sweeden'
                        ]
            }), hide_index=True)
        with c2:
            st.write("**Schema & data dimension**")
            st.code(
'''
  transactions_df.printSchema()

  print("Columns:", len(transactions_df.columns))
  print("Partitions:", transactions_df.rdd.getNumPartitions())
''', language='python'
            )
            st.write("""
                        **Output**

                                root
                                |-- InvoiceNo: string (nullable = true)
                                |-- StockCode: string (nullable = true)
                                |-- Description: string (nullable = true)
                                |-- Quantity: integer (nullable = true)
                                |-- InvoiceDate: string (nullable = true)
                                |-- UnitPrice: double (nullable = true)
                                |-- CustomerID: integer (nullable = true)
                                |-- Country: string (nullable = true)

                                Columns: 8
                                Partitions: 2

                        Không nên mặc định dùng `df.count()` chỉ để xem kích thước. `count()` có thể quét toàn bộ dữ liệu nguồn, đặc biệt tốn thời gian với file lớn hoặc nguồn không có metadata hữu ích.
                    """)

    elif title == 'Eda02. Null profiling':
        c1, c2, c3 = st.columns([7, 8, 3.5], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Khi đếm `null-values` ở tất cả các cột; thay vì gọi `filter().count()` cho từng cột, tạo một aggregation cho tất cả cột cần kiểm tra.

                        Một số lưu ý khác:
                        - `count(col)` bỏ qua `null`, nhưng `count(when(...))` đếm đúng số `null`.
                        - `NaN` trong cột floating-point không giống `NULL`; cần kiểm tra riêng.
                        - Nếu có hàng nghìn cột, đừng tạo một `aggregation` khổng lồ; hãy chia thành các nhóm cột để tránh query plan quá lớn.

                        Ta vẫn sử dụng dữ liệu `transactions_df` đã giới thiệu ở `Eda01.`
                    ''')
        with c2:
            null_df =  {'col': {0: 'InvoiceNo',
                                1: 'StockCode',
                                2: 'Description',
                                3: 'Quantity',
                                4: 'InvoiceDate',
                                5: 'UnitPrice',
                                6: 'CustomerID',
                                7: 'Country'},
                                'null_cnt': {0: 0, 1: 0, 2: 299, 3: 0, 4: 0, 5: 0, 6: 34935, 7: 0},
                                'null_perc': {0: 0.0,
                                1: 0.0,
                                2: 0.3,
                                3: 0.0,
                                4: 0.0,
                                5: 0.0,
                                6: 34.85,
                                7: 0.0}}
            st.write("**Giải pháp**")
            st.code(
'''
  agg_exprs = [F.count(F.lit(1)).alias("_rows")]

  agg_exprs += [
                F.count(
                    F.when(F.col(c).isNull(), 1)
                ).alias(c)
                for c in transactions_df.columns
            ]

  stats = transactions_df.agg(*agg_exprs).first().asDict()
  n = stats.pop("_rows")
  pd.DataFrame({
        "col": stats.keys(),
        "null_cnt": stats.values(),
        "null_perc": [round(v / n * 100, 2) for v in stats.values()]
    })
'''
            )
        c3.write("**Output**")
        c3.dataframe(null_df, hide_index=True)

    elif title == 'Eda03. Numeric distribution':
        c1, c2 = st.columns([3, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Ví dụ ở đây ta sẽ quan tâm 2 cột numeric là `Quantity` và `UnitPrice`.

                        Trong các bước triển khai nâng cao, ta có thể tạo cột `amount` bằng cách `UnitPrice * Quantity`

                        Ngoài ra, EDA nâng cao ta sẽ phân tích luôn cả `negative rate` và `zero_rate`
                    ''')

        with c2:
            st.code(
'''
  numeric_cols = ['Quantity', 'UnitPrice']
  transactions_df.select(*[
        expr
        for c in numeric_cols
        for expr in [
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
            F.round(F.mean(c).alias(f"{c}_mean"), 2).alias(f'{c}_avg'),
            F.round(F.stddev(c).alias(f"{c}_std"), 2).alias(f'{c}_std')
        ]
    ]).show()
''', language='python'
            )
            st.write("**Output**")
            st.dataframe(pd.DataFrame({
'Quantity_min': {0: -74215},
 'Quantity_max': {0: 74215},
 'Quantity_avg': {0: 8.89},
 'Quantity_std': {0: 336.15},
 'UnitPrice_min': {0: 0.0},
 'UnitPrice_max': {0: 16888.02},
 'UnitPrice_avg': {0: 5.39},
 'UnitPrice_std': {0: 120.74}
            }), hide_index=True)

    elif title == 'Eda04. Quantiles & outlier screening':
        c1, c2 = st.columns([4, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Khi xét đến phân vị (`quantiles`) thì `approxQuantile()` là lựa chọn thực tế để ước lượng percentile mà không phải sort toàn bộ dataset để lấy quantile chính xác.

                        Trong bài toán này, ta vẫn dùng `transactions_df`, ở đây ta muốn xét các phân vị trong `transaction_amount`
                    ''')
        with c2:
            st.code(
"""
  transactions_df = transactions_df.withColumn(
                                        "transaction_amount", 
                                        col("Quantity") * col("UnitPrice"))
  quantiles = transactions_df.approxQuantile(
                                    "transaction_amount",
                                    [0.01, 0.25, 0.50, 0.75, 0.95, 0.99],
                                    0.01
                                )
  quantiles
""", language = 'python'
            )
            st.write("""
                    **Output**

                        [-77183.6, 3.36, 8.42, 16.94, 50.98, 77183.6]
                    """)

    elif title == 'Eda05. Drop duplicates then Repartition':
        c1, c2 = st.columns([6, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Với dữ liệu `transactions_df` này, ta sẽ
                        - Kiểm tra duplicate theo `InvoiceNo`
                        - Ước tính số `InvoiceNo` distinct
                    ''')
        with c2:
            st.code(
'''
transactions_df.groupBy("InvoiceNo").count().filter(F.col("count") > 1) \\
                .orderBy(F.desc("count")).show(5)
'''
            )
            st.write("**Output**")
            _, oc1, _ = st.columns([1,2,1])
            with oc1:
                st.dataframe(pd.DataFrame({'InvoiceNo': {0: '537434',
                                            1: '538071',
                                            2: '538349',
                                            3: '537638',
                                            4: '537237'},
                                            'count': {0: 675, 1: 652, 2: 620, 3: 601, 4: 597}}), hide_index=True)
            st.divider()
            st.code(
"""
transactions_df.agg(
                    F.count("InvoiceNo").alias("non_null_ids"),
                    F.approx_count_distinct("InvoiceNo").alias("approx_distinct_ids")
                ).show()
""", language='python'
            )
            st.write("**Output**")
            _, oc2, _ = st.columns(3)
            with oc2:
                st.dataframe(pd.DataFrame({'non_null_ids': {0: 100247}, 'approx_distinct_ids': {0: 4963}}), hide_index=True)

    elif title == 'Eda06. Skewness notes':
        c1, c2 = st.columns([6, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong banking, một vài customer hoặc merchant có thể chiếm tỷ trọng giao dịch rất lớn. Để biết mức độ tập trung của hot keys, có thể lấy tổng số records của top keys so với tổng dữ liệu. Đừng chỉ nhìn số lượng distinct customer: 1 triệu customer không có nghĩa là phân phối giao dịch đồng đều.

                        Kiểm tra dimension trước khi join. Đây là một trong những kỹ năng quan trọng nhất để tránh join làm tăng số lượng dòng ngoài dự kiến.
                    ''')