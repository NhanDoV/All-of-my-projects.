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
            'GA03. Transformation on Nested Structs',
            'GA04. Dynamic Skewed Join',
            'GA05. Massive Scale Percentile Approx' 
        ],
    'GroupPivot': [
            'GP01. Pivot Student Subject Marks',
            'GP02. Purchase per category for each customer',
            'GP03. Dynamic High-Cardinality'
        ],
    'Timestamp Processor': [
            'TP01. Extract daypart',
            'TP02. Previous & Next date',
            "TP03. Transactions's statistics over 30 days",
            'TP04. Event-Time Deduplication',
            'TP05. Time Decay vs Multi-Touch Funnel',
        ],
    'Regex Clause': [
            'RC01. Find spending-behaviors pattern',
            'RC02. rlike, lower & trim',
            'RC03. regexp + extract/replace/instr/count',
            'RC04. Customers used E-wallet & gmail',
            'RC05. Valid-IP & confused transaction-description'
        ],
    'Window': [
            'Wd01. Rising Temperatures',
            'Wd02. Distinguish rank, dense_rank & row_num',
            'Wd03. End-to-end Sales Analytic Pipeline',
            'Wd04. User Session Island Detection',
            'Wd05. Cumulative Revenue by Dynamic_Window',
    ]
}

# Cột slidebar để chọn topic và title
topic_titles = list(topic_title_dict.keys())

with st.sidebar:
    topic = st.selectbox('Chọn topic', topic_titles)
    title = st.selectbox('Chọn title', topic_title_dict[topic])

# Main app
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
        left_op, right_op = st.columns(2, gap='medium')
        with left_op:
            st.write("##### :orange[Input]")
            _, ic, _ = st.columns([1,9,1])
            with ic:
                st.dataframe(df, hide_index=True)
        with right_op:
            st.write("##### :orange[Output]")            
            res = df.copy()
            _, oc, _ = st.columns([1,9,1])
            res = df.sort_values(by='salary', ascending=False).head(5)
            with oc:
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
            st.write("**:green[Khi nào nên dùng đến]**")
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

                        :orange[→ repartition phân phối theo key không đồng nghĩa với chia nhỏ một hot key thành nhiều partition]
                    """,)
        with c2:
            st.write(""" 
                        **:blue[Salt-key giải quyết ntn??]**

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
                    - Fact có `customer_id = 121` được chia ngẫu nhiên vào `salt 0 - 9`.
                    - Customer 121 ở bảng dimension được nhân thành 10 bản sao, mỗi bản mang một salt tương ứng.
                    - Các customer còn lại vẫn dùng `salt 0` ở cả hai bảng.
                    - Khi join, mỗi giao dịch vẫn khớp với customer tương ứng, nhưng workload của hot key có cơ hội được phân phối qua nhiều partition.
                    
                    :red[**Lưu ý**: salting làm tăng số bản ghi của bảng dimension đối với hot key, và không đảm bảo tuyệt đối workload cân bằng. Số salt cần chọn dựa trên mức độ skew và tài nguyên cluster].
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

    elif title == 'Wd03. End-to-end Sales Analytic Pipeline':
        c1, c2 = st.columns([7, 5], gap='large')
        with c1:
            st.write(f"""
                     ##### :blue[{title}]
                     Cho 3 bảng `customers_df`, `orders_df` và `products_df`, hãy xây dựng một quy trình `Data Mart` đạt chuẩn sản xuất `production-grade` bằng cách kết hợp các kỹ thuật như `filtering`, `multiple table joins`, `aggregations` và `window ranking`, theo đó
                     - **:yellow[Data cleaning]:** Lọc ra những đơn hàng (`orders`) có `amount` không hợp lệ (bị `NULL` hoặc giá trị âm) hoặc `order_date` bị `NULL`
                     - **:yellow[Relational Enrichment]:** Join cả 3 bảng đã cho với nhau theo key phù hợp
                     - **:yellow[Customer Metrics]:** Với mỗi `country` và `customer_name`, tính `total_spent = round(sum(amount), 2)` và `order_count = count(order_id)`
                     - **:yellow[Country Ranking]:** Tạo biến `spending_rank` bằng `F.dense_rank()` phân hoạch trên `country` và sắp xếp theo `total_spent DESC`
                     - **:yellow[Top Performers Filter]:** Lọc các dữ liệu theo `spending_rank <= 2` (chỉ lấy top 2 các dữ liệu theo từng nhóm)
                     - **:yellow[Final Ordering]:** Sắp xếp theo `country ASC`, `spending_rank ASC` và `customer_name ASC`
                    """)
            st.divider()
            st.write("##### :green[Input]")
            customers_df = pd.DataFrame({
                'customer_id': [1,2,3,4,5],
                'customer_name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Elena'],
                'country': ['USA', 'USA', 'USA', 'UK', 'UK']
            })
            orders_df = pd.DataFrame({
                'order_id': [101, 102, 103, 104, 105, 106, 107],
                'customer_id': [1, 1, 2, 3, 4, 5, 5],
                'product_id': [50, 51, 50, 50, 52, 52, 52],
                'order_date': ['2024-01-10', '2024-01-15', '2024-01-12', '2024-01-14', '2024-01-20', '2024-01-22', 'NULL'],
                'amount': [500, 300, 400, 150, 900, 750, 999]
            })
            products_df = pd.DataFrame({
                'product_id': [50, 51, 52],
                'product_name': ['Laptop', 'Monitor', 'Phone'],
                'category': ['Electronics', 'Electronics', 'Electronics']
            })

            cust_col, prod_col = st.columns([3, 5], gap='medium')
            cust_col.write("**:orange[products_df]**")
            cust_col.dataframe(products_df, hide_index=True)

            cust_col.write("**:orange[customers_df]**")
            cust_col.dataframe(customers_df, hide_index=True)

            prod_col.write("**:orange[orders_df]**")
            prod_col.dataframe(orders_df, hide_index=True)


        with c2:
            st.code(
'''
from pyspark.sql.window import Window

valid_od = orders.filter(
                        F.col('amount').isNotNull() & 
                        (F.col('amount') > 0) &
                        F.col('order_date').isNotNull() 
                )
df_agg = valid_od.join(customers_df, on='customer_id', how='inner') \\
                .join(products_df, on='product_id', how='inner') \\
                .groupBy('country', 'customer_name') \\
                .agg(
                    F.round(F.sum('amount'), 2).alias('total_spent'),
                    F.count('order_id').alias('order_count')
                )
wd_rnk = Window.partitionBy('country')\\
                .orderBy(F.col('total_spent').desc())

df_agg.withColumn('spending_rank', F.dense_rank().over(wd_rnk)) \\
      .filter( F.col('spending_rank') <= 2 )
      .orderBy(
            F.col('country').asc(), F.col('spending_rank').asc(),
            F.col('customer_name').asc()
        ) 
'''
            )
            out_df = pd.DataFrame({
                'country': ['UK', 'UK', 'USA', 'USA'],
                'customer_name': ['Diana', 'Elena', 'Alice', 'Bob'],
                'total_spent': [900, 750, 800, 400],
                'order_count': [1, 1, 2, 1],
                'spending_rank': [1, 2, 1, 2]
            })
            st.write("**:green[Output]**")
            _, oc, _ = st.columns([1, 9, 1])
            oc.dataframe(out_df, hide_index=True)

    elif title == 'Wd04. User Session Island Detection':
        c1, c2 = st.columns([7, 11], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Các hệ thống **:green[Product Analytics]** và **:green[Growth Analytics]** thường sẽ thực hiện quá trình **:orange[sessionization]** 
                        trên dữ liệu `clickstream` thô được thu thập từ `webapp` hoặc `mobile`

                        - Một `user session` được định nghĩa là một chuỗi các sự kiện liên tục của cùng một người dùng, trong đó 
                        `khoảng thời gian` không hoạt động (`inactivity gap`) giữa 2 sự kiện liên tiếp **:red[nhỏ hơn 30 phút (1,800 giây)]**

                        - Nếu khoảng cách giữa 2 sự kiện liên tiếp **:yellow[lớn hơn hoặc bằng 30 phút]**, hệ thống sẽ xem nó là sự khởi đầu của một `session mới`

                        Dựa vào bảng `raw_events_df` dưới đây, hãy
                        - Tạo `session_id` dựa trên `{{user_id}}{{session_number}}` (e.g, `u001_1, u001_2`)
                        - Tách ra giá trị bắt đầu (`session_start`) và kết thúc (`session_end`) của mỗi phiên
                        - Tính `session_event_count`: đếm các sự kiện trong phiên (`session`)
                    
                        **:blue[Input dataframe]**
                    ''')

            _, ic, _ = st.columns([1, 4, 1])
            with ic:
                st.dataframe(pd.DataFrame({
                    'user_id': ['u001', 'u001', 'u001', 'u001', 'u002'],
                    'event_ts': ['2024-01-15 09:00:00', '2024-01-01 09:12:00', '2024-01-15 09:45:00', '2024-01-15 09:58:00', '2024-01-15 10:00:00'],
                    'event_type': ['page_view', 'click', 'page_view', 'purchase', 'page_view']
                }), hide_index=True)
        with c2:
            st.code(
'''
 from pyspark.sql.window import Window
 
 lag_wd = Window.partitionBy('user_id').orderBy('event_ts')
 tm_wdw = Window.partitionBy('user_id').orderBy('event_ts')\\
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)

 event_with_ssn_df = df.withColumn('prev_ts', F.lag('event_ts', 1).over(lag_wd))\\
                       .withColumn('is_new_session', 
                            F.when(F.col('prev_ts').isNull() | 
                                (F.unix_timestamp('event_ts') - F.unix_timestamp('prev_ts') >= 1800),
                            1).otherwise(0)
                      ).withColumn('sess_num', F.sum('is_new_session').over(tm_wdw))\\
                      .withColumn('session_id',
                                  F.concat_ws('_', F.col('user_id'), F.col('sess_num').cast('string'))
                                )
 ssn_agg = event_with_ssn_df.groupBy('user_id', 'session_id')\\
                            .agg(
                                F.min('event_ts').alias('session_start'),
                                F.max('event_ts').alias('session_end'), F.count('*').alias('event_count')
                            )

 event_with_ssn_df.join(ssn_agg, ['user_id', 'session_id'], 'left')\\
                  .select('user_id', 'session_id',
                          F.date_format('event_ts', 'yyyy-MM-dd HH:mm:ss').alias('event_ts'),
                          'event_type', 
                          F.date_format('session_start', 'yyyy-MM-dd HH:mm:ss').alias('session_start'),
                          F.date_format('session_end', 'yyyy-MM-dd HH:mm:ss').alias('session_end'),
                    ).orderBy('user_id', 'event_ts').show()
'''
            )
            st.write("**Output**")
            _, oc = st.columns([1, 9])
            with oc:
                st.dataframe(pd.DataFrame({
                    'user_id': ['u001', 'u001', 'u001', 'u001', 'u002'],
                    'session_id': ['u001_1', 'u001_2', 'u001_3', 'u001_3', 'u002_1'],
                    'event_ts': ['2024-01-01 09:12:00', '2024-01-15 09:00:00', '2024-01-15 09:45:00', '2024-01-15 09:58:00', '2024-01-15 10:00:00'],
                    'event_type': ['click', 'page_view', 'page_view', 'purchase', 'page_view'],
                    'session_start': ['2024-01-01 09:12:00', '2024-01-15 09:00:00', '2024-01-15 09:45:00', '2024-01-15 09:45:00', '2024-01-15 10:00:00'],
                    'session_end': ['2024-01-01 09:12:00', '2024-01-15 09:00:00', '2024-01-15 09:58:00', '2024-01-15 09:58:00', '2024-01-15 10:00:00']
                }), hide_index=True)

    elif title == 'Wd05. Cumulative Revenue by Dynamic_Window':
        c1, c2 = st.columns([7, 9], gap='medium')
        with c1:
            st.write(f"""
                    ##### :blue[{title}]
                    Hệ thống báo cáo tài chính yêu cầu các dashboard :blue[doanh thu host] theo :green[ngày], theo dõi :yellow[tổng doanh thu tích lũy] :green[suốt đời (`lifetime`)] , :blue[trung bình động] theo lịch, :blue[tổng tích lũy từ đầu tháng] (MTD), và :blue[xếp hạng theo quốc gia] hàng ngày. Trong dữ liệu `marketplace` thực tế, :red[host không có booking mỗi ngày (dữ liệu lịch thưa thớt)]. 

                    Cho bảng `host_daily_revenue_df` chứa các bản ghi doanh thu host theo ngày, hãy tính toán các chỉ số sau:
                    - `running_total`: Tổng doanh thu tích lũy không giới hạn (unbounded) của mỗi host, tính từ ngày booking đầu tiên đến dòng hiện tại.
                    - `rolling_7d_avg`: Trung bình động 7 ngày theo lịch (bao gồm 6 ngày lịch liền trước + ngày hiện tại; không phải 7 dòng gần nhất).
                    - `mtd_cumulative`: Tổng doanh thu tích lũy từ đầu tháng (`Month-to-Date [MTD]`), reset về 0 tại mỗi ranh giới tháng theo lịch.
                    - `country_daily_rank`: Xếp hạng doanh thu hàng ngày của host trong cùng quốc gia, sử dụng `dense_rank()` sắp xếp theo `daily_revenue` giảm dần (DESC).
                    
                    Sắp xếp kết quả cuối cùng theo `host_id ASC`, `revenue_date ASC`. 

                    **:orange[Input dataframe: host_daily_revenue_df]**
                    """)
            data = {
                    "host_daily_revenue_df": [
                        {
                        "host_id": "h001",
                        "country": "US",
                        "revenue_date": "2024-01-01",
                        "daily_revenue": 500
                        },
                        {
                        "host_id": "h001",
                        "country": "US",
                        "revenue_date": "2024-01-03",
                        "daily_revenue": 300
                        },
                        {
                        "host_id": "h001",
                        "country": "US",
                        "revenue_date": "2024-01-05",
                        "daily_revenue": 450
                        },
                        {
                        "host_id": "h002",
                        "country": "US",
                        "revenue_date": "2024-01-01",
                        "daily_revenue": 700
                        },
                        {
                        "host_id": "h002",
                        "country": "US",
                        "revenue_date": "2024-01-07",
                        "daily_revenue": 200
                        }
                    ]
                }
            _, ic, _ = st.columns([1, 9, 1])
            with ic:
                st.dataframe(pd.DataFrame(data['host_daily_revenue_df']), hide_index=True)
        with c2:
            st.code(
'''
  df = host_daily_revenue_df.withColumn("revenue_date", F.to_date("revenue_date"))
                            .withColumn(
                               "date_epoch_days",
                               F.datediff(F.col("revenue_date"), F.lit("1970-01-01").cast("date"))
                            )
  w_running = Window.partitionBy("host_id").orderBy("date_epoch_days")
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)    
  w_rolling_7d = Window.partitionBy("host_id").orderBy("date_epoch_days")
                        .rangeBetween(-6, Window.currentRow)
  w_mtd = Window.partitionBy("host_id", F.year("revenue_date"), F.month("revenue_date")) \\
                .orderBy("date_epoch_days")\\
                .rowsBetween(Window.unboundedPreceding, Window.currentRow))
  w_country_rank = Window.partitionBy("country", "revenue_date").orderBy(F.desc("daily_revenue"))

  df.withColumn("running_total", F.sum("daily_revenue").over(w_running)) \\
    .withColumn("rolling_7d_avg", F.avg("daily_revenue").over(w_rolling_7d)) \\
    .withColumn("mtd_cumulative", F.sum("daily_revenue").over(w_mtd)) \\
    .withColumn("country_daily_rank", F.dense_rank().over(w_country_rank)) \\
    .select(
        "host_id", "country", 
        F.date_format("revenue_date", "yyyy-MM-dd").alias("revenue_date"), 
        "daily_revenue",
        F.round("running_total", 2).alias("running_total"),
        F.round("rolling_7d_avg", 2).alias("rolling_7d_avg"),
        F.round("mtd_cumulative", 2).alias("mtd_cumulative"),
        "country_daily_rank"
    ).orderBy("host_id", "revenue_date"))
'''
            )
            _, oc = st.columns([1, 29], gap='small')
            with oc:
                st.dataframe(pd.DataFrame({
                    'host_id': ['h001', 'h001', 'h001', 'h002', 'h002'],
                    'country': ['US', 'US', 'US', 'US', 'US'],
                    'revenue_date': ['2024-01-01', '2024-01-03', '2024-01-05', '2024-01-01', '2024-01-07'],
                    'daily_revenue': ['500', '300', '450', '700', '200'],
                    'running_total': ['500', '800', '1,250', '700', '900'],
                    'rolling_7d_avg': ['500.0', '400.0', '416.67', '700.0', '450.0'],
                    'mtd_cumulative': ['500', '800', '1,250', '700', '900'],
                    'country_daily_rank': ['2', '1', '1', '1', '1']
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
                    - `education` nếu `tx_description` có chứa `khóa học, học phí, ngoại ngữ, ...`
                    - `investment` nếu `tx_description` có chứa `chứng khoán, cổ phiếu, đầu tư, ...`
                    - `gamming` nếu `tx_description` có chứa `game, nạp, vật phẩm, ...`                    
                    """)
            st.write("**:green[SYNTAX NOTES]**")
            st.code('''
  F.col('_').rlike(pattern)         --> return True/False
  F.regexp_extract(F.col('_'),
                     pattern, replacement
                    )               --> pull a pice out   
  F.split(F.col('_'), pattern)      --> split by a regex-delimeter
                    ''', language='python')
            st.write('**:green[Patterns rules]**')
            st.dataframe(
                pd.DataFrame({
                    'pattern': ['^/$', '.', '\\d / \\D', '\\w / \\W', '\\s / \\S', '[abc]', '[^abc]', '(...)'],
                    'read it as': ['start/end of string', 
                                   'any single char', 'digit/ non digit', 
                                   'word (letter/digits/underscore) / not',
                                   'whitespace or not', 'one of a,b or c',
                                   'anything except a/b/c', 'capture group'],
                    'examples': ['^hi$ matches exactly "hi"', 
                                 'a.c matches "abc", "axc"', "\\d\\d is 2 digits",
                                 '\\w+ is word chunk', '\\s+ is one or more spaces',
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

    elif title == "RC04. Customers used E-wallet & gmail":
        c1, c2 = st.columns([6, 5],  gap='medium')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong dữ liệu `transactions_df`, có các trường: `name`, `payment_content`, `email` và `amount`
                        
                        Người ta muốn check xem có bao nhiêu \% số giao dịch cũng như \% tổng số tiền giao dịch đến từ những nhóm:
                        - users sử dụng `gmail` đăng ký tài khoản
                        - users sử dụng `ví điện tử (E-wallet)` như zalo, momo, ...
                    ''')
            ipc, opc = st.columns([7, 4], gap='small')
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
                    'amount': ['75,000', '92,000', '111,000', '68,000', '80,000', '69,000', '89,000']
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
                            Trong đó, cột `measure_by` sẽ có 2 giá trị
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

    elif title == "RC02. rlike, lower & trim":
        c1, c2 = st.columns([5, 7], gap='large')
        with c1:
            st.write(f"""
                    ##### :blue[{title}]
                    Ở phần này, chúng ta sẽ tìm hiểu thêm về `trim` và `lower` 
                    - :green[F.trim()] : dùng để xóa khoảng trắng trong text
                    - :green[F.lower()] : dùng để đưa toàn bộ text về chữ in thường
                    - :green[F.col().rlike(...)]: trả về :blue[True]/:red[False] 

                    Trong ví dụ dưới đây, ta sẽ minh họa dữ liệu đó như sau:

                    ```python
                    from pyspark.sql import SparkSession
                    from pyspark.sql import functions as F

                    spark = SparkSession.builder.appName('Nhan').getOrCreate()
                    df = spark.createDataFrame(
                        data = [
                            (101, ' Tony Adam', '0912-345-678'),
                            (102, '  Hilary Bobby', 'unknow'),
                            (103, 'Mario Carlos', '0988.111.888'),
                            (104, '   Le Cun  ', '0905555239')
                        ],
                        schema = ['emp_id', 'full_name', 'phone_number']
                    )
                    # -----------
                    story_df = spark.createDataFrame(
                            data = [
                                (101, '123 is not equal 122'),
                                (102, '123/12 St Andrew ...'),
                                (103, 'today is not good, 1+1=2')
                            ],
                            schema = ['story_id', 'description']
                    )                 
                    ```
                    """)
        with c2:
            st.markdown(
'''

    ```python
    df.withColumn('name_normalized', F.trim('full_name')) \\
        .withColumn('lower_name_norm', F.lower('name_normalized'))\\
        .withColumn('phone_start_with_09', F.col('phone_number').rlike('^09'))
    ```

    **:blue[Output]**:

            +------+--------------+------------+---------------+---------------+-------------------+
            |emp_id|full_name     |phone_number|name_normalized|lower_name_norm|phone_start_with_09|
            +------+--------------+------------+---------------+---------------+-------------------+
            |101   | Tony Adam    |0912-345-678|Tony Adam      |tony adam      |true               |
            |102   |  Hilary Bobby|unknow      |Hilary Bobby   |hilary bobby   |false              |
            |103   |Mario Carlos  |0988.111.888|Mario Carlos   |mario carlos   |true               |
            |104   |   Le Cun     |0905555239  |Le Cun         |le cun         |true               |
            +------+--------------+------------+---------------+---------------+-------------------+

    Dưới đây là một ví dụ khác để hiểu rõ hơn :green[F.col(...).rlike()] hoạt động, để check văn bản bắt đầu hoặc kết thúc bởi các chữ số (digit)

    ```python
    # Chúng ta có thể dùng \\d hay [0-9] đều như nhau
    story_df.withColumn('start_at_num', F.col('description').rlike(r'^\\d')) \\
            .withColumn('end_by_num', F.col('description').rlike(r'\\d$'))
    ```
    **:blue[Output]**:

            +--------+------------------------+------------+----------+
            |story_id|description             |start_at_num|end_by_num|
            +--------+------------------------+------------+----------+
            |1001    |123 is not equal 122    |true        |true      |
            |1002    |123/12 St Andrew ...    |true        |false     |
            |1003    |today is not good, 1+1=2|false       |true      |
            +--------+------------------------+------------+----------+
'''
            )

    elif title == "RC03. regexp + extract/replace/instr/count":
        c1, c2 = st.columns([4, 7], gap='large')
        with c1:
            st.markdown(f"""
                    ##### :blue[{title}]
                    Cả 4 phương thức :green[_replace], :green[_extract], :green[_instr] và :green[_count] trong :green[regexp] của :orange[Pysparks]
                    đều được dùng để xử lý chuỗi với các mục đích riêng biệt
                    - :green[regexp_extract]: trích xuất

                            Syntax: regexp_extract(col, pattern, group_index)

                    - :green[regexp_replace]: thay thế các substrings khớp với pattern

                            Syntax: regexp_replace(col, pattern, replacement)

                    - :green[regexp_count]: đếm số lần mà pattern xuất hiện

                            Syntax: regexp_count(col, pattern)

                    - :green[regexp_instr]: trả về vị trí mà substring đầu tiên khớp pattern

                            Syntax: regexp_instr(col, pattern)
                                                
                    **:violet[Create sample dataframe]**
                    ```python
                    df = spark.createDataFrame(
                        data = [
                            (91, 'Đơn hàng: ORD123-HN [số lượng: 121] from TN'),
                            (92, 'Đơn hàng: ORD598-SG [số lượng: 238] from QN'),
                            (93, 'Đơn hàng: UNKNOWN-XX [số lượng: 0] from ZZ')
                        ], schema = ['oid', 'description']
                    )
                    ```
                """)
        with c2:
            st.write('''
                    ```python
                    df.withColumn('order_id', F.regexp_extract('description', r'ORD(\\d{3})', 1))\\
                       .withColumn('full_order_id', F.regexp_extract('description', r'ORD(\\d{3})', 0))\\
                       .withColumn('from', F.regexp_extract('description', r'from (\\w+)', 1))\\
                       .withColumn('to', F.regexp_extract('description', r'-(\\w+)', 1))\\
                       .withColumn('quantity', F.regexp_extract('description', r'\\[số lượng: (\\d+)\\]', 1))
                    ``` 
                    ''')
            n1, oc1 = st.columns([2, 19])
            with n1:
                st.write("**:green[Output]**")
            with oc1:
                st.dataframe(pd.DataFrame({
                    'oid': ['91', '92', '93'],
                    'description': ['Đơn hàng: ORD123-HN [số lượng: 121] from TN', 
                                    'Đơn hàng: ORD598-SG [số lượng: 238] from QN', 
                                    'Đơn hàng: UNKNOWN-XX [số lượng: 0] from ZZ'],
                    'order_id': ['123', '598', ''],
                    'full_order_id': ['ORD123', 'ORD598', ''],
                    'from': ['TN', 'QN', 'ZZ'],
                    'to': ['HN', 'SG', 'XX'],
                    'quantity': ['121', '238', '0']
                }), hide_index=True)
            
            st.divider()
            st.write('''
                    ```python
                    df.withColumn('cleaned_text', 
                                    F.trim(
                                            F.regexp_replace('description', 'Đơn hàng:', ''))
                        ).withColumn('removed_num', F.regexp_replace('cleaned_text', r'(\d+)', '')) \\
                         .withColumn('count_digits', F.regexp_count('description', F.lit(r'\d'))) \\
                         .withColumn('1st-idx_hyphen', F.regexp_instr('description', F.lit('-'))) \\
                         .drop('description')                    
                    ```
                    ''')
            n2, oc2 = st.columns([3, 19])
            with n2:
                st.write("**:green[Output]**")
            with oc2:
                st.dataframe(pd.DataFrame({
                    'oid': ['91', '92', '93'],
                    'cleaned_text': ['ORD123-HN [số lượng: 121] from TN',
                                     'ORD598-SG [số lượng: 238] from QN',
                                     'UNKNOWN-XX [số lượng: 0] from ZZ'],
                    'removed_num': ['ORD-HN [số lượng: ] from TN',
                                'ORD-SG [số lượng: ] from QN',
                                'UNKNOWN-XX [số lượng: ] from ZZ'],
                    'count_digits': ['6', '6', '1'],
                    '1st-idx_hyphen': ['17', '17', '18']
                }), hide_index=True)

    elif title == "RC05. Valid-IP & confused transaction-description":
        c1, c2 = st.columns([3, 7],  gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong dữ liệu :red[ground truth] của `transactions_df`, có các trường: `tx_id`, `tx_content`, `IP_address` và `is_fraud`
                        
                        Người ta muốn sử dụng :yellow[Feature Engineering] để tạo thêm các biến quan sát:
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
  suspicious_patt = r"(?i)(Team\\s*Member\\s*Discount|Employee\\s*Benifit|Internal\\s*Staff|Payroll\\s*Override|VIP\\s*Internal|Bypass|Test\\s*Transaction)"
  ipv4_patt = r"^(?:(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)\\.){3}(?:25[0-5]|2[0-4]\d|1\\d\\d|[1-9]?\\d)$"

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

    elif title == 'TP04. Event-Time Deduplication':
        c1, c2 = st.columns([4, 3], gap='large')
        with c1:
            data = {
                    "payment_events_df": [
                        {
                        "payment_id": "P001",
                        "user_id": "U100",
                        "amount": 99.99,
                        "event_time": "2024-01-15 10:00:00"
                        },
                        {
                        "payment_id": "P001",
                        "user_id": "U100",
                        "amount": 99.99,
                        "event_time": "2024-01-15 10:05:00"
                        },
                        {
                        "payment_id": "P002",
                        "user_id": "U200",
                        "amount": 250,
                        "event_time": "2024-01-15 10:10:00"
                        },
                        {
                        "payment_id": "P003",
                        "user_id": "U300",
                        "amount": 45,
                        "event_time": "2024-01-15 08:30:00"
                        }
                    ]
                }
            st.write(
                f"""
                ##### :blue[{title}]
                Các kiến trúc ingestion thanh toán và messaging nhận luồng sự kiện với cơ chế retry mạng và semantics at-least-once delivery. Để duy trì tính nhất quán dữ liệu, pipeline phải loại bỏ trùng lặp sự kiện dựa trên business key (như `payment_id`) trong một cửa sổ theo event-time, đồng thời xử lý được các sự kiện đến muộn và không đúng thứ tự.
                
                Hãy triển khai giải pháp deduplication theo `event-time` với các yêu cầu sau:
                - `Watermark Threshold`: Thiết lập ngưỡng `watermark` 2 giờ so với timestamp sự kiện lớn nhất đã quan sát được (`max_ts - 2 hours`).
                - `Late-Arrival Pruning`: Loại bỏ các sự kiện cũ (stale) có `event_time` nằm trước ngưỡng **:red[watermark 2 giờ]**.
                - `Exact Deduplication`: Với các sự kiện có cùng `payment_id` nằm trong cửa sổ hợp lệ, chỉ giữ lại lần xuất hiện sớm nhất (sắp xếp theo `event_time ASC`).

                Sắp xếp kết quả cuối cùng theo `payment_id ASC`.

                **:orange[Input dataframe]**
                """
            )
            _, ic, _ = st.columns([1, 4, 1])
            with ic:
                st.dataframe(pd.DataFrame(data['payment_events_df']), hide_index=True)
            
        with c2:
            st.code(
'''
from pyspark.sql.window import Window

df = payment_events_df.withColumn("ts", F.to_timestamp("event_time"))
    
max_ts = df.agg(F.max("ts")).collect()[0][0]
watermark_boundary = max_ts.timestamp() - (2 * 3600)

valid_events = df.filter(F.unix_timestamp("ts") >= watermark_boundary)
wd = Window.partitionBy("payment_id").orderBy("ts")

deduped = valid_events.withColumn("rn", F.row_number().over(wd)) \\
                      .filter(F.col("rn") == 1) \\
                      .select(
                            "payment_id", "user_id", "amount",
                            F.date_format(
                                "ts", "yyyy-MM-dd HH:mm:ss"
                            ).alias("event_time")
                        ).orderBy("payment_id")
'''
            )
            st.write("**:blue[Output]**")
            _, oc, _ = st.columns([1, 6, 1])
            with oc:
                st.dataframe(pd.DataFrame({
                    'payment_id': ['P001', 'P002', 'P003'],
                    'user_id': ['U100', 'U200', 'U300'],
                    'amount': ['99.99', '250.0', '45.0'] 
                }), hide_index=True)

    elif title == 'TP05. Time Decay vs Multi-Touch Funnel':
        c1, c2 = st.columns([11, 14], gap='large')
        with c1:
            st.write(f"""
            ##### :blue[{title}]
            Các hệ thống phân tích marketing và thu hút khách hàng triển khai các mô hình **:green[Multi-Touch Attribution]** (`MTA`) để đo lường hiệu quả của từng kênh trong phễu chuyển đổi mua hàng của khách hàng. 
            
            Theo mô hình Time-Decay Attribution, các touchpoint xảy ra gần thời điểm chuyển đổi cuối cùng sẽ được phân bổ credit theo cấp số nhân cao hơn so với các touchpoint xuất hiện sớm hơn. Cho dữ liệu các tương tác touchpoint marketing dẫn đến việc khách hàng đặt chỗ (booking), hãy thực hiện các bước sau:
            - Tính `Recency`: Với mỗi touchpoint, tính `days_before_booking` là số ngày phân số giữa `touchpoint_ts` và `booking_ts` theo công thức `((booking_seconds - touchpoint_seconds) / 86400.0)`.
            - `Exponential Decay`: Tính trọng số thô (raw weight) với hệ số suy giảm $\lambda = 0.1$:

            $$\\qquad \\text{{weight}} = e^{{ -\lambda \\times \\text{{days\\_before\\_booking}} }} $$

            - `Chuẩn hóa Credit theo từng Booking:` Tổng hợp các raw weight theo từng booking, sau đó tính `credit đã chuẩn hóa cho mỗi touchpoint`: $$ \\text{{credit}} = \\frac{{\\text{{weight}} }}{{ \\sum \\text{{weights}} }} $$
            - `Tổng hợp theo Channel`: Aggregate toàn bộ các booking để tạo bảng tóm tắt attribution theo kênh (`channel`), bao gồm các cột: `channel, total_credit_allocated, total_touchpoints, unique_bookings` và `avg_credit`

            Sắp xếp kết quả cuối cùng theo `total_credit_allocated DESC, channel ASC`.

            **:orange[Input dataframe]**
            """)

            data = {
                    "touchpoints_df": [
                        {
                            "user_id": "U001",
                            "booking_id": "B001",
                            "channel": "paid_search",
                            "touchpoint_ts": "2024-01-10 10:00:00",
                            "booking_ts": "2024-01-15 12:00:00"
                        },
                        {
                            "user_id": "U001",
                            "booking_id": "B001",
                            "channel": "social_ad",
                            "touchpoint_ts": "2024-01-13 14:00:00",
                            "booking_ts": "2024-01-15 12:00:00"
                        },
                        {
                            "user_id": "U001",
                            "booking_id": "B001",
                            "channel": "email",
                            "touchpoint_ts": "2024-01-14 09:00:00",
                            "booking_ts": "2024-01-15 12:00:00"
                        },
                        {
                            "user_id": "U002",
                            "booking_id": "B002",
                            "channel": "organic_search",
                            "touchpoint_ts": "2024-01-12 08:00:00",
                            "booking_ts": "2024-01-15 18:00:00"
                        },
                        {
                            "user_id": "U002",
                            "booking_id": "B002",
                            "channel": "direct",
                            "touchpoint_ts": "2024-01-15 10:00:00",
                            "booking_ts": "2024-01-15 18:00:00"
                        }
                    ]
                }

            _, ic, _ = st.columns([1, 9, 1])
            with ic:
                st.dataframe(pd.DataFrame(data['touchpoints_df']), hide_index=True)

        with c2:
            st.code(
'''
LAMBDA = 0.1

weighted = touchpoints_df.withColumn("t_ts", F.to_timestamp("touchpoint_ts") )\\
                   .withColumn("b_ts", F.to_timestamp("booking_ts") ) \\
                   .withColumn(
                        "days_before_booking",
                        F.greatest(
                            (F.unix_timestamp("b_ts") - F.unix_timestamp("t_ts")) / 86400.0,
                            F.lit(0.0) )
                   ).withColumn("raw_weight",
                                F.exp(-LAMBDA * F.col("days_before_booking")) )

weights_sum = weighted.groupBy("user_id", "booking_id")\\
                      .agg(F.sum("raw_weight").alias("total_weight"))

credited = weighted.join(weights_sum, on=["user_id", "booking_id"], how="left") \\
                   .withColumn("credit", 
                                F.round(F.col("raw_weight") / F.col("total_weight"), 4)\\
                   ).groupBy("channel")\\
                   .agg(
                        F.round(F.sum("credit"), 4).alias("total_credit_allocated"),
                        F.count("*").alias("total_touchpoints"),
                        F.countDistinct("booking_id").alias("unique_bookings"),
                        F.round(F.avg("credit"), 4).alias("avg_credit")
                    ).orderBy(F.desc("total_credit_allocated"), "channel")
''', language='python'
            )
            st.write("**:blue[Output]**")
            _, oc, _ = st.columns([1, 19, 1])
            with oc:
                st.dataframe(pd.DataFrame({
                    'channel': ['direct', 'organic_search', 'email', 'social_ad', 'paid_search'],
                    'total_credit_allocated': ['0.5765', '0.4235', '0.3851', '0.3558', '0.2592'],
                    'total_touchpoints': ['1', '1', '1', '1', '1'],
                    'unique_bookings': ['1', '1', '1', '1', '1'],
                    'avg_credit': ['0.5765', '0.4235', '0.3851', '0.3558', '0.2592']
                }), hide_index=True)

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

    elif title == 'GA02. Sliding Event-Time Window':
        c1, c2 = st.columns([2, 3], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Với `events_df` dưới đây; trong các bài toán giám sát thời gian thực (`Real time monitoring system`), ta thường gom nhóm dữ liệu telemetry 
                        của các sự kiện theo các `sliding event-time window` để tính toán các chỉ số thông lượng (`throughput metrics`) 
                        trên thời gian thực.
                    ''')
            st.write('**:green[Input dataframe]**')
            events_df = pd.DataFrame({
                'event_id': ['1', '2', '3', '4'],
                'event_type': ['CLICK', 'CLICK', 'CLICK', 'PURCHASE'],
                'event_ts': ['2025-01-01 10:02:00', '2025-01-01 10:06:00', '2025-01-01 10:08:00', '2025-01-01 10:09:00'],
                'value': ['1', '2', '3', '50'],
            })
            _, ic, _ = st.columns([1, 9, 1])
            with ic:
                st.dataframe(events_df, hide_index=True,
                             column_config = {'event_id': st.column_config.TextColumn(width=80), 
                                              'event_type': st.column_config.TextColumn(width=100),
                                              'value': st.column_config.TextColumn(width=60)
                                            }
                            )
            st.write("""
                        Sử dụng `F.window()` để:
                        - Nhóm các sự kiện theo `event_type` trên một cửa sổ với `window_size = 10 mins` và `5 mins` dựa trên cột `event_ts`
                        - Trích xuất `window_start = window.start()` với định dạng `timestamp` và `window_end = window.end()`
                        - Tính `event_count`: tổng số sự kiện với `count('*')` và `total_value`: tổng số giá trị được làm tròn 2 chữ số thập phân
                        - Sắp xếp kết quả theo `window_start (ASC)` và `event_type (ASC)`
                    """)
        with c2:
            st.code(
'''
  ts_df = events_df.withColumn('event_ts', F.to_timestamp('event_ts')) \\
                   .groupBy(
                      F.col('event_type'), 
                      F.window(F.col('event_ts'), '10 minutes', '5 minutes')
                   ).agg(
                      F.count('*').alias('event_count'),
                      F.round( F.sum('value'), 2 ).alias('total_value')  
                    )
  filtered_df = ts_df.select('event_type',
                    F.date_format( F.col('window.start'), 'yyyy-MM-dd HH:mm:ss').alias('window_start'),
                    F.date_format( F.col('window.end'), 'yyyy-MM-dd HH:mm:ss').alias('window_end'),
                    'event_count', 'total_value'
             ).orderBy(F.col('window_start').asc(), F.col('event_type').asc())
'''
            )
            st.write("**:blue[Output]**")
            _, df_name1, oc1 = st.columns([1, 1, 9])
            with df_name1:
                st.write("**:orange[ts_df]**")
            with oc1:
                st.dataframe(pd.DataFrame({
                    'event_type': ['CLICK', 'CLICK', 'CLICK', 'PURCHASE', 'PURCHASE'],
                    'window': ['(2025-01-01 10:00:00, 2025-01-01 10:10:00)', 
                            '(2025-01-01 10:05:00, 2025-01-01 10:15:00)',
                            '(2025-01-01 09:55:00, 2025-01-01 10:05:00)',
                            '(2025-01-01 10:00:00, 2025-01-01 10:10:00)',
                            '(2025-01-01 10:05:00, 2025-01-01 10:15:00)'
                            ],
                    'event_count': [3, 2, 1, 1, 1],
                    'total_value': [6, 5, 1, 50, 50]
                }), hide_index=True)

            _, df_name2, oc2 = st.columns([1, 1, 6])
            with df_name2:
                st.write("**:red[filtered_df]**")
            with oc2:
                st.dataframe(pd.DataFrame({'event_type': {0: 'CLICK', 1: 'CLICK', 2: 'PURCHASE', 3: 'CLICK', 4: 'PURCHASE'},
                                           'window_start': {0: '2025-01-01 09:55:00', 1: '2025-01-01 10:00:00', 2: '2025-01-01 10:00:00', 3: '2025-01-01 10:05:00', 4: '2025-01-01 10:05:00'},
                                           'window_end': {0: '2025-01-01 10:05:00', 1: '2025-01-01 10:10:00', 2: '2025-01-01 10:10:00', 3: '2025-01-01 10:15:00', 4: '2025-01-01 10:15:00'},
                                           'event_count': {0: 1, 1: 3, 2: 1, 3: 2, 4: 1},
                                           'total_value': {0: 1, 1: 6, 2: 50, 3: 5, 4: 50}
                                        }), hide_index=True)            

    elif title == 'GA03. Transformation on Nested Structs':
        c1, c2 = st.columns([5, 7], gap='large')
        with c1:
            st.write(f"""
                      ##### :blue[{title}] 
                      Trong pipeline xử lý những bài toán thương mại điện tử **:green[E-commerce shopping cart]**, ta thường nhận dữ liệu ở dạng :green[bán cấu trúc] (`semi-structured`).
                      Trong đó, mỗi đơn hàng chứa trong một mảng lồng (:red[nested arrays]) các cấu trúc chứa thông tin về sản phẩm, ví dụ

                            {{"item_id": "i1", "category":"book" ,..., "quantity":"2"}}
 
                      Việc :yellow[san phẳng cấu trúc (unnest)] các mảng này bằng `F.explode()` có thể tạo lên đến hàng tỷ bản ghi trung gian, 
                      sau đó bắt buộc hệ thống phải thực hiện thêm một lần `.groupBy()`; điều này gây nên sự tốn kém khi tổng hợp dữ liệu theo đơn hàng.

                      Đối với các hệ thống `Spark` hiệu năng cao; cách tiếp cận được ưu tiên là sử dụng các `Higher-Order Array Functions` để xử lý trực tiếp trên mảng mà không phát sinh dữ liệu trung gian trong quá trình xử lý

                      Trong bài toán dưới đây, ta sẽ dùng bảng `orders_df`; **:red[KHÔNG]** sử dụng `explode()`, hãy:
                      - Tính `total_order_value`: tổng của `quantity * unit_price` cho tất cả các sản phẩm với `F.aggregate`
                      - Tính `expensive_item_count`: chỉ đếm những sản phẩm có đơn giá cao (`unit_price > 20.0`) với `F.size( F.filter() )`
                      - Tìm `discounted_order_value`: Giảm 10% cho những sản phẩm điện tử (`category = Electronics`) với `F.transform()` sau đó kết tính tổng giá trị đơn hàng với `F.aggregate()`
                      - Kiểm `is_premium_order` bằng cách sử dụng `F.forall()` để đánh giá `True` nếu và chỉ nếu mỗi sản phẩm có `unit_price > 50.0`
                     """)
            st.write("**:blue[Input dataframe]**")
            data = {
                    "orders_df": [
                        {
                        "order_id": "O001",
                        "customer_id": "C001",
                        "items": [
                            {
                            "item_id": "i1",
                            "category": "Books",
                            "quantity": 2,
                            "unit_price": 15
                            },
                            {
                            "item_id": "i2",
                            "category": "Electronics",
                            "quantity": 1,
                            "unit_price": 150
                            }
                        ]
                        },
                        {
                        "order_id": "O002",
                        "customer_id": "C002",
                        "items": [
                            {
                            "item_id": "i3",
                            "category": "Clothing",
                            "quantity": 3,
                            "unit_price": 25
                            },
                            {
                            "item_id": "i4",
                            "category": "Clothing",
                            "quantity": 1,
                            "unit_price": 30
                            }
                        ]
                        },
                        {
                        "order_id": "O003",
                        "customer_id": "C003",
                        "items": [
                            {
                            "item_id": "i5",
                            "category": "Electronics",
                            "quantity": 2,
                            "unit_price": 80
                            },
                            {
                            "item_id": "i6",
                            "category": "Electronics",
                            "quantity": 1,
                            "unit_price": 60
                            }
                        ]
                        }
                    ]
                }
            _, ic, _ = st.columns([1, 19, 1])
            ip_df = pd.DataFrame(data['orders_df'])
            ip_df['items'] = ip_df['items'].astype(str)
            with ic:
                st.dataframe(ip_df, hide_index=True, 
                             column_config = {
                                              'order_id': st.column_config.TextColumn(width=72),
                                              'customer_id': st.column_config.TextColumn(width=90)
                                            }
                            )
        with c2:
            st.code(
'''
  total_value = F.aggregate(F.col("items"), F.lit(0.0),
                            lambda acc, x: acc + (x["quantity"] * x["unit_price"]) )    
  expensive_count = F.size( F.filter(F.col("items"), lambda x: x["unit_price"] > 20.0) )
  discounted_items = F.transform( F.col("items"),
                          lambda x: F.when(
                             x["category"] == "Electronics",
                             F.named_struct(
                               F.lit('category'), x['category'], F.lit('item_id'), x['item_id'],
                               F.lit('quantity'), x['quantity'].cast('long'),
                               F.lit('unit_price'), F.round(x['unit_price'].cast('double') * 0.9, 2)
                            )).otherwise(
                            F.named_struct(
                                F.lit("category"), x["category"], F.lit("item_id"), x["item_id"],
                                F.lit("quantity"), x["quantity"].cast("long"),
                                F.lit("unit_price"), x["unit_price"].cast("double")
                        ) ) )
  discounted_total = F.aggregate(discounted_items, F.lit(0.0),
                                 lambda acc, x: acc + (x["quantity"] * x["unit_price"]) )
  premium_flag = F.forall(F.col("items"), lambda x: x["unit_price"] > 50.0)
    
  orders_df.select('order_id', F.round(total_value, 2).alias("total_order_value"),
                    expensive_count.alias("expensive_item_count"),
                    F.round(discounted_total, 2).alias("discounted_order_value"),
                    premium_flag.alias("is_premium_order")
                  ).orderBy("order_id")
'''
            )
            st.write("**:green[Output]**")
            _, oc = st.columns([1, 9])
            with oc:
                st.dataframe(pd.DataFrame({
                    'order_id': ['O001', 'O002', 'O003'],
                    'total_order_value': ['100', '105', '220'],
                    'expensive_item_count': ['1', '2', '2'],
                    'discounted_order_value': ['165', '105', '198'],
                    'is_premium_order': ['False', 'False', 'True']
                    }), hide_index=True, 
                        column_config = {'order_id': st.column_config.TextColumn(width=80)}
                )

    elif title == 'GA04. Dynamic Skewed Join':
        c1, c2 = st.columns([3, 4.5], gap='medium')
        with c1:
            st.write(f"""
                        ##### :blue[{title}]
                        Các :green[bảng thanh toán] và :green[bảng giao dịch] thường có :red[khối lượng lớn (massive)] thường gặp hiện tượng :red[lệch khóa (key skew)] nghiêm trọng khi :yellow[chỉ một số ít merchant doanh nghiệp lớn chiếm tỷ trọng giao dịch không tương xứng]. 
                        
                        Các phép join phân tán thông thường :red[sẽ đẩy toàn bộ bản ghi của khóa bị lệch về cùng một partition của executor], dẫn đến tình trạng :red[straggler] nặng và lỗi `OOM`.

                        Hãy triển khai pipeline salting động được tối ưu với các yêu cầu sau:
                        - **:yellow[Phát hiện lệch động (Dynamic Skew Detection):]** Tự động xác định các khóa `merchant` có tần suất cao (ví dụ: các khóa có số lượng ≥ 2 trong tập dữ liệu mẫu này) mà không hardcode danh sách `merchant ID`.
                        - **:yellow[Salting có cô lập (Isolated Salting):]** Tách dataset thành hai tập con: `skewed` và `non-skewed`. Chỉ salt các giao dịch bị lệch bằng số nguyên ngẫu nhiên trong khoảng từ 0 đến SALT_BUCKETS - 1 (sử dụng 5 buckets). Đồng thời nhân bản các dòng `merchant lookup` tương ứng bằng `F.sequence()` và `F.explode()`.
                        - **:yellow[Thực thi tối ưu (Optimized Execution):]** `Join` tập `skewed` trên khóa composite đã salt, join tập `non-skewed` bằng broadcast join, sau đó union hai kết quả lại với nhau.
                        - **:yellow[Sắp xếp kết quả]** cuối cùng theo `txn_id` tăng dần (ASC).

                        **:blue[Input dataframe]**
                    """)
            data = {
                    "transactions_df": [
                        {
                        "txn_id": "t001",
                        "merchant_id": "M_AMAZON",
                        "amount": 59.99
                        },
                        {
                        "txn_id": "t002",
                        "merchant_id": "M_AMAZON",
                        "amount": 120
                        },
                        {
                        "txn_id": "t003",
                        "merchant_id": "M_SMALL_CO",
                        "amount": 14.5
                        },
                        {
                        "txn_id": "t004",
                        "merchant_id": "M_WALMART",
                        "amount": 89
                        },
                        {
                        "txn_id": "t005",
                        "merchant_id": "M_WALMART",
                        "amount": 22.75
                        }
                    ],
                    "merchants_df": [
                        {
                        "merchant_id": "M_AMAZON",
                        "merchant_name": "Amazon",
                        "category": "Retail"
                        },
                        {
                        "merchant_id": "M_WALMART",
                        "merchant_name": "Walmart",
                        "category": "Retail"
                        },
                        {
                        "merchant_id": "M_SMALL_CO",
                        "merchant_name": "Small Co",
                        "category": "Services"
                        }
                    ]
                }

            le, ri = st.columns([3, 4], gap='medium')
            with le:
                st.write("**:orange[transactions_df]**")
                st.dataframe(pd.DataFrame(data['transactions_df']), hide_index=True)
            with ri:
                st.write("**:orange[merchants_df]**")
                st.dataframe(pd.DataFrame(data['merchants_df']), hide_index=True)
        with c2:
            st.code(
'''
 from pyspark.sql.types import IntegerType

 SALT_BUCKETS, SKEW_THRESHOLD = 5, 2
 key_counts = transactions_df.groupBy("merchant_id").agg(F.count("*").alias("cnt"))
 skewed_keys = [row["merchant_id"] 
                    for row in key_counts.filter(F.col("cnt") >= SKEW_THRESHOLD).collect()]

 txn_skewed = transactions_df.filter(F.col("merchant_id").isin(skewed_keys))
 txn_normal = transactions_df.filter(~F.col("merchant_id").isin(skewed_keys))

 txn_skewed_salted = txn_skewed.withColumn("salt", 
                                            (F.rand() * SALT_BUCKETS).cast(IntegerType())
                               ).withColumn("salted_merchant_id",
                                       F.concat_ws("_", F.col("merchant_id"), F.col("salt").cast("string"))
                               )
 merchants_exploded = merchants_df.filter(F.col("merchant_id").isin(skewed_keys))\\
                                  .withColumn("salt_array", F.sequence(F.lit(0), F.lit(SALT_BUCKETS - 1)))
                                  .withColumn("salt", F.explode("salt_array"))
                                  .withColumn("salted_merchant_id",
                                      F.concat_ws("_", F.col("merchant_id"), F.col("salt").cast("string"))
                                   ).drop("salt_array", "salt")
 result_skewed = txn_skewed_salted.join(merchants_exploded, on="salted_merchant_id", how="left")\\
                                  .select("txn_id", txn_skewed_salted["merchant_id"], 
                                          "amount", "merchant_name", "category")

 merchants_normal = merchants_df.filter(~F.col("merchant_id").isin(skewed_keys))
 result_normal = txn_normal.join(F.broadcast(merchants_normal), on="merchant_id", how="left")\\
                           .select("txn_id", "merchant_id", "amount", "merchant_name", "category")

 result_skewed.unionByName(result_normal).orderBy("txn_id")
''', language='python'
            )

            oc1, oc2, _ = st.columns([1, 6, 1], gap='medium')
            with oc1:
                st.write("**:orange[Output]**")
            with oc2:
                st.dataframe(pd.DataFrame({
                    'txn_id': ['t001', 't002', 't003', 't004', 't005'],
                    'merchant_id': ['M_AMAZON', 'M_AMAZON', 'M_SMALL_CO', 'M_WALMART', 'M_WALMART'],
                    'amount': ['59.99', '120.0', '14.5', '89.0', '22.75'],
                    'merchant_name': ['Amazon', 'Amazon', 'Small Co', 'Walmart', 'Walmart'],
                    'category': ['Retail', 'Retail', 'Services', 'Retail', 'Retail']
                }), hide_index=True)

    elif title == 'GA05. Massive Scale Percentile Approx':
        c1, c2 = st.columns([5, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Các hệ thống :green[ad tech], :green[gaming] và :green[tài chính] có :red[thông lượng cao] ghi nhận hàng tỷ bản ghi latency sự kiện mỗi ngày. Việc tính toán percentile chính xác bằng `percentile()` đòi hỏi sắp xếp phân tán với độ phức tạp $O(N \log N)$ kèm theo chi phí network shuffle rất lớn. Các :green[pipeline streaming] và :green[batch production] thường sử dụng :orange[approximation sketches] (như :green[Greenwald-Khanna] hoặc :yellow[T-Digest] thông qua `percentile_approx()`) để ước lượng quantile với sai số có giới hạn chỉ trong một lần aggregation duy nhất.

                        Cho dữ liệu :orange[telemetry latency] của :red[ad auction], hãy thực hiện các bước sau:
                        - Group by theo `campaign_id`, `ad_format` và `region`.
                        - Tính `total_events` (sử dụng `count(*)`).
                        - Tính đồng thời `p50_latency`, `p90_latency`, `p95_latency` và `p99_latency` trong một lần pass duy nhất bằng `F.percentile_approx("latency_ms", [0.50, 0.90, 0.95, 0.99], 10000)`.
                        - Tách và làm tròn từng `quantile` về 2 chữ số thập phân.
                        
                        Sắp xếp kết quả cuối cùng theo `campaign_id ASC, ad_format ASC, region ASC`.
                    ''')
            data = {
                    "ad_auction_events_df": [
                        {
                        "event_id": "E001",
                        "campaign_id": "CAM001",
                        "ad_format": "banner",
                        "region": "US-EAST",
                        "latency_ms": 45.2
                        },
                        {
                        "event_id": "E002",
                        "campaign_id": "CAM001",
                        "ad_format": "banner",
                        "region": "US-EAST",
                        "latency_ms": 120.5
                        },
                        {
                        "event_id": "E003",
                        "campaign_id": "CAM001",
                        "ad_format": "video",
                        "region": "US-WEST",
                        "latency_ms": 250
                        },
                        {
                        "event_id": "E004",
                        "campaign_id": "CAM002",
                        "ad_format": "banner",
                        "region": "US-EAST",
                        "latency_ms": 38
                        },
                        {
                        "event_id": "E005",
                        "campaign_id": "CAM002",
                        "ad_format": "banner",
                        "region": "US-EAST",
                        "latency_ms": 95.3
                        }
                    ]
                }

            st.write('**:orange[Input dataframe]**')

            _, ic, _ = st.columns([1, 9, 1])
            with ic:
                st.dataframe(pd.DataFrame(data['ad_auction_events_df']), 
                             hide_index=True,
                             column_config = {
                                    'event_id': st.column_config.TextColumn(width=80),
                                    'campaign_id': st.column_config.TextColumn(width=100),
                                    'ad_format': st.column_config.TextColumn(width=90),
                                    'region': st.column_config.TextColumn(width=80),
                                    'latency_ms': st.column_config.TextColumn(width=120)
                                }
                            )

        with c2:
            st.code(
'''
 QUANTILES = [0.50, 0.90, 0.95, 0.99]
 ACCURACY = 10000
    
 ad_auction_events_df.groupBy("campaign_id", "ad_format", "region")
                     .agg(
                         F.percentile_approx("latency_ms", QUANTILES, ACCURACY).alias("pcts"),
                         F.count("*").alias("total_events")
                        )
                    .withColumn("p50_latency", F.round(F.col("pcts").getItem(0), 2))
                    .withColumn("p90_latency", F.round(F.col("pcts").getItem(1), 2))
                    .withColumn("p95_latency", F.round(F.col("pcts").getItem(2), 2))
                    .withColumn("p99_latency", F.round(F.col("pcts").getItem(3), 2))
                    .select("campaign_id", "ad_format", "region", "total_events", 
                            "p50_latency", "p90_latency", "p95_latency", "p99_latency")
                    .orderBy("campaign_id", "ad_format", "region")
'''
            )
            st.write("**:blue[Output]**")
            _, oc = st.columns([1, 19])
            with oc:
                st.dataframe(pd.DataFrame({
                    'campaign_id': ['CAM001', 'CAM001', 'CAM002'],
                    'ad_format': ['banner', 'video', 'banner'],
                    'region': ['US-EAST', 'US-WEST', 'US-EAST'],
                    'total_events': ['2', '1', '2'],
                    'p50_latency': ['45.2', '250', '38'],
                    'p90_latency': ['120.5', '250', '95.3'],
                    'p95_latency': ['120.5', '250', '95.3'],
                    'p99_latency': ['120.5', '250', '95.3']
                }), hide_index=True)

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

    elif title == 'GP03. Dynamic High-Cardinality':
        c1, c2 = st.columns([6, 7], gap='medium')
        with c1:
            st.write(f"""
                        ##### :blue[{title}]
                        Các hệ thống phân tích và báo cáo :green[DataMart] thường sử dụng dữ liệu `metric` được lưu dưới dạng **:green[Entity-Attribute-Value (EAV)]**;
                        trong đó, các loại metric mới sẽ được bổ sung trong quá trình vận hành.
                        
                        Việc phải `hard-code` danh sách các cột metric có thể khiến `pipeline` được chỉnh sửa thủ công mỗi khi xuất hiện một `metric mới`

                        Trong khi đó, việc sử dụng `spark_df.pivot()` mà không chỉ định đích danh các giá trị sẽ buộc `Spark` phải thực hiện thêm một lần quét toàn bộ dữ liệu để xác định tất cả các giá trị cần pivot, làm tăng đáng kể chi phí xử lý

                        Cho bảng `content_metrics_df`, hãy chuyển nó về `wide-format` bằng cách:
                        - Thu thập danh sách tên các metric rồi sắp xếp tăng dần trước khi đưa nó vào `driver`
                        - Pivot dữ liệu lên `metric_name` và nhập danh sách tên các metric đã thu thập trước đó
                        - Điền các giá trị khuyết bởi `0.0`
                        - Với mỗi `(content_id, metric_date)`, hãy sắp xếp theo `content_id ASC` và `metric_date ASC`. Lưu ý danh sách các cột trong đầu ra sẽ có thứ tự như sau: `['content_id', 'metric_date'] + sorted(metric cols)`
                    """)
            st.write('**:green[Input dataframe]**')
            _, ic, _ = st.columns([1, 6, 1])
            with ic:
                st.dataframe(pd.DataFrame({
                    'content_id': ['vid_88', 'vid_99', 'vid_99', 'vid_99', 'vid_99'],
                    'metric_date': ['2024-02-01', '2024-02-01', '2024-02-01', '2024-02-02', '2024-02-02'],
                    'metric_name': ['ratings', 'views', 'completion_rate', 'views', 'watch_time_minutes'],
                    'metric_value': ['4.8', '5,000', '0.85', '6,200', '1,240']
                }), hide_index=True)
        with c2:
            st.code(
'''
  cnt_metric_df = cnt_metric_df.withColumn('metric_date',
                                            F.date_format('metric_date', 'yyyy-MM-dd'))
  sorted_metrics = sorted([
                    r['metric_name'] for r in
                                    cnt_metric_df.select('metric_name').distinct().collect()
                ])
  selected_cols = ['content_id', 'metric_date'] + sorted_metrics

  pivoted_df = cnt_metric_df.groupBy('content_id', 'metric_date')\\
                            .pivot('metric_name', sorted_metrics) \\
                            .agg(F.first('metric_value'))\\
                            .fillna(0, subset=sorted_metrics) \\
                            .select(selected_cols) \\
                            .orderBy('content_id', 'metric_date')
'''
            )
            st.write("**:blue[Output]**")
            _, oc, _ = st.columns([1, 9, 1])
            with oc:
                st.dataframe(pd.DataFrame({
                    'content_id': ['vid_88', 'vid_99', 'vid_99'],
                    'metric_date': ['2024-02-01', '2024-02-01', '2024-02-02'],
                    'completion_rate': ['0', '0.85', '0'],
                    'ratings': ['4.8', '0', '0'],
                    'views': ['0', '5,000', '6,200'],
                    'watch_time_minutes': ['0', '0', '1,240']
                }), hide_index=True)

elif topic == "EDA":
    if title == 'Eda01. Schema, columns & partition count':
        c1, c2 = st.columns([9, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong **:red[EDA]** (chỉ bàn về Performance & Data Quality), chứ không chỉ EDA thông thường; mỗi phép phân tích phải trả lời được ít nhất một câu hỏi:
                        - Data có vấn đề gì: :yellow[null, duplicate, outlier, skew]?
                        - Phép xử lý nào có thể gây :orange[shuffle, spill] hoặc :orange[OOM]?
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
                                Partitions: 4

                        Không nên mặc định dùng `df.count()` chỉ để xem kích thước. `count()` có thể quét toàn bộ dữ liệu nguồn, đặc biệt tốn thời gian với file lớn hoặc nguồn không có metadata hữu ích.
                    """)

    elif title == 'Eda02. Null profiling':
        c1, c2, c3 = st.columns([7, 8, 3.5], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Nhắc lại:
                        - Đôi khi, chúng ta được yêu cầu đếm số dòng trong dữ liệu, lúc này `df.count()` là một lựa chọn hợp lý hơn so với `df.agg(F.count('*'))`
                        - Chỉ sử dụng `df.agg().first()` nếu ta muốn lưu nhiều giá trị cùng lúc như

                                df.agg(F.count('*').alias('count'),
                                       F.sum('amount').alias('t_amount'),
                                       F.avg('amount').alias('avg_amount')     
                                    )
                        
                        Như vậy, khi đếm `null-values` ở tất cả các cột; thay vì gọi `filter().count()` cho từng cột, ta sẽ tạo một `aggregation` cho tất cả cột cần kiểm tra.

                        Một số lưu ý khác:
                        - `count(col)` bỏ qua `null`, nhưng `count(when(...))` đếm đúng số `null`.
                        - `NaN` trong cột floating-point không giống `NULL`; cần kiểm tra riêng.
                        - Nếu có hàng nghìn cột, đừng tạo một `aggregation` khổng lồ; hãy chia thành các nhóm cột để tránh `query plan` quá lớn.

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
  # Tạo list lưu việc đếm số dòng trên toàn dữ liệu 
  agg_exprs = [F.count(F.lit(1)).alias("_rows")]

  # Append F.count các cột khác rỗng với các cột trong dữ liệu
  agg_exprs += [
                F.count(
                    F.when(F.col(c).isNull(), 1)
                ).alias(c)
                for c in transactions_df.columns
            ]

  # Apply list các agg-expression vào aggregation
  stats = transactions_df.agg(*agg_exprs).first().asDict()

  # Trích xuất số dòng
  n = stats.pop("_rows")

  # Display dataframe tổng kết việc đếm cũng như % các cột null
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

                        Ngoài ra, đối với 2 cột này, ta sẽ đặc biệt chú ý những giá trị bất thường (**:yellow[abnormal values]**) nếu chúng bị âm hoặc bằng 0.

                        Trong các bước triển khai nâng cao, ta có thể tạo cột `amount` bằng cách `UnitPrice * Quantity` (xem ở `Eda04`)
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
            _, oc1 = st.columns([1, 9])
            with oc1:
                st.dataframe(pd.DataFrame({
                        'Quantity_min': {0: -80995},
                        'Quantity_max': {0: 74215},
                        'Quantity_avg': {0: 8.89},
                        'Quantity_std': {0: 218.08},
                        'UnitPrice_min': {0: -11062.06},
                        'UnitPrice_max': {0: 38970.0},
                        'UnitPrice_avg': {0: 5.39},
                        'UnitPrice_std': {0: 96.76}
                }), hide_index=True)

            st.code(
'''
transactions_df.agg(
                    F.count(F.when(F.col('Quantity') < 0, 1)).alias('Quantity < 0').alias('n_negative_quant'),
                    F.count(F.when(F.col('UnitPrice') < 0, 1)).alias('UnitPrice < 0').alias('n_negative_price'),
                    F.count(F.when(F.col('Quantity') == 0, 1)).alias('Quantity = 0').alias('n_zero_quant'),
                    F.count(F.when(F.col('UnitPrice') == 0, 1)).alias('UnitPrice = 0').alias('n_zero_price')
                ).show()
'''
            )
            st.write("**Output**")
            _, oc2 = st.columns([1, 6])
            with oc2:
                st.dataframe(pd.DataFrame({
                    'n_negative_quant': ['10,624'],
                    'n_negative_price': ['2'],
                    'n_zero_quant': ['0'],
                    'n_zero_price': ['2,515']
                }), hide_index=True)

    elif title == 'Eda04. Quantiles & outlier screening':
        c1, c2 = st.columns([4, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Khi xét đến phân vị (`quantiles`) thì `approxQuantile()` là lựa chọn thực tế để ước lượng percentile mà không phải sort toàn bộ dataset để lấy quantile chính xác.

                        Trong bài toán này, ta vẫn dùng `transactions_df`, ở đây ta muốn xét các phân vị trong `'transaction_amount'` với thuật toán `approxQuantile`

                        **:yellow[Syntax]**

                            spark_dataframe.approxQuantile(
                                                    column,
                                                    probabilities,
                                                    relativeError
                                                )
                    ''')
        with c2:
            st.code(
"""
  transactions_df = transactions_df.withColumn(
                                        "transaction_amount", 
                                        col("Quantity") * col("UnitPrice"))

  # Here we evaluate these levels (1 %, 25 %, 50 %, 75 %, 99 %) and error = 0.01
  quantiles = transactions_df.approxQuantile(
                                     "transaction_amount",
                                     [0.01, 0.25, 0.50, 0.75, 0.95, 0.99],
                                     0.01
                                )
  quantiles
""", language = 'python'
            )
            st.write("""
                    **:blue[Output]**

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
            _, oc1, _ = st.columns([1,2,3])
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
            _, oc2, _ = st.columns([1,2,2])
            with oc2:
                st.dataframe(pd.DataFrame({'non_null_ids': {0: 100247}, 'approx_distinct_ids': {0: 4963}}), hide_index=True)

    elif title == 'Eda06. Skewness notes':
        c1, c2 = st.columns([6, 7], gap='large')
        with c1:
            st.write(f'''
                        ##### :blue[{title}]
                        Trong banking, một vài `customer` hoặc `merchant` có thể chiếm tỷ trọng giao dịch rất lớn. Để biết mức độ tập trung của `hot keys`, có thể lấy tổng số records của `top keys` so với tổng dữ liệu. Đừng chỉ nhìn số lượng distinct customer: 1 triệu customer không có nghĩa là phân phối giao dịch đồng đều.

                        Ngoài ra, khi sử dụng `.groupBy()` với `F.count()` sẽ cần shuffle index, do đó không nên chạy liên tục trên dataset lớn

                        Kiểm tra dimension trước khi join. Đây là một trong những kỹ năng quan trọng nhất để tránh join làm tăng số lượng dòng ngoài dự kiến. Trước khi join, ta cần xác định `grain` của mỗi table. Chẳng hạn, nếu `dimension` dự kiến là `1 row / customer` nhưng thực tế đến `4-5 rows/ customer` thì việc thực hiện `join` sẽ nhân số dòng fact tương ứng lên 4-5 lần.

                        Nếu chỉ cần check một số `customer keys` có tồn tại trong bảng dim hay không, ta có thể sử dụng `left anti join` thay thế
                    ''')

        with c2:
            st.code(
'''
  tx_uniq_cust = transactions_df.select('CustomerID').distinct()
  cs_uniq_cust = customers_df.select('CustomerID').distinct()

  missing_keys = tx_uniq_cust.join(cs_uniq_cust, on='customer_id', how='left_anti')
'''
            )

            st.write('Với phương thức này, dữ liệu vẫn có thể `shuffle` và gây tốn resource; `.distinct()` cũng khá tốn cost; hãy dùng khi yêu cầu kiểm tra coverage thực sự cần thiết.')

