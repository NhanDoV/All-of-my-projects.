Source: [Kaggle SalesDataset](https://www.kaggle.com/datasets/yassinealouini/m5-sales-hierarchy-dataset)

## Mở đầu
Ở bài toán này, mục đích không hướng đến phân tích doanh số hay phân cụm khách hàng, mà là tối ưu hàng tồn kho 

| Nhóm thông tin       | Cột có sẵn                                                                                  | Dùng được cho gì                              |
|----------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------|
| Demand               | `demand_units`, `demand_forecast_units`                                                     | Forecast, tính safety stock, service level    |
| Inventory position   | `on_hand_units`, `on_order_units`, `safety_stock_units`                                     | Inventory optimization, (s,S), base-stock     |
| Replenishment        | `order_qty_units`, `arrivals_units`, `lead_time_days`                                       | Mô phỏng / tối ưu chính sách đặt hàng         |
| Service & cost       | `fill_rate`, `stockout`, `holding_cost_usd`, `stockout_cost_usd`, `ordering_cost_usd`, `total_cost_usd` | Tối ưu chi phí, trade-off service vs cost |
| Policy hiện tại      | `policy` (`ml_reorder`, `minmax`, `base_stock`)                                             | So sánh / benchmark các chính sách            |
| External factors     | `promo_flag`, `holiday_flag`, `weather_index`, `competitor_price_index`                     | Feature cho ML forecast / dynamic policy      |