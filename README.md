# Du doan gia nha - Boston Housing

Du an hoc may du doan gia nha (MEDV) tu bo du lieu Boston Housing.

## Cau truc thu muc

```
.
├── data/
│   ├── raw/                # Du lieu goc (HousingData.csv)
│   └── processed/          # Du lieu da qua tien xu ly
├── notebooks/              # Jupyter notebook phan tich tham duong
├── src/
│   ├── data/
│   │   └── make_dataset.py     # Doc du lieu goc, lo boi MEDV=50, luu processed
│   ├── features/
│   │   └── build_features.py   # Imputer (median) + StandardScaler
│   └── models/
│       ├── train_model.py      # Chia 3 tap, huan luyen LinearRegression, tinh metric
│       └── plot_results.py     # Residual plot, du doan vs thuc te
├── models/                 # Mo hinh da luu (.joblib)
├── reports/
│   ├── figures/            # Bieu do, ket qua truc quan
│   ├── coefficients.csv    # System hoi quy
│   └── results.csv         # Metric tren val/test
├── requirements.txt
├── .gitignore
└── README.md
```

## Cach chay

```bash
# 1. Tao moi truong ao (khuyen khich)
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Tien xu ly du lieu
python src/data/make_dataset.py

# 3. Huan luyen mo hinh
python src/models/train_model.py

# 4. Ve bieu do
python src/models/plot_results.py
```

## Mo ta du lieu

File `data/raw/HousingData.csv` gom 13 dac trung + cot muc tieu `MEDV` (gia nha, don vi 1000 USD):

| Cot     | Y nghia                                        |
|---------|------------------------------------------------|
| CRIM    | Ty le toi pham theo thit tran                  |
| ZN      | Ty le dat o dat vung dan cu (> 25,000 sq ft)   |
| INDUS   | Ty le dien tich kinh doanh khong buon ban      |
| CHAS    | Bien gia gioi song Charles River (1 = co)      |
| NOX     | Nong do khi oxide nitric (phan 10 trieu)       |
| RM      | So phong moi can ho                            |
| AGE     | Ty le nha xay dung truoc 1940                  |
| DIS     | Khoang cach den 5 trung tam viec lam Boston    |
| RAD     | Chi so tiep can cao toc                        |
| TAX     | Thue tai san theo gia tri `$10,000`            |
| PTRATIO | Ty le hoc sinh/giao vien theo thi tran         |
| B       | Chi so nhan khau da Philippe                   |
| LSTAT   | Ty le dan thu nhap thap (%)                    |
| MEDV    | Gia tri trung binh nha (x1000 USD) - Muc tieu  |

## Quy trinh xu ly du lieu

- Lo boi: bo 16 dong `MEDV = 50` (gia tri bi cat tran trong dataset goc, gay nhieu cho hoi quy) -> con 490 dong
- Dien gia tri thieu: `SimpleImputer(strategy="median")` - median chi duoc tinh tren tap train de tranh data leakage
- Chuan hoa: `StandardScaler()` fit tren train
- Chia 3 tap theo ty le **70 / 15 / 15** (train / val / test, `random_state=42`):
  - **Train (70%)** - hoc tham so mo hinh
  - **Val (15%)** - dung trong qua trinh can bang/tuning
  - **Test (15%)** - chi dung mot lan de danh gia cuoi cung

## Mo hinh

Linear Regression (OLS - line hoi quy it nha binh phuong, trung thuc hoa bien so thieu - MSE).

## Loss functions va metric danh gia

| Chi so | Y nghia | Ghi chu |
|---|---|---|
| **MSE** | Loss trung binh binh phuong - sai so phuong, ham mat mat chu dao (phat nang sai so lon) | Ham mat mat cua OLS |
| **RMSE** | Can hai MSE - cung don vi gia nha (x1000 USD) | De doc ket qua |
| **MAE** | Sai so trung binh tuyet doi - it nhay outlier hon MSE | Do sai so thuc te |
| **MAPE** | Sai so trung binh phan tram | De dien giai nghiep vu |
| **R2** | Ti le phuong sai cua y duoc giai thich | Quan trong khi so sanh mo hinh |

Metric tinh tren tap **val** trong qua trinh y nghia cuoi cung tren tap **test**. Ket qua luan tai `reports/results.csv`.
