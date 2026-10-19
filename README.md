# Du doan gia nha - Boston Housing

Du an hoc may du doan gia nha (MEDV) tu bo du lieu Boston Housing.

## Cau truc thu muc

```
.
├── data/
│   ├── raw/                # Du lieu goc (HousingData.csv)
│   └── processed/          # Du lieu da qua tien xu ly
├── notebooks/              # Jupyter notebook phan tich tam giac
├── src/                    # Ma nguon
│   └── train.py            # Script huan luyen mo hinh
├── models/                 # Mo hinh da luu (.joblib)
├── reports/
│   └── figures/            # Bieu do, ket qua truc quan
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

# 2. Huan luyen mo hinh
python src/train.py
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

## Mo hinh

Sooth lap cac mo hinh: Linear Regression, Random Forest. Chon mo hinh tot nhat theo do tung duoi R2 (su dung du lieu kiem tra thu dung 80/20).
