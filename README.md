# 2d_wave_equ

gmsh(https://gmsh.info/) 等で生成した2次元の三角形非構造格子メッシュ（.msh）を用いて，Dirichlet境界条件の2次元波動方程式を有限体積法で解くプログラム．

参考資料：数値流体力学 第2版 (https://www.morikita.co.jp/books/mid/091972) p341ぐらいから非構造格子での離散化の話 交差拡散項については付録Fにも記載されています．大学の図書館にあります．

Jax(https://github.com/google/jax) について・・・イメージで言うとnumpyを高速化しやすいやつ

このJaxを使う深層学習ライブラリにFlax(https://github.com/google/flax) があります．（FlaxはTensorFlow,PyTorchみたいな感じ）

## 実行方法

```python
python mesh_parser.py
```
響板.msh（プログラムのfilenameを変更することで読み取るメッシュを指定できる）を読み込んでメッシュの情報をnumpy，jax numpyで扱える形式にする．
parse_（メッシュファイル名）のフォルダが作成される．

```python
python 2d_wave_equ.py
```
param.csv（プログラムのpd.read_csv()を変更することで読み取るパラメータファイルを指定できる）で書かれているパラメータを使用して2次元の波動方程式を有限体積法で解く．
result/(パラメータ_実行時刻)のフォルダが作成される．

```python
python visualization_2d_wave_equ.py
```
指定したresult/（パラメータ_実行時刻）フォルダに格納されているデータを可視化する．プログラム中のdirectory = "（パラメータ_時刻）"を適宜可視化したいフォルダ名に変更する．
visualization/（パラメータ_実行時刻）フォルダが作成される．

```python
python data_to_wave.py
```
指定したresult/（パラメータ_実行時刻）フォルダに格納されている変位データを音データ（.wav）に変換する．プログラム中のdirectory_name = "（パラメータ_時刻）"を適宜音に変換したいフォルダ名に変更する．
wave/（パラメータ_実行時刻）フォルダが作成される．