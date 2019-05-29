---
jupyter:
  jupytext:
    format: ipynb,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.0
---

# 流速計の結果の座標変換

流速ベクトルの直交基底を変換する.

> [ＥＭＡＮの物理学・物理数学・基底ベクトルの変換](https://eman-physics.net/math/linear08.html)

## 制約

* 2次元平面上
* 入力はベクトル
* 出力はベクトル

## 必要関数

* ベクトル変換関数

# ```python
def transform_coordinate(matrix, matrix_original)->Callable[[Vector], Vector]:
    def transform(v: Vector)->Vector:
        pass
    return transform
# ```

* 真北からの時計回りの回転角に基づき基底ベクトルを作る関数

# ```python
def basis_by_North(rotate_from_N1, rotate_from_N2) -> List[List[Number]]:
    pass
# ```

* pandas.DataFrameからベクトルを作る関数

# ```python
def vectors_from_dataframe(*use_column_names)-> Callable[[pandas.Dataframe], List[Vector]]:
    def apply(dataframe) -> Vector:
        pass
    return apply
# ```

* ベクトルからpandas.DataFrameを作る関数

# ```python

def dataframe_from_vectors(**kwargs)-> Callable[[List[Vector]], pandas.DataFrame]:
    def apply(vector) -> pandas.Dataframe:
        pass
    return apply
# ```

## Usecese

* DataFrameに対し座標変換をかけて新たなデータフレームを得る
    1. DataFrameの必要な列から座標ベクトルの配列を作る
    2. 各座標ベクトルに座標変換関数を適用し, 新しい座標ベクトルの配列を作る
    3. 座標ベクトルの配列からDataFrameを作る

例えば, 流速計のx,y方向がそれぞれ真北から東に54度の方向, それと直交する南東方向であり, 
これらをそれぞれ真北と真東での座標に変換したい場合, 

# ```python
basis_N_and_E = basis_by_North(0,90)
original_basis = basis_by_North(54, 54+90)
transformed = pip(
    vectors_from_dataframe("x", "y"),
    lambda array: map(transform_coordinate(basis_N_and_E, original_basis), array),
    list,
    dataframe_from_vectors(columns=["N", "E"])
)(df)

# ```

```python
import numpy as np
import pandas as pd
import func_helper.func_helper.dictionary as dictionary
import func_helper.func_helper.dataframe as dataframe
from func_helper import pip, identity

from matdat import Figure, Subplot
import matdat.matdat.plot as plot

import os,sys

sys.path.append(os.pardir)
```

```python
from toko_physical_measurement.core.transform_coodinate import *
```

```python
def _moc_df():
    df = pip(
        dictionary.over_iterator(
            x=np.sin,
            y=np.cos,
            r=identity
        ),
        pd.DataFrame
    )(np.arange(0,np.pi*2,0.1))
    return df

```

```python
def __test_get_basis_vector_matrix():
    mat = basis_vector_matrix([1,1,0],[0,1,0],[0,0,1])
    expect = np.array([
        [1,0,0],
        [1,1,0],
        [0,0,1]
    ])
    
    assert(np.array_equal(mat,expect)) 

__test_get_basis_vector_matrix()

    
def __test_transform_coordinate():
    new_basis = [[1,1],[0,1]]
    original_coord = [1,1]
    new_coord = [1,0]
    assert(np.array_equal(
        transform_coordinate(new_basis)(original_coord),
        new_coord
    ))
    
__test_transform_coordinate()


def __test_get_rotation_from_North():
    
    def _offset(offset_degree):
        return lambda f: lambda d: f(d+offset_degree)
    
    def _clockwise(f):
        return lambda rotate: f(-rotate)
    
    
    @_offset(-90)
    @_clockwise
    def clockwise_rotate_from_North(anticlockwise_from_x_axis):
        return anticlockwise_from_x_axis
    
    assert(clockwise_rotate_from_North(0) == 90)
    assert(clockwise_rotate_from_North(90) == 0)
    assert(clockwise_rotate_from_North(135) == (-45))
    assert(clockwise_rotate_from_North(-45) == (135))
    assert(clockwise_rotate_from_North(180) == (-90))
    assert(clockwise_rotate_from_North(-180) == (270))
    
__test_get_rotation_from_North()

    
def __test_transform_basis_by_North():
    mat = basis_by_North(0,45)
    expect = np.array([
        [0,1],
        [np.sqrt(1/2),np.sqrt(1/2)]
    ])
    assert(np.allclose(mat, expect))
    
    mat = basis_by_North(0,90)
    expect = np.array([
        [0,1],
        [1,0]
    ])
    assert(np.allclose(mat, expect))
    
    new_coord = transform_coordinate(mat)([1,1])
    assert(np.allclose(new_coord, [1,1]))
    
    new_coord = transform_coordinate(mat)([1,0.5])
    assert(np.allclose(new_coord, [0.5,1]))
    
    
__test_transform_basis_by_North()



def __test_vectors_from_dataframe():
    df = pd.DataFrame({
        "x" : [0,1,2],
        "y" : [0,0,0],
        "z" : [1,1,1]
    })
        
    data_matrix = vectors_from_dataframe("x","y")(df)
    expect = [
        np.array([0,0]),
        np.array([1,0]),
        np.array([2,0])
    ]
    assert(np.array_equal(data_matrix, expect))
    
__test_vectors_from_dataframe()

def __test_dataframe_from_vectors():
    data_matrix = [
        [0,0],
        [1,0],
        [2,0]
    ]
    df = dataframe_from_vectors(columns=["x","y"])(data_matrix)
    
    assert(np.array_equal(df.columns, ["x","y"]))
    assert(np.array_equal(df["x"], [0,1,2]))
    assert(np.array_equal(df["y"], [0,0,0]))
    assert(np.array_equal(df.index, [0,1,2]))
    
    df = dataframe_from_vectors()(data_matrix)
    assert(np.array_equal(df.columns, [0,1]))
    
__test_dataframe_from_vectors()
```

```python
def __test_plot_transformed():
    df = _moc_df()
    df_new = pip(
        vectors_from_dataframe("x","y"),
        lambda array: map(transform_coordinate([[1,1],[0,1]]), array),
        list,
        dataframe_from_vectors(columns=["x","y"])
    )(df)
    
    
    Figure().add_subplot(
        Subplot().add(
            data=df,
            x="x",
            y="y",
            plot=[plot.scatter(c=lambda df: df.index)]
        ),
        Subplot().add(
            data=df_new,
            x="x",
            y="y",
            plot=[plot.scatter(c=lambda df: df.index)]
        )
    ).show(size=(6,6), column=2)
    
__test_plot_transformed()
```

```python
def __test_plot_transformed_by_North():
    df = _moc_df()
    
    df_new = pip(
        vectors_from_dataframe("x","y"),
        lambda array: map(transform_coordinate(basis_by_North(0,90)), array),
        list,
        dataframe_from_vectors(columns=["x","y"])
    )(df)
    
    Figure().add_subplot(
        Subplot().add(
            data=df,
            x="x",
            y="y",
            plot=[plot.scatter(c=lambda df: df.index)]
        ),
        Subplot().add(
            data=df_new,
            x="x",
            y="y",
            plot=[plot.scatter(c=lambda df: df.index)]
        )
    ).show(size=(6,6), column=2)
    
__test_plot_transformed_by_North()
```

```python

```
