
# coding: utf-8

# # 流速計の結果の座標変換
# 
# 流速ベクトルの直交基底を変換する.
# 
# > [ＥＭＡＮの物理学・物理数学・基底ベクトルの変換](https://eman-physics.net/math/linear08.html)
# 
# ## 制約
# 
# * 2次元平面上
# * 入力はベクトル
# * 出力はベクトル
# 
# ## 必要関数
# 
# * ベクトル変換関数
# 
# ```python
# def transform_coordinate(matrix, matrix_original)->Callable[[Vector], Vector]:
#     def transform(v: Vector)->Vector:
#         pass
#     return transform
# ```
# 
# * 真北からの時計回りの回転角に基づき基底ベクトルを作る関数
# 
# ```python
# def basis_by_North(rotate_from_N1, rotate_from_N2) -> List[List[Number]]:
#     pass
# ```
# 
# * pandas.DataFrameからベクトルを作る関数
# 
# ```python
# def vectors_from_dataframe(*use_column_names)-> Callable[[pandas.Dataframe], List[Vector]]:
#     def apply(dataframe) -> Vector:
#         pass
#     return apply
# ```
# 
# * ベクトルからpandas.DataFrameを作る関数
# 
# ```python
# 
# def dataframe_from_vectors(**kwargs)-> Callable[[List[Vector]], pandas.DataFrame]:
#     def apply(vector) -> pandas.Dataframe:
#         pass
#     return apply
# ```
# 
# ## Usecese
# 
# * DataFrameに対し座標変換をかけて新たなデータフレームを得る
#     1. DataFrameの必要な列から座標ベクトルの配列を作る
#     2. 各座標ベクトルに座標変換関数を適用し, 新しい座標ベクトルの配列を作る
#     3. 座標ベクトルの配列からDataFrameを作る
# 
# 例えば, 流速計のx,y方向がそれぞれ真北から東に54度の方向, それと直交する南東方向であり, 
# これらをそれぞれ真北と真東での座標に変換したい場合, 
# 
# ```python
# basis_N_and_E = basis_by_North(0,90)
# original_basis = basis_by_North(54, 54+90)
# transformed = pip(
#     vectors_from_dataframe("x", "y"),
#     lambda array: map(transform_coordinate(basis_N_and_E, original_basis), array),
#     list,
#     dataframe_from_vectors(columns=["N", "E"])
# )(df)
# 
# ```

# In[ ]:


import numpy as np
import pandas as pd
from func_helper import pip, identity


def basis_vector_matrix(*e):
    """
    Make matrix of basis vectors.
    
    Parameters
    ----------
    *e: List[Number]
        Basis vector.
        
    Returns
    -------
    numpy.ndarray
    
    Usage
    -----
    basis_matrix = basis_vector_matrix([1,1,0],[0,1,1],[1,0,1])
    # basis_matrix == np.array([
        [1,0,1],
        [1,1,0],
        [0,1,1]
    ])
    """
    return np.array(e).T


def transform_coordinate(basis_vector_list, original_basis_vector_list=None):
    """
    Transform coordinate by changeing two sets of basis vectors.
    
    Parameters
    ----------
    basis_vector_list: List[List[Number]]
        List of numbers list. Inner list represents the new basis vectors.
        
    original_basis_vector_list, optional: List[List[Number]]
        List of numbers list. Inner list represents the original basis vectors.
        Default value is Decalt coordinate basis vectors.
        
    Returns
    -------
    transform: Callable[[List[Number]], List[Number]]
        A function returns list of number from list of number.
        
    Usage
    -----
    original_basis = [[1,0],[0,1]]
    new_basis = [[1,1],[0,1]]
    transform = transform_coordinate(new_basis, original_basis)
    
    original_coordinate = [1,1]
    new_coordinate = transform(original_coordinate)
    # new_coordinate == numpy.array([1., 0.])
    """
    
    new_basis = basis_vector_matrix(*basis_vector_list)
    
    original = np.eye(*new_basis.shape) if original_basis_vector_list is None else basis_vector_matrix(*original_basis_vector_list)
    
    transform_matrix = np.linalg.inv(new_basis) @ original
    
    def transform(v):
        _v = v if type(v) is np.array else np.array(v)
        return transform_matrix @ _v
    
    return transform


def _offset(offset_degree):
    return lambda f: lambda d: f(d+offset_degree)
    
def _clockwise(f):
    return lambda rotate: f(-rotate)
    

def basis_by_North(clockwise_degree_from_N1, clockwise_degree_from_N2):
    """
    Make basis normalized vectors from clockwise rotation degrees from North.
    
    Parameters
    ----------
    clockwise_degree_from_N1: Number
        Rotation degree from North of the first basis vector.
    
    closkwise_degree_from_N2: Number
        Rotation degree from North of the second basis vector.
        
    Returns
    -------
    List[List[Number]]
    
    Usage
    -----
    basis_by_North(0, 45)
    # > [[0,1], [numpy.sqrt(1/2), numpy.sqrt(1/2)]]
    """
    
    @_offset(-90)
    @_clockwise
    def degree_to_radiun(degree):
        return degree/180*np.pi
    
    return [
        [np.cos(degree_to_radiun(clockwise_degree_from_N1)),np.sin(degree_to_radiun(clockwise_degree_from_N1))],
        [np.cos(degree_to_radiun(clockwise_degree_from_N2)),np.sin(degree_to_radiun(clockwise_degree_from_N2))]
    ]


def vectors_from_dataframe(*columns):
    """
    Make list of np.ndarray from pandas.DataFrame.
    
    Parameters
    ----------
    *columns: Union[str, Number]
        Column names extract from dataframe.
        
    Returns
    -------
    Callable[[pandas.DataFrame], List[numpy.ndarray]]
    
    Usage
    -----
    vectors_from_dataframe("x","y")(
        pandas.Dataframe({
            "x" : [0,1,2],
            "y" : [0,0,0],
            "z" : [1,1,1]
        })
    )
    # > [numpy.array([0,0]),
    #    numpy.array([1,0]),
    #    numpy.array([2,0])]
    """
    return lambda df: [np.array(v) for v in zip(*[list(df[x].values) for x in columns])]

def dataframe_from_vectors(**kwargs):
    """
    Make pandas.DataFrame from list of array-like.
    
    Parameters
    ----------
    **kwargs
        Keyword arguments compatible to pandas.DataFrame()
        
    Returns
    -------
    Callable[[List[ArrayLike]], pandas.DataFrame]
    
    Usage
    -----
    dataframe_from_vectors(columns=["x","y"])(
        [[0,0],
         [1,0],
         [2,0]]
    )
    # > pandas.DataFrame:
    #   | "x" | "y" |
    # 0 |  0  |  0  |
    # 1 |  1  |  0  |
    # 2 |  2  |  0  |
    """
    return lambda vectors: pd.DataFrame(vectors, **kwargs)


