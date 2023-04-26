use std::ops::{Add, Mul, Index, IndexMut};
use std::fmt;
use std::borrow::Borrow;

#[derive(Clone)]
struct Matrix<const M: usize, const N: usize> {
    data: [[f64; N]; M]
}

type Vector<const M: usize> = Matrix<M, 1>;

impl<const M: usize, const N: usize> Index<usize> for Matrix<M, N> {
    type Output = [f64; N]
    
    fn index(&self, idx:usize) -> &Self::Output{
        return &self.data[idx]
    }
}

impl<const M: usize, const N: usize> IndexMut<usize> for Matrix<M, N> {
    type Output = [f64; N]
    
    fn index(&mut self, idx:usize) -> &mut Self::Output{
        return &mut self.data[idx]
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    
    fn zeros() -> Self {
        Matrix {
            data: [[0.0; N]; M],
        }
    }
    
    fn add(&self, other: &Self) -> Matrix<M,N> {
        let mut result = Matrix::zeros();
        
        for i in 0..N {
            for j in 0..M {
                result[i][j] = self[i][j] + other[i][j]
            }
        }
        result
    }
    
    fn multiply<const P: usize>(&self, other: &Matrix<N, P>) -> Matrix<M, P> {
        let mut result = Matrix::<M, P>::zeros();
        
        for self_row in 0..M{
            for self_col in 0..N {
                for other_col in 0..P{
                    result[self_row][other_col] += 
                        &(self[self_row][self_col] * other[self_col][other_col])
                }
            }
        }
        
        result
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<&Matrix<N,P>> for &Matrix<M, N>{
    type Output = Matrix<M, P>;

    fn mul(self, other: &Matrix<N, P>) ->
     Matrix<M, P> {
        self.multiply(&other)
    }
}

impl<const M: usize, const N: usize> Add<&Matrix<M, N>> for &Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn add(self, other: &Matrix<M, N>) -> Matrix<M, N> {
        self.add(other)
    }
}