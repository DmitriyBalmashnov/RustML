use std::ops::{Add, Mul, Sub, Index, IndexMut};
use std::fmt;
extern crate nalgebra as na;
use self::na::Dyn;

use self::na::Matrix as nam;

#[derive(Clone)]
pub struct Matrix<const M: usize, const N: usize> {
    data: [[f64; N]; M]
}

pub type Vector<const M: usize> = Matrix<M, 1>;

impl<const M: usize, const N: usize> Index<usize> for Matrix<M, N> {
    type Output = [f64; N];
    
    fn index(&self, idx:usize) -> &Self::Output{
        return &self.data[idx]
    }
}

impl<const M: usize, const N: usize> IndexMut<usize> for Matrix<M, N> {    
    fn index_mut(&mut self, idx:usize) -> &mut Self::Output{
        return &mut self.data[idx]
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    
    pub fn zeros() -> Self {
        Matrix {
            data: [[0.0; N]; M],
        }
    }
    
    fn add(&self, other: &Self) -> Matrix<M,N> {
        let mut result = Matrix::zeros();
        
        for i in 0..M {
            for j in 0..N {
                result[i][j] = self[i][j] + other[i][j]
            }
        }
        result
    }

    fn subtract(&self, other: &Self) -> Matrix<M, N> {
        let mut result = Matrix::zeros();
        
        for i in 0..M {
            for j in 0..N {
                result[i][j] = self[i][j] - other[i][j]
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

    pub fn sum_rows(&self) -> Matrix<N, 1> {
        let mut result = Matrix::<N, 1>::zeros();
        for i in 0..M {
            for j in 0..N {
                result[0][j] += result[i][j];
            }
        }
        return result;
    }

    pub fn transpose(&self) -> Matrix<N, M> {
        let mut result = Matrix::<N, M>::zeros();
        for i in 0..M{
            for j in 0..N{
                result[j][i] = self[i][j]
            }
        }
        return result
    }

    pub fn from_data(data: [[f64; N]; M]) -> Self {
        Self { data }
    }

    pub fn to_data(self) -> [[f64; N]; M] {
        return self.data.clone()
    }

    pub fn pseudo_inverse(&self) -> Matrix<N, M>{
        //Use nalgebra svd implementation
        let flat_data: Vec<f64> = self.data.iter().flatten().map(|x| *x).collect::<Vec<f64>>();
        let matrix: nam<f64, Dyn, Dyn, na::VecStorage<f64, Dyn, Dyn>> = nam::<f64, Dyn, Dyn, na::VecStorage<f64, Dyn, Dyn>>::from_row_slice(M, N, flat_data.as_slice());
        let svd = matrix.svd(true, true);

        let ps_i = svd.pseudo_inverse(f64::EPSILON).unwrap();
        let mut result = Matrix::<N, M>::zeros();
        for i in 0..N { for j in 0..M { result[i][j] = ps_i[(i, j)]}}

        return result;
    }
}

impl<const M: usize> Matrix<M, M> {
    pub fn identity() -> Matrix<M, M> {
        let mut result = Matrix::zeros();
        for i in 0..M {
            result[i][i] = 1.0;
        }
        return result
    }

    pub fn inverse(&self) -> Option<Matrix<M, M>> {
        let mut result = Matrix::<M,M>::identity();
        let mut curr = self.clone();
        for i in 0..M {
            let (max_idx, max) = curr.max_abs_in_col(i, i);
            if max == 0.0 {
                return None
            }
            if max_idx > i {
                //perform hoisting;
                //swap max to current_row for all columns in the max_row
                for j in 0..M {
                    (curr[i][j], curr[max_idx][j]) = (curr[max_idx][j], curr[i][j]);
                    (result[i][j], result[max_idx][j]) = (result[max_idx][j], result[i][j]);
                }
            }
            //perform gaussian elimination
        }
        return Some(result)
    }

    fn max_abs_in_col(&self, j: usize, min_index: usize) -> (usize, f64) {
        let mut max = 0.0;
        let mut max_idx = 0;

        for i in min_index..M {
            let abs_val = self[i][j].abs();
            if abs_val > max {
                max = abs_val;
                max_idx = i;
            }
        }

        return (max_idx, max)
    }
}


impl<const M: usize> Vector<M> {

    pub fn length(self) -> f64{
        return f64::sqrt(self.dot(&self))
    }

    pub fn pow(&self, power: f64) -> Self{
        let mut new_data = [[0.0; 1];M];
        for i in 0..M {
            new_data[i][0] = f64::powf(self[i][0], power);
        }
        return Vector{data:new_data}
    }

    pub fn dot(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..M {
            sum += self[i][0] * other[i][0]
        }
        return sum
    }

    pub fn from_array(data: [f64; M]) -> Self {
        let mut new_data = [[0.0; 1];M];
        for i in 0..M { 
            new_data[i][0] = data[i]
        }
        return Vector{data:new_data}
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<&Matrix<N,P>> for &Matrix<M, N>{
    type Output = Matrix<M, P>;

    fn mul(self, other: &Matrix<N, P>) -> Matrix<M, P> {
        return self.multiply(&other)
    }
}

impl<const M: usize, const N: usize> Mul<&Matrix<M, N>> for f64 {
    type Output = Matrix<M, N>;

    fn mul(self, other: &Matrix<M, N>) -> Matrix<M, N> {
        let mut result = Matrix::<M,N>::zeros();
        for i in 0..M {
            for j in 0..N {
                result[i][j] = self*other[i][j]
            }
        }
        return result
    }
}

impl<const M: usize, const N: usize> Add<&Matrix<M, N>> for &Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn add(self, other: &Matrix<M, N>) -> Matrix<M, N> {
        return self.add(other)
    }
}

impl<const M: usize, const N: usize> Sub<&Matrix<M, N>> for &Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn sub(self, other: &Matrix<M,N>) -> Matrix<M, N> {
        return self.subtract(other)
    }
}

impl<const M: usize, const N: usize> fmt::Display for Matrix<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for col in self.data {
            write!(f, "| ")?;

            for val in col {
                write!(f, "{:>8}", format!("{:.2}", val))?;
            }

            writeln!(f, " |")?;
        }

        Ok(())
    }
}


#[cfg(test)]
mod test {
    use mat::{Matrix, Vector};

    #[test]
    fn additition() {
        let a = Matrix::from_data([[1.0, 2.0, 5.0], [3.0, 4.0, 3.0]]);
        let b = Matrix::from_data([[5.0, 6.0, 9.0], [7.0, 8.0, 8.0]]);

        let c = &a + &b;

        let expected = [[6.0, 8.0, 14.0], [10.0, 12.0, 11.0]];
        assert_eq!(c.data, expected)
    }

    
    #[test]
    fn subtraction() {
        let a = Matrix::from_data([[1.0, 2.0, 3.0], [3.0, 4.0, 9.0]]);
        let b = Matrix::from_data([[8.0, 7.0, 3.0], [6.0, 5.0, 2.0]]);

        let c = &a - &b;

        let expected = [[-7.0, -5.0, 0.0], [-3.0, -1.0, 7.0]];
        assert_eq!(c.data, expected)
    }

    #[test]
    fn multiplication_square() {
        let a = Matrix::from_data([[2.0, 0.0], [0.0, 9.0]]);
        let b = &a * &a;

        let expected = [[4.0, 0.0], [0.0, 81.0]];
        assert_eq!(b.data, expected)
    }

    #[test]
    fn multiplication_uneq() {
        let a = Matrix::from_data([[2.0, 1.0], [4.0, 3.0]]);
        let b = Vector::from_data([[3.0], [7.0]]);

        let c = &a * &b;

        let expected = [[13.0], [33.0]];
        assert_eq!(c.data, expected)
    }

    #[test]
    fn dot() {
        let a = Vector::from_data([[3.0], [2.0]]);
        let b = Vector::from_data([[5.0], [7.0]]);
        let c = a.dot(&b);

        let expected = 29.0;
        assert_eq!(c, expected)
    }

    #[test]
    fn length() {
        let a = Vector::from_data([[4.0], [3.0]]).length();
        let expected = 5.0;

        assert_eq!(a, expected)
    }
}
