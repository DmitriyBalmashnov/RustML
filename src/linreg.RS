use mat::{Matrix, Vector};

struct LinearRegressor<const M: usize, const N: usize> {
    y_hat: Vector<N>
}

impl<const M: usize, const N: usize> LinearRegressor<M, N> {
    fn train(x: &Matrix<M, N>, y:&Vector<M>, learning_rate: f64) -> Self {
        let initial_theta = Vector::<N>::zeros();
        let mut curr_model = LinearRegressor{y_hat: initial_theta};
        let mut prev_error = f64::MAX;
        let mut curr_error = 0.0;
        while (prev_error - curr_error).abs() > f64::EPSILON {
            
            prev_error = curr_error;
            curr_error = (y - &curr_model.predict(x)).length();
        }
        return curr_model;
    }

    fn predict(&self, x: &Matrix<M, N>) -> Vector<M> {
        return x * &self.y_hat;
    }

    fn update(&mut self, x: &Matrix<M, N>, y:&Vector<M>, learning_rate: f64) {
        let y_hat = self.predict(x);
        for training_example in 0..M {
            
        }

    }
}

#[cfg(test)]
mod test {
    use linreg::LinearRegressor;
    use mat::{Matrix, Vector};

    #[test]
    fn simple_fit() {
        let x = Matrix::from_data([[1.0, 0.0], [1.0, 4.0]]);
        let y = Vector::from_data([[0.0], [4.0]]);

        let model = LinearRegressor::train(&x, &y, 0.01);
    }

}
