use mat::{Matrix, Vector};

struct LinearRegressor<const M: usize, const N: usize> {
    theta: Vector<N>
}

impl<const M: usize, const N: usize> LinearRegressor<M, N> {
    fn train(x: &Matrix<M, N>, y:&Vector<M>, learning_rate: f64) -> Self {
        let initial_theta = Vector::<N>::zeros();
        let mut curr_model = LinearRegressor{theta: initial_theta};
        let mut prev_error = f64::MAX;
        let mut curr_error = 0.0;
        while (prev_error - curr_error).abs() > f64::EPSILON {
            curr_model.theta = &curr_model.theta - &curr_model.calc_new_gradient(x, y, learning_rate);
            prev_error = curr_error;
            curr_error = (y - &curr_model.predict(x)).length();
        }
        return curr_model;
    }

    fn predict(&self, x: &Matrix<M, N>) -> Vector<M> {
        return x * &self.theta;
    }

    fn calc_new_gradient(&self, x: &Matrix<M, N>, y:&Vector<M>, learning_rate: f64) -> Vector<N>{
        let y_hat = self.predict(x);
        let mut gradient = Vector::<N>::zeros();
        for i in 0..M {
            for j in 0..N{
                gradient[j][0] += (y_hat[i][0] - y[i][0]) * x[i][j]
            }
        }

        gradient = learning_rate/(M as f64) * &gradient;
        return gradient
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

        let model = LinearRegressor::train(&x, &y, 0.00001);
        let expected_theta = [[0.0], [1.0]];
        let resulting_theta = model.theta.to_data();

        assert_eq!(expected_theta, resulting_theta)
    }

}
