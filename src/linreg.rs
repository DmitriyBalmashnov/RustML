use mat::{Matrix, Vector};
use optimizer::Optimizer;

pub struct LinearRegressor<const N: usize> {
    pub theta: Vector<N>
}

impl<const N: usize> LinearRegressor<N> {
    pub fn train<const M: usize>(x: &Matrix<M, N>, y:&Vector<M>, opt: Optimizer) -> Self {
        let initial_theta = Vector::<N>::zeros();
        let mut curr_model = LinearRegressor{theta: initial_theta};
        let mse = opt.optimize(&mut curr_model, x, y);
        println!("Finished training, MSE: {mse}");
        return curr_model;
    }

    pub fn batch_predict<const M: usize>(&self, x: &Matrix<M, N>) -> Vector<M> {
        return x * &self.theta;
    }

    pub fn predict(&self, x: &Vector<N>) -> f64 {
        return (&x.transpose() * &self.theta)[0][0];
    }

    pub fn gradient<const M: usize>(&self, x: &Matrix<M, N>, y:&Vector<M>, y_hat:&Vector<M>) -> Vector<N> {
        let mut gradient = Vector::<N>::zeros();
        for i in 0..M {
            for j in 0..N{
                gradient[j][0] += (y_hat[i][0] - y[i][0]) * x[i][j]
            }
        }
        return 1.0/(M as f64) * &gradient;
    }
}

#[cfg(test)]
mod test {
    use linreg::{LinearRegressor};
    use optimizer::{Optimizer, AdamParams};
    use mat::{Matrix, Vector};

    #[test]
    fn simple_fit() {
        let x = Matrix::from_data([[1.0, 0.0], [1.0, 4.0]]);
        let y = Vector::from_data([[0.0], [4.0]]);

        let model = LinearRegressor::train(&x, &y, Optimizer::NaiveGradient(0.001));
        let expected_theta = Vector::from_data([[0.0], [1.0]]);
        let resulting_theta = model.theta;
        let error = (&expected_theta - &resulting_theta).length();

        assert!(error < 0.0001, "{}, {} unequal", expected_theta, resulting_theta)
    }

    #[test]
    fn simple_fit_adam() {
        let x = Matrix::from_data([[1.0, 0.0], [1.0, 4.0]]);
        let y = Vector::from_data([[0.0], [4.0]]);

        let model = LinearRegressor::train(&x, &y, Optimizer::Adam(AdamParams::default()));
        let expected_theta = Vector::from_data([[0.0], [1.0]]);
        let resulting_theta = model.theta;
        let error = (&expected_theta - &resulting_theta).length();

        assert!(error < 0.0001, "{}, {} unequal", expected_theta, resulting_theta)
    }

    #[test]
    fn simple_fit_psi() {
        let x = Matrix::from_data([[1.0, 0.0], [1.0, 4.0]]);
        let y = Vector::from_data([[0.0], [4.0]]);

        let model = LinearRegressor::train(&x, &y, Optimizer::PseudoInverse);
        let expected_theta = Vector::from_data([[0.0], [1.0]]);
        let resulting_theta = model.theta;
        let error = (&expected_theta - &resulting_theta).length();

        assert!(error < 0.0001, "{}, {} unequal", expected_theta, resulting_theta)
    }

    #[test]
    fn simple_fit_result() {
        let x = Matrix::from_data([[1.0, 0.0], [1.0, 4.0]]);
        let y = Vector::from_data([[0.0], [4.0]]);

        let model = LinearRegressor::train(&x, &y, Optimizer::NaiveGradient(0.001));
        let result = model.predict(&Vector::from_data([[1.0], [4.0]]));
        let expected_result = 4.0;
        let error = (result-expected_result).abs();

        assert!(error < 0.0001, "{}, {} unequal", result, expected_result)
    }

    #[test]
    fn simple_fit_bias() {
        let x = Matrix::from_data([[1.0, 0.0], [1.0, 4.0]]);
        let y = Vector::from_data([[2.0], [0.0]]);

        let model = LinearRegressor::train(&x, &y, Optimizer::NaiveGradient(0.001));

        let expected_theta = Vector::from_data([[2.0], [-0.5]]);
        let resulting_theta = model.theta;
        let error = (&expected_theta - &resulting_theta).length();

        assert!(error < 0.0001, "{}, {} unequal", expected_theta, resulting_theta)
    }

}
