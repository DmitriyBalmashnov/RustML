use mat::{Matrix, Vector};

pub struct LinearRegressor<const M: usize, const N: usize> {
    theta: Vector<N>
}

pub enum Optimizer{
    NaiveGradient(f64),
    Adam(AdamParams),
    PseudoInverse
}

impl Optimizer {
    fn optimize<const M: usize, const N: usize>(&self, linreg: &mut LinearRegressor<M, N>, x: &Matrix<M, N>, y: &Vector<M>) -> f64 {
        match self {
            Self::NaiveGradient(learning_rate) => naive(*learning_rate, linreg, x, y),
            Self::Adam(params) => adam(params, linreg, x, y),
            Self::PseudoInverse => pseudo_inverse_solve(linreg, x, y),
        }
    }
}
fn naive<const M: usize, const N: usize>(learning_rate: f64, linreg: &mut LinearRegressor<M, N>, x: &Matrix<M, N>, y: &Vector<M>) -> f64 {
    let mut prev_error = f64::MAX;
    let mut curr_error = 0.0;
    while (prev_error - curr_error).abs() > f64::EPSILON {
        let y_hat = &linreg.batch_predict(x);
        linreg.theta = &linreg.theta - &(learning_rate * &linreg.gradient(x, y, y_hat));
        prev_error = curr_error;
        curr_error = (y - y_hat).length();
    }
    return curr_error
}

fn adam<const M: usize, const N: usize>(params: &AdamParams, linreg: &mut LinearRegressor<M, N>, x: &Matrix<M, N>, y: &Vector<M>) -> f64 {
    let mut timestep = 0;
    let mut prev_error = f64::MAX;
    let mut curr_error = 0.0;
    let mut first_moment = Vector::<N>::zeros();
    let mut second_moment = Vector::<N>::zeros();

    while (prev_error - curr_error).abs() > f64::EPSILON {
        timestep += 1;
        let y_hat = &linreg.batch_predict(x);
        let grad = &linreg.gradient(x, y, y_hat);
        first_moment = &(params.decay_first_moment * &first_moment) + &((1.0 - params.decay_first_moment) * grad);
        second_moment = &(params.decay_second_moment* &second_moment) + &((1.0 - params.decay_second_moment) * &(grad.pow(2.0)));
        let bias_corr_f = 1.0/(1.0 - f64::powi(params.decay_first_moment,timestep)) * &first_moment;
        let bias_corr_m = 1.0/(1.0 - f64::powi(params.decay_second_moment, timestep)) * &second_moment;
        
        let mut update = Vector::<N>::zeros();
        for i in 0..N {
            update[i][0] = params.learning_rate * bias_corr_f[i][0] / (f64::sqrt(bias_corr_m[i][0]) + f64::EPSILON);
        }
        linreg.theta = &linreg.theta - &update;
        prev_error = curr_error;
        curr_error = (y - y_hat).length();
        if timestep % 10000 == 0{
            println!("timestep: {timestep}, error: {curr_error}")
        }
    }
    return curr_error
}

fn pseudo_inverse_solve<const M: usize, const N: usize>(linreg: &mut LinearRegressor<M, N>, x: &Matrix<M, N>, y: &Vector<M>) -> f64 {
    let pseudo_inverse = x.pseudo_inverse();
    let theta_hat = &pseudo_inverse* &y;
    linreg.theta = theta_hat;
    let y_hat = linreg.batch_predict(x);
    return (y-&y_hat).length();
}

pub struct AdamParams{
    learning_rate: f64,
    decay_first_moment: f64,
    decay_second_moment: f64,
}

impl Default for AdamParams{
    fn default() -> Self {
        return AdamParams { learning_rate: 0.9, decay_first_moment: 0.9, decay_second_moment: 0.999 }
    }
}

impl<const M: usize, const N: usize> LinearRegressor<M, N> {
    pub fn train(x: &Matrix<M, N>, y:&Vector<M>, opt: Optimizer) -> Self {
        let initial_theta = Vector::<N>::zeros();
        let mut curr_model = LinearRegressor{theta: initial_theta};
        opt.optimize(&mut curr_model, x, y);
        return curr_model;
    }

    pub fn batch_predict(&self, x: &Matrix<M, N>) -> Vector<M> {
        return x * &self.theta;
    }

    pub fn predict(&self, x: &Vector<N>) -> f64 {
        return (&x.transpose() * &self.theta)[0][0];
    }

    pub fn gradient(&self, x: &Matrix<M, N>, y:&Vector<M>, y_hat:&Vector<M>) -> Vector<N> {
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
    use linreg::{LinearRegressor, Optimizer, AdamParams};
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
