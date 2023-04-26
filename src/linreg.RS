struct LinearRegressor<const N: usize> {
    y_hat: Vector<N>
}

impl<const N: usize> for LinearRegressor<N> {
    fn train(X: &Matrix<M, N>, y:&Vector<M>, learning_rate: f64) -> Self {
        theta = Vector<N>::zeros();

    }

    fn predict(self, X: &Matrix<M, N>) -> Vector<M> {
        return &X * &self.y_hat
    }
}
