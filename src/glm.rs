trait GLM<N> {
    fn predict(&self, x: Vector<N>) -> f64;
    fn batch_predict<const M: usize>(&self, x: Matrix<M, N>) -> Vector<M>;
    fn gradient<const M: usize>(&self, x: &Matrix<M, N>, y:&Vector<M>, y_hat:&Vector<M>) -> Vector<N> {
        let mut gradient = Vector::<N>::zeros();
        for i in 0..M {
            for j in 0..N{
                gradient[j][0] += (y_hat[i][0] - y[i][0]) * x[i][j]
            }
        }
        return 1.0/(M as f64) * &gradient;
    }
}