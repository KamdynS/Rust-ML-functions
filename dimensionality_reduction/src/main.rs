extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::{Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::*;

/// Perform mean normalization on the dataset
fn mean_normalize(data: &Array2<f64>) -> Array2<f64> {
    let means = data.mean_axis(Axis(0)).unwrap();
    data - &means
}

/// Compute the covariance matrix of the data
fn covariance_matrix(data: &Array2<f64>) -> Array2<f64> {
    let n_samples = data.nrows() as f64;
    let data = mean_normalize(data);
    let covariance = data.t().dot(&data) / (n_samples - 1.0);
    covariance
}

/// Perform PCA, returning the principal components
fn pca(data: &Array2<f64>, n_components: usize) -> Array2<f64> {
    let covariance = covariance_matrix(data);
    let (eigenvalues, eigenvectors) = covariance.eig().unwrap();
    let mut eigen_pairs: Vec<_> = eigenvalues.iter()
                                             .zip(eigenvectors.gencolumns().into_iter())
                                             .collect();
    // Sort eigenvalues and corresponding eigenvectors
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    // Select the top `n_components` eigenvectors
    let top_components: Array2<f64> = Array2::from_shape_fn((data.ncols(), n_components), |(i, j)| {
        eigen_pairs[j].1[i]
    });
    // Project the data onto the top components
    data.dot(&top_components)
}


fn main() {
    println!("Hello, world!");
}
