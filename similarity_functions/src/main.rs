extern crate rayon;
use rayon::prelude::*;

/// Computes the dot product of two vectors.
fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.par_iter()
     .zip(b.par_iter())
     .map(|(x, y)| x * y)
     .sum()
}

/// Calculates the norm (magnitude) of a vector.
fn norm(a: &Vec<f64>) -> f64 {
    a.par_iter()
     .map(|x| x.powi(2))
     .sum::<f64>()
     .sqrt()
}

/// Calculates the cosine similarity between two vectors.
/// This function panics if the vectors are of different lengths.
fn cosine_similarity(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length");
    }
    let dot = dot_product(a, b);
    let norm_a = norm(a);
    let norm_b = norm(b);

    // Avoid division by zero by checking for zero norms.
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculates the Euclidean distance between two vectors.
fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length");
    }
    a.iter()
     .zip(b.iter())
     .map(|(x, y)| (x - y).powi(2))
     .sum::<f64>()
     .sqrt()
}

/// Calculates the Manhattan distance between two vectors.
fn manhattan_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length");
    }
    a.iter()
     .zip(b.iter())
     .map(|(x, y)| (x - y).abs())
     .sum()
}

/// Calculates the Pearson correlation coefficient between two vectors.
fn pearson_correlation(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length");
    }
    let n = a.len() as f64;
    let sum_a = a.iter().sum::<f64>();
    let sum_b = b.iter().sum::<f64>();
    let sum_a_sq: f64 = a.iter().map(|&x| x.powi(2)).sum();
    let sum_b_sq: f64 = b.iter().map(|&x| x.powi(2)).sum();
    let sum_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let numerator = n * sum_ab - sum_a * sum_b;
    let denominator = ((n * sum_a_sq - sum_a.powi(2)) * (n * sum_b_sq - sum_b.powi(2))).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}


fn main() {
    // Example usage of the cosine similarity function.
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![4.0, 5.0, 6.0];

    let similarity = cosine_similarity(&vec1, &vec2);
    println!("Cosine similarity: {:.4}", similarity);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = dot_product(&vec1, &vec2);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&vec1, &vec2);
        assert_eq!(result, 1.0); // Cosine similarity of identical vectors is 1
    }

    #[test]
    fn test_euclidean_distance() {
        let vec1 = vec![0.0, 0.0];
        let vec2 = vec![3.0, 4.0];
        let result = euclidean_distance(&vec1, &vec2);
        assert_eq!(result, 5.0); // 3-4-5 triangle property
    }
}