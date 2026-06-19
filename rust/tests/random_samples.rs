use rmc_core::random::{
    cauchy_pdf, cauchy_sample, exclusive_uniform_int_pdf, exponential_pdf, exponential_pdf_bounded,
    exponential_sample, exponential_sample_bounded, normal_pdf, safe_exponential_pdf,
    safe_exponential_sample, uniform_int_pdf, uniform_pdf, uniform_sample,
};

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 1e-14 * scale,
        "actual={actual:?} expected={expected:?}"
    );
}

#[test]
fn simple_closed_form_helpers_match_analytical_values() {
    assert_eq!(uniform_sample(0.5, 0.0, 1.0), 0.5);
    assert_eq!(uniform_pdf(0.0, 1.0), 1.0);
    assert_eq!(uniform_int_pdf(0, 9), 0.1);
    assert_eq!(exclusive_uniform_int_pdf(0, 10), 0.1);
}

#[test]
fn floating_sample_helpers_match_closed_forms() {
    assert_close(uniform_sample(0.5, -2.0, 2.0), 0.0);
    assert_close(uniform_pdf(-2.0, 2.0), 0.25);
    assert_close(exponential_sample(1.0, 2.0, 3.0), 3.0);
    assert_close(exponential_pdf(3.0, 2.0, 3.0), 2.0);
    assert_close(exponential_sample_bounded(0.0, 1.25, -1.0, 1.0), -1.0);
    assert_close(
        exponential_pdf_bounded(-1.0, 1.25, -1.0, 1.0),
        1.25 / (1.0 - (-2.5_f64).exp()),
    );
    assert_close(safe_exponential_sample(0.25, 0.0, 2.0, 6.0), 3.0);
    assert_close(safe_exponential_pdf(3.5, 0.0, 2.0, 6.0), 0.25);
    assert_close(
        normal_pdf(-2.0, -2.0, 1.25),
        1.0 / (1.25 * (2.0 * std::f64::consts::PI).sqrt()),
    );
    assert_close(cauchy_sample(0.5, -2.0, 1.25), -2.0);
    assert_close(
        cauchy_pdf(-2.0, -2.0, 1.25),
        1.0 / (std::f64::consts::PI * 1.25),
    );
}

#[test]
fn integer_pdfs_match_closed_forms() {
    assert_eq!(uniform_int_pdf(-2, 2), 0.2);
    assert_eq!(exclusive_uniform_int_pdf(-2, 2), 0.25);
}

#[test]
fn sample_ranges_follow_cpp_preconditions() {
    assert_eq!(uniform_sample(0.0, -2.0, 1.0), -2.0);
    assert!(exponential_sample(0.5, 2.0, 3.0) >= 3.0);

    for lambda in [-1.5, 0.0, 1.25] {
        let sample = safe_exponential_sample(0.625, lambda, -5.0, -2.0);
        assert!((-5.0..=-2.0).contains(&sample));
    }
}
