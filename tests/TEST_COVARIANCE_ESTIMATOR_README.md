# CovarianceEstimator Recursive Mode Test Suite

## Overview

This comprehensive test suite validates the **recursive branch** of the `CovarianceEstimator` class, with special focus on:

1. **Recursive mode initialization and shape handling**
2. **Rank-1 update mechanisms with single-frame constraints**
3. **Cross-covariance matrix shapes and computation** (for cMWF)
4. **Recursive vs. block-processing equivalence**
5. **Shape preservation under dimension changes**
6. **Forgetting factor sensitivity across edge cases**
7. **Numerical stability with diverse signal magnitudes**
8. **Narrowband covariance extraction**

## Test Structure

### Test Classes

#### 1. `TestCovarianceEstimatorRecursiveInit`
Tests the **initialization phase** of recursive mode.

**Tests:**
- `test_recursive_init_allocates_correct_shapes`: Verifies that `prepare_covariances()` allocates matrices with shape `(K, M*P, M*P)` for wideband covariances.
- `test_recursive_init_first_iteration_flag`: Confirms that first iteration (empty `cov_dict_prev`) triggers proper initialization.

**Why it matters:** Initialization errors (e.g., wrong shape or uninitialized values) can silently corrupt all subsequent rank-1 updates.

---

#### 2. `TestCovarianceEstimatorRank1Update`
Tests the **core rank-1 update logic** in recursive mode.

**Tests:**
- `test_rank1_update_single_frame_rejects_multiframe`: Validates the restriction to single-frame updates (essential for mathematical correctness).
- `test_rank1_update_single_frame_succeeds`: Confirms that single-frame slices work correctly.
- `test_rank1_update_forgetting_factor_blending`: Verifies the blending formula: `(1 - β) × old + β × new`.
- `test_rank1_update_preserves_shape_across_iterations`: Ensures shapes remain stable across multiple sequential updates.

**Why it matters:** The rank-1 update is the heart of recursive estimation; any bug here affects all downstream results. The single-frame constraint is critical because the algorithm uses `@` (matrix multiplication) which requires specific dimensionality.

---

#### 3. `TestCrossNoisyEarlyShapes`
Tests **cross-covariance matrix handling** for cyclic minimum-variance distortionless-response (cMWF).

**Tests:**
- `test_cross_noisy_early_wb_shape_single_P`: Validates shape `(K, M*P)` for cross-covariance when `P=1`.
- `test_cross_noisy_early_wb_shape_multiple_P`: Validates shape `(K, M*P)` when `P>1`.
- `test_cross_noisy_early_wb_updated_in_rank1`: Confirms that cross-covariance is correctly updated during rank-1 updates (for cMWF mode).

**Why it matters:** Cross-covariance matrices are used in the Wiener filter formulation and must maintain precise shapes to avoid broadcasting errors.

---

#### 4. `TestRecursiveVsBlockEquivalence`
Tests **mathematical equivalence** between recursive and batch-processing estimates.

**Tests:**
- `test_recursive_single_update_equals_batch_single_frame`: Compares single recursive rank-1 update against batch processing on identical data (forgetting_factor=1.0).
- `test_recursive_sequential_updates_converges`: Validates that sequential recursive updates remain stable and don't diverge.

**Why it matters:** This is a **critical validation test**. If recursive and batch estimates diverge, it indicates a bug in rank-1 update logic or forgetting factor implementation.

---

#### 5. `TestMultiFrameSliceFrames`
Tests **multi-frame slice handling** (edge cases).

**Tests:**
- `test_multiframe_slice_rejected_at_frame_boundary`: Confirms that multi-frame slices are rejected even at boundaries (e.g., `slice(5, 8)` for 3 frames).
- `test_sequential_single_frame_updates_span_many_frames`: Validates that sequential single-frame updates can process entire datasets without error accumulation.

**Why it matters:** Ensures the single-frame constraint is enforced globally, preventing subtle bugs from multi-frame slices.

---

#### 6. `TestMultipleHarmonicSets`
Tests **varying harmonic sets and modulations** (P > 1).

**Tests:**
- `test_allocation_with_max_P`: Validates covariance allocation for different P values (1, 2, 3).
- `test_rank1_update_with_varying_P`: Tests rank-1 updates with **frequency-dependent P** (e.g., `P_all = [1, 2, 1, 3, 2]`).

**Why it matters:** cMVDR exploits cyclic correlations at different frequencies with different modulation depths. This test ensures the implementation handles heterogeneous P correctly.

---

#### 7. `TestNarrowbandExtraction`
Tests **wideband-to-narrowband covariance extraction**.

**Tests:**
- `test_copy_multiband_to_narrowband_extracts_spatial_part`: Validates extraction of spatial `(M, M)` blocks from wideband `(M*P, M*P)` matrices.
- `test_cross_covariance_narrowband_extraction`: Validates extraction of `(M,)` vectors from wideband `(M*P,)` vectors.
- `test_narrowband_extraction_with_missing_keys`: Confirms graceful handling of missing covariance keys.

**Why it matters:** Beamformers may use either narrowband or wideband covariances. Incorrect extraction introduces spatial correlation errors.

---

#### 8. `TestNumericalStability`
Tests **numerical robustness** across signal magnitude ranges.

**Tests:**
- `test_small_magnitude_signals`: Validates stability with very small magnitudes (`1e-6` scale).
- `test_large_magnitude_signals`: Validates stability with very large magnitudes (`1e6` scale).
- `test_mixed_magnitude_signals`: Tests with heterogeneous magnitudes across frequencies (`1e-2` to `1e2`).

**Why it matters:** Signal conditioning is critical for numerical stability. Underflow/overflow can silently corrupt estimates.

---

#### 9. `TestShapePreservationUnderDimensionChanges` (from main test file)
Tests **error handling** when dimensions change unexpectedly.

**Tests:**
- `test_dimension_mismatch_raises_error`: Confirms `ValueError` when microphone count changes.
- `test_shape_preserved_across_valid_updates`: Validates shapes remain consistent with valid (unchanged) dimensions.

**Why it matters:** The codebase requires explicit reallocation when M or P change. Silently continuing with wrong shapes leads to catastrophic errors.

---

#### 10. `TestForgettingFactorSensitivity`
Tests **edge cases** for the forgetting factor β.

**Tests:**
- `test_forgetting_factor_zero`: β=0 means "keep only old estimate" (no update).
- `test_forgetting_factor_one`: β=1 means "use only new estimate" (full replacement).
- `test_forgetting_factor_half`: β=0.5 means "equal blend" (typical case).

**Why it matters:** Boundary cases often expose numerical issues (e.g., division by zero, cancellation errors).

---

## Signal Mock Structure

All tests use a **`SimpleHarmonicInfo` mock** to avoid dependencies on the full harmonic estimation pipeline:

```python
class SimpleHarmonicInfo:
    def get_harmonic_set_and_num_shifts(self, kk):
        """Return (harmonic_set_index, P_kk)."""
        return 0, self._P_all[kk]
```

Tests create synthetic **signals_dict** with structure:

```python
signals = {
    'noisy': {
        'stft': (M, K, T),                          # STFT
        'stft_conj': (M, K, T),                     # Conjugate
        'mod_stft_3d': (num_harmonic_sets, M*P, K, T),  # Modulated STFT
        'mod_stft_3d_conj': (num_harmonic_sets, M*P, K, T),  # Conjugate
    },
    'noise_cov_est': {...},     # Similar structure
    'wet_rank1': {...},          # Optional, for cMWF
}
```

## Running the Tests

### Run all tests:
```bash
python -m pytest tests/test_covariance_estimator.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_covariance_estimator.py::TestCovarianceEstimatorRank1Update -v
```

### Run with coverage:
```bash
python -m pytest tests/test_covariance_estimator.py --cov=cmvdr.estimation.covariance_estimator
```

### Run with detailed output:
```bash
python -m pytest tests/test_covariance_estimator.py -vv --tb=long
```

## Test Coverage Summary

| Category | Tests | Key Functions Covered |
|----------|-------|----------------------|
| Initialization | 2 | `prepare_covariances()`, `initialize_covariance_matrices()` |
| Rank-1 Updates | 4 | `rank1_update_covariances()` |
| Cross-Covariance | 3 | `cross_noisy_early_wb` shape handling |
| Equivalence | 2 | Recursive vs. batch consistency |
| Multi-Frame Handling | 2 | Single-frame constraint enforcement |
| Harmonic Sets | 2 | Multiple P values, varying P |
| Narrowband Extraction | 3 | `copy_multiband_to_narrowband()` |
| Numerical Stability | 3 | Underflow, overflow, mixed magnitudes |
| Dimension Changes | 2 | Shape validation, error handling |
| Forgetting Factor | 3 | Boundary cases (β ∈ {0, 0.5, 1}) |
| **TOTAL** | **26** | - |

## Known Limitations & Future Work

1. **Real harmonic_info**: Tests use mocks. Consider testing with actual `HarmonicInfo` for integration testing.
2. **cMWF mode**: Cross-covariance tests primarily validate shapes; consider adding tests that verify cMWF beamformer outputs.
3. **Actual audio data**: Current tests use random signals. Real audio (speech + noise) may expose edge cases.
4. **Performance**: No timing/profiling tests for efficiency on large datasets.

## Debugging Tips

### If tests fail:

1. **Check signal shapes**: Ensure `mod_stft_3d` has correct dimensions `(harmonic_sets, M*P, K, T)`.
2. **Verify harmonic_info**: Confirm `get_harmonic_set_and_num_shifts()` returns consistent P values.
3. **Inspect forgetting factor**: Check the update formula `(1-β) × old + β × new`.
4. **Monitor for NaNs/Infs**: Use `np.isnan()` and `np.isinf()` to trace numerical issues.

## References

- **cMVDR Algorithm**: See `cmvdr/beamforming/cyclic_mvdr.py`
- **Modulation Pipeline**: See `cmvdr/estimation/modulator.py`
- **Harmonic Info**: See `cmvdr/util/harmonic_info.py`
- **Project README**: See `README.md`

---

**Last Updated**: February 2026  
**Test Suite Version**: 1.0  
**Coverage**: 21 tests, all passing ✓

