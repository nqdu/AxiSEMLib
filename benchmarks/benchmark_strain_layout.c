/* Compare the two C-order layouts used for the strain derivative sums.
 *
 * Build and run on the local CPU:
 *   cc -O3 -march=native -std=c11 benchmarks/benchmark_strain_layout.c \
 *      -o /tmp/benchmark_strain_layout
 *   /tmp/benchmark_strain_layout
 *
 * Optional arguments are the number of time samples and calls per trial.
 * The numerical work matches the xi and eta derivative loop in sem_funcs.py.
 * Compilation, allocation, and layout conversion are outside timed regions.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { COMPONENTS = 3, N = 5, TRIALS = 9 };
typedef void (*derivative_kernel)(const double *, const double *,
                                  const double *, double *, size_t);
static volatile double benchmark_sink = 0.0;

static double *allocate(size_t count)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, count * sizeof(double)) != 0) {
        fprintf(stderr, "allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static double sample(uint64_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return (double)(*state & 0xffffu) / 65536.0 - 0.5;
}

static double seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

/* u is (component, eta, xi, time); out is (component, eta, xi, derivative, time). */
__attribute__((noinline))
static void derivatives_time_last(const double *restrict u,
                                  const double *restrict G,
                                  const double *restrict GT,
                                  double *restrict out, size_t nt)
{
    for (size_t c = 0; c < COMPONENTS; ++c) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                double *dxi = out + ((((c * N + j) * N + i) * 2) * nt);
                double *deta = dxi + nt;
                for (size_t t = 0; t < nt; ++t) {
                    dxi[t] = 0.0;
                    deta[t] = 0.0;
                }
                for (size_t k = 0; k < N; ++k) {
                    const double wx = GT[i * N + k];
                    const double we = G[k * N + j];
                    const double *ux = u + (((c * N + j) * N + k) * nt);
                    const double *ue = u + (((c * N + k) * N + i) * nt);
                    for (size_t t = 0; t < nt; ++t) {
                        dxi[t] += wx * ux[t];
                        deta[t] += we * ue[t];
                    }
                }
            }
        }
    }
}

/* u is (time, component, eta, xi); out is (time, component, eta, xi, derivative). */
__attribute__((noinline))
static void derivatives_time_first(const double *restrict u,
                                   const double *restrict G,
                                   const double *restrict GT,
                                   double *restrict out, size_t nt)
{
    for (size_t t = 0; t < nt; ++t) {
        for (size_t c = 0; c < COMPONENTS; ++c) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < N; ++i) {
                    double dxi = 0.0;
                    double deta = 0.0;
                    for (size_t k = 0; k < N; ++k) {
                        dxi += GT[i * N + k] * u[((t * COMPONENTS + c) * N + j) * N + k];
                        deta += G[k * N + j] * u[((t * COMPONENTS + c) * N + k) * N + i];
                    }
                    const size_t index = (((t * COMPONENTS + c) * N + j) * N + i) * 2;
                    out[index] = dxi;
                    out[index + 1] = deta;
                }
            }
        }
    }
}

/* Same time-first layout and loop order, with the five k terms explicit. */
__attribute__((noinline))
static void derivatives_time_first_unrolled(const double *restrict u,
                                            const double *restrict G,
                                            const double *restrict GT,
                                            double *restrict out, size_t nt)
{
    for (size_t t = 0; t < nt; ++t) {
        for (size_t c = 0; c < COMPONENTS; ++c) {
            const double *plane = u + (t * COMPONENTS + c) * N * N;
            for (size_t j = 0; j < N; ++j) {
                const double *row = plane + j * N;
                for (size_t i = 0; i < N; ++i) {
                    double dxi = GT[i * N] * row[0];
                    dxi += GT[i * N + 1] * row[1];
                    dxi += GT[i * N + 2] * row[2];
                    dxi += GT[i * N + 3] * row[3];
                    dxi += GT[i * N + 4] * row[4];

                    double deta = G[j] * plane[i];
                    deta += G[N + j] * plane[N + i];
                    deta += G[2 * N + j] * plane[2 * N + i];
                    deta += G[3 * N + j] * plane[3 * N + i];
                    deta += G[4 * N + j] * plane[4 * N + i];

                    const size_t index = (((t * COMPONENTS + c) * N + j) * N + i) * 2;
                    out[index] = dxi;
                    out[index + 1] = deta;
                }
            }
        }
    }
}

static int compare_doubles(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double benchmark(derivative_kernel kernel,
                        const double *u, const double *G, const double *GT,
                        double *out, size_t nt, size_t calls)
{
    const double start = seconds();
    for (size_t call = 0; call < calls; ++call) {
        kernel(u, G, GT, out, nt);
        benchmark_sink += out[0];
    }
    return (seconds() - start) * 1e6 / (double)calls;
}

int main(int argc, char **argv)
{
    const size_t nt = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 2048;
    const size_t calls = argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : 100;
    if (argc > 3 || nt == 0 || calls == 0) {
        fprintf(stderr, "usage: %s [time_samples] [calls_per_trial]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t u_count = COMPONENTS * N * N * nt;
    const size_t out_count = 2 * u_count;
    double *u_last = allocate(u_count);
    double *u_first = allocate(u_count);
    double *out_last = allocate(out_count);
    double *out_first = allocate(out_count);
    double *out_unrolled = allocate(out_count);
    double *G = allocate(N * N);
    double *GT = allocate(N * N);
    uint64_t state = 77;

    for (size_t i = 0; i < N * N; ++i) {
        G[i] = sample(&state);
        GT[i] = sample(&state);
    }
    for (size_t c = 0; c < COMPONENTS; ++c) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t t = 0; t < nt; ++t) {
                    const double value = sample(&state);
                    u_last[(((c * N + j) * N + i) * nt) + t] = value;
                    u_first[(((t * COMPONENTS + c) * N + j) * N) + i] = value;
                }
            }
        }
    }

    derivatives_time_last(u_last, G, GT, out_last, nt);
    derivatives_time_first(u_first, G, GT, out_first, nt);
    derivatives_time_first_unrolled(u_first, G, GT, out_unrolled, nt);
    double max_error = 0.0;
    for (size_t c = 0; c < COMPONENTS; ++c) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t d = 0; d < 2; ++d) {
                    for (size_t t = 0; t < nt; ++t) {
                        const double old_value = out_last[((((c * N + j) * N + i) * 2 + d) * nt) + t];
                        const size_t index = ((((t * COMPONENTS + c) * N + j) * N + i) * 2) + d;
                        const double rolled_error = fabs(old_value - out_first[index]);
                        const double unrolled_error = fabs(old_value - out_unrolled[index]);
                        if (rolled_error > max_error) max_error = rolled_error;
                        if (unrolled_error > max_error) max_error = unrolled_error;
                    }
                }
            }
        }
    }
    if (max_error > 1e-12) {
        fprintf(stderr, "layout comparison failed: max error %.3g\n", max_error);
        return EXIT_FAILURE;
    }

    for (int warmup = 0; warmup < 10; ++warmup) {
        derivatives_time_last(u_last, G, GT, out_last, nt);
        derivatives_time_first(u_first, G, GT, out_first, nt);
        derivatives_time_first_unrolled(u_first, G, GT, out_unrolled, nt);
    }

    derivative_kernel kernels[3] = {
        derivatives_time_last,
        derivatives_time_first,
        derivatives_time_first_unrolled
    };
    const double *inputs[3] = {u_last, u_first, u_first};
    double *outputs[3] = {out_last, out_first, out_unrolled};
    double timings[3][TRIALS];
    for (int trial = 0; trial < TRIALS; ++trial) {
        for (int order = 0; order < 3; ++order) {
            const int variant = (trial + order) % 3;
            timings[variant][trial] = benchmark(kernels[variant], inputs[variant],
                                                G, GT, outputs[variant], nt, calls);
        }
    }
    for (int variant = 0; variant < 3; ++variant) {
        qsort(timings[variant], TRIALS, sizeof(double), compare_doubles);
    }
    const double last_us = timings[0][TRIALS / 2];
    const double first_us = timings[1][TRIALS / 2];
    const double unrolled_us = timings[2][TRIALS / 2];
    printf("n=%d nt=%zu calls=%zu max_error=%.3g\n", N, nt, calls, max_error);
    printf("time last:                %.2f us/call\n", last_us);
    printf("time first, rolled:       %.2f us/call\n", first_us);
    printf("time first, unrolled:     %.2f us/call\n", unrolled_us);
    printf("unrolled / time last:     %.2fx\n", unrolled_us / last_us);
    printf("unrolled / rolled:        %.2fx\n", unrolled_us / first_us);
    printf("checksum: %.6g\n", benchmark_sink);

    free(u_last);
    free(u_first);
    free(out_last);
    free(out_first);
    free(out_unrolled);
    free(G);
    free(GT);
    return EXIT_SUCCESS;
}
