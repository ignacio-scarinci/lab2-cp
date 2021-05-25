/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include "mt19937ar.h"

char t1[] = "(https://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];


/***
 * Photon
 ***/

static void photon(uint n_primarios)
{
    __m256 albedo = _mm256_set1_ps(MU_S / (MU_S + MU_A));
    __m256 albedo_1 = _mm256_set1_ps(1.0f - (MU_S / (MU_S + MU_A)));
    __m256 shells_per_mfp = _mm256_set1_ps (1e4 / MICRONS_PER_SHELL / (MU_A + MU_S));
    __m256 negativo = _mm256_set1_ps(-1.0f);
    __m256 shell_ = _mm256_set1_ps(SHELLS -1.0f);
    __m256 umbral_w = _mm256_set1_ps(0.001f);
    __m256 umbral_a = _mm256_set1_ps(0.01f);
    __m256 factor_ruleta = _mm256_set1_ps(10.0f);
    __m256 zero =  _mm256_set1_ps(0.0f);
    int fotones = 0;

    while (fotones < n_primarios){
    fotones += 8;
    /* launch */
    __m256 x = _mm256_set1_ps(0.0f);
    __m256 y = _mm256_set1_ps(0.0f);
    __m256 z = _mm256_set1_ps(0.0f);
    __m256 u = _mm256_set1_ps(0.0f);
    __m256 v = _mm256_set1_ps(0.0f);
    __m256 w = _mm256_set1_ps(1.0f);
    __m256 weight = _mm256_set1_ps(1.0f);
    __m256 t = _mm256_set1_ps(0.0f);

    float xi1, xi2, tt;

    for (;;) {
      __m256 aleatorio = _mm256_set_ps(genrand_real1(),
                               genrand_real1(),
                               genrand_real1(),
                               genrand_real1(),
                               genrand_real1(),
                               genrand_real1(),
                               genrand_real1(),
                               genrand_real1());
      t = _mm256_mul_ps(negativo, _mm256_log_ps(aleatorio)); /* move */
      x = _mm256_add_ps(x, _mm256_mul_ps(t, u));
      y = _mm256_add_ps(y, _mm256_mul_ps(t, v));
      z = _mm256_add_ps(z, _mm256_mul_ps(t,  w));

      __m256 shell = _mm256_sqrt_ps(_mm256_add_ps(
                                    _mm256_add_ps(_mm256_mul_ps(x,x),  _mm256_mul_ps(y,y)),
                                    _mm256_mul_ps(z,z)));

      shell = _mm256_mul_ps(shell, shells_per_mfp);

      shell =_mm256_blendv_ps(shell, shell_, _mm256_cmp_ps (shell, shell_, 14 ));

      __m256 calor = _mm256_mul_ps(albedo_1, weight);

      weight = _mm256_mul_ps(weight, albedo);

      for (uint i=0; i<8; i=i+1){
        heat[(int)shell[i]] = heat[(int)shell[i]] + calor[i];
     }

     __m256 u_w = _mm256_cmp_ps (  weight, umbral_w,1 );
     int menor = _mm256_movemask_ps(u_w);

     if (menor > 0){
       weight = _mm256_blendv_ps(weight, _mm256_mul_ps(weight, factor_ruleta), u_w);
      __m256 aleatorio = _mm256_set_ps(genrand_real1(),
                              genrand_real1(),
                              genrand_real1(),
                              genrand_real1(),
                              genrand_real1(),
                              genrand_real1(),
                              genrand_real1(),
                              genrand_real1());
      __m256 u_a = _mm256_cmp_ps( umbral_a,  aleatorio, 14);
      weight = _mm256_blendv_ps(weight, zero, u_a);
      if (_mm256_movemask_ps(weight) == 0){
        break;
      }
    }
        /* New direction, rejection method */
        for (uint i=0; i<8; i=i+1){
        do {
            xi1 = 2.0f * genrand_real1() - 1.0f;
            xi2 = 2.0f * genrand_real1() - 1.0f;
            tt = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < tt);

        u[i] = 2.0f * tt - 1.0f;
        v[i] = xi1 * sqrtf((1.0f - u[i] * u[i]) / tt);
        w[i] = xi2 * sqrtf((1.0f - u[i] * u[i]) / tt);
      }
     }
    }
}

/***
 * Main matter
 ***/

int main(void)
{
    // heading
    printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);

    // configure RNG
    init_genrand(SEED);
    // start timer
    double start = wtime();
    
    // simulation
    photon(PHOTONS);

    for (uint i=0; i<SHELLS; ++i){
        heat2[i] = heat[i] * heat[i];
    }

    // stop timer
    double end = wtime();
    assert(start <= end);
    double elapsed = end - start;

    printf("# %lf seconds\n", elapsed);
    printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed);

    printf("# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");
    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
               heat[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

    return 0;
}
