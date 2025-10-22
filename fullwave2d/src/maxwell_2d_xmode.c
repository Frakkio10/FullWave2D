/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/*                                                             */
/* Author: Emilio Blanco                                       */
/*                                                             */
/* Date:   March 1st, 2016                                     */
/*                                                             */
/* Routine: maxwell_2d_xmode.c                                 */
/*                                                             */
/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/* This subroutine solves two-dimensional maxwell's equations  */
/* for a magnetized plasma in X-mode.                          */
/*                                                             */
/* The emitter/receiver antena plane is located at X = 0       */
/*                                                             */
/* #include "fdtd_2d_xmode.h"                                        */
/* maxwell_2d_xmode (struct inputdata *data)                   */
/*                                                             */
/* To call: maxwell_2d_xmode (&data);                          */
/*                                                             */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/*                                                             */
/* BE SURE YOU INCLUDE fdtd_2d_xmode.h file into your code. This     */
/* file declares struct inputdata with the parameters used in  */
/* the routine maxwell. User has to assign the values he want  */
/* to use in the computation.                                 */
/*                                                             */
/*                                                             */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/*-------------------------------------------------------------*/
/* Input: struct inputdata data                                */
/*                                                             */
/* double f0;     Probing frequency (Hz)                       */
/* int nt;        Number of temporal iterations                */
/* int nx;        Number of grid points along X axis           */
/* int ny;        Number of grid points along Y axis           */
/* double dx;     Distance between consecutive grid points (m) */
/* int yante;     Vertical position of beam waist (X=0 plane)  */
/* int waist;     Beam waist (grid points)                     */
/* double angle;  Antena tilt angle (degrees)                  */
/* double **ne;   Plasma density at each position (m-3)        */
/*                                                             */
/* ------------------------------------------------------------*/
/* Output: struct inputdata data                               */
/*                                                             */
/* double *ampl_ant; Amplitud recibida en la antena            */
/* double *fase_ant; Fase recibida en la antena                */
/*-------------------------------------------------------------*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "fdtd_2d.h"


static void set_update_coef_hzx_hzy (double **chz, double **dhz, double **sigmastar,
				     int nfilas, int ncols, double dt) {
  int i, j;
  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      chz[j][i] = (1.0 - sigmastar[j][i]*dt/2.0/U0)/(1.0 + sigmastar[j][i]*dt/2.0/U0);
      dhz[j][i] = S/(1.0 + sigmastar[j][i]*dt/2.0/U0);
    }
  }
}



static void set_update_coef_ex_ey (double **cez, double **dez, double **sigma,
				     int nfilas, int ncols, double dt) {
  int i, j;
  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      cez[j][i] = (1.0 - sigma[j][i]*dt/2.0/E0)/(1.0 + sigma[j][i]*dt/2.0/E0);
      dez[j][i] = S/(1.0 + sigma[j][i]*dt/2.0/E0);
    }
  }

}

static void set_update_coef_jx_jy (double **cj, double **dj, double **wp2, double **omega,
				     int nfilas, int ncols, double dt) {
  int i, j;
  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      cj[j][i] = dt*wp2[j][i]/C*0.5;
      dj[j][i] = dt*omega[j][i];
    }
  }
}



int maxwell_2d_xmode (struct inputdata *data) {

  double **hz, **hzx, **hzy, **hz_inc, **hzx_inc, **hzy_inc;
  double **ey, **ex, **ey_inc, **ex_inc;
  double **cex, **dex, **cey, **dey;
  double **jx, **jy, **wp2, **omega;

  double **chzx, **dhzx, **chzy, **dhzy, **cj, **dj;

  double **sigmax, **sigmay, **sigmastarx, **sigmastary;


  double **I1_emi, **I2_emi, **I_emi, **Q_emi;
  double **I1_rec, **I2_rec, **I_rec, **Q_rec;
  double **ampl_emi, **phase_emi;
  double **ampl_rec, **phase_rec;

  int i, j, n;

  FILE *f, *f2;

  int nt, nx, ny;

  double f0, dx, dt;
  double wdt, wt;
  double dfase, fase;
  double I_ant, Q_ant;

  int npml        = data->npml;
  double reflmax  = data->reflmax;
  int TFSF        = data->TFSF;
  int xante        = data->xante;

  nx = (data->nx) - 1 + 2*TFSF; /* Numero de celdas eje X */
  ny = (data->ny) - 1 + 2*TFSF; /* Numero de celdas eje Y */
  nt = (data->nt);
  dx = (data->dx);

  f0    = (data->f0);
  dt    = S*dx/C;

  wdt = 2.0*PI*f0*dt;
  wt  = wdt;
  I_ant = 0.0;
  Q_ant = 0.0;
  *(data->ampl_ant) = 0.0;
  *(data->fase_ant) = 0.0;


  /* Memory allocation. Variables are initialized to zero */

  hz = hzx = hzy = hz_inc = hzx_inc = hzy_inc = NULL;
  chzx = dhzx = chzy = dhzy = sigmastarx = sigmastary = NULL;
  jx = jy = cj = dj = wp2 = omega = ey = ey_inc = cey = NULL;
  dey = sigmax = ex = ex_inc = cex = dex = sigmay = NULL;

  // ampl_inc = phase_inc = NULL;

  I1_emi = I2_emi = I_emi = Q_emi = I1_rec = I2_rec = I_rec = Q_rec = NULL;
  ampl_emi = phase_emi = ampl_rec = phase_rec = NULL;


  hz         = memory (ny + 1, nx + 1);
  hzx        = memory (ny + 1, nx + 1);
  hzy        = memory (ny + 1, nx + 1);
  hz_inc     = memory (ny + 1, nx + 1);
  hzx_inc    = memory (ny + 1, nx + 1);
  hzy_inc    = memory (ny + 1, nx + 1);
  chzx       = memory (ny + 1, nx + 1);
  dhzx       = memory (ny + 1, nx + 1);
  chzy       = memory (ny + 1, nx + 1);
  dhzy       = memory (ny + 1, nx + 1);
  sigmastarx = memory (ny + 1, nx + 1);
  sigmastary = memory (ny + 1, nx + 1);

  jx    = memory (ny + 1, nx + 1);
  jy    = memory (ny + 1, nx + 1);
  cj    = memory (ny + 1, nx + 1);
  dj    = memory (ny + 1, nx + 1);
  wp2     = memory (ny + 1, nx + 1);
  omega = memory (ny + 1, nx + 1);


  ey     = memory (ny + 1, nx);
  ey_inc = memory (ny + 1, nx);
  cey    = memory (ny + 1, nx);
  dey    = memory (ny + 1, nx);
  sigmax = memory (ny + 1, nx);

  ex     = memory (ny, nx + 1);
  ex_inc = memory (ny, nx + 1);
  cex    = memory (ny, nx + 1);
  dex    = memory (ny, nx + 1);
  sigmay = memory (ny, nx + 1);



  I1_emi = memory (ny + 1, 1);
  I2_emi = memory (ny + 1, 1);
  I_emi  = memory (ny + 1, 1);
  Q_emi  = memory (ny + 1, 1);
  I1_rec = memory (ny + 1, 1);
  I2_rec = memory (ny + 1, 1);
  I_rec  = memory (ny + 1, 1);
  Q_rec  = memory (ny + 1, 1);

  ampl_emi   = memory (ny + 1, 1);
  phase_emi  = memory (ny + 1, 1);
  ampl_rec   = memory (ny + 1, 1);
  phase_rec  = memory (ny + 1, 1);

  /* Electric and magnetic conductivity in PML */
  set_sigma (sigmax, ny + 1, nx, dx, nx, npml, reflmax, 'x', FALSE, 'X');
  set_sigma (sigmay, ny, nx + 1, dx, ny, npml, reflmax, 'y', FALSE, 'X');
  set_sigma (sigmastarx, ny + 1, nx + 1, dx, nx, npml, reflmax, 'x', TRUE, 'X');
  set_sigma (sigmastary, ny + 1, nx + 1, dx, ny, npml, reflmax, 'y', TRUE, 'X');

  /* Update coefficient for hz, ey, ex */
  set_update_coef_ex_ey (cex, dex, sigmay, ny, nx + 1, dt);
  set_update_coef_ex_ey (cey, dey, sigmax, ny + 1, nx, dt);
  set_update_coef_hzx_hzy (chzx, dhzx, sigmastarx, ny + 1, nx + 1, dt);
  set_update_coef_hzx_hzy (chzy, dhzy, sigmastary, ny + 1, nx + 1, dt);



  /* Set plasma frequency squared wp2 */
  for (j = TFSF; j <= ny - TFSF; j++) {
    for (i = TFSF; i <= nx - TFSF; i++) {
	wp2[j][i] = ((data->ne)[j-TFSF][i-TFSF])*E*E/E0/ME;
	if (i == TFSF)
	  wp2[j][i] = 0.0;   /* Avoid non-zero density at TFSF interface */
    }
  }

  /* Set cyclotron frequency */
  for (j = TFSF; j <= ny - TFSF; j++)
    for (i = TFSF; i <= nx - TFSF; i++)
      omega[j][i] = ((data->b0)[j-TFSF][i-TFSF])*E/ME;

  /* Update coefficient for jx, jy */
  set_update_coef_jx_jy (cj, dj, wp2, omega, ny + 1, nx + 1, dt);

  // initialize file in which to store some data during the loop
  if (data->save_diag){
    char fname[200];
    sprintf(fname, "%s/ez_t.dat", data->outp_dir);
    f = fopen(fname, "w");
    char fname2[200];
    sprintf(fname2, "%s/ant_signal_t.dat", data->outp_dir);
    f2 = fopen(fname2, "w");
  }

  /* ------------------------------------------------------------ */
  /* --------------- Begin temporal iterations ------------------ */
  /* ------------------------------------------------------------ */

  n = 1;
  while (n <= nt) {

    if (n%100 == 0) printf("current time step: %d\n", n);

    /* ----------------------------------------------------------- */
    /*               Calcula las corrientes jx, jy                 */
    /* ----------------------------------------------------------- */
    for (j = 1; j <= ny-1; j++) {
      for (i = 1; i <= nx-1; i++) {
	jx[j][i] += cj[j][i]*(ex[j][i] + ex[j-1][i]) - dj[j][i]*jy[j][i];
	jy[j][i] += cj[j][i]*(ey[j][i] + ey[j][i-1]) + dj[j][i]*jx[j][i];
      }
    }

    /* --------------------------------------------------------- */
    /*               Calcula el campo magnetico hz               */
    /* --------------------------------------------------------- */
    for (j = 1; j <= ny-1; j++) {
      for (i = 1; i <= nx-1; i++) {
 	hzx[j][i] = chzx[j][i]*hzx[j][i] - dhzx[j][i]*(ey[j][i] - ey[j][i-1]);
	hzy[j][i] = chzy[j][i]*hzy[j][i] + dhzy[j][i]*(ex[j][i] - ex[j-1][i]);
	hz[j][i] = hzx[j][i] + hzy[j][i];
	// note that the '_inc' field is unaffected by the plasma because not
	// multiplied by dx * jz
 	hzx_inc[j][i] = chzx[j][i]*hzx_inc[j][i] - dhzx[j][i]*(ey_inc[j][i] - ey_inc[j][i-1]);
	hzy_inc[j][i] = chzy[j][i]*hzy_inc[j][i] + dhzy[j][i]*(ex_inc[j][i] - ex_inc[j-1][i]);
	hz_inc[j][i] = hzx_inc[j][i] + hzy_inc[j][i];
      }
    }

    /* -------------------------------------------------- */
    /*            Consistencia TF / SF para hz            */
    /* -------------------------------------------------- */
    for (j = 1; j <= ny - 1; j++) {
      hzx[j][TFSF] += dhzx[j][TFSF]*ey_inc[j][TFSF-1];
      hz[j][TFSF]  += dhzx[j][TFSF]*ey_inc[j][TFSF-1];
    }

    /* ----------------------------------------------------------------- */
    /*   Conductor (magnetico) perfecto en los limites computacionales   */
    /* ----------------------------------------------------------------- */
    /* planos: x = 0, x = nx */
    for (j = 0; j <= ny; j++) {
      hz[j][0]  = jx[j][0]  = jy[j][0]  = hz_inc[j][0]  = 0.0;
      hz[j][nx] = jx[j][nx] = jy[j][nx] = hz_inc[j][nx] = 0.0;
    }

    /* planos: y = 0, y = ny */
    for (i = 0; i <= nx; i++) {
      hz[0][i] = jx[0][i] = jy[0][i] = hz_inc[0][i] = 0.0;
      hz[ny][i] = jx[ny][i] = jy[ny][i] = hz_inc[ny][i] = 0.0;
    }

    /* ------------------------------------------------- */
    /*              Source the input wave                */
    /* ------------------------------------------------- */
    // the first factor ensures a smooth rise of Ez
    for (j = npml + 1; j <= ny - npml - 1; j++)
      hz_inc[j][xante] = (1.0-exp(-(double)(n)/200))*data->ampl_inc[j][0]*cos(wt+data->phase_inc[j][0]);


    /* -------------------------------------------------- */
    /*            Calcula el campo electrico ex           */
    /* -------------------------------------------------- */
    for (j = 0; j <= ny-1; j++) {
      for (i = 0; i <= nx; i++) {
	ex[j][i] = cex[j][i]*ex[j][i] + dex[j][i]*(hz[j+1][i] - hz[j][i] - dx*0.5*(jx[j+1][i]+jx[j][i]));
	ex_inc[j][i] = cex[j][i]*ex_inc[j][i] + dex[j][i]*(hz_inc[j+1][i] - hz_inc[j][i]);
      }
    }

    /* ---------------------------------------------------- */
    /*             Calcula el campo electrico ey            */
    /* ---------------------------------------------------- */
    for (j = 0; j <= ny; j++) {
      for (i = 0; i <= nx-1; i++) {
	ey[j][i] = cey[j][i]*ey[j][i] - dey[j][i]*(hz[j][i+1] - hz[j][i] + dx*0.5*(jy[j][i+1]+jy[j][i]));
	ey_inc[j][i] = cey[j][i]*ey_inc[j][i] - dey[j][i]*(hz_inc[j][i+1] - hz_inc[j][i]);
      }
    }

    /* ---------------------------------------------------- */
    /*            Consistencia TF/SF para ey                */
    /* ---------------------------------------------------- */
    for (j = 0; j < ny - 1; j++)
      ey[j][TFSF-1] += dey[j][TFSF-1]*hz_inc[j][TFSF];

    /* ----------------------------------------------------- */
    /*      Amplitude and phase in detection plane           */
    /* ----------------------------------------------------- */
    if ((n % 2) == 0) {     /* n es par */
      I_ant = 0.0;
      Q_ant = 0.0;
      for (j = npml + 1; j <= ny - npml - 1; j++) {
        I2_emi[j][0] = hz_inc[j][xante];
        I2_rec[j][0] = hz[j][xante];
      }

      for (j = npml + 1; j <= ny - npml - 1; j++) {
        /* Calcula terminos I, Q */
        I_emi[j][0] = (I1_emi[j][0]*sin(wt) - I2_emi[j][0]*sin(wt-wdt))/sin(wdt);
        Q_emi[j][0] = (I1_emi[j][0]*cos(wt) - I2_emi[j][0]*cos(wt-wdt))/sin(wdt);

        I_rec[j][0] = (I1_rec[j][0]*sin(wt) - I2_rec[j][0]*sin(wt-wdt))/sin(wdt);
        Q_rec[j][0] = (I1_rec[j][0]*cos(wt) - I2_rec[j][0]*cos(wt-wdt))/sin(wdt);

        ampl_emi[j][0]  = sqrt(pow(I_emi[j][0], 2.0) + pow (Q_emi[j][0], 2.0));
        // phase_emi[j][0] = atan2 (Q_emi[j][0], I_emi[j][0]);

        ampl_rec[j][0]  = sqrt(pow(I_rec[j][0], 2.0) + pow (Q_rec[j][0], 2.0));
        // phase_rec[j][0] = atan2 (Q_rec[j][0], I_rec[j][0]);

        I_ant += ampl_emi[j][0]*ampl_rec[j][0]*cos(/*phase_emi[j][0]*/-phase_rec[j][0]-data->phase_inc[j][0])/(ny-1.0);
        Q_ant += ampl_emi[j][0]*ampl_rec[j][0]*sin(/*phase_emi[j][0]*/-phase_rec[j][0]-data->phase_inc[j][0])/(ny-1.0);

        *(data->ampl_ant) = sqrt(pow(I_ant, 2.0) + pow (Q_ant, 2.0));
        *(data->fase_ant) = atan2(Q_ant, I_ant);

      }
    }
    else {
      for (j = npml + 1; j <= ny - npml - 1; j++) {
        I1_emi[j][0] = hz_inc[j][xante];
        I1_rec[j][0] = hz[j][xante];
      }
    }

    wt += wdt;
    if (wt > PI)
      wt -= 2.0*PI;

      /* ----------------------------------------------------- */
      /*                    Diagnostics                        */
      /* ----------------------------------------------------- */

      // save diagnostics every given number of timesteps
      if (data->save_diag) {

        // for animations of the electric field:
        // if ((n%50) == 0) save_2d_arr_to_file(nx, ny, hz, f);
        // for time trace of the recieved signal:
        if ((n%10) == 0) {
          fprintf(f2, "%f %f\n", *(data->ampl_ant), *(data->fase_ant));
        }
      }


      n++;
    }
  // printf("nx = %d, ny = %d\n", nx, ny);

  /* ------------------------------------------------------- */
  /*                  End temporal iterations                */
  /* ------------------------------------------------------- */

  if (data->save_diag){

    // last time frame
    save_2d_arr_to_file(nx, ny, hz, f);

    fclose(f);
    fclose(f2);
  }

  /* Libera la memoria reservada */


  memory_free (hz, ny + 1); hz = NULL;
  memory_free (hzx, ny + 1); hzx = NULL;
  memory_free (hzy, ny + 1); hzy = NULL;
  memory_free (hz_inc, ny + 1); hz_inc = NULL;
  memory_free (hzx_inc, ny + 1); hzx_inc = NULL;
  memory_free (hzy_inc, ny + 1); hzy_inc = NULL;

  memory_free (chzx, ny + 1); chzx = NULL;
  memory_free (dhzx, ny + 1); dhzx = NULL;
  memory_free (chzy, ny + 1); chzy = NULL;
  memory_free (dhzy, ny + 1); dhzy = NULL;

  memory_free (sigmastarx, ny + 1); sigmastarx = NULL;
  memory_free (sigmastary, ny + 1); sigmastary = NULL;

  memory_free (jx, ny + 1); jx = NULL;
  memory_free (jy, ny + 1); jy = NULL;
  memory_free (cj, ny + 1); cj = NULL;
  memory_free (dj, ny + 1); dj = NULL;
  memory_free (wp2, ny + 1); wp2 = NULL;
  memory_free (omega, ny + 1); omega = NULL;

  memory_free (ey, ny + 1); ey = NULL;
  memory_free (ey_inc, ny + 1); ey_inc = NULL;
  memory_free (cey, ny + 1); cey = NULL;
  memory_free (dey, ny + 1); dey = NULL;
  memory_free (sigmax, ny + 1); sigmax = NULL;

  memory_free (ex, ny); ex = NULL;
  memory_free (ex_inc, ny); ex_inc = NULL;
  memory_free (cex, ny); cex = NULL;
  memory_free (dex, ny); dex = NULL;
  memory_free (sigmay, ny); sigmay = NULL;

  memory_free (I1_emi, ny + 1); I1_emi = NULL;
  memory_free (I2_emi, ny + 1); I2_emi = NULL;
  memory_free (I_emi, ny + 1); I_emi = NULL;
  memory_free (Q_emi, ny + 1); Q_emi = NULL;
  memory_free (I1_rec, ny + 1); I1_rec = NULL;
  memory_free (I2_rec, ny + 1); I2_rec = NULL;
  memory_free (I_rec, ny + 1); I_rec = NULL;
  memory_free (Q_rec, ny + 1); Q_rec = NULL;

  memory_free (ampl_emi, ny + 1); ampl_emi = NULL;
  memory_free (phase_emi, ny + 1); phase_emi = NULL;
  memory_free (ampl_rec, ny + 1); ampl_rec = NULL;
  memory_free (phase_rec, ny + 1); phase_rec = NULL;

  return 0;
}
