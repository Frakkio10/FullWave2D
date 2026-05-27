/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/*                                                             */
/* Author: Emilio Blanco                                       */
/*                                                             */
/* Date:   March 1st, 2016                                     */
/*                                                             */
/* Routine: maxwell_2d_omode.c                                 */
/*                                                             */
/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/* This subroutine solves two-dimensional maxwell's equations  */
/* for a magnetized plasma in O-mode.                          */
/*                                                             */
/* The emitter/receiver antena plane is located at X = 0       */
/*                                                             */
/* #include "fdtd_2d.h"                                        */
/* maxwell_2d_omode (struct inputdata *data)                   */
/*                                                             */
/* To call: maxwell_2d_omode (&data);                          */
/*                                                             */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/*                                                             */
/* BE SURE YOU INCLUDE fdtd_2d.h file into your code. This     */
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

static void set_update_coef_hx_hy (double **chy, double **dhy, double **sigmastar,
			 int nfilas, int ncols, double dt) {
  int i, j;
  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      chy[j][i] = (1.0 - sigmastar[j][i]*dt/2.0/U0)/(1.0 + sigmastar[j][i]*dt/2.0/U0);
      dhy[j][i] = S/(1.0 + sigmastar[j][i]*dt/2.0/U0);
    }
  }
}



static void set_update_coef_ezx_ezy (double **cez, double **dez, double **sigma,
			 int nfilas, int ncols, double dt) {
  int i, j;
  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      cez[j][i] = (1.0 - sigma[j][i]*dt/2.0/E0)/(1.0 + sigma[j][i]*dt/2.0/E0);
      dez[j][i] = S/(1.0 + sigma[j][i]*dt/2.0/E0);
    }
  }

}


int maxwell_2d_omode (struct inputdata *data) {


  double **ez, **ezx, **ezy, **ez_inc, **ezx_inc, **ezy_inc;
  double **hy, **hx, **hy_inc, **hx_inc;
  double **jz, **wp2;
  double **chx, **dhx, **chy, **dhy;
  double **cezx, **dezx, **cezy, **dezy;
  double **sigmax, **sigmay, **sigmastarx, **sigmastary;


  double **I1_emi, **I2_emi, **I_emi, **Q_emi;
  double **I1_rec, **I2_rec, **I_rec, **Q_rec;
  double **ampl_emi, **phase_emi;
  double **ampl_rec, **phase_rec;

  int i, j, n;

  FILE *f, *f2, *f_inc, *f_anim;
  FILE *f_recv = NULL;

  int nt, nx, ny;

  double f0, dx, dt;
  double wdt, wt;
  double dfase, fase;
  double I_ant, Q_ant;

  int npml        = data->npml;
  double reflmax  = data->reflmax;
  int TFSF        = data->TFSF;
  int xante        = data->xante;
  int n_recv       = data->n_recv;
  int *yrecv       = data->yrecv;
  int recv_width   = data->recv_width;

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
  ez = ezx = ezy = ez_inc = ezx_inc = ezy_inc = NULL;
  hy = hx = hy_inc = hx_inc = jz = wp2 = NULL;
  chx = dhx = chy = dhy = cezx = dezx = cezy = dezy = NULL;
  sigmax = sigmay = sigmastarx = sigmastary = NULL;
  // ampl_inc = phase_inc = NULL;

  I1_emi = I2_emi = I_emi = Q_emi = I1_rec = I2_rec = I_rec = Q_rec = NULL;
  ampl_emi = phase_emi = ampl_rec = phase_rec = NULL;

  ez      = memory (ny + 1, nx + 1);
  ezx     = memory (ny + 1, nx + 1);
  ezy     = memory (ny + 1, nx + 1);
  ez_inc  = memory (ny + 1, nx + 1);
  ezx_inc = memory (ny + 1, nx + 1);
  ezy_inc = memory (ny + 1, nx + 1);
  cezx    = memory (ny + 1, nx + 1);
  dezx    = memory (ny + 1, nx + 1);
  cezy    = memory (ny + 1, nx + 1);
  dezy    = memory (ny + 1, nx + 1);
  sigmax  = memory (ny + 1, nx + 1);
  sigmay  = memory (ny + 1, nx + 1);
  jz      = memory (ny + 1, nx + 1);
  wp2     = memory (ny + 1, nx + 1);

  hy         = memory (ny + 1, nx);
  hy_inc     = memory (ny + 1, nx);
  chy        = memory (ny + 1, nx);
  dhy        = memory (ny + 1, nx);
  sigmastarx = memory (ny + 1, nx);

  hx         = memory (ny, nx + 1);
  hx_inc     = memory (ny, nx + 1);
  chx        = memory (ny, nx + 1);
  dhx        = memory (ny, nx + 1);
  sigmastary = memory (ny, nx + 1);



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
  set_sigma (sigmax, ny + 1, nx + 1, dx, nx, npml, reflmax, 'x', FALSE, 'O');
  set_sigma (sigmay, ny + 1, nx + 1, dx, ny, npml, reflmax, 'y', FALSE, 'O');
  set_sigma (sigmastarx, ny + 1, nx, dx, nx, npml, reflmax, 'x', TRUE, 'O');
  set_sigma (sigmastary, ny, nx + 1, dx, ny, npml, reflmax, 'y', TRUE, 'O');

  /* Update coefficient for ez, hy, hx */
  set_update_coef_ezx_ezy (cezx, dezx, sigmax, ny + 1, nx + 1, dt);
  set_update_coef_ezx_ezy (cezy, dezy, sigmay, ny + 1, nx + 1, dt);
  set_update_coef_hx_hy (chx, dhx, sigmastary, ny, nx + 1, dt);
  set_update_coef_hx_hy (chy, dhy, sigmastarx, ny + 1, nx, dt);




  /* Se plasma frequency squared wp2 */
  for (j = TFSF; j <= ny - TFSF; j++)
    for (i = TFSF; i <= nx - TFSF; i++)
	wp2[j][i] = ((data->ne)[j-TFSF][i-TFSF])*E*E/E0/ME;


  // initialize file in which to store some data during the loop
  if (data->save_diag){
    char fname[200];
    sprintf(fname, "%s/ez_t.dat", data->outp_dir);
    f = fopen(fname, "w");
		char fname_inc[200];
    sprintf(fname_inc, "%s/ez_t_inc.dat", data->outp_dir);
    f_inc = fopen(fname_inc, "w");
    char fname2[200];
    sprintf(fname2, "%s/ant_signal_t.dat", data->outp_dir);
    f2 = fopen(fname2, "w");
		char fname_anim[200];
		sprintf(fname_anim, "%s/ez_anim.dat", data->outp_dir);
		f_anim = fopen(fname_anim, "w");
    if (n_recv > 0) {
      char fname_recv[200];
      sprintf(fname_recv, "%s/recv_ampl_phase.dat", data->outp_dir);
      f_recv = fopen(fname_recv, "w");
    }
  }

  /* ------------------------------------------------------------ */
  /* --------------- Begin temporal iterations ------------------ */
  /* ------------------------------------------------------------ */

  n = 1;
  while (n <= nt) {

    if (n%100 == 0) printf("current time step: %d\n", n);

    /* --------------------------------------------------------- */
    /*   Calcula el campo electrico ez. Se excluye la frontera   */
    /* --------------------------------------------------------- */
    for (j = 1; j < ny; j++) {
      for (i = 1; i < nx; i++) {
        ezx[j][i] = cezx[j][i]*ezx[j][i] + dezx[j][i]*(hy[j][i] - hy[j][i-1] - dx*jz[j][i]);
        ezy[j][i] = cezy[j][i]*ezy[j][i] - dezy[j][i]*(hx[j][i] - hx[j-1][i]);
        ez[j][i]  = ezx[j][i] + ezy[j][i];
        // note that the '_inc' field is unaffected by the plasma because not
        // multiplied by dx * jz
        ezx_inc[j][i] = cezx[j][i]*ezx_inc[j][i] + dezx[j][i]*(hy_inc[j][i] - hy_inc[j][i-1]);
        ezy_inc[j][i] = cezy[j][i]*ezy_inc[j][i] - dezy[j][i]*(hx_inc[j][i] - hx_inc[j-1][i]);
        ez_inc[j][i]  = ezx_inc[j][i] + ezy_inc[j][i];
      }
    }

    /* -------------------------------------------------- */
    /*            Consistencia TF / SF para ez            */
    /* -------------------------------------------------- */
    /* Pared izquierda y derecha */
    for (j = TFSF; j <= ny - TFSF; j++) {
      ezx[j][TFSF]    -= dezx[j][TFSF]*hy_inc[j][TFSF-1];
      ez[j][TFSF]     -= dezx[j][TFSF]*hy_inc[j][TFSF-1];
      ezx[j][nx-TFSF] += dezx[j][nx-TFSF]*hy_inc[j][nx-TFSF];
      ez[j][nx-TFSF]  += dezx[j][nx-TFSF]*hy_inc[j][nx-TFSF];
    }
    
    /* Pared superior e inferior */
    for (i = TFSF; i <= nx - TFSF; i++) {
      ezy[TFSF][i] += dezy[TFSF][i]*hx_inc[TFSF-1][i];
      ez[TFSF][i]  += dezy[TFSF][i]*hx_inc[TFSF-1][i];
      ezy[ny-TFSF][i] -= dezy[ny-TFSF][i]*hx_inc[ny-TFSF][i];
      ez[ny-TFSF][i]  -= dezy[ny-TFSF][i]*hx_inc[ny-TFSF][i];      
    }
    

    /* -------------------------------------------------------- */
    /*     Conductor perfecto en los limites computacionales    */
    /* -------------------------------------------------------- */
    /* planos: x = 0, x = nx */
    for (j = 0; j < ny; j++) {
      ez[j][0] = ezx[j][0] = ezy[j][0] = ez_inc[j][0] = ezx_inc[j][0] = ezy_inc[j][0]  = 0.0;
      ez[j][nx]= ezx[j][nx]= ezy[j][nx]= ez_inc[j][nx]= ezx_inc[j][nx]= ezy_inc[j][nx] = 0.0;
    }

    /* planos: y = 0, y = ny */
    for (i = 0; i < nx; i++) {
      ez[0][i] = ezx[0][i] = ezy[0][i] = ez_inc[0][i] = ezx_inc[0][i] = ezy_inc[0][i]  = 0.0;
      ez[ny][i]= ezx[ny][i]= ezy[ny][i]= ez_inc[ny][i]= ezx_inc[ny][i]= ezy_inc[ny][i] = 0.0;
    }

    /* ------------------------------------------------- */
    /*              Source the input wave                */
    /* ------------------------------------------------- */
    // the first factor ensures a smooth rise of Ez
    for (j = npml + 1; j <= ny - npml - 1; j++)
      ez_inc[j][xante] = (1.0-exp(-(double)(n)/200))*data->ampl_inc[j][0]*cos(wt+data->phase_inc[j][0]);


    /* -------------------------------------------------- */
    /*            Calcula el campo magnetico hx           */
    /* -------------------------------------------------- */
    for (j = 0; j <= ny-1; j++) {
      for (i = 0; i <= nx ; i++) {
        hx[j][i] = chx[j][i]*hx[j][i] - dhx[j][i] * (ez[j+1][i] - ez[j][i]);
        hx_inc[j][i] = chx[j][i]*hx_inc[j][i] - dhx[j][i] * (ez_inc[j+1][i] - ez_inc[j][i]);
      }
    }

    /* ---------------------------------------------------- */
    /*             Calcula el campo magnetico hy            */
    /* ---------------------------------------------------- */
    for (j = 0; j <= ny; j++) {
      for (i = 0; i <= nx-1; i++) {
        hy[j][i] = chy[j][i]*hy[j][i] + dhy[j][i]*(ez[j][i+1] - ez[j][i]);
        hy_inc[j][i] = chy[j][i]*hy_inc[j][i] + dhy[j][i]*(ez_inc[j][i+1] - ez_inc[j][i]);
      }
    }

    /* ---------------------------------------------------- */
    /*            Consistencia TF/SF para hx y hy           */
    /* ---------------------------------------------------- */
/* Pared superior e inferior (hx) */
    for (i = TFSF; i <= nx - TFSF; i++) {
      hx[TFSF-1][i]  += dhx[TFSF-1][i]*ez_inc[TFSF][i];
      hx[ny-TFSF][i] -= dhx[ny-TFSF][i]*ez_inc[ny-TFSF][i];
    }

    /* Pared izquierda y derecha (hy) */
    for (j = TFSF; j < ny - TFSF; j++) {
      hy[j][TFSF-1]  -= dhy[j][TFSF-1]*ez_inc[j][TFSF];
      hy[j][nx-TFSF] += dhy[j][nx-TFSF]*ez_inc[j][nx-TFSF];
    }

    /* ----------------------------------------------------- */
    /*          Calcula la densidad de corriente jz          */
    /* ----------------------------------------------------- */
    for (j = 0; j <= ny; j++)
      for (i = 0; i <= nx; i++)
        jz[j][i] += dt*wp2[j][i]/C*ez[j][i];


    /* ----------------------------------------------------- */
    /*      Amplitude and phase in detection plane           */
    /* ----------------------------------------------------- */
    if ((n % 2) == 0) {     /* n es par */
      I_ant = 0.0;
      Q_ant = 0.0;
      for (j = npml + 1; j <= ny - npml - 1; j++) {
				// if desired, restrict the the collection surface of the antenna:
				// if (data->ampl_inc[j][0] > 1./2.){
	        I2_emi[j][0] = ez_inc[j][xante];
	        I2_rec[j][0] = ez[j][xante];
			// 	}
			// 	else{
			// 		I2_emi[j][0] = 0.0;
			// 		I2_rec[j][0] = 0.0;
			// 	}
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
        phase_rec[j][0] = atan2 (Q_rec[j][0], I_rec[j][0]);

        I_ant += ampl_emi[j][0]*ampl_rec[j][0]*cos(/*phase_emi[j][0]*/-phase_rec[j][0]-data->phase_inc[j][0])/(ny-1.0);
        Q_ant += ampl_emi[j][0]*ampl_rec[j][0]*sin(/*phase_emi[j][0]*/-phase_rec[j][0]-data->phase_inc[j][0])/(ny-1.0);

        *(data->ampl_ant) = sqrt(pow(I_ant, 2.0) + pow (Q_ant, 2.0));
        *(data->fase_ant) = atan2(Q_ant, I_ant);

      }
    }
    else {
      for (j = npml + 1; j <= ny - npml - 1; j++) {
				// if (data->ampl_inc[j][0] > 1./2.){
	        I1_emi[j][0] = ez_inc[j][xante];
	        I1_rec[j][0] = ez[j][xante];
				// }
				// else{
				// 	I1_emi[j][0] = 0.0;
				// 	I1_rec[j][0] = 0.0;
				// }
      }
    }
    /* PCR: per-receiver IQ collection */
    if ((n % 2) == 0 && n_recv > 0) {
        int r;
        for (r = 0; r < n_recv; r++) {
            double I_r = 0.0, Q_r = 0.0;
            int jmin = yrecv[r] - recv_width;
            int jmax = yrecv[r] + recv_width;
            if (jmin < npml + 1)       jmin = npml + 1;
            if (jmax > ny - npml - 1)  jmax = ny - npml - 1;
            for (j = jmin; j <= jmax; j++) {
                I_r += ampl_emi[j][0]*ampl_rec[j][0]
                      *cos(-phase_rec[j][0] - data->phase_inc[j][0]);
                Q_r += ampl_emi[j][0]*ampl_rec[j][0]
                      *sin(-phase_rec[j][0] - data->phase_inc[j][0]);
            }
            I_r /= (jmax - jmin + 1);
            Q_r /= (jmax - jmin + 1);
            data->ampl_recv[r] = sqrt(I_r*I_r + Q_r*Q_r);
            data->fase_recv[r] = atan2(Q_r, I_r);
        }
        if (data->save_diag && (n % 10) == 0) {
            for (r = 0; r < n_recv; r++)
                fprintf(f_recv, "%e %e ", data->ampl_recv[r], data->fase_recv[r]);
            fprintf(f_recv, "\n");
        }
    }

    wt += wdt;
    if (wt > PI)
      wt -= 2.0*PI;

      /* ----------------------------------------------------- */
      /*                    Diagnostics                        */
      /* ----------------------------------------------------- */

      // save diagnostics every few timesteps
      if (data->save_diag) {

				// time trace of the recieved signal (save every 10 steps):
				if ((n%10) == 0) {
					fprintf(f2, "%f %f\n", *(data->ampl_ant), *(data->fase_ant));
				}
        // uncomment for animations of the electric field:
        // if ((n%50) == 0) save_2d_arr_to_file(nx, ny, ez, f);

				// not tested sufficiently, might not working as expected:
        // if ((n%10) == 0) save_2d_arr_compressed(nx, ny, ez, 5, f_anim);

      }

      n++;
    }
  // printf("nx = %d, ny = %d\n", nx, ny);

  /* ------------------------------------------------------- */
  /*                  End temporal iterations                */
  /* ------------------------------------------------------- */


  /* Libera la memoria reservada */

  if (data->save_diag){

    // last time frame
    save_2d_arr_to_file(nx, ny, ez, f);
    save_2d_arr_to_file(nx, ny, ez_inc, f_inc);

    fclose(f);
    fclose(f2);
    fclose(f_inc);
    fclose(f_anim);
    if (f_recv) fclose(f_recv);
  }

  memory_free (ez, ny + 1);
  memory_free (ezx, ny + 1);
  memory_free (ezy, ny + 1);
  memory_free (ez_inc, ny + 1);
  memory_free (ezx_inc, ny + 1);
  memory_free (ezy_inc, ny + 1);
  memory_free (cezx, ny + 1);
  memory_free (dezx, ny + 1);
  memory_free (cezy, ny + 1);
  memory_free (dezy, ny + 1);
  memory_free (sigmax, ny + 1);
  memory_free (sigmay, ny + 1);
  memory_free (jz, ny + 1);
  memory_free (wp2, ny + 1);

  memory_free (hy, ny + 1);
  memory_free (hy_inc, ny + 1);
  memory_free (chy, ny + 1);
  memory_free (dhy, ny + 1);
  memory_free (sigmastarx, ny + 1);

  memory_free (hx, ny);
  memory_free (hx_inc, ny);
  memory_free (chx, ny);
  memory_free (dhx, ny);
  memory_free (sigmastary, ny);

  // memory_free (ampl_inc, ny + 1);
  // memory_free (phase_inc, ny + 1);

  memory_free (I1_emi, ny + 1);
  memory_free (I2_emi, ny + 1);
  memory_free (I_emi, ny + 1);
  memory_free (Q_emi, ny + 1);
  memory_free (I1_rec, ny + 1);
  memory_free (I2_rec, ny + 1);
  memory_free (I_rec, ny + 1);
  memory_free (Q_rec, ny + 1);

  memory_free (ampl_emi, ny + 1);
  memory_free (phase_emi, ny + 1);
  memory_free (ampl_rec, ny + 1);
  memory_free (phase_rec, ny + 1);

  ez = ezx = ezy = ez_inc = ezx_inc = ezy_inc = NULL;
  hy = hx = hy_inc = hx_inc = jz = wp2 = NULL;
  chx = dhx = chy = dhy = cezx = dezx = cezy = dezy = NULL;
  sigmax = sigmay = sigmastarx = sigmastary = NULL;
  // ampl_inc = phase_inc = NULL;

  I1_emi = I2_emi = I_emi = Q_emi = I1_rec = I2_rec = I_rec = Q_rec = NULL;
  ampl_emi = phase_emi = ampl_rec = phase_rec = NULL;

  return 0;
}
