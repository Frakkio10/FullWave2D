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
/* #include "fdtd_2d_omode.h"                                        */
/* maxwell_2d_omode (struct inputdata *data)                   */
/*                                                             */
/* To call: maxwell_2d_omode (&data);                          */
/*                                                             */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/*                                                             */
/* BE SURE YOU INCLUDE fdtd_2d_omode.h file into your code. This     */
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
#include "fdtd_2d_omode.h"

/* Velocidad de la luz en vacio (m/s) */
#define C 2.99792458e8

/* Carga del electron (C) */
#define E 1.602176487e-19

/* Masa del electron (kg) */
#define ME 9.10938215e-31

/* Permitividad del espacio libre (F/m) */
#define E0 8.854187817e-12

/* Permeabilidad magnetica del espacio libre (H/m) */
#define U0 1.2566370614e-6

/* Impedancia del espacio libre (ohm) */
#define Z0 376.730313461

/* Numero PI */
#define PI 3.141592653589

/* Numero de celdas de la capa PML */
#define NXPML 8

/* Total-Fields/Scattered-Fields interface position */
#define TFSF (NXPML + 10)

/* Posicion de la antena receptora */
#define XANT (TFSF - 1)

/* Maximo coeficiente de reflexion para capa PML */
#define REFLMAX 0.0001

/* Factor de estabilidad de Courant */
#define S 0.5


static double **memory (int filas, int cols) {
  /* Reserva memoria para alojar un array bi-dimensional con 'filas' */
  /* filas y 'cols' columnas. Devuelve un puntero a puntero.Cada     */
  /* elemento del array de punteros apunta a cada una de las filas   */
  /* del array bi-dimensional                                        */
  /* Ademas inicializa el array bidimensional con ceros              */
  int i, j;
  double **pf;

  /* Reserva memoria para el array de punteros */
  if ((pf = (double **) malloc (filas * sizeof (double *)))== NULL) {
    printf("Insuficiente memoria\n");
    exit(-1);
  }

  /* Reserva memoria para cada fila con cols columnas */
  for (j = 0; j < filas; j ++)
    if ((pf[j] = (double *)malloc (cols * sizeof(double))) == NULL) {
      printf("Insuficiente memoria. Exit\n");
      exit(-1);
    }

  /* Inicializa el array bidimensional a cero */
  for (j = 0; j < filas; j++)
    for (i = 0; i < cols; i++)
      pf[j][i] = 0.0;

  return pf;

}

static void memory_free (double **pf, int filas) {

  int j;
  for (j = 0; j < filas; j++) {
    free(pf[j]);
    pf[j] = NULL;
  }
  free (pf);
}


static void set_sigmax (double **sigmax, int nfilas, int ncols, double dx, int nx) {
  int i, j;
  double sigmamax = -3.0*log(REFLMAX)/(2.0*Z0*NXPML*dx);

  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      if (i <= NXPML)
	sigmax[j][i] = sigmamax*pow((double)(i - NXPML)/(double)(NXPML), 2.0);
      else if (i >= nx-NXPML)
	sigmax[j][i] = sigmamax*pow((double)(i - (nx-NXPML))/(double)(NXPML), 2.0);
      else
	sigmax[j][i] = 0.0;
    }
  }
}

static void set_sigmay (double **sigmay, int nfilas, int ncols, double dx, int ny) {
  int i, j;
  double sigmamax = -3.0*log(REFLMAX)/(2.0*Z0*NXPML*dx);

  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      if (j <= NXPML)
	sigmay[j][i] = sigmamax*pow((double)(j - NXPML)/(double)(NXPML), 2.0);
      else if (j >= ny-NXPML)
	sigmay[j][i] = sigmamax*pow((double)(j - (ny-NXPML))/(double)(NXPML), 2.0);
      else
	sigmay[j][i] = 0.0;
    }
  }
}


static void set_sigmastarx (double **sigmastarx, int nfilas, int ncols, double dx, int nx) {
  double sigmamax = -3.0*log(REFLMAX)/(2.0*Z0*NXPML*dx);
  int i, j;

  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      if (i <= NXPML-1)
	sigmastarx[j][i] = sigmamax*pow((double)(i + 0.5 - NXPML)/(double)(NXPML), 2.0)*Z0*Z0;
      else if (i >= nx-NXPML)
	sigmastarx[j][i] = sigmamax*pow((double)(i + 0.5 - (nx-NXPML))/(double)(NXPML), 2.0)*Z0*Z0;
      else
	sigmastarx[j][i] = 0.0;
    }
  }
}

static void set_sigmastary (double **sigmastary, int nfilas, int ncols, double dx, int ny) {
  double sigmamax = -3.0*log(REFLMAX)/(2.0*Z0*NXPML*dx);
  int i, j;

  for (j = 0; j <= nfilas - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {
      if (j <= NXPML-1)
	sigmastary[j][i] = sigmamax*pow((double)(j + 0.5 - NXPML)/(double)(NXPML), 2.0)*Z0*Z0;
      else if (j >= ny-NXPML)
	sigmastary[j][i] = sigmamax*pow((double)(j + 0.5 - (ny-NXPML))/(double)(NXPML), 2.0)*Z0*Z0;
      else
	sigmastary[j][i] = 0.0;
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

int maxwell_2d_omode (struct inputdata *data) {

  double **ez, **ezx, **ezy, **ez_inc, **ezx_inc, **ezy_inc;
  double **hy, **hx, **hy_inc, **hx_inc;
  double **jz, **wp2;
  double **chx, **dhx, **chy, **dhy;
  double **cezx, **dezx, **cezy, **dezy;
  double **sigmax, **sigmay, **sigmastarx, **sigmastary;
  double **ampl_inc, **phase_inc;

  double **I1_emi, **I2_emi, **I_emi, **Q_emi;
  double **I1_rec, **I2_rec, **I_rec, **Q_rec;
  double **ampl_emi, **phase_emi;
  double **ampl_rec, **phase_rec;

  int i, j, n;
  FILE *fp;

  int nt, nx, ny, yante, waist;
  double f0, dx, dt, angle;
  double wdt, wt;
  double dfase, fase;
  double I_ant, Q_ant;

  nx = (data->nx) - 1 + 2*TFSF; /* Numero de celdas eje X */
  ny = (data->ny) - 1 + 2*TFSF; /* Numero de celdas eje Y */
  nt = (data->nt);
  dx = (data->dx);
  angle = (data->angle)*PI/180.0;     /* angulo en radianes */
  yante = (data->yante) + TFSF;
  waist = (data->waist);
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
  sigmax = sigmay = sigmastarx = sigmastary = ampl_inc = phase_inc = NULL;

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

  ampl_inc   = memory (ny + 1, 1);
  phase_inc  = memory (ny + 1, 1);

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
  set_sigmax (sigmax, ny + 1, nx + 1, dx, nx);
  set_sigmay (sigmay, ny + 1, nx + 1, dx, ny);
  set_sigmastarx (sigmastarx, ny + 1, nx, dx, nx);
  set_sigmastary (sigmastary, ny, nx + 1, dx, ny);

  /* Update coefficient for ez, hy, hx */
  set_update_coef_ezx_ezy (cezx, dezx, sigmax, ny + 1, nx + 1, dt);
  set_update_coef_ezx_ezy (cezy, dezy, sigmay, ny + 1, nx + 1, dt);
  set_update_coef_hx_hy (chx, dhx, sigmastary, ny, nx + 1, dt);
  set_update_coef_hx_hy (chy, dhy, sigmastarx, ny + 1, nx, dt);

  /* Set amplitude and phase distribution at antena plane */
  dfase = 2.0*PI*f0/C*dx*sin(angle);
  fase = 0.0;
  for (j = 0; j <= ny; j++) {
    double aux;
    aux = cos(angle);
    aux = aux*(j - yante)/(double)(waist);
    aux = aux*aux;
    ampl_inc[j][0] = exp(-aux);
    phase_inc[j][0] = fase;
    fase -= dfase;
    if (fase < -PI)
      fase += 2.0*PI;
  }


  /* Se plasma frequency squared wp2 */
  for (j = TFSF; j <= ny - TFSF; j++)
    for (i = TFSF; i <= nx - TFSF; i++)
	wp2[j][i] = ((data->ne)[j-TFSF][i-TFSF])*E*E/E0/ME;

  /* ------------------------------------------------------------ */
  /* --------------- Begin temporal iterations ------------------ */
  /* ------------------------------------------------------------ */

  n = 1;
  while (n <= nt) {

    /* --------------------------------------------------------- */
    /*   Calcula el campo electrico ez. Se excluye la frontera   */
    /* --------------------------------------------------------- */
    for (j = 1; j < ny; j++) {
      for (i = 1; i < nx; i++) {
	ezx[j][i] = cezx[j][i]*ezx[j][i] + dezx[j][i]*(hy[j][i] - hy[j][i-1] - dx*jz[j][i]);
	ezy[j][i] = cezy[j][i]*ezy[j][i] - dezy[j][i]*(hx[j][i] - hx[j-1][i]);
	ez[j][i]  = ezx[j][i] + ezy[j][i];
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
    for (j = NXPML + 1; j <= ny - NXPML - 1; j++)
      ez_inc[j][XANT] = (1.0-exp(-(double)(n)/200))*ampl_inc[j][0]*cos(wt+phase_inc[j][0]);


    /* -------------------------------------------------- */
    /*            Calcula el campo magnetico hx           */
    /* -------------------------------------------------- */
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx + 1; i++) {
	hx[j][i] = chx[j][i]*hx[j][i] - dhx[j][i] * (ez[j+1][i] - ez[j][i]);
	hx_inc[j][i] = chx[j][i]*hx_inc[j][i] - dhx[j][i] * (ez_inc[j+1][i] - ez_inc[j][i]);
      }
    }

    /* ---------------------------------------------------- */
    /*             Calcula el campo magnetico hy            */
    /* ---------------------------------------------------- */
    for (j = 0; j < ny + 1; j++) {
      for (i = 0; i < nx; i++) {
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
      for (j = NXPML + 1; j <= ny - NXPML - 1; j++) {
	I2_emi[j][0] = ez_inc[j][XANT];
	I2_rec[j][0] = ez[j][XANT];
      }

      for (j = NXPML + 1; j <= ny - NXPML - 1; j++) {
	/* Calcula terminos I, Q */
	I_emi[j][0] = (I1_emi[j][0]*sin(wt)-I2_emi[j][0]*sin(wt-wdt))/sin(wdt);
	Q_emi[j][0] = (I1_emi[j][0]*cos(wt)-I2_emi[j][0]*cos(wt-wdt))/sin(wdt);

	I_rec[j][0] = (I1_rec[j][0]*sin(wt)-I2_rec[j][0]*sin(wt-wdt))/sin(wdt);
	Q_rec[j][0] = (I1_rec[j][0]*cos(wt)-I2_rec[j][0]*cos(wt-wdt))/sin(wdt);

	ampl_emi[j][0]  = sqrt(pow(I_emi[j][0], 2.0) + pow (Q_emi[j][0], 2.0));
	phase_emi[j][0] = atan2 (Q_emi[j][0], I_emi[j][0]);

	ampl_rec[j][0]  = sqrt(pow(I_rec[j][0], 2.0) + pow (Q_rec[j][0], 2.0));
	phase_rec[j][0] = atan2 (Q_rec[j][0], I_rec[j][0]);

	I_ant += ampl_emi[j][0]*ampl_rec[j][0]*cos(/*phase_emi[j][0]*/-phase_rec[j][0]-phase_inc[j][0])/(ny-1.0);
	Q_ant += ampl_emi[j][0]*ampl_rec[j][0]*sin(/*phase_emi[j][0]*/-phase_rec[j][0]-phase_inc[j][0])/(ny-1.0);

	*(data->ampl_ant) = sqrt(pow(I_ant, 2.0) + pow (Q_ant, 2.0));
	*(data->fase_ant) = atan2(Q_ant, I_ant);

      }
    }
    else {
      for (j = NXPML + 1; j <= ny - NXPML - 1; j++) {
	I1_emi[j][0] = ez_inc[j][XANT];
	I1_rec[j][0] = ez[j][XANT];
      }
    }

    wt += wdt;
    if (wt > PI)
      wt -= 2.0*PI;

    n++;
  }

  /* ------------------------------------------------------- */
  /*                  End temporal iterations                */
  /* ------------------------------------------------------- */

  fp = fopen("output.dat", "w");
  for (j = ny; j >= 0; j--){
    for (i = 0; i <= nx; i++){
      fprintf(fp, " %f", ez[j][i]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);


  /* Libera la memoria reservada */

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

  memory_free (ampl_inc, ny + 1);
  memory_free (phase_inc, ny + 1);

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
  sigmax = sigmay = sigmastarx = sigmastary = ampl_inc = phase_inc = NULL;

  I1_emi = I2_emi = I_emi = Q_emi = I1_rec = I2_rec = I_rec = Q_rec = NULL;
  ampl_emi = phase_emi = ampl_rec = phase_rec = NULL;


  return 0;

}
