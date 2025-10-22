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
/* for a magnetized plasma in O-mode.                          */
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
/* to used in the computation.                                 */
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
#include "fdtd_2d_xmode.h"

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
      if (i <= NXPML-1)
	sigmax[j][i] = sigmamax*pow((double)(i + 0.5 - NXPML)/(double)(NXPML), 2.0);
      else if (i >= nx-NXPML)
	sigmax[j][i] = sigmamax*pow((double)(i + 0.5 - (nx-NXPML))/(double)(NXPML), 2.0);
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
      if (j <= NXPML-1)
	sigmay[j][i] = sigmamax*pow((double)(j + 0.5 - NXPML)/(double)(NXPML), 2.0);
      else if (j >= ny-NXPML)
	sigmay[j][i] = sigmamax*pow((double)(j + 0.5 - (ny-NXPML))/(double)(NXPML), 2.0);
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
      if (i <= NXPML)
	sigmastarx[j][i] = sigmamax*pow((double)(i - NXPML)/(double)(NXPML), 2.0)*Z0*Z0;
      else if (i >= nx-NXPML)
	sigmastarx[j][i] = sigmamax*pow((double)(i - (nx-NXPML))/(double)(NXPML), 2.0)*Z0*Z0;
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
      if (j <= NXPML)
	sigmastary[j][i] = sigmamax*pow((double)(j - NXPML)/(double)(NXPML), 2.0)*Z0*Z0;
      else if (j >= ny-NXPML)
	sigmastary[j][i] = sigmamax*pow((double)(j - (ny-NXPML))/(double)(NXPML), 2.0)*Z0*Z0;
      else
	sigmastary[j][i] = 0.0;
    }
  }
}


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
  double **ampl_inc, **phase_inc;

  double **I1_emi, **I2_emi, **I_emi, **Q_emi;
  double **I1_rec, **I2_rec, **I_rec, **Q_rec;
  double **ampl_emi, **phase_emi;
  double **ampl_rec, **phase_rec;

  int i, j, n;
  /* FILE *fp; */

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

  hz = hzx = hzy = hz_inc = hzx_inc = hzy_inc = NULL;
  chzx = dhzx = chzy = dhzy = sigmastarx = sigmastary = NULL;
  jx = jy = cj = dj = wp2 = omega = ey = ey_inc = cey = NULL;
  dey = sigmax = ex = ex_inc = cex = dex = sigmay = NULL;
  ampl_inc = phase_inc = I1_emi = I2_emi = I_emi = NULL;
  Q_emi = I1_rec = I2_rec = I_rec = Q_rec = ampl_emi = NULL;
  phase_emi = ampl_rec = phase_rec = NULL;


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
  wp2   = memory (ny + 1, nx + 1);
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
  set_sigmax (sigmax, ny + 1, nx, dx, nx);
  set_sigmay (sigmay, ny, nx + 1, dx, ny);
  set_sigmastarx (sigmastarx, ny + 1, nx + 1, dx, nx);
  set_sigmastary (sigmastary, ny + 1, nx + 1, dx, ny);

  /* Update coefficient for hz, ey, ex */
  set_update_coef_ex_ey (cex, dex, sigmay, ny, nx + 1, dt);
  set_update_coef_ex_ey (cey, dey, sigmax, ny + 1, nx, dt);
  set_update_coef_hzx_hzy (chzx, dhzx, sigmastarx, ny + 1, nx + 1, dt);
  set_update_coef_hzx_hzy (chzy, dhzy, sigmastary, ny + 1, nx + 1, dt);

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

  /* ------------------------------------------------------------ */
  /* --------------- Begin temporal iterations ------------------ */
  /* ------------------------------------------------------------ */
  
  n = 1;
  while (n <= nt) {

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
    for (j = NXPML + 1; j <= ny - NXPML - 1; j++)
      hz_inc[j][XANT] = (1.0-exp(-(double)(n)/200))*ampl_inc[j][0]*cos(wt+phase_inc[j][0]);

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
      for (j = NXPML + 1; j <= ny - NXPML - 1; j++) {
	I2_emi[j][0] = hz_inc[j][XANT];
	I2_rec[j][0] = hz[j][XANT];
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
	
	I_ant += ampl_emi[j][0]*ampl_rec[j][0]*cos(-phase_rec[j][0]-phase_inc[j][0])/(ny-1.0);
	Q_ant += ampl_emi[j][0]*ampl_rec[j][0]*sin(-phase_rec[j][0]-phase_inc[j][0])/(ny-1.0);

	*(data->ampl_ant) = sqrt(pow(I_ant, 2.0) + pow (Q_ant, 2.0));
	*(data->fase_ant) = atan2(Q_ant, I_ant);

      }
    }

    else {
      for (j = NXPML + 1; j <= ny - NXPML - 1; j++) {
	I1_emi[j][0] = hz_inc[j][XANT];
	I1_rec[j][0] = hz[j][XANT];
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
  /*
  fp = fopen("prueba.txt", "w");
  for (j = ny-1; j >= 0; j--){
    for (i = 0; i <= nx-1; i++){
      fprintf(fp, " %f", hz[j][i]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
  */
  
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

  memory_free (ampl_inc, ny + 1); ampl_inc = NULL;
  memory_free (phase_inc, ny + 1); phase_inc = NULL;

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

